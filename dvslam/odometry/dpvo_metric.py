"""DPVO + metric depth prior behind the Odometry interface (plan v3 items 1.1, 1.2; §4.2, §4.3, §4.5).

Scale convention. BA units = metric · e^g, with one global log-scale g:
  * new patch inverse depth z = 1 / (d_prior · e^g)                                   (§4.3 step 2)
  * BA unary on every patch with a prior: r = log(d_p · e^g) − log d,  w = λ · conf · Huber(r; 0.3)   (§4.5)
    (CUDA kernel in third_party/dpvo/dpvo/fastba/ba_cuda.cu, built as `cuda_ba_metric`)
  * emitted increment: ΔT_t from one BA state (as dpvo_mono), translation × e^{−g_t}, applied once (§4.2)
  * g is updated after the increment of frame t is emitted, so it only affects later increments.
(§4.5 writes the initialisation as "prior × e^{−g}"; that is inconsistent with its own unary and output rule,
 which both imply BA = metric · e^g. This module follows the unary/output rule. With g = 0 at start they agree.)

g filter: each frame passes slot n − KEYFRAME_INDEX of the graph exactly once, just before the cull test. There
m_k = Huber location of log(d_BA / d_prior) over its patches with a prior is taken (parallax gate: the
keyframe-test flow ≥ g_min_flow quarter-res px). g stays 0 (state BOOTSTRAP) until g_min_samples samples
exist, then g ← g + α · clip(m_k − g, ±g_clip), α = 1 / (g_tau_s · 30 Hz).

Depth schedule (§4.3), deterministic: until DPVO initialises (n == 8) depth runs synchronously on every frame.
Afterwards a simulated latest-wins worker takes the newest frame k whenever idle and its result arrives
`prior_lag` frames later (prior_lag = 0: synchronous on every frame). On arrival: if k is still a graph slot,
its patches get the prior at their centres (weight w0 = conf); every other slot of the last OPT window and
every newly created patch gets the latest depth map forward-warped (z-buffer, quarter resolution) with the
3×3 max/min ≤ 1.1 consistency test and weight w0 · exp(−age / prior_tau_s); a warped prior only replaces a
patch prior of lower weight. Prior / weight buffers live per patch slot and are shifted with the slots on
culling (A-N5).

tracking_state (F11): BOOTSTRAP before DPVO init and before g is valid; DEGRADED when the BA raised (upstream
swallowed it in a bare except, dpvo.py:355) or the newest pose is not finite; LOST after 3 consecutive
DEGRADED frames → the DPVO graph is reset (g kept) and re-bootstraps; increments while LOST or re-bootstrapping
repeat the last good increment (source motion_model).

1.2 (ring=1): fixed-capacity edge buffer of e_max slots (no per-frame torch.cat; when full, the edges whose
source patch is oldest are evicted first, i.e. an adaptive removal window), vectorised slot shift on culling
(no Python loop), the update operator on the padded buffer with static group tables and GPU neighbours
(dpvo/static_graph.py: no torch.unique, no CPU neighbour sort, no index.max() syncs), keyframe flow test with
one host sync. cuda_graph=1 additionally captures reproject → corr → update operator as one CUDA graph.

Options (all via --opt k=v): see DPVOMetricOdometry.__init__.
"""
import sys
import time
from pathlib import Path

import numpy as np

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from dvslam.odometry.dpvo_mono import (DEFAULT_CONFIG, DEFAULT_WEIGHTS, DPVO_ROOT, causal_increment,
                                       pose7_to_matrix, resolve_pose_cw)

STREAM_EXT = str(DPVO_ROOT / 'build/stream_ext')
FPS = 30.0
D_MIN, D_MAX = 0.1, 10.0          # valid prior depth range (m)
LOST_AFTER = 3                    # consecutive DEGRADED frames


# ---------------------------------------------------------------------------------------------- pure helpers
def shift_slots_loop(k: int, n: int, linear: list, ring: list) -> None:
    """Upstream keyframe() slot shift (dpvo.py:287-297): slots k+1..n-1 move down by one, one slot at a time.
    linear: tensors/arrays indexed by graph slot; ring: (tensor, modulus) pairs indexed by slot % modulus."""
    for i in range(k, n - 1):
        for t in linear:
            t[i] = t[i + 1]
        for t, mod in ring:
            t[i % mod] = t[(i + 1) % mod]


def shift_slots_vec(k: int, n: int, linear: list, ring: list) -> None:
    """Vectorised equivalent of shift_slots_loop (1.2). Valid while n - k < every ring modulus (no wrap
    collision); DPVO culls k = n - KEYFRAME_INDEX, so n - k = 4 < 36."""
    if n - 1 <= k:
        return
    for t in linear:
        t[k:n - 1] = t[k + 1:n].copy() if isinstance(t, np.ndarray) else t[k + 1:n].clone()
    for t, mod in ring:
        assert n - k < mod
        import torch
        dst = torch.arange(k, n - 1, device=t.device) % mod
        t[dst] = t[(dst + 1) % mod].clone()


def huber_location(x: np.ndarray, delta: float = 0.05, iters: int = 5) -> float:
    """Robust 1-D location (Huber M-estimate, IRLS from the median)."""
    mu = float(np.median(x))
    for _ in range(iters):
        a = np.abs(x - mu)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
        mu = float((w * x).sum() / w.sum())
    return mu


class GFilter:
    """Single global log-scale g (§4.5): fixed at 0 until `min_samples` gated samples, then a clipped
    first-order filter with time constant tau_s."""

    def __init__(self, tau_s: float = 10.0, min_samples: int = 5, clip: float = 0.05, enabled: bool = True):
        self.alpha = 1.0 / max(tau_s * FPS, 1.0)
        self.min_samples, self.clip, self.enabled = min_samples, clip, enabled
        self.g, self.samples = 0.0, []

    @property
    def valid(self) -> bool:
        return len(self.samples) >= self.min_samples

    def add(self, m: float) -> None:
        self.samples.append(m)
        if self.enabled and len(self.samples) > self.min_samples:
            self.g += self.alpha * float(np.clip(m - self.g, -self.clip, self.clip))


class DepthWorker:
    """Deterministic latest-wins depth thread simulation: idle → takes the newest frame k, result at k + lag."""

    def __init__(self, lag: int):
        self.lag = int(lag)
        self.job = None    # (frame, ready_at)

    def due(self, t: int):
        if self.job is not None and self.job[1] <= t:
            job, self.job = self.job, None
            return job[0]
        return None

    def submit(self, frame: Frame, t: int) -> None:
        if self.job is None:
            self.job = (frame, t + self.lag)


# ---------------------------------------------------------------------------------------------- engine
def _engine_class():
    """DPVO subclass; built lazily so CPU-only imports of this module never touch CUDA extensions."""
    import torch
    from dpvo import fastba
    from dpvo import projective_ops as pops
    from dpvo.dpvo import DPVO, Id, autocast
    from dpvo.lietorch import SE3
    from dpvo.static_graph import JJ_WIN, static_update

    class DPVOMetric(DPVO):
        def __init__(self, cfg, network, ht, wd, ring=False, e_max=0, prior_lambda=0.0, prior_huber=0.3,
                     cuda_graph=False):
            super().__init__(cfg, network, ht=ht, wd=wd, viz=False)
            assert not cfg.LOOP_CLOSURE and not cfg.CLASSIC_LOOP_CLOSURE
            N, M = self.N, self.M
            self.prior_d_ = torch.zeros(N, M, device='cuda')      # metric prior depth at the patch centre
            self.prior_w_ = torch.zeros(N, M, device='cuda')      # base weight (conf · decay); 0 = no prior
            self.prior_src_ = np.full(N, -1, dtype=np.int64)      # frame the slot's direct prior came from
            self.prior_decay_ = np.zeros(N)                       # 1 = direct prior, exp(-age/τ) = warped, 0 = none
            self.prior_lambda, self.prior_huber = float(prior_lambda), float(prior_huber)
            self.log_g = 0.0
            self.init_hook = None          # callable(engine, patches) -> None, may set patch depth + prior
            self.ba_calls = self.ba_failures = 0
            self.ba_failed_last = False
            self.kf_samples = []           # (tstamp, flow, z_BA (M,), d_prior (M,), w (M,)) per frame at slot n-4
            self.n_culled = 0
            self._lmbda = torch.as_tensor([1e-4], device='cuda')
            self.ring = bool(ring)
            self.cuda_graph = bool(cuda_graph) and self.ring
            self._graph = None
            if self.ring:
                E = int(e_max)
                assert E > 0
                self.E = E
                pg = self.pg
                pg.ii = torch.zeros(E, dtype=torch.long, device='cuda')
                pg.jj = torch.zeros(E, dtype=torch.long, device='cuda')
                pg.kk = torch.zeros(E, dtype=torch.long, device='cuda')
                pg.net = torch.zeros(1, E, self.DIM, **self.kwargs)
                pg.target = torch.zeros(1, E, 2, device='cuda')
                pg.weight = torch.zeros(1, E, 2, device='cuda')
                self.valid = torch.zeros(E, dtype=torch.bool, device='cuda')
                self.stamp = torch.zeros(E, dtype=torch.long, device='cuda')
                self.evicted = torch.zeros((), dtype=torch.long, device='cuda')
                self._jj_base = torch.zeros((), dtype=torch.long, device='cuda')   # n - JJ_WIN (graph input)

        # ---------------------------------------------------------------- edges (1.2 ring buffer)
        def n_edges(self) -> int:
            return int(self.valid.sum()) if self.ring else int(self.pg.ii.numel())

        def append_factors(self, ii, jj):
            if not self.ring:
                return super().append_factors(ii, jj)
            pg, E_new = self.pg, ii.numel()
            if E_new > self.E:
                raise ValueError(f'e_max={self.E} < {E_new} edges added by one frame (M={self.M})')
            prio = torch.where(self.valid, pg.ii * (1 << 40) + self.stamp, torch.full_like(pg.ii, -1))
            slots = torch.topk(prio, E_new, largest=False, sorted=False).indices
            self.evicted += (prio[slots] >= 0).sum()
            pg.ii[slots] = self.ix[ii]
            pg.jj[slots] = jj
            pg.kk[slots] = ii
            pg.net[:, slots] = 0
            self.valid[slots] = True
            self.stamp[slots] = self.counter

        def remove_factors(self, m, store: bool):
            if not self.ring:
                return super().remove_factors(m, store)
            self.valid &= ~m
            self._sanitize()

        def _sanitize(self):
            """Padding edges point at frame 0 / patch 0 (valid indices, zero weight, private groups)."""
            inv = ~self.valid
            self.pg.ii.masked_fill_(inv, 0)
            self.pg.jj.masked_fill_(inv, 0)
            self.pg.kk.masked_fill_(inv, 0)

        # ---------------------------------------------------------------- update + BA with prior
        def _net_step(self):
            """reproject → corr → update operator on the padded edge buffer (fixed shapes; graph-capturable)."""
            pg = self.pg
            coords = self.reproject()
            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, pg.kk % (self.M * self.pmem)]
                net, delta, weight = static_update(self.network.update, pg.net, ctx, corr, pg.ii, pg.jj, pg.kk,
                                                   self.valid, self.M * self.pmem, self._jj_base)
            pg.net.copy_(net)
            target = coords[..., self.P // 2, self.P // 2] + delta.float()
            weight = weight.float() * self.valid.view(1, -1, 1)
            return target, weight

        def _net_step_graphed(self):
            if self._graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                net0 = self.pg.net.clone()
                with torch.cuda.stream(s):
                    for _ in range(2):             # warm-up (allocator / cuBLAS handles) outside the graph
                        self._net_step()
                torch.cuda.current_stream().wait_stream(s)
                self.pg.net.copy_(net0)
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._g_target, self._g_weight = self._net_step()
                self.pg.net.copy_(net0)            # capture ran the step once; restore the state
            self._graph.replay()
            return self._g_target, self._g_weight

        def update(self):
            pg = self.pg
            if self.ring:
                self._jj_base.fill_(self.n - JJ_WIN)
                target, weight = self._net_step_graphed() if self.cuda_graph else self._net_step()
            else:
                coords = self.reproject()
                with autocast(enabled=True):
                    corr = self.corr(coords)
                    ctx = self.imap[:, pg.kk % (self.M * self.pmem)]
                    pg.net, (delta, weight, _) = self.network.update(pg.net, ctx, corr, None, pg.ii, pg.jj, pg.kk)
                weight = weight.float()
                target = coords[..., self.P // 2, self.P // 2] + delta.float()
            pg.target, pg.weight = target, weight

            self.ba_calls += 1
            self.ba_failed_last = False
            try:
                if not self.ring and (pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() \
                        and not self.ran_global_ba[self.n]:
                    self._DPVO__run_global_BA()   # never taken in VO mode (kept for parity with upstream)
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    pd = (self.prior_d_ * float(np.exp(self.log_g))).view(-1)
                    pw = self.prior_w_.view(-1)
                    prior_disp = torch.where(pw > 0, 1.0 / pd.clamp(min=1e-6), torch.zeros_like(pd))
                    fastba.BA_prior(self.poses, self.patches, self.intrinsics, target, weight, self._lmbda,
                                    pg.ii, pg.jj, pg.kk, t0, self.n, M=self.M, iterations=2,
                                    prior_disp=prior_disp, prior_w=pw, prior_lambda=self.prior_lambda,
                                    prior_huber=self.prior_huber, eff_impl=False)
            except Exception as e:  # upstream: bare except + print (dpvo.py:355-356). Recorded → DEGRADED (F11)
                self.ba_failures += 1
                self.ba_failed_last = True
                self.last_ba_error = repr(e)

            if not self.ring:   # visualisation point cloud, as upstream (kept so ring=0 timings stay faithful)
                points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
                points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
                pg.points_[:len(points)] = points[:]

        # ---------------------------------------------------------------- keyframe / culling
        def _flow_pair(self, i, j):
            """motionmag(i, j) + motionmag(j, i) over each frame's own M patches, one host sync (ring mode)."""
            M = self.M
            p = torch.arange(M, device='cuda')
            kk = torch.cat([i * M + p, j * M + p])
            ii = torch.cat([torch.full_like(p, i), torch.full_like(p, j)])
            jj = torch.cat([torch.full_like(p, j), torch.full_like(p, i)])
            flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
            return float(flow[:, :M].mean() + flow[:, M:].mean())

        def keyframe(self):
            pg, KI = self.pg, self.cfg.KEYFRAME_INDEX
            i, j = self.n - KI - 1, self.n - KI + 1
            m = self._flow_pair(i, j) if self.ring else self.motionmag(i, j) + self.motionmag(j, i)
            k = self.n - KI
            zdw = torch.stack([pg.patches_[k, :, 2, 1, 1], self.prior_d_[k], self.prior_w_[k]]).cpu().numpy()
            self.kf_samples.append((int(pg.tstamps_[k]), m / 2, *zdw))

            if m / 2 < self.cfg.KEYFRAME_THRESH:
                t0, t1 = pg.tstamps_[k - 1], pg.tstamps_[k]
                dP = SE3(pg.poses_[k]) * SE3(pg.poses_[k - 1]).inv()
                pg.delta[t1] = (t0, dP)

                if self.ring:
                    self.remove_factors(self.valid & ((pg.ii == k) | (pg.jj == k)), store=False)
                    above = pg.ii > k
                    pg.kk.sub_(above.long() * self.M)
                    pg.ii.sub_(above.long())
                    pg.jj.sub_((pg.jj > k).long())
                    self._sanitize()
                else:
                    to_remove = (pg.ii == k) | (pg.jj == k)
                    self.remove_factors(to_remove, store=False)
                    pg.kk[pg.ii > k] -= self.M
                    pg.ii[pg.ii > k] -= 1
                    pg.jj[pg.jj > k] -= 1

                linear = [pg.tstamps_, pg.colors_, pg.poses_, pg.patches_, pg.intrinsics_,
                          self.prior_d_, self.prior_w_, self.prior_src_, self.prior_decay_]   # A-N5: priors move with slots
                ring = [(self.imap_, self.pmem), (self.gmap_, self.pmem), (self.fmap1_[0], self.mem),
                        (self.fmap2_[0], self.mem)]
                (shift_slots_vec if self.ring else shift_slots_loop)(k, self.n, linear, ring)
                self.n -= 1
                self.m -= self.M
                self.n_culled += 1

            if self.ring:
                self.remove_factors(self.valid & (self.ix[pg.kk] < self.n - self.cfg.REMOVAL_WINDOW), store=True)
            else:
                to_remove = self.ix[pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
                self.remove_factors(to_remove, store=True)

        # ---------------------------------------------------------------- frame entry (upstream __call__)
        def __call__(self, tstamp, image, intrinsics):
            if (self.n + 1) >= self.N:
                raise Exception(f'The buffer size is too small ({self.N}).')
            image = 2 * (image[None, None] / 255.0) - 0.5
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                fmap, gmap, imap, patches, _, clr = self.network.patchify(
                    image, patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, return_color=True)

            self.tlist.append(tstamp)
            self.pg.tstamps_[self.n] = self.counter
            self.pg.intrinsics_[self.n] = intrinsics / self.RES
            clr = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)
            self.pg.index_[self.n + 1] = self.n + 1
            self.pg.index_map_[self.n + 1] = self.m + self.M

            if self.n > 1:
                if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                    P1 = SE3(self.pg.poses_[self.n - 1])
                    P2 = SE3(self.pg.poses_[self.n - 2])
                    *_, a, b, c = [1] * 3 + self.tlist
                    fac = (c - b) / (b - a)
                    xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                    self.pg.poses_[self.n] = (SE3.exp(xi) * P1).data
                else:
                    self.pg.poses_[self.n] = self.poses[self.n - 1]

            # upstream depth init (dpvo.py:426-430), then the prior overrides patches it covers (§4.3 step 2)
            patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
            if self.is_initialized:
                s = torch.median(self.pg.patches_[self.n - 3:self.n, :, 2])
                patches[:, :, 2] = s
            self.prior_d_[self.n] = 0
            self.prior_w_[self.n] = 0
            self.prior_src_[self.n] = -1
            self.prior_decay_[self.n] = 0.0
            if self.init_hook is not None:
                self.init_hook(self, patches)
            self.pg.patches_[self.n] = patches

            self.imap_[self.n % self.pmem] = imap.squeeze()
            self.gmap_[self.n % self.pmem] = gmap.squeeze()
            self.fmap1_[:, self.n % self.mem] = torch.nn.functional.avg_pool2d(fmap[0], 1, 1)
            self.fmap2_[:, self.n % self.mem] = torch.nn.functional.avg_pool2d(fmap[0], 4, 4)

            self.counter += 1
            if self.n > 0 and not self.is_initialized:
                if self.motion_probe() < 2.0:
                    self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                    return
            self.n += 1
            self.m += self.M

            self.append_factors(*self._DPVO__edges_forw())
            self.append_factors(*self._DPVO__edges_back())

            if self.n == 8 and not self.is_initialized:
                self.is_initialized = True
                for _ in range(12):
                    self.update()
            elif self.is_initialized:
                for _ in range(self.cfg.LATEST_UPDATES):
                    self.update()
                self.keyframe()

    return DPVOMetric


# ---------------------------------------------------------------------------------------------- odometry
class DPVOMetricOdometry:
    def __init__(self, patches: int = 96, seed: int = 0,
                 prior: str = 'perturbed', prior_bias: float = 0.02, prior_jitter: float = 0.03, prior_seed: int = 0,
                 prior_lambda: float = 10.0, prior_huber: float = 0.3, prior_lag: int = 3, prior_tau_s: float = 0.5,
                 init_from_prior: bool = True, warp_ratio: float = 1.1,
                 g_filter: bool = True, g_tau_s: float = 10.0, g_min_samples: int = 5, g_clip: float = 0.05,
                 g_min_flow: float = 0.5,
                 ring: bool = True, e_max: int = 0, cuda_graph: bool = False,
                 motion_damping: float = 0.5, latest_updates: int = 1,
                 weights: str | Path = DEFAULT_WEIGHTS, config: str | Path = DEFAULT_CONFIG, depth_ckpt: str = '',
                 depth_fp16: bool = False):
        """prior: none | perturbed | oracle (perturbed with b = 0, jitter 0) | da3 | dav2ft.
        prior_lambda: unary strength λ (BA weight units); e_max: edge-buffer capacity (0 = 400·M, no eviction
        in practice; ring mode only); ring=0 runs the upstream (torch.cat) edge path."""
        if str(DPVO_ROOT) not in sys.path:
            sys.path.insert(0, str(DPVO_ROOT))
        if cuda_graph:   # cuda_corr / lietorch_backends built to launch on the current stream (setup_stream.py)
            if 'lietorch_backends' in sys.modules and STREAM_EXT not in str(sys.modules['lietorch_backends'].__file__):
                raise RuntimeError('cuda_graph=1 needs the stream-aware extensions loaded before any DPVO import')
            sys.path.insert(0, STREAM_EXT)
        import torch
        from dpvo.config import cfg as base_cfg
        from dpvo.net import VONet

        self._torch = torch
        self.cfg = base_cfg.clone()
        self.cfg.merge_from_file(str(config))
        self.cfg.PATCHES_PER_FRAME = int(patches)
        self.cfg.MOTION_DAMPING = float(motion_damping)
        self.cfg.LATEST_UPDATES = int(latest_updates)
        self.seed = int(seed)
        self.prior_name = str(prior)
        self.prior_kw = dict(bias=float(prior_bias), jitter=float(prior_jitter), seed=int(prior_seed),
                             ckpt=str(depth_ckpt), fp16=bool(depth_fp16))
        self.use_prior = self.prior_name != 'none'
        self.prior_lambda = float(prior_lambda) if self.use_prior else 0.0
        self.prior_huber = float(prior_huber)
        self.prior_lag, self.prior_tau_s = int(prior_lag), float(prior_tau_s)
        self.init_from_prior, self.warp_ratio = bool(init_from_prior), float(warp_ratio)
        self.g_kw = dict(tau_s=float(g_tau_s), min_samples=int(g_min_samples), clip=float(g_clip),
                         enabled=bool(g_filter) and self.use_prior)
        self.g_min_flow = float(g_min_flow)
        self.ring, self.cuda_graph = bool(ring), bool(cuda_graph)
        self.e_max = int(e_max) if int(e_max) > 0 else 400 * int(patches)

        state = torch.load(str(weights), map_location='cpu')
        state = {k.replace('module.', ''): v for k, v in state.items() if 'update.lmbda' not in k}
        self.network = VONet()
        self.network.load_state_dict(state)
        self.network.cuda().eval()
        self._Engine = _engine_class()
        self._depth_model, self._depth_K = None, None
        self.slam = None

    # ------------------------------------------------------------------ depth prior
    def _depth_predictor(self, K: np.ndarray):
        name, kw = self.prior_name, self.prior_kw
        if name in ('perturbed', 'oracle'):
            from dvslam.depth.perturbed import PerturbedGTDepth
            b, j = (0.0, 0.0) if name == 'oracle' else (kw['bias'], kw['jitter'])
            return PerturbedGTDepth(bias=b, jitter_logsd=j, seed=kw['seed'])
        if self._depth_model is not None and np.allclose(self._depth_K, K):
            return self._depth_model
        if name == 'da3':
            from dvslam.depth import build
            self._depth_model = build('da3m', K, fp16=kw['fp16'])
        elif name == 'dav2ft':
            from dvslam.depth.dav2_ft import DAv2FT
            self._depth_model = DAv2FT(ckpt=kw['ckpt'] or None, K=K)
        else:
            raise ValueError(f'unknown prior {name!r}')
        self._depth_K = K.copy()
        return self._depth_model

    def _predict(self, frame: Frame):
        """→ (depth (H, W) cuda float32 metric, conf (H, W) cuda float32), timed separately (depth thread)."""
        torch = self._torch
        t0 = time.perf_counter()
        d, c = self.depth.predict(frame)
        d = torch.from_numpy(np.ascontiguousarray(d[:self.H, :self.W], dtype=np.float32)).cuda()
        c = torch.ones_like(d) if c is None else \
            torch.from_numpy(np.ascontiguousarray(c[:self.H, :self.W], dtype=np.float32)).cuda()
        ok = torch.isfinite(d) & (d > D_MIN) & (d < D_MAX)
        self.depth_ms.append((time.perf_counter() - t0) * 1e3)
        return torch.where(ok, d, torch.zeros_like(d)), torch.where(ok, c, torch.zeros_like(c))

    def _sample_centres(self, depth, conf, centres):
        """depth/conf at patch centres; centres (M, 2) quarter-res (x, y), full-res pixel = 4 · (x, y) (K / 4)."""
        x = (centres[:, 0] * 4).round().long().clamp(0, self.W - 1)
        y = (centres[:, 1] * 4).round().long().clamp(0, self.H - 1)
        d, c = depth[y, x], conf[y, x]
        return d, torch_where(self._torch, d > 0, c, 0.0)

    def _cloud(self, depth, conf):
        """Quarter-res back-projection of a depth map, once per map: X (P, 3) metric, valid (P,), conf (P,)."""
        torch = self._torch
        fx, fy, cx, cy = self.K4
        dq, cq = depth[::4, ::4], conf[::4, ::4]
        h, w = dq.shape
        vv, uu = torch.meshgrid(torch.arange(h, device='cuda', dtype=torch.float32),
                                torch.arange(w, device='cuda', dtype=torch.float32), indexing='ij')
        X = torch.stack([(uu - cx) / fx * dq, (vv - cy) / fy * dq, dq], -1).view(-1, 3)
        return X, (dq > 0).view(-1), cq.reshape(-1), (h, w)

    def _warp(self, cloud, T_tgt_src: np.ndarray, centres, g: float):
        """Forward-warp a back-projected depth map into a target frame (BA units, quarter res); z-buffer; 3×3
        max/min ≤ warp_ratio consistency at each patch centre. Fixed shapes, no host sync.
        Returns metric depth (M,), conf (M,) (0 = invalid)."""
        torch = self._torch
        fx, fy, cx, cy = self.K4
        X, valid, cq, (h, w) = cloud
        eg = float(np.exp(g))
        T = torch.as_tensor(T_tgt_src, dtype=torch.float32).cuda(non_blocking=True)
        Xt = (X * eg) @ T[:3, :3].T + T[:3, 3]
        zt = Xt[:, 2]
        zc = zt.clamp(min=1e-3)
        ut = torch.round(fx * Xt[:, 0] / zc + cx)
        vt = torch.round(fy * Xt[:, 1] / zc + cy)
        ok = valid & (zt > 1e-3) & (ut >= 0) & (ut < w) & (vt >= 0) & (vt < h)
        idx = torch.where(ok, vt * w + ut, torch.full_like(ut, h * w)).long()        # invalid -> spill slot
        zbuf = torch.full((h * w + 1,), float('inf'), device='cuda').scatter_reduce_(0, idx, zt, 'amin')
        cbuf = torch.zeros(h * w + 1, device='cuda').scatter_(0, idx, cq)
        zbuf, cbuf = zbuf[:-1].view(h, w), cbuf[:-1].view(h, w)
        xi = centres[:, 0].round().long().clamp(1, w - 2)
        yi = centres[:, 1].round().long().clamp(1, h - 2)
        o = torch.arange(-1, 2, device='cuda')
        win = zbuf[(yi[:, None, None] + o[None, :, None]), (xi[:, None, None] + o[None, None, :])].reshape(-1, 9)
        mn, mx = win.min(1).values, win.max(1).values
        good = torch.isfinite(win).all(1) & (mx <= self.warp_ratio * mn)
        d = torch.where(good, win.mean(1) / eg, torch.zeros_like(mn))
        c = torch.where(good, cbuf[yi, xi], torch.zeros_like(mn))
        return d, c

    def _pose_cw(self, ts: int, slot_of: dict, poses_cw: np.ndarray):
        try:
            return resolve_pose_cw(ts, slot_of, poses_cw, self._deltas())
        except KeyError:
            return None

    def _init_hook(self, eng, patches):
        """Called inside __call__ for the new slot n: set patch inverse depth and prior from the sync depth of
        this frame, else from the latest depth map warped with the motion-model pose (§4.3)."""
        torch = self._torch
        n, t = eng.n, eng.counter          # engine frame index of the new frame
        centres = patches[0, :, :2, 1, 1]
        d = None
        if self._cur_depth is not None:   # synchronous depth for this very frame
            d, c = self._sample_centres(*self._cur_depth, centres)
            w = c
            eng.prior_src_[n] = t
            eng.prior_decay_[n] = 1.0
        elif self._latest is not None and eng.is_initialized:
            k, _, _, cloud = self._latest
            slot_of, poses_cw = self._state()
            T_k = self._pose_cw(k, slot_of, poses_cw)
            if T_k is not None:
                T_n = pose7_to_matrix(eng.pg.poses_[n].double().cpu().numpy())
                d, c = self._warp(cloud, T_n @ np.linalg.inv(T_k), centres, eng.log_g)
                decay = float(np.exp(-abs(t - k) / FPS / self.prior_tau_s))
                w = c * decay
                eng.prior_decay_[n] = decay
        if d is None:
            return
        ok = d > 0
        eng.prior_d_[n] = d
        eng.prior_w_[n] = torch.where(ok, w, torch.zeros_like(w))
        if self.init_from_prior:
            z = 1.0 / (d.clamp(min=1e-6) * float(np.exp(eng.log_g)))
            cur = patches[0, :, 2]
            patches[0, :, 2] = torch.where(ok[:, None, None], z[:, None, None].expand_as(cur), cur)
            self.n_init_prior += ok.sum()          # GPU accumulator, read in finalize()
            self.n_init_total += int(ok.numel())

    def _attach(self, frame: Frame) -> None:
        """Async result for frame k arrived (§4.3, OK state): direct prior if k is still a slot, warp into the
        recent slots otherwise / additionally; keep it as the latest map for new patches."""
        torch, eng = self._torch, self.slam
        k = frame.index - self.offset
        depth, conf = self._predict(frame)
        cloud = self._cloud(depth, conf)
        self._latest = (k, depth, conf, cloud)
        pg, n = eng.pg, eng.n
        slot_of, poses_cw = self._state()
        T_k = self._pose_cw(k, slot_of, poses_cw)
        for s in range(max(n - eng.cfg.OPTIMIZATION_WINDOW - 2, 0), n):
            ts = int(pg.tstamps_[s])
            centres = pg.patches_[s, :, :2, 1, 1]
            if ts == k:
                d, w = self._sample_centres(depth, conf, centres)
                eng.prior_src_[s] = k
                eng.prior_decay_[s] = 1.0
                self.n_attach_direct += 1
            else:
                decay = float(np.exp(-abs(ts - k) / FPS / self.prior_tau_s))
                if T_k is None or decay <= eng.prior_decay_[s] + 1e-6:   # slot already has an equal/better prior
                    continue
                d, c = self._warp(cloud, poses_cw[s] @ np.linalg.inv(T_k), centres, eng.log_g)
                w = c * decay
                better = (d > 0) & (w > eng.prior_w_[s])
                d = torch.where(better, d, eng.prior_d_[s])
                w = torch.where(better, w, eng.prior_w_[s])
                eng.prior_decay_[s] = decay
                self.n_attach_warp += 1
            eng.prior_d_[s] = d
            eng.prior_w_[s] = torch.where(d > 0, w, torch.zeros_like(w))
        if k not in slot_of:
            self.n_attach_culled += 1

    # ------------------------------------------------------------------ protocol
    def _new_engine(self):
        torch = self._torch
        self.slam = None
        torch.cuda.empty_cache()
        self.slam = self._Engine(self.cfg, self.network, self.H, self.W, ring=self.ring, e_max=self.e_max,
                                 prior_lambda=self.prior_lambda, prior_huber=self.prior_huber,
                                 cuda_graph=self.cuda_graph)
        self.slam.log_g = self.g.g
        self.slam.init_hook = self._init_hook if self.use_prior else None
        self._delta_cache = {}
        self._latest = None
        self.worker = DepthWorker(self.prior_lag)

    def reset(self, camera: PinholeCamera) -> None:
        torch = self._torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        self.camera = camera
        self.H, self.W = camera.height - camera.height % 16, camera.width - camera.width % 16
        K = camera.K
        self.intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32, device='cuda')
        self.K4 = tuple(float(v) / 4 for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        self.depth = self._depth_predictor(K) if self.use_prior else None
        self.g = GFilter(**self.g_kw)
        self.offset = 0
        self._new_engine()
        self.inc, self.state, self.source = {}, {}, {}
        self.cuda_ms, self.wall_ms, self.edges, self.peak_mb, self.g_hist = {}, {}, {}, {}, {}
        self.depth_ms = []
        self._pending = (0.0, 0.0)
        self.consec_degraded = 0
        self.last_good = np.eye(4)
        self.n_resets = 0
        self.n_init_prior = self.n_init_total = 0
        self.n_attach_direct = self.n_attach_warp = self.n_attach_culled = 0
        self.ba_calls = self.ba_failures = 0
        self.n_culled = 0
        self.g_samples = []

    def step(self, frame: Frame) -> PoseEstimate | None:
        torch = self._torch
        import cv2

        img = cv2.imread(str(frame.image_path))
        img = torch.from_numpy(np.ascontiguousarray(img[:self.H, :self.W])).permute(2, 0, 1).cuda()
        t = frame.index
        eng = self.slam
        te = t - self.offset
        if te != eng.counter:
            raise RuntimeError(f'frame index {t} (engine {te}) != DPVO counter {eng.counter}')

        # depth schedule: sync until DPVO init (or prior_lag == 0), else deterministic async worker
        self._cur_depth = None
        if self.use_prior:
            if eng.is_initialized and self.prior_lag > 0:
                due = self.worker.due(t)
                if due is not None:
                    self._timed(lambda: self._attach(due))
            else:
                self._cur_depth = self._predict(frame)
                self._latest = (te, *self._cur_depth, self._cloud(*self._cur_depth))

        torch.cuda.reset_peak_memory_stats()
        self._timed(lambda: eng(te, img, self.intrinsics), t)
        self._cur_depth = None
        if self.use_prior and eng.is_initialized and self.prior_lag > 0:
            self.worker.submit(frame, t)
        self.edges[t] = eng.n_edges()
        self.peak_mb[t] = torch.cuda.max_memory_allocated() / 2**20

        # g samples (one per frame passing slot n-4), then the increment, then the g update (§4.2 rule)
        new_samples, eng.kf_samples = eng.kf_samples, []
        slot_of, poses_cw = self._state()
        timings = {'total': self.wall_ms[t], 'cuda_event': self.cuda_ms[t]}
        est = self._emit(frame, t, te, slot_of, poses_cw, timings)
        for ts, flow, z, dp, w in new_samples:
            ok = (w > 0) & (dp > 0) & (z > 0)
            if ok.sum() >= 5 and flow >= self.g_min_flow:
                m = huber_location(np.log(1.0 / z[ok]) - np.log(dp[ok]))
                self.g.add(m)
                self.g_samples.append((ts + self.offset, m))
        eng.log_g = self.g.g
        self.g_hist[t] = self.g.g
        return est

    def _emit(self, frame, t, te, slot_of, poses_cw, timings):
        eng = self.slam
        if t == 0:
            self.state[t], self.source[t] = 'BOOTSTRAP', 'motion_model'
            return None
        if not eng.is_initialized:
            dT = self.last_good if self.n_resets else np.eye(4)
            return self._record(t, frame, dT, 'LOST' if self.n_resets and te < 2 else 'BOOTSTRAP', 'motion_model',
                                False, timings)
        failed = eng.ba_failed_last or not np.isfinite(poses_cw).all()
        if failed:
            self.consec_degraded += 1
        else:
            self.consec_degraded = 0
        if self.consec_degraded >= LOST_AFTER:      # F11: reset, re-bootstrap, keep g
            self._accumulate_engine_stats()
            self.n_resets += 1
            self.offset = t + 1
            self._new_engine()
            self.consec_degraded = 0
            return self._record(t, frame, self.last_good, 'LOST', 'motion_model', False, timings)
        if te < 1 or (te - 1) not in slot_of and (te - 1) not in self._deltas():
            return self._record(t, frame, self.last_good, 'BOOTSTRAP', 'motion_model', False, timings)
        dT = causal_increment(resolve_pose_cw(te - 1, slot_of, poses_cw, self._deltas()),
                              resolve_pose_cw(te, slot_of, poses_cw, self._deltas()))
        if not np.isfinite(dT).all():
            return self._record(t, frame, self.last_good, 'DEGRADED', 'motion_model', False, timings)
        dT = dT.copy()
        dT[:3, 3] *= np.exp(-self.g.g)               # e^{-g}, applied exactly once (§4.2)
        if failed:
            state = 'DEGRADED'
        elif self.use_prior and not self.g.valid:
            state = 'BOOTSTRAP'
        else:
            state = 'OK'
        self.last_good = dT
        return self._record(t, frame, dT, state, 'tracked', te in slot_of, timings)

    def _record(self, t, frame, dT, state, source, is_kf, timings):
        self.inc[t], self.state[t], self.source[t] = dT, state, source
        return PoseEstimate(t, frame.timestamp, dT, tracking_state=state, source=source, is_keyframe=is_kf,
                            timings_ms=timings)

    def _timed(self, fn, t=None):
        """Tracker-thread work (prior attach/warp, then the DPVO step) timed by CUDA events and wall clock. The
        depth network itself is excluded (`_predict` logs it in depth_ms: it runs on the depth thread).
        Calls without t accumulate into the next call with t (the frame's total)."""
        torch = self._torch
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d0 = sum(self.depth_ms)
        w0 = time.perf_counter()
        ev0.record()
        with torch.no_grad():
            fn()
        ev1.record()
        torch.cuda.synchronize()
        dd = sum(self.depth_ms) - d0
        wall, cuda = (time.perf_counter() - w0) * 1e3 - dd, ev0.elapsed_time(ev1) - dd
        pw, pc = self._pending
        if t is None:
            self._pending = (pw + wall, pc + cuda)
        else:
            self._pending = (0.0, 0.0)
            self.wall_ms[t], self.cuda_ms[t] = wall + pw, cuda + pc

    def _accumulate_engine_stats(self):
        self.ba_calls += self.slam.ba_calls
        self.ba_failures += self.slam.ba_failures
        self.n_culled += self.slam.n_culled

    def finalize(self) -> dict:
        n = max(self.inc) + 1 if self.inc else 1
        idx = np.arange(n)
        self._accumulate_engine_stats()
        f = lambda d: np.array([d.get(i, np.nan) for i in idx], dtype=np.float64)  # noqa: E731
        evicted = int(self.slam.evicted) if self.ring else 0
        gs = np.array(self.g_samples, dtype=np.float64).reshape(-1, 2)
        return {
            'index': idx,
            'tracking_state': np.array([self.state.get(i, 'BOOTSTRAP') for i in idx]),
            'source': np.array([self.source.get(i, 'motion_model') for i in idx]),
            'cuda_event_ms': f(self.cuda_ms), 'wall_ms': f(self.wall_ms), 'edges': f(self.edges),
            'peak_gpu_mb': f(self.peak_mb), 'log_g': f(self.g_hist),
            'g_samples': gs,
            'depth_ms': np.array(self.depth_ms, dtype=np.float64),
            'ba_calls': np.array(self.ba_calls), 'ba_failures': np.array(self.ba_failures),
            'resets': np.array(self.n_resets), 'culled': np.array(self.n_culled),
            'init_from_prior_fraction': np.array(float(self.n_init_prior) / max(self.n_init_total, 1)),
            'attach_direct': np.array(self.n_attach_direct), 'attach_warp': np.array(self.n_attach_warp),
            'attach_culled': np.array(self.n_attach_culled), 'edges_evicted': np.array(evicted),
            'e_max': np.array(self.e_max if self.ring else -1),
        }

    # ------------------------------------------------------------------ helpers
    def _state(self):
        pg, n = self.slam.pg, self.slam.n
        slot_of = {int(pg.tstamps_[i]): i for i in range(n)}
        poses_cw = np.stack([pose7_to_matrix(v) for v in pg.poses_[:n].double().cpu().numpy()]) if n \
            else np.zeros((0, 4, 4))
        return slot_of, poses_cw

    def _deltas(self) -> dict:
        for ts, (t0, dP) in self.slam.pg.delta.items():
            if ts not in self._delta_cache:
                self._delta_cache[ts] = (int(t0), pose7_to_matrix(dP.data.reshape(-1).double().cpu().numpy()))
        return self._delta_cache


def torch_where(torch, cond, a, b: float):
    return torch.where(cond, a, torch.full_like(a, b))
