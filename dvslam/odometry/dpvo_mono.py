"""DPVO, monocular and depth-free, behind the Odometry interface (plan v3 item 0.6, E-mono-B).

Causal hook (§4.2): after each DPVO `__call__` (i.e. after `update()` and `keyframe()`), frames t-1 and t are
looked up in `pg.tstamps_`, their T_cw are read from `pg.poses_`, and the increment
ΔT_t = inv(T_wc_{t-1}) · T_wc_t = T_cw_{t-1} · inv(T_cw_t) is emitted from that single BA state. Culling only
removes slot n-4, so t-1 is still in the graph at time t; a frame that is not (removed earlier, or skipped by
the pre-initialisation motion probe) is resolved through DPVO's own fixed deltas `pg.delta`, exactly as
`terminate()` does. `terminate()` output is kept as `final_T_wc` in `finalize()` for reference only.

Pre-initialisation (DPVO initialises at n == 8 graph frames, after the motion probe accepts them): there is no
BA state, so the increment is identity with tracking_state='BOOTSTRAP', source='motion_model'. The pair in which
initialisation completes is the first 'OK' / 'tracked' one. eval_pairs's tracked-only figure excludes BOOTSTRAP.

Scale is arbitrary (Sim(3) gauge); apps/eval_pairs.py's windowed rescale (G0 oracle) fits it per 1 s / 2 s.

Options: patches (M), seed, motion_damping (upstream 0.5), latest_updates (update() calls per new frame,
upstream 1), weights, config (DPVO yaml), rho_lag (A-N3 re-measure lag, 4).
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate

REPO_ROOT = Path(__file__).resolve().parents[2]
DPVO_ROOT = REPO_ROOT / 'third_party/dpvo'
DEFAULT_WEIGHTS = DPVO_ROOT / 'models/dpvo.pth'
DEFAULT_CONFIG = DPVO_ROOT / 'config/default.yaml'


def pose7_to_matrix(v) -> np.ndarray:
    """lietorch SE3 data (tx, ty, tz, qx, qy, qz, qw) -> 4x4 float64."""
    v = np.asarray(v, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def resolve_pose_cw(ts: int, slot_of: dict, poses_cw: np.ndarray, deltas: dict) -> np.ndarray:
    """T_cw of frame `ts` in the current DPVO state.

    slot_of: frame index -> graph slot; poses_cw: (n, 4, 4) T_cw per slot; deltas: frame index -> (t0, dP) with
    T_cw(ts) = dP · T_cw(t0) (DPVO `pg.delta`, as `DPVO.get_pose` uses it).
    """
    chain = np.eye(4)
    while ts not in slot_of:
        t0, dP = deltas[ts]
        chain = chain @ dP
        ts = t0
    return chain @ poses_cw[slot_of[ts]]


def causal_increment(T_cw_prev: np.ndarray, T_cw_cur: np.ndarray) -> np.ndarray:
    """ΔT = inv(T_wc_prev) · T_wc_cur = T_cw_prev · inv(T_cw_cur): pose of camera t in camera t-1."""
    return T_cw_prev @ np.linalg.inv(T_cw_cur)


class DPVOMonoOdometry:
    def __init__(self, patches: int = 96, seed: int = 0, motion_damping: float = 0.5, latest_updates: int = 1,
                 weights: str | Path = DEFAULT_WEIGHTS, config: str | Path = DEFAULT_CONFIG, rho_lag: int = 4):
        if str(DPVO_ROOT) not in sys.path:
            sys.path.insert(0, str(DPVO_ROOT))   # vendored package + in-place built extensions
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
        self.rho_lag = int(rho_lag)

        state = torch.load(str(weights), map_location='cpu')
        state = {k.replace('module.', ''): v for k, v in state.items() if 'update.lmbda' not in k}
        self.network = VONet()
        self.network.load_state_dict(state)
        self.network.cuda().eval()
        self.slam = None

    # ------------------------------------------------------------------ protocol
    def reset(self, camera: PinholeCamera) -> None:
        torch = self._torch
        from dpvo.dpvo import DPVO

        self.slam = None
        torch.cuda.empty_cache()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        self.camera = camera
        self.H, self.W = camera.height - camera.height % 16, camera.width - camera.width % 16
        K = camera.K
        self.intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32, device='cuda')
        self.slam = DPVO(self.cfg, self.network, ht=self.H, wd=self.W, viz=False)
        self._delta_cache = {}
        self.inc, self.bootstrap, self.probe_skipped = {}, {}, {}
        self.cuda_ms, self.wall_ms, self.edges, self.peak_mb = {}, {}, {}, {}
        self.rho, self.inc_lag = {}, {}

    def step(self, frame: Frame) -> PoseEstimate | None:
        torch = self._torch
        import cv2

        img = cv2.imread(str(frame.image_path))   # BGR, as DPVO's own image_stream feeds it
        img = torch.from_numpy(np.ascontiguousarray(img[:self.H, :self.W])).permute(2, 0, 1).cuda()
        t = frame.index
        if t != self.slam.counter:
            raise RuntimeError(f'frame index {t} != DPVO counter {self.slam.counter}; frames must be 0, 1, 2, ...')

        n_before = self.slam.n
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        w0 = time.perf_counter()
        ev0.record()
        with torch.no_grad():
            self.slam(t, img, self.intrinsics)
        ev1.record()
        torch.cuda.synchronize()
        self.wall_ms[t] = (time.perf_counter() - w0) * 1e3
        self.cuda_ms[t] = ev0.elapsed_time(ev1)
        self.peak_mb[t] = torch.cuda.max_memory_allocated() / 2**20
        self.edges[t] = int(self.slam.pg.ii.numel())
        self.probe_skipped[t] = (not self.slam.is_initialized) and self.slam.n == n_before

        slot_of, poses_cw = self._state()
        if self.slam.is_initialized and t - self.rho_lag >= 1:
            self._remeasure(t - self.rho_lag, slot_of, poses_cw)
        if t == 0:
            return None

        timings = {'total': self.wall_ms[t], 'cuda_event': self.cuda_ms[t]}
        if not self.slam.is_initialized:
            self.bootstrap[t] = True
            self.inc[t] = np.eye(4)
            return PoseEstimate(t, frame.timestamp, np.eye(4), tracking_state='BOOTSTRAP', source='motion_model',
                                timings_ms=timings)
        self.bootstrap[t] = False
        dT = causal_increment(resolve_pose_cw(t - 1, slot_of, poses_cw, self._deltas()),
                              resolve_pose_cw(t, slot_of, poses_cw, self._deltas()))
        self.inc[t] = dT
        return PoseEstimate(t, frame.timestamp, dT, tracking_state='OK', source='tracked', is_keyframe=t in slot_of,
                            timings_ms=timings)

    def finalize(self) -> dict:
        """Per-frame diagnostics (index = frame t; pair values refer to the pair (t-1, t), NaN at t = 0)."""
        n = self.slam.counter
        idx = np.arange(n)
        boot = np.array([self.bootstrap.get(i, True) for i in idx])
        skipped = np.array([self.probe_skipped.get(i, False) for i in idx])
        deltas = self.slam.pg.delta
        post_init = ~boot & (idx > 0)
        culled = np.array([i in deltas for i in idx]) & ~skipped & post_init
        rho = np.array([self.rho.get(i, np.nan) for i in idx])
        inc_lag = np.stack([self.inc_lag.get(i, np.full((4, 4), np.nan)) for i in idx])
        with self._torch.no_grad():
            poses, _ = self.slam.terminate()   # reference only, never scored (§4.2)
        final_T_wc = np.stack([pose7_to_matrix(p) for p in poses])
        f = lambda d: np.array([d.get(i, np.nan) for i in idx], dtype=np.float64)  # noqa: E731
        return {
            'index': idx,
            'preinit_pair': boot & (idx > 0),
            'probe_skipped': skipped,
            'culled': culled,
            'culled_fraction': np.array(culled.sum() / max(post_init.sum(), 1)),
            'cuda_event_ms': f(self.cuda_ms),
            'wall_ms': f(self.wall_ms),
            'edges': f(self.edges),
            'peak_gpu_mb': f(self.peak_mb),
            'rho': rho,
            'rel_increment_lag': inc_lag,
            'rho_lag': np.array(self.rho_lag),
            'final_T_wc': final_T_wc,
        }

    # ------------------------------------------------------------------ helpers
    def _state(self):
        pg, n = self.slam.pg, self.slam.n
        slot_of = {int(pg.tstamps_[i]): i for i in range(n)}
        poses_cw = np.stack([pose7_to_matrix(v) for v in pg.poses_[:n].double().cpu().numpy()]) if n else np.zeros((0, 4, 4))
        return slot_of, poses_cw

    def _deltas(self) -> dict:
        for ts, (t0, dP) in self.slam.pg.delta.items():   # each entry is written once and never changed
            if ts not in self._delta_cache:
                self._delta_cache[ts] = (int(t0), pose7_to_matrix(dP.data.reshape(-1).double().cpu().numpy()))
        return self._delta_cache

    def _remeasure(self, t: int, slot_of: dict, poses_cw: np.ndarray) -> None:
        """A-N3: ρ_t = ‖causal ΔT_t‖ / ‖ΔT_t re-read from the BA state rho_lag frames later‖ (tracked pairs only)."""
        if self.bootstrap.get(t, True):
            return
        d = self._deltas()
        lag = causal_increment(resolve_pose_cw(t - 1, slot_of, poses_cw, d), resolve_pose_cw(t, slot_of, poses_cw, d))
        self.inc_lag[t] = lag
        den = np.linalg.norm(lag[:3, 3])
        self.rho[t] = np.linalg.norm(self.inc[t][:3, 3]) / den if den > 0 else np.nan
