"""KLT + sliding-window BA with a depth-prior factor and one global log-scale g: tracker "C" (plan v3 1.3, §4.2, §4.5).

Extends the 0.7 tracker (`KLTPnPOdometry`). Its front end, PnP pair solve and ray-anchored frame-to-frame refinement
are reused unchanged.
  keyframes  : new keyframe when tracks < kf_ratio · n_features, median flow since the last keyframe > kf_px, or
               kf_max frames passed. Live corners are re-tracked host keyframe -> new keyframe (kf_refine) so
               keyframe observations carry no chained-LK drift.
  landmarks  : anchored at the host keyframe (pixel + log inverse depth x). New landmarks start at d_p · ŝ_k, where
               ŝ_k is the robust BA/prior ratio of older landmarks seen in that keyframe (kf_align; initial value only).
               Corners without a valid prior are triangulated (DLT) once the ray angle to a later keyframe > tri_deg.
  window BA  : last `window` keyframes (oldest pose fixed) plus their landmarks. Residuals: reprojection (/ px_sigma)
               and the prior factor r = log d_p − log d (/ σ_j), Huber. scipy `least_squares` (trf, lsmr) with an
               analytic sparse Jacobian. prior_per_kf: the landmarks of one host share a weight of 1/prior_sigma²
               (σ_j = prior_sigma·sqrt(n_host)), because the prior error is one jitter per frame, not per pixel.
               An update that moves a pose > ba_max_move_m / ba_max_turn_deg is discarded (divergence guard).
               S4 options (default off): prior_obs scales the window's prior weight by w = clip(prior_obs_ref / θ_w,
               prior_obs_min, prior_obs_max) (σ_j / sqrt(w)), θ_w = median over prior landmarks of the largest
               host-to-observer ray angle (deg) before the BA: low parallax → stronger prior. One weight per window
               BA (MegaSaM weights once per video; no per-window precedent). The global scale direction itself is
               an exact gauge of the reprojection term (Hessian eigenvalue 0), so parallax measures how well the
               geometry fixes the relative depths, not the scale. prior_loss ∈ {huber, cauchy} replaces the prior
               residual r = log d_p − log d by ψ(r) (robust_log_residual, scale prior_loss_c in log units); the
               BA's shared ba_loss / huber still applies on top of ψ / σ_j.
               ba_guard (default off, EXP-003): depths are clipped to [BA_DEPTH_LO, BA_DEPTH_HI] inside the residual
               (a zero-parallax landmark without prior can run x → −∞; exp(−x) then overflows in trial steps, which
               trf rejects at the cost of evaluations) and landmarks accepted outside that range become unmapped.
  g          : one global log-scale, BA units = metric · e^g. After each BA, m_k = median(log d_BA − log d_p) over
               parallax-gated landmarks; after g_min_m samples g ← g + (m_k − g) / g_tau. Earlier increments are
               tagged BOOTSTRAP. Deviation from §4.5: g is NOT inside the prior factor by default (g_in_prior=False).
               With g in the factor, the BA is pulled to d_p·e^g, m_k − g only measures the visual/prior disagreement,
               and the filter integrates it into a random walk (measured: g → −0.39 over 600 frames).
  output     : ΔT_t = inv(T_{t−1}) · T_t from one map state at t (BA units), then translation × e^{−g} once (§4.2).
               g and BA updates happen after ΔT_t is emitted. An increment that jumps > max_jump_m / max_jump_deg from
               the previous one is treated as a PnP failure (motion model, DEGRADED). jump_ref_tracked (default
               off, EXP-003): the gate applies only when the previous pair was tracked. Without it a wrong increment
               that passed the gate is repeated by the motion model and every later correct solve differs from it by
               > max_jump_m, so the tracker never recovers (measured: room4 with quality=0.001 lk_win=15, 1,500 pairs).

Prior: `prior` ∈ {perturbed (b = 2 %, jitter 3 %), oracle (GT depth), da3 (DA3METRIC-L), dav2_ft, ssl, cache (maps
cached on disk by scripts/exp003_cache_depth.py; prior_ckpt = the cache directory)}. `prior_seed`
seeds the perturbed jitter. `ba=False kf_align=False` is the 0.7 sensor-depth path with a prior (baseline).
EXP-007 front-end options (default off): flow_cache=<dir> moves tracks by dense flow cached offline instead of LK
(klt_flow.py; diagnostic); flow_cache=raft_small|raft_large evaluates torchvision RAFT on the GPU when needed
(EXP-008); photo=all|fallback refines the output increment of every pair /
of motion-model pairs only by direct photometric LM on exposure- and vignette-normalised images with the SSL depth
of frame t-1 rescaled to map units (klt_photo.py). A refined increment is kept only when its robust cost dropped,
>= 50 points remain and it moved < photo_max_m (metric) / photo_max_deg from the tracker's increment; on a tracked
pair only the output changes (the map pose stays the PnP one), on a motion-model pair the pose follows it.
EXP-008: photo_tol / photo_max_pts (early stop, point cap); photo_ref_max > 1 refines frame t against a photometric
reference frame (anchored at t-1 when the reference is older than photo_ref_max frames or a refinement was rejected)
and outputs the difference of the two reference-relative poses; 1 = the pair refinement. flow_mode='hybrid' uses the
flow only to re-acquire tracks on frames where LK keeps < flow_hybrid_frac of them.
EXP-013 (CONDITIONAL, default off): out_lag=L emits pair (k-1, k) after frame k+L (klt_lag.py; smooth = fixed-lag
constant-acceleration RTS on the step, window = causal increment corrected by the later window poses); flush() emits
the last L pairs at the end of a sequence, so drivers must call it (scripts/exp003_track_log.py does).
EXP-018 (default off): photo_bidir=True also refines the pair from frame t to frame t-1 (points and SSL depth of frame
t, the same map-unit ratio); each direction passes the acceptance test on its own, and when both pass the output is
their midpoint (mean translation, geodesic rotation midpoint). Needs photo_ref_max=1. Asks the prior for frame t's
depth one step earlier than the next anchor does (a cached prior is a lookup; a live network prior runs twice).
S1 (plan v3.4 §14.4, default numpy): photo_backend=torch|torch32 runs the photometric LM with klt_photo_torch.py
(GPU when available; per-point work in float64 / float32). Points, depths and acceptance are unchanged.
S2 (plan v3.4 §14.4, default off): photo_feat=<ckpt> runs that LM with a learned PhotoNet (photo_learned.py:
per-pixel weights and / or an alignment feature of each frame); needs photo_backend=torch|torch32 and photo_levels=1
(the net is trained on level 1 only). photo_feat / photo_backend with photo=off is a ValueError.
EXP-011: wide=<size> tracks on tumvi.wide_K(size) images (same f = 190 and principal ray, larger canvas; the central
512 crop equals the standard image); the depth prior is still predicted on the standard image (built with K_OUT and
512 x 512) and embedded in the canvas centre (no prior outside: those corners are triangulated); wide_fov_deg caps
detection and tracks by angle. wide with a flow_cache directory is a ValueError (cached flow is of the standard image).
G0c (plan v3.4 §14.4, default off): pt_subset=left|right|blocks_a|blocks_b|random_a|random_b runs the tracker on one
point subset (subset_mask): corners are detected only inside the mask, the photometric points (both directions) are
only those inside it, and n_features is halved (every subset covers half the image: same corner density as the full
tracker). Region subsets also drop a track once it leaves the mask (as wide_fov_deg), so the two halves never share a
point; random subsets keep a track wherever it moves (membership is fixed at detection).
Threads: OpenCV's own pool (pthreads) ignores OMP_NUM_THREADS and used ~8.7 cores for a job that needs ~1.6 (EXP-003,
same output, same speed). reset() caps it with cv2.setNumThreads, which is PROCESS-WIDE (every later OpenCV call in
the process, not only this tracker; deliberate, it protects the shared workstation): cv_threads=None (default) uses
OMP_NUM_THREADS when that is set, else leaves OpenCV's pool alone; cv_threads=N > 0 sets N; cv_threads=0 leaves the
pool as it is (it does NOT call cv2.setNumThreads(0), which would disable OpenCV threading).
`head_inputs` (BA-unit ΔT^raw, g, BA depths of frame t−1) is exposed for the precision head (klt_head.py).
"""
import os
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from dvslam.odometry.klt_pnp import _STATE_CODE, KLTPnPOdometry

BA_DEPTH_LO, BA_DEPTH_HI = 0.01, 1000.0    # metres; ba_guard range of landmark depths in the window BA
PT_SUBSETS = ('off', 'left', 'right', 'blocks_a', 'blocks_b', 'random_a', 'random_b')


def subset_mask(mode: str, w: int, h: int, block: int = 128, cell: int = 16, seed: int = 0) -> np.ndarray:
    """G0c point subset: (h, w) bool, True where corners are detected and photometric points used.
    left / right: column < w / 2 / the rest. blocks_a / blocks_b: the two colours of a block×block checkerboard.
    random_a / random_b (control): every cell×cell cell is kept with probability 1/2, independently in a and b (seed
    streams (seed, 0) / (seed, 1)): each subset holds half the image like a region subset, and the two overlap in a
    quarter of the cells (half of each), as two independent draws would."""
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    if mode in ('left', 'right'):
        return (uu < w / 2) == (mode == 'left')
    if mode in ('blocks_a', 'blocks_b'):
        return ((uu // block + vv // block) % 2 == 0) == (mode == 'blocks_a')
    if mode in ('random_a', 'random_b'):
        keep = np.random.default_rng([seed, int(mode == 'random_b')]).random((-(-h // cell), -(-w // cell))) < 0.5
        return keep[vv // cell, uu // cell]
    raise ValueError(f'pt_subset must be one of {PT_SUBSETS[1:]}, got {mode!r}')


def make_prior(name: str, K: np.ndarray, seed: int = 0, bias: float = 0.02, jitter: float = 0.03,
               ckpt: str | None = None, hw: tuple[int, int] | None = None):
    """hw = (H, W) of the image the prior predicts on; only `cache` uses it (checked against <scene>.meta.json)."""
    if name == 'perturbed':
        from dvslam.depth.perturbed import PerturbedGTDepth
        return PerturbedGTDepth(bias=bias, jitter_logsd=jitter, seed=seed)
    if name == 'oracle':
        from dvslam.depth.perturbed import PerturbedGTDepth
        return PerturbedGTDepth(bias=0.0, jitter_logsd=0.0, seed=seed)
    if name == 'da3':
        from dvslam.depth import build
        return build('da3m', K)
    if name == 'dav2_ft':
        from dvslam.depth.dav2_ft import DAv2FT
        return DAv2FT(ckpt=ckpt, K=K)
    if name == 'ssl':
        from dvslam.depth.ssl import SSLDepth
        return SSLDepth(ckpt=ckpt, K=K)
    if name == 'cache':
        from dvslam.odometry.klt_depth_cache import CachedDepth
        return CachedDepth(ckpt, hw)
    raise ValueError(f'prior must be perturbed|oracle|da3|dav2_ft|ssl|cache, got {name!r}')


def sample_depth(depth: np.ndarray, px: np.ndarray, dmin: float = 0.2, dmax: float = 6.0,
                 edge: float = 1.05) -> np.ndarray:
    """Bilinear depth at sub-pixel px (n, 2); NaN where the 3x3 neighbourhood is invalid or straddles an edge."""
    h, w = depth.shape
    x, y = px[:, 0].astype(np.float64), px[:, 1].astype(np.float64)
    ui, vi = np.clip(np.round(x).astype(int), 1, w - 2), np.clip(np.round(y).astype(int), 1, h - 2)
    patch = np.stack([depth[vi + dv, ui + du] for dv in (-1, 0, 1) for du in (-1, 0, 1)], 1)
    ok = (patch.min(1) > dmin) & (patch.max(1) < dmax) & (patch.max(1) < edge * np.maximum(patch.min(1), 1e-6))
    x0, y0 = np.clip(np.floor(x).astype(int), 0, w - 2), np.clip(np.floor(y).astype(int), 0, h - 2)
    fx, fy = x - x0, y - y0
    d = ((1 - fy) * ((1 - fx) * depth[y0, x0] + fx * depth[y0, x0 + 1])
         + fy * ((1 - fx) * depth[y0 + 1, x0] + fx * depth[y0 + 1, x0 + 1])).astype(np.float64)
    d[~ok] = np.nan
    return d


def huber_location(x: np.ndarray, c: float = 1.345, iters: int = 10) -> float:
    """Huber M-estimate of location with MAD scale; equivariant to shifts of x (exactly, up to rounding)."""
    mu = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - mu)))
    if s <= 0:
        return mu
    for _ in range(iters):
        r = (x - mu) / s
        w = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
        mu = float((w * x).sum() / w.sum())
    return mu


def window_parallax(R, t, bear, x, host, o_lm, o_kf, pri) -> float:
    """S4: median over the prior landmarks `pri` of the largest ray angle (deg) between the host keyframe and any
    keyframe observing the landmark (poses R, t and log inverse depths x of the window before the BA; scale-free).
    x is clipped to the ba_guard depth range and NaN angles are ignored, so an extreme depth cannot give NaN."""
    x = np.clip(x, -np.log(BA_DEPTH_HI), -np.log(BA_DEPTH_LO))
    pw = np.einsum('nij,nj->ni', R[host], bear * np.exp(-x)[:, None]) + t[host]
    a, b = pw[o_lm] - t[host[o_lm]], pw[o_lm] - t[o_kf]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    ang = np.zeros(len(x))
    np.fmax.at(ang, o_lm, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return float(np.median(ang[pri])) if len(pri) else float('nan')


def prior_obs_weight(theta_deg: float, ref_deg: float, w_min: float, w_max: float) -> float:
    """S4 observability weight of the window's depth prior: w = clip(ref_deg / θ_w, w_min, w_max), applied as
    σ_j / sqrt(w). Low parallax (geometry barely fixes depth) → stronger prior; θ_w not finite or 0 → w_max."""
    if not np.isfinite(theta_deg) or theta_deg <= 0:
        return float(w_max)
    return float(np.clip(ref_deg / theta_deg, w_min, w_max))


def robust_log_residual(r: np.ndarray, c: float, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """S4: ψ(r) = c · sign(s) · sqrt(2ρ(s)), s = r / c, and dψ/dr, for the log-depth prior residual r (log units).
    Σψ² = 2c² Σρ(s), so least squares on ψ minimises the robust cost. huber: ρ = s²/2 (|s| ≤ 1), |s| − 1/2;
    cauchy: ρ = log(1 + s²) / 2. |s| ≤ 1 (huber) or s → 0 (cauchy): ψ = r, dψ/dr = 1."""
    s = np.asarray(r, float) / c
    a = np.abs(s)
    if kind == 'huber':
        rho2 = np.where(a <= 1, a * a, 2 * a - 1)
        d = np.where(a <= 1, 1.0, 1 / np.sqrt(np.maximum(2 * a - 1, 1.0)))
    elif kind == 'cauchy':
        rho2 = np.log1p(a * a)
        d = np.where(a < 1e-6, 1.0, a / ((1 + a * a) * np.sqrt(np.maximum(rho2, 1e-300))))
    else:
        raise ValueError(f'kind must be huber|cauchy, got {kind!r}')
    return c * np.sign(s) * np.sqrt(rho2), d


def _skew(v: np.ndarray) -> np.ndarray:
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2], S[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    S[..., 1, 0], S[..., 2, 0], S[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return S


class KLTWindowOdometry(KLTPnPOdometry):
    def __init__(self, prior: str = 'perturbed', prior_seed: int = 0, prior_bias: float = 0.02,
                 prior_jitter: float = 0.03, window: int = 8, kf_ratio: float = 0.5, kf_px: float = 15.0,
                 kf_max: int = 12, ba: bool = True, prior_sigma: float = 0.05, prior_per_kf: bool = True,
                 px_sigma: float = 0.5, huber: float = 2.0, ba_loss: str = 'huber', ba_nfev: int = 20,
                 g_tau: float = 20.0, g_min_m: int = 5, g_gate_deg: float = 1.0, g_in_prior: bool = False,
                 tri_deg: float = 1.0, kf_refine: float = 2.0, kf_align: bool = True, max_jump_m: float = 0.10,
                 max_jump_deg: float = 20.0, ba_max_move_m: float = 0.05,
                 ba_max_turn_deg: float = 3.0, prior_ckpt: str | None = None, ba_guard: bool = False,
                 cv_threads: int | None = None, jump_ref_tracked: bool = False, flow_cache: str | None = None,
                 photo: str = 'off', photo_cell: int = 8, photo_grad: float = 0.01, photo_levels: int = 2,
                 photo_iters: int = 10, photo_huber: float = 0.03, photo_max_m: float = 0.02,
                 photo_max_deg: float = 1.0, photo_tol: float = 0.0, photo_max_pts: int = 0,
                 flow_mode: str = 'replace', flow_hybrid_frac: float = 0.7, photo_ref_max: int = 1, wide: int = 0,
                 wide_fov_deg: float = 0.0, out_lag: int = 0, out_lag_mode: str = 'smooth', out_lag_q: float = 0.002,
                 out_lag_r: float = 0.001, photo_bidir: bool = False, pt_subset: str = 'off', pt_block: int = 128,
                 pt_cell: int = 16, pt_seed: int = 0, photo_backend: str = 'numpy', prior_obs: bool = False,
                 prior_obs_ref: float = 1.0, prior_obs_min: float = 0.5, prior_obs_max: float = 2.0,
                 prior_loss: str = 'off', prior_loss_c: float = 0.05, photo_feat: str = '', **klt_kw):
        super().__init__(mode='sensor_depth', tri_deg=tri_deg, kf_ratio=kf_ratio, **klt_kw)
        self.prior_name, self.prior_seed, self.prior_bias, self.prior_jitter = prior, prior_seed, prior_bias, prior_jitter
        self.prior_ckpt = prior_ckpt                    # dav2_ft checkpoint; None = the model's default weights
        self.window, self.kf_px, self.kf_max, self.ba = max(2, int(window)), kf_px, kf_max, ba
        self.prior_sigma, self.px_sigma, self.huber, self.ba_nfev = prior_sigma, px_sigma, huber, ba_nfev
        self.g_tau, self.g_min_m, self.g_gate_cos = g_tau, g_min_m, np.cos(np.radians(g_gate_deg))
        self.ba_loss, self.g_in_prior = ba_loss, g_in_prior
        self.kf_align, self.prior_per_kf = kf_align, prior_per_kf
        self.max_jump_m, self.max_jump_deg = max_jump_m, max_jump_deg
        self.ba_max_move_m, self.ba_max_turn_deg = ba_max_move_m, ba_max_turn_deg
        self.ba_guard, self.cv_threads, self.jump_ref_tracked = ba_guard, cv_threads, jump_ref_tracked
        if photo not in ('off', 'all', 'fallback'):
            raise ValueError(f'photo must be off|all|fallback, got {photo!r}')
        self.flow_cache, self.photo = flow_cache, photo
        self.photo_cell, self.photo_grad = photo_cell, photo_grad
        self.photo_levels, self.photo_iters = photo_levels, photo_iters
        self.photo_huber, self.photo_max_m, self.photo_max_deg = photo_huber, photo_max_m, photo_max_deg
        self.photo_tol, self.photo_max_pts = photo_tol, photo_max_pts
        if flow_mode not in ('replace', 'hybrid'):
            raise ValueError(f'flow_mode must be replace|hybrid, got {flow_mode!r}')
        self.flow_mode, self.flow_hybrid_frac, self.photo_ref_max = flow_mode, flow_hybrid_frac, photo_ref_max
        self.wide, self.wide_fov_deg = wide, wide_fov_deg
        if wide and flow_cache and flow_cache not in ('raft_small', 'raft_large'):
            raise ValueError('wide with a cached flow directory is not supported: the cache holds flow of the standard '
                             'image, the tracker would sample it at canvas pixels (flow_cache=raft_small|raft_large '
                             'computes it on the canvas)')
        if out_lag_mode not in ('smooth', 'window'):
            raise ValueError(f'out_lag_mode must be smooth|window, got {out_lag_mode!r}')
        self.out_lag, self.out_lag_mode = int(out_lag), out_lag_mode
        self.out_lag_q, self.out_lag_r = out_lag_q, out_lag_r
        if photo_bidir and photo_ref_max != 1:
            raise ValueError('photo_bidir needs photo_ref_max=1 (pair refinement)')
        self.photo_bidir = bool(photo_bidir)
        if photo_backend not in ('numpy', 'torch', 'torch32'):
            raise ValueError(f'photo_backend must be numpy|torch|torch32, got {photo_backend!r}')
        self.photo_backend = photo_backend
        if photo == 'off' and (photo_feat or photo_backend != 'numpy'):
            raise ValueError('photo_feat / photo_backend need photo=all|fallback (they have no effect with photo=off)')
        if photo_feat and photo_backend == 'numpy':
            raise ValueError('photo_feat needs photo_backend=torch|torch32')
        if photo_feat and photo_levels != 1:
            raise ValueError(f'photo_feat needs photo_levels=1 (PhotoNet is trained on level 1), got {photo_levels}')
        self.photo_feat, self._photo_net = photo_feat, None
        if prior_loss not in ('off', 'huber', 'cauchy'):
            raise ValueError(f'prior_loss must be off|huber|cauchy, got {prior_loss!r}')
        self.prior_obs, self.prior_obs_ref = bool(prior_obs), prior_obs_ref
        self.prior_obs_min, self.prior_obs_max = prior_obs_min, prior_obs_max
        self.prior_loss, self.prior_loss_c = prior_loss, prior_loss_c
        if pt_subset not in PT_SUBSETS:
            raise ValueError(f'pt_subset must be one of {PT_SUBSETS}, got {pt_subset!r}')
        self.pt_subset, self.pt_block, self.pt_cell, self.pt_seed = pt_subset, int(pt_block), int(pt_cell), int(pt_seed)
        if pt_subset != 'off':                          # half the image: same corner density as the full tracker
            self.n_features = max(1, round(self.n_features / 2))
        self.flow = self.photo_norm = None
        self.kf_refine = kf_refine                     # max px change of the host->keyframe re-track (0 = off)
        self.prior = None

    # ------------------------------------------------------------------ setup
    def reset(self, camera: PinholeCamera) -> None:
        super().reset(camera)
        self.detect_mask, self._woff = None, 0
        K_prior, hw_prior = self.K, (camera.height, camera.width)   # the prior always sees the standard image
        if self.wide:                                  # EXP-011: track on the wide canvas, same f and principal ray
            from dvslam.data import tumvi
            if not np.allclose(self.K, tumvi.K_OUT):
                raise ValueError('wide needs the standard TUM-VI cam0 camera K_OUT')
            self.K, self._woff = tumvi.wide_K(self.wide), tumvi.wide_offset(self.wide)
            self.Kinv, self.size = np.linalg.inv(self.K), (self.wide, self.wide)
        if self.wide_fov_deg > 0:                       # detect / keep corners only within this angle of the axis
            w, h = self.size
            uu, vv = np.meshgrid(np.arange(w), np.arange(h))
            r = np.hypot((uu - self.K[0, 2]) / self.K[0, 0], (vv - self.K[1, 2]) / self.K[1, 1])
            self.detect_mask = np.where(np.degrees(np.arctan(r)) <= self.wide_fov_deg, 255, 0).astype(np.uint8)
        self.track_mask, self._pt_mask = self.detect_mask, None   # track_mask: tracks that leave it are dropped
        if self.pt_subset != 'off':                     # G0c point subset
            self._pt_mask = subset_mask(self.pt_subset, *self.size, self.pt_block, self.pt_cell, self.pt_seed)
            base = self.detect_mask if self.detect_mask is not None else np.uint8(255)
            self.detect_mask = np.where(self._pt_mask, base, 0).astype(np.uint8)
            if not self.pt_subset.startswith('random'):
                self.track_mask = self.detect_mask
        n = self.cv_threads if self.cv_threads is not None else os.environ.get('OMP_NUM_THREADS')
        if n:
            cv2.setNumThreads(int(n))
        if self.prior is None:
            self.prior = make_prior(self.prior_name, K_prior, self.prior_seed, self.prior_bias, self.prior_jitter,
                                    self.prior_ckpt, hw=hw_prior)
        self.g, self.n_m, self.n_jump, self.n_ba_reject = 0.0, 0, 0, 0
        self.fail_run = 0                               # consecutive untracked frames, for the reported state only
        self.last_prior = None                         # (frame index, depth map) of the last prior evaluation
        self.head_inputs = None
        self.prior_obs_log = []                         # S4: (newest keyframe, θ_w deg, weight) per window BA
        if self.flow_cache and self.flow is None:
            from dvslam.odometry.klt_flow import FlowCache, OnlineRAFT
            if self.flow_cache in ('raft_small', 'raft_large'):    # computed on demand on the GPU (EXP-008)
                self.flow = OnlineRAFT(self.flow_cache.split('_')[1])
            else:
                self.flow = FlowCache(self.flow_cache, hw=(self.size[1], self.size[0]))
        if self.photo != 'off' and self.photo_norm is None:
            from dvslam.odometry.klt_photo import PhotoNorm
            self.photo_norm = PhotoNorm(wide=self.wide)
        self._photo_s, self._prev_frame, self._cur_path, self._photo_prep = None, None, None, None
        self._photo_ref = None
        self._lag_pending, self._lag_pose, self._lag_pushed = [], {}, 0
        self._lag_R, self._lag_Rout = np.eye(3), np.eye(3)
        if self.out_lag:
            from dvslam.odometry.klt_lag import FixedLagCA
            self._lag_ca = FixedLagCA(self.out_lag_q, self.out_lag_r, self.out_lag)
        self._clear_window()
        for k in ('g', 'ba_ms', 'n_window', 'n_ba_lm', 'fail', 'photo', 'flow_used'):
            self.diag[k] = []

    def _clear_window(self) -> None:
        self._clear_tracks()
        self.uid = np.empty(0, np.int64)
        cap = 4096
        self.lm_host, self.lm_x = np.full(cap, -1, np.int64), np.full(cap, np.nan)
        self.lm_hpx, self.lm_lp = np.zeros((cap, 2)), np.full(cap, np.nan)
        self.lm_alive = np.zeros(cap, bool)
        self.next_uid = 0
        self.kfs: list[int] = []
        self.kf_poses, self.kf_prior, self.kf_obs = {}, {}, {}   # kf -> T_wc / prior map / (uids, px)
        self.kf_gray = {}
        self.since_kf = 0

    def _grow(self, n: int) -> None:
        cap = len(self.lm_host)
        if self.next_uid + n <= cap:
            return
        new = max(2 * cap, self.next_uid + n)
        pad = new - cap
        self.lm_host = np.r_[self.lm_host, np.full(pad, -1, np.int64)]
        self.lm_x, self.lm_lp = np.r_[self.lm_x, np.full(pad, np.nan)], np.r_[self.lm_lp, np.full(pad, np.nan)]
        self.lm_hpx = np.vstack([self.lm_hpx, np.zeros((pad, 2))])
        self.lm_alive = np.r_[self.lm_alive, np.zeros(pad, bool)]

    def _keep(self, m: np.ndarray) -> None:
        super()._keep(m)
        self.uid = self.uid[m]

    def _detect(self, gray: np.ndarray, index: int, T_wc: np.ndarray) -> int:
        n = super()._detect(gray, index, T_wc)
        self._grow(n)
        self.uid = np.r_[self.uid, np.arange(self.next_uid, self.next_uid + n)]
        self.next_uid += n
        return n

    def _predict_prior(self, frame: Frame) -> np.ndarray:
        if self.last_prior is None or self.last_prior[0] != frame.index:
            self.last_prior = (frame.index, self._prior_map(frame))
        return self.last_prior[1]

    def _prior_map(self, frame: Frame) -> np.ndarray:
        """Depth prior of frame (the network sees the standard 512 image); wide: embedded in the canvas, 0 outside."""
        d = np.asarray(self.prior.predict(frame)[0], np.float32)
        if not self.wide:
            return d
        out = np.zeros((self.wide, self.wide), np.float32)
        o = self._woff
        out[o:o + d.shape[0], o:o + d.shape[1]] = d
        return out

    # ------------------------------------------------------------------ geometry
    def _bearing(self, px: np.ndarray) -> np.ndarray:
        return np.c_[px, np.ones(len(px))] @ self.Kinv.T

    def _world_points(self, uids: np.ndarray) -> np.ndarray:
        X = np.full((len(uids), 3), np.nan)
        ok = self.lm_alive[uids] & np.isfinite(self.lm_x[uids])
        u = uids[ok]
        pc = self._bearing(self.lm_hpx[u]) * np.exp(-self.lm_x[u])[:, None]
        for h in np.unique(self.lm_host[u]):
            m = self.lm_host[u] == h
            T = self.kf_poses[h]
            X[np.flatnonzero(ok)[m]] = pc[m] @ T[:3, :3].T + T[:3, 3]
        return X

    def _refresh_map(self) -> None:
        self.X = self._world_points(self.uid)

    # ------------------------------------------------------------------ keyframes
    def _need_kf(self) -> bool:
        if len(self.px) < self.kf_ratio * self.n_features or self.since_kf >= self.kf_max:
            return True
        uids, px = self.kf_ref
        _, a, b = np.intersect1d(self.uid, uids, assume_unique=True, return_indices=True)
        if len(a) < self.min_inliers:
            return True
        return float(np.median(np.linalg.norm(self.px[a] - px[b], axis=1))) > self.kf_px

    def _add_keyframe(self, gray: np.ndarray, frame: Frame, T_wc: np.ndarray) -> None:
        k = frame.index
        depth = self._predict_prior(frame)
        self.kfs.append(k)
        self.kf_poses[k], self.kf_prior[k], self.kf_gray[k] = T_wc.copy(), depth, gray
        if self.kf_refine:
            self._refine_from_host(gray)
        live = self.lm_alive[self.uid]
        self.kf_obs[k] = (self.uid[live].copy(), self.px[live].astype(np.float64))
        n0 = len(self.uid)
        self._detect(gray, k, T_wc)
        new = self.uid[n0:]
        hpx = self.px[n0:].astype(np.float64)
        lp = np.log(sample_depth(depth, hpx, self.depth_min, self.depth_max, self.depth_edge))
        self.lm_host[new], self.lm_hpx[new], self.lm_lp[new], self.lm_alive[new] = k, hpx, lp, True
        self.lm_x[new] = -(lp + self._g_ba())          # d = d_p · e^g  ->  x = log(1/d) = −(log d_p + g)
        if self.ba:
            self._triangulate_new(k)
        while len(self.kfs) > self.window:
            self._drop_oldest()
        if self.ba and len(self.kfs) >= 2:
            self._bundle_adjust()
        if self.kf_align and len(new):
            self._align_new(k, depth, new, lp)
        self._refresh_map()
        self.since_kf = 0
        self.kf_ref = (self.uid.copy(), self.px.copy())   # all tracks at the newest keyframe (flow criterion)

    def _align_new(self, k: int, depth: np.ndarray, new: np.ndarray, lp: np.ndarray) -> None:
        """Initialise the new keyframe's landmarks at d_p · ŝ_k, ŝ_k = robust BA/prior depth ratio of the older
        landmarks seen in keyframe k (removes that frame's prior jitter from the new points; initial value only,
        the prior factor keeps log d_p)."""
        uids, px = self.kf_obs[k]
        if len(uids) < self.min_inliers:
            return
        X = self._world_points(uids)
        z = self._project(self.kf_poses[k], np.nan_to_num(X))[1]
        dp = sample_depth(depth, px, self.depth_min, self.depth_max, self.depth_edge)
        ok = np.isfinite(X[:, 0]) & (z > 0) & np.isfinite(dp)
        if ok.sum() < self.min_inliers:
            return
        s = huber_location(np.log(z[ok]) - np.log(dp[ok]))
        v = np.isfinite(lp)
        self.lm_x[new[v]] = -(lp[v] + s)

    def _refine_from_host(self, gray: np.ndarray) -> None:
        """Re-track live corners directly from their host keyframe (chained-LK init), so keyframe observations
        carry no accumulated frame-to-frame drift. Falls back to the chained position when the check fails."""
        hosts = self.lm_host[self.uid]
        for h in np.unique(hosts):
            if h not in self.kf_gray or h == self.kfs[-1]:
                continue
            i = np.flatnonzero((hosts == h) & self.lm_alive[self.uid])
            p0 = self.lm_hpx[self.uid[i]].astype(np.float32).reshape(-1, 1, 2)
            init = self.px[i].reshape(-1, 1, 2).copy()
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.kf_gray[h], gray, p0, init.copy(), winSize=self.lk['winSize'],
                                                 maxLevel=1, criteria=self.lk['criteria'],
                                                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.kf_gray[h], p1, p0.copy(), winSize=self.lk['winSize'],
                                                  maxLevel=1, criteria=self.lk['criteria'],
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < self.fb_px)
            ok &= np.linalg.norm((p1 - init).reshape(-1, 2), axis=1) < self.kf_refine
            self.px[i[ok]] = p1.reshape(-1, 2)[ok]

    def _triangulate_new(self, k: int) -> None:
        """Landmarks without depth: DLT between host and keyframe k once the ray angle exceeds tri_deg."""
        uids, px = self.kf_obs[k]
        m = np.isnan(self.lm_x[uids]) & (self.lm_host[uids] != k)
        if not m.any():
            return
        uids, px = uids[m], px[m]
        Tk = self.kf_poses[k]
        Pk = self.K @ np.linalg.inv(Tk)[:3]
        for h in np.unique(self.lm_host[uids]):
            j = self.lm_host[uids] == h
            u, hp, kp = uids[j], self.lm_hpx[uids[j]], px[j]
            Th = self.kf_poses[h]
            bh = self._bearing(hp) @ Th[:3, :3].T
            bk = self._bearing(kp) @ Tk[:3, :3].T
            cos = (bh * bk).sum(1) / np.linalg.norm(bh, axis=1) / np.linalg.norm(bk, axis=1)
            sel = cos < self.tri_cos
            if not sel.any():
                continue
            Xh = cv2.triangulatePoints(self.K @ np.linalg.inv(Th)[:3], Pk, hp[sel].T, kp[sel].T)
            X = (Xh[:3] / Xh[3]).T
            uv0, z0 = self._project(Th, X)
            uv1, z1 = self._project(Tk, X)
            ok = (z0 > 0) & (z1 > 0) & (np.linalg.norm(uv0 - hp[sel], axis=1) < self.ransac_px)
            ok &= np.linalg.norm(uv1 - kp[sel], axis=1) < self.ransac_px
            self.lm_x[u[sel][ok]] = -np.log(z0[ok])

    def _drop_oldest(self) -> None:
        """Remove the oldest keyframe; its landmarks move to the earliest remaining keyframe that observes them."""
        h = self.kfs.pop(0)
        todo = np.zeros(len(self.lm_host), bool)
        todo[np.flatnonzero(self.lm_alive & (self.lm_host == h))] = True
        for k in self.kfs:
            uids, px = self.kf_obs[k]
            m = todo[uids]
            if not m.any():
                continue
            u, p = uids[m], px[m]
            X = self._world_points_from(u, h)
            _, zk = self._project(self.kf_poses[k], np.nan_to_num(X)) if len(u) else (None, None)
            lp = np.log(sample_depth(self.kf_prior[k], p, self.depth_min, self.depth_max, self.depth_edge))
            has = np.isfinite(X[:, 0]) & (zk > 1e-6)
            x = np.where(has, -np.log(np.where(has, zk, 1.0)), -(lp + self._g_ba()))
            self.lm_host[u], self.lm_hpx[u], self.lm_lp[u], self.lm_x[u] = k, p, lp, x
            self.kf_obs[k] = (uids[~m], px[~m])
            todo[u] = False
        self.lm_alive[todo] = False
        for d in (self.kf_poses, self.kf_prior, self.kf_obs, self.kf_gray):
            d.pop(h, None)
        self.kf_poses = {k: self.kf_poses[k] for k in self.kfs}

    def _world_points_from(self, u: np.ndarray, h: int) -> np.ndarray:
        X = np.full((len(u), 3), np.nan)
        ok = np.isfinite(self.lm_x[u])
        pc = self._bearing(self.lm_hpx[u[ok]]) * np.exp(-self.lm_x[u[ok]])[:, None]
        T = self.kf_poses[h]
        X[ok] = pc @ T[:3, :3].T + T[:3, 3]
        return X

    # ------------------------------------------------------------------ window BA
    def _ba_problem(self):
        kfs = self.kfs
        pos = {k: i for i, k in enumerate(kfs)}
        o_uid, o_kf, o_px = [], [], []
        for k in kfs:
            uids, px = self.kf_obs[k]
            m = self.lm_alive[uids] & np.isfinite(self.lm_x[uids]) & np.isin(self.lm_host[uids], kfs)
            o_uid.append(uids[m]), o_kf.append(np.full(m.sum(), pos[k])), o_px.append(px[m])
        o_uid, o_kf, o_px = np.concatenate(o_uid), np.concatenate(o_kf), np.concatenate(o_px)
        lms, o_lm = np.unique(o_uid, return_inverse=True)
        return pos, lms, o_lm, o_kf, o_px

    def _bundle_adjust(self) -> None:
        t0 = time.perf_counter()
        pos, lms, o_lm, o_kf, o_px = self._ba_problem()
        n_kf, n_lm, n_obs = len(self.kfs), len(lms), len(o_lm)
        if n_lm < self.min_inliers:
            self._log_ba(t0, 0)
            return
        R0 = np.stack([self.kf_poses[k][:3, :3] for k in self.kfs])
        t0w = np.stack([self.kf_poses[k][:3, 3] for k in self.kfs])
        host = np.array([pos[h] for h in self.lm_host[lms]])
        bear = self._bearing(self.lm_hpx[lms])
        lp = self.lm_lp[lms]
        pri = np.flatnonzero(np.isfinite(lp))
        # per-keyframe prior noise is common-mode (one jitter per frame): with prior_per_kf the landmarks of one
        # host share a total weight of 1 / prior_sigma², i.e. σ_j = prior_sigma · sqrt(n_host)
        n_host = np.bincount(host[pri], minlength=n_kf)
        sig = self.prior_sigma * (np.sqrt(n_host[host[pri]]) if self.prior_per_kf else np.ones(len(pri)))
        if self.prior_obs:                              # S4: one observability weight per window, σ_j / sqrt(w)
            theta = window_parallax(R0, t0w, bear, self.lm_x[lms], host, o_lm, o_kf, pri)
            w = prior_obs_weight(theta, self.prior_obs_ref, self.prior_obs_min, self.prior_obs_max)
            self.prior_obs_log.append((self.kfs[-1], theta, w))
            sig = sig / np.sqrt(w)
        oh = host[o_lm]
        n_p = 6 * (n_kf - 1)
        g_ba = self._g_ba()
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x_lo, x_hi = -np.log(BA_DEPTH_HI), -np.log(BA_DEPTH_LO)

        def unpack(p):
            om = np.zeros((n_kf, 3))
            v = np.zeros((n_kf, 3))
            om[1:], v[1:] = p[:n_p].reshape(-1, 6)[:, :3], p[:n_p].reshape(-1, 6)[:, 3:]
            R = R0 @ Rotation.from_rotvec(om).as_matrix()
            return R, t0w + v, p[n_p:]

        def geom(p):
            R, t, x = unpack(p)
            xo = np.clip(x[o_lm], x_lo, x_hi) if self.ba_guard else x[o_lm]
            ph = bear[o_lm] * np.exp(-xo)[:, None]
            pw = np.einsum('nij,nj->ni', R[oh], ph) + t[oh]
            pk = np.einsum('nji,nj->ni', R[o_kf], pw - t[o_kf])
            return R, x, ph, pk

        def fun(p):
            _, x, _, pk = geom(p)
            z = pk[:, 2]
            uv = np.c_[fx * pk[:, 0] / z + cx, fy * pk[:, 1] / z + cy]
            r_px = ((uv - o_px) / self.px_sigma).ravel()
            r_pr = lp[pri] + g_ba + x[pri]
            if self.prior_loss != 'off':
                r_pr = robust_log_residual(r_pr, self.prior_loss_c, self.prior_loss)[0]
            return np.r_[r_px, r_pr / sig]

        def jac(p):
            R, x, ph, pk = geom(p)
            X, Y, z = pk[:, 0], pk[:, 1], pk[:, 2]
            Jp = np.zeros((n_obs, 2, 3))
            Jp[:, 0, 0], Jp[:, 0, 2] = fx / z, -fx * X / z ** 2
            Jp[:, 1, 1], Jp[:, 1, 2] = fy / z, -fy * Y / z ** 2
            Jp /= self.px_sigma
            RkT = np.transpose(R[o_kf], (0, 2, 1))
            RkTRh = RkT @ R[oh]
            blocks = [(np.einsum('nij,nj->ni', Jp, np.einsum('nij,nj->ni', RkTRh, -ph))[:, :, None],
                       n_p + o_lm[:, None] + np.zeros((1, 1), int), np.ones(n_obs, bool))]
            dh = np.concatenate([-RkTRh @ _skew(ph), RkT], axis=2)           # d pk / d(ω_h, v_h)
            dk = np.concatenate([_skew(pk), -RkT], axis=2)                   # d pk / d(ω_k, v_k)
            for d, which in ((dh, oh), (dk, o_kf)):
                cols = 6 * (which - 1)[:, None] + np.arange(6)[None]
                blocks.append((Jp @ d, cols, which > 0))
            rows, cols, vals = [], [], []
            r2 = 2 * np.arange(n_obs)
            for val, col, m in blocks:
                nc = val.shape[2]
                for a in range(2):
                    rows.append(np.repeat(r2[m] + a, nc)), cols.append(col[m].ravel()), vals.append(val[m, a].ravel())
            rows.append(2 * n_obs + np.arange(len(pri))), cols.append(n_p + pri)
            if self.prior_loss != 'off':
                vals.append(robust_log_residual(lp[pri] + g_ba + x[pri], self.prior_loss_c, self.prior_loss)[1] / sig)
            else:
                vals.append(1.0 / sig)
            return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(2 * n_obs + len(pri), n_p + n_lm))

        p0 = np.r_[np.zeros(n_p), self.lm_x[lms]]
        try:
            res = least_squares(fun, p0, jac=jac, method='trf', loss=self.ba_loss, f_scale=self.huber,
                                max_nfev=self.ba_nfev, x_scale='jac', tr_solver='lsmr')
            p = res.x
        except (ValueError, np.linalg.LinAlgError):
            self._log_ba(t0, n_lm)
            return
        R, t, x = unpack(p)
        _, _, _, pk = geom(p)
        move = np.linalg.norm(t - t0w, axis=1).max() * np.exp(-self.g)
        turn = np.degrees(np.linalg.norm(Rotation.from_matrix(np.transpose(R0, (0, 2, 1)) @ R).as_rotvec(), axis=1)).max()
        if (not np.isfinite(p).all() or (pk[:, 2] <= 0).mean() > 0.2 or move > self.ba_max_move_m
                or turn > self.ba_max_turn_deg):
            self.n_ba_reject += 1                       # diverged / implausible window update: keep the old state
            self._log_ba(t0, n_lm)
            return
        for i, k in enumerate(self.kfs):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R[i], t[i]
            self.kf_poses[k] = T
        if self.ba_guard:
            x = np.where((x < x_lo) | (x > x_hi), np.nan, x)   # non-physical depth: unmapped from now on
        self.lm_x[lms] = x
        self._update_g(lms, host, o_lm, o_kf, R, t, x)
        self._log_ba(t0, n_lm)

    def _update_g(self, lms, host, o_lm, o_kf, R, t, x) -> None:
        """m_k = median(log d_BA − log d_p) over landmarks with parallax; slow 1-D filter on g (§4.5)."""
        ph = self._bearing(self.lm_hpx[lms]) * np.exp(-x)[:, None]
        pw = np.einsum('nij,nj->ni', R[host], ph) + t[host]
        a = pw[o_lm] - t[host][o_lm]
        b = pw[o_lm] - t[o_kf]
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        par = np.zeros(len(lms), bool)
        par[o_lm[cos < self.g_gate_cos]] = True
        ok = par & np.isfinite(self.lm_lp[lms])
        if ok.sum() < self.min_inliers:
            return
        m = float(np.median(-x[ok] - self.lm_lp[lms][ok]))
        self.n_m += 1
        if self.n_m > self.g_min_m:
            self.g += (m - self.g) / self.g_tau

    def _g_ba(self) -> float:
        """g inside the prior factor / landmark init. Default 0: BA units stay anchored to the prior and g is a slow
        output correction for the BA-vs-prior offset. g_in_prior=True puts g into the factor as written in §4.5;
        then m_k − g only measures the visual/prior disagreement and the filter integrates it (g random-walks)."""
        return self.g if self.g_in_prior else 0.0

    def _log_ba(self, t0: float, n_lm: int) -> None:
        self._ba_ms += (time.perf_counter() - t0) * 1e3
        self._ba_lm = n_lm

    # ------------------------------------------------------------------ EXP-007 front-end options
    def _track(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """_track_inner, then (wide_fov_deg, pt_subset regions) drop the tracks that left track_mask."""
        self._flow_used = False
        if self.track_mask is not None and len(self.px):    # wide_fov_deg / pt_subset: drop tracks that leave it
            p0, p1 = self._track_inner(gray)
            w, h = self.size
            ui = np.clip(np.round(p1[:, 0]).astype(int), 0, w - 1)
            vi = np.clip(np.round(p1[:, 1]).astype(int), 0, h - 1)
            inside = self.track_mask[vi, ui] > 0
            self._keep(inside)
            return p0[inside], p1[inside]
        return self._track_inner(gray)

    def _track_inner(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """flow_cache set: move tracks by the cached dense flow t-1 -> t, check with the backward flow (fb_px).
        flow_mode='hybrid' (EXP-008): LK first; only on a frame where LK keeps < flow_hybrid_frac of the tracks are the
        lost tracks re-acquired from the flow (diag flow_used = frames that needed a flow evaluation)."""
        if self.flow is None or len(self.px) == 0:
            return super()._track(gray)
        from dvslam.odometry.klt_flow import bilinear
        if self.flow_mode == 'hybrid':
            p1, ok = self._lk(gray)
            if ok.mean() >= self.flow_hybrid_frac:
                self._keep(ok)
                return self.px.copy(), p1[ok]
        else:
            p1, ok = np.zeros((len(self.px), 2), np.float32), np.zeros(len(self.px), bool)
        self._flow_used = True
        fwd, bwd = self.flow.pair(self._cur_path, self.prev_gray, gray)
        p0 = self.px.astype(np.float64)
        f1 = p0 + bilinear(fwd, p0)
        p0b = f1 + bilinear(bwd, f1)
        w, h = self.size
        okf = (np.linalg.norm(p0b - p0, axis=1) < self.fb_px) & (f1[:, 0] >= 0) & (f1[:, 0] <= w - 1)
        okf &= (f1[:, 1] >= 0) & (f1[:, 1] <= h - 1)
        use = okf & ~ok                                 # hybrid: LK survivors keep their LK position
        p1 = p1.astype(np.float32)
        p1[use] = f1[use]
        ok = ok | okf
        self._keep(ok)
        return self.px.copy(), p1[ok]

    def _photo_refine(self, frame: Frame, gray: np.ndarray, inc: np.ndarray, head_prev) -> np.ndarray | None:
        """Photometric LM refinement of the BA-unit increment (klt_photo.py); None when skipped or not accepted."""
        from dvslam.odometry.klt_photo import prepare
        pf, nm = self._prev_frame, self.photo_norm
        if self._photo_prep is None or self._photo_prep[0] != pf.index:   # frame t-1 normally prepared at step t-1
            I0, M0 = nm(pf.image_path, self.prev_gray, 1.0)
            self._photo_prep = (pf.index, prepare(I0, M0, self.photo_levels), M0)
        _, P0, M0 = self._photo_prep
        I1, M1 = nm(frame.image_path, gray, 1.0)
        P1 = prepare(I1, M1, self.photo_levels)
        self._photo_prep = (frame.index, P1, M1)          # reference of the next pair
        ref = self._photo_ref
        if ref is None or ref['last'] != pf.index or frame.index - ref['index'] > self.photo_ref_max:
            ref = self._photo_anchor(pf, P0, M0, head_prev)   # reference = frame t-1, A_{t-1} = I
            if ref is None:
                return None
        A0 = np.linalg.inv(inc) @ ref['A']              # X_t = inv(ΔT) X_{t-1} = inv(ΔT) A_{t-1} X_ref
        ratio = ref['e'] / nm.exposure(frame.image_path)
        R, t, info = self._photo_lm(ref['P'], P1, ratio, ref['pts'], ref['z'], self.K, A0[:3, :3], A0[:3, 3],
                                     self.photo_levels, self.photo_iters, self.photo_huber, self.photo_tol)
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = R, t
        new = np.linalg.inv(A @ np.linalg.inv(ref['A']))
        if self.photo_bidir:                            # EXP-018: refine t -> t-1 too, combine the accepted ones
            self._photo_ref = None                      # (photo_ref_max == 1: the next pair re-anchors anyway)
            got = [T for T, i in ((new, info), self._photo_back(frame, P0, P1, M1, inc)) if self._photo_ok(i, T, inc)]
            if not got:
                return None
            if len(got) == 1:
                return got[0]
            mid = got[0].copy()
            half = Rotation.from_rotvec(0.5 * Rotation.from_matrix(got[0][:3, :3].T @ got[1][:3, :3]).as_rotvec())
            mid[:3, :3] = got[0][:3, :3] @ half.as_matrix()
            mid[:3, 3] = 0.5 * (got[0][:3, 3] + got[1][:3, 3])
            return mid
        if not self._photo_ok(info, new, inc):
            self._photo_ref = None                      # re-anchor on the next pair
            return None
        ref.update(A=A, last=frame.index)
        self._photo_ref = ref
        return new

    def _photo_ok(self, info: dict | None, new: np.ndarray, inc: np.ndarray) -> bool:
        """Acceptance of a refined increment: cost dropped, >= 50 points, within photo_max_m / photo_max_deg of inc."""
        if info is None:
            return False
        dt = np.linalg.norm(new[:3, 3] - inc[:3, 3]) * np.exp(-self.g)
        dr = np.degrees(np.linalg.norm(Rotation.from_matrix(inc[:3, :3].T @ new[:3, :3]).as_rotvec()))
        return not (info['n'] < 50 or not info['cost'] < info['cost0'] or dt > self.photo_max_m
                    or dr > self.photo_max_deg)

    def _photo_lm(self, *args):
        """klt_photo.refine_prepared, or (photo_backend=torch|torch32, S1) its PyTorch port with the same arguments;
        photo_feat (S2): the port with the learned features / weights of photo_learned.LearnedPhoto."""
        if self.photo_backend == 'numpy':
            from dvslam.odometry.klt_photo import refine_prepared
            return refine_prepared(*args)
        import torch

        dtype = torch.float64 if self.photo_backend == 'torch' else torch.float32
        if self.photo_feat:
            if self._photo_net is None:
                from dvslam.odometry.photo_learned import LearnedPhoto
                self._photo_net = LearnedPhoto(self.photo_feat)
            return self._photo_net(*args, dtype=dtype)
        from dvslam.odometry.klt_photo_torch import refine_prepared_torch
        return refine_prepared_torch(*args, dtype=dtype)

    def _photo_back(self, frame: Frame, P0: list, P1: list, M1: np.ndarray, inc: np.ndarray):
        """EXP-018 photo_bidir: the same LM from frame t to frame t-1 (points and SSL depth of frame t, same map-unit
        ratio _photo_s); X_{t-1} = ΔT X_t, so the result is the increment itself. (ΔT, info), info None if skipped."""
        from dvslam.odometry.klt_photo import select_points_g
        pts = self._pt_select(select_points_g(P1[0][..., 1], P1[0][..., 2], M1, self.photo_cell, self.photo_grad,
                                              self.photo_max_pts)[0])
        z = sample_depth(self._prior_map(frame), pts, self.depth_min, self.depth_max, self.depth_edge) * self._photo_s
        v = np.isfinite(z)
        if v.sum() < 50:
            return inc, None
        nm = self.photo_norm
        ratio = nm.exposure(frame.image_path) / nm.exposure(self._prev_frame.image_path)
        R, t, info = self._photo_lm(P1, P0, ratio, pts[v], z[v], self.K, inc[:3, :3], inc[:3, 3], self.photo_levels,
                                     self.photo_iters, self.photo_huber, self.photo_tol)
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = R, t
        return out, info

    def _photo_anchor(self, pf: Frame, P0: list, M0: np.ndarray, head_prev) -> dict | None:
        """Photometric reference at frame t-1: points, depths in map units (SSL depth × median z_map / d_p)."""
        from dvslam.odometry.klt_photo import select_points_g
        d0 = self._prior_map(pf)
        if head_prev is not None:                       # SSL depth -> map units: median z_map / d_p, tracked points
            dp = sample_depth(d0, head_prev[0], self.depth_min, self.depth_max, self.depth_edge)
            ok = np.isfinite(dp) & (head_prev[1] > 0)
            if ok.sum() >= 5:
                self._photo_s = float(np.median(head_prev[1][ok] / dp[ok]))
        if self._photo_s is None:
            return None
        pts = self._pt_select(select_points_g(P0[0][..., 1], P0[0][..., 2], M0, self.photo_cell, self.photo_grad,
                                              self.photo_max_pts)[0])
        z = sample_depth(d0, pts, self.depth_min, self.depth_max, self.depth_edge) * self._photo_s
        v = np.isfinite(z)
        if v.sum() < 50:
            return None
        return {'index': pf.index, 'last': pf.index, 'P': P0, 'pts': pts[v], 'z': z[v], 'A': np.eye(4),
                'e': self.photo_norm.exposure(pf.image_path)}

    def _pt_select(self, pts: np.ndarray) -> np.ndarray:
        """pt_subset: the photometric points (integer pixel centres) inside the subset mask; all of them when off.
        photo_max_pts (if set) caps the points before this selection."""
        if self._pt_mask is None:
            return pts
        return pts[self._pt_mask[pts[:, 1].astype(int), pts[:, 0].astype(int)]]

    # ------------------------------------------------------------------ main step
    def _plausible(self, inc: np.ndarray, g: float) -> bool:
        """Reject an increment that jumps more than max_jump_m (metric) / max_jump_deg from the previous one."""
        dt = np.linalg.norm(inc[:3, 3] - self.last_inc_ba[:3, 3]) * np.exp(-g)
        drot = np.degrees(np.linalg.norm(Rotation.from_matrix(self.last_inc_ba[:3, :3].T @ inc[:3, :3]).as_rotvec()))
        return dt <= self.max_jump_m and drot <= self.max_jump_deg

    def finalize(self) -> dict[str, np.ndarray]:
        out = super().finalize()
        out['n_jump_reject'], out['n_ba_reject'] = np.array(self.n_jump), np.array(self.n_ba_reject)
        return out

    def _start_window(self, gray, frame: Frame, T_wc: np.ndarray) -> None:
        self._clear_window()
        self._add_keyframe(gray, frame, T_wc)
        self.state, self.n_fail = 'OK', 0

    def step(self, frame: Frame) -> PoseEstimate | None:
        t0 = time.perf_counter()
        self._ba_ms, self._ba_lm = 0.0, 0
        path = frame.image_path
        if self.wide:
            from dvslam.data import tumvi
            path = tumvi.wide_path(path, self.wide)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        if self.prev_gray is None:
            self._start_window(gray, frame, np.eye(4))
            self.prev_gray = gray
            self.last_inc_ba = np.eye(4)
            self._prev_frame = frame
            self._lag_record(frame.index, np.eye(4))
            return None

        self._cur_path = frame.image_path
        prev_px, cur_px = self._track(gray)
        self.px = cur_px.astype(np.float32)
        n_tracks = len(self.px)
        m = np.isfinite(self.X[:, 0])
        head_prev = (prev_px[m].astype(np.float64), self._project(self.T_prev, self.X[m])[1]) if m.any() else None

        g_out = self.g                                  # the g of this output; updated only after ΔT_t
        solved = self._solve_pair(prev_px, self.T_prev, self.T_prev @ self.last_inc_ba)
        inc, source, n_inl, rms = self.last_inc_ba, 'motion_model', 0, np.nan
        fail = '' if solved is not None else self.fail_reason
        gate = not (self.jump_ref_tracked and self.fail_run > 0)   # fail_run: untracked frames up to t−1
        if solved is not None and gate and not self._plausible(solved[0], g_out):
            solved = None                               # degenerate PnP (e.g. a flipped solution): treat as a failure
            self.n_jump += 1
            fail = 'jump'
        if solved is not None:
            inc, T_cur, n_inl, rms = solved
            source, self.state, self.n_fail = 'tracked', 'OK', 0
        else:
            self.n_fail += 1
            self.state = 'DEGRADED' if self.n_fail < 3 else 'LOST'
            T_cur = self.T_prev @ inc
        photo = 0                                       # 0 not tried, 1 refined, 2 rejected
        if self.photo == 'all' or (self.photo == 'fallback' and source == 'motion_model'):
            ref = self._photo_refine(frame, gray, inc, head_prev)
            photo = 2 if ref is None else 1
            if ref is not None:
                inc = ref
                if source == 'motion_model':
                    T_cur = self.T_prev @ inc
        # Reported state (labels only; re-init decisions above are unchanged). _start_window resets n_fail on every
        # re-init, so LOST is derived from the run of untracked frames instead (F11).
        self.fail_run = 0 if source == 'tracked' else self.fail_run + 1
        state = 'OK' if source == 'tracked' else ('DEGRADED' if self.fail_run < 3 else 'LOST')
        if state == 'OK' and self.ba and self.n_m <= self.g_min_m:
            state = 'BOOTSTRAP'                         # g not yet valid (§4.3 item 4); no g is estimated without BA
        out = inc.copy()
        out[:3, 3] *= np.exp(-g_out)
        self.head_inputs = {'T_raw': inc.copy(), 'g': g_out, 'tracked': source == 'tracked',
                            'prev_px': head_prev[0] if head_prev else np.empty((0, 2)),
                            'prev_z': head_prev[1] if head_prev else np.empty(0)}

        # map maintenance after ΔT_t is fixed
        self.since_kf += 1
        is_kf = False
        if self.state == 'LOST' or len(self.px) < self.min_inliers:
            self._start_window(gray, frame, T_cur)
            is_kf = True
        elif source == 'tracked' and self._need_kf():
            self._add_keyframe(gray, frame, T_cur)
            T_cur = self.kf_poses[frame.index]          # BA may have moved the newest keyframe
            is_kf = True
        self.T_prev, self.last_inc_ba, self.prev_gray, self._prev_frame = T_cur, inc, gray, frame
        ms = (time.perf_counter() - t0) * 1e3
        d = self.diag
        for k, v in (('index', frame.index), ('state', _STATE_CODE[state]), ('n_tracks', n_tracks),
                     ('n_mapped', int(np.isfinite(self.X[:, 0]).sum())), ('n_inliers', n_inl),
                     ('is_keyframe', is_kf), ('reproj_rms_px', rms), ('total_ms', ms), ('g', g_out),
                     ('ba_ms', self._ba_ms), ('n_window', len(self.kfs)), ('n_ba_lm', self._ba_lm), ('fail', fail),
                     ('photo', photo), ('flow_used', self._flow_used)):
            d[k].append(v)
        est = PoseEstimate(frame.index, frame.timestamp, np.asarray(out, np.float64), state, source, is_kf,
                           {'total': ms, 'ba': self._ba_ms})
        if not self.out_lag:
            return est
        self._lag_record(frame.index, T_cur)
        self._lag_pending.append((est, inc.copy(), g_out))
        out = self._lag_emit(final=False)               # out_lag: the pair of frame index − out_lag, if any
        return out[0] if out else None

    # ------------------------------------------------------------------ EXP-013 fixed output lag (conditional)
    def _lag_record(self, index: int, T_wc: np.ndarray) -> None:
        """out_lag: pose of frame `index` as recorded now, plus its offset to the newest keyframe (window mode)."""
        if not self.out_lag:
            return
        k = self.kfs[-1] if self.kfs else None
        rel = np.linalg.inv(self.kf_poses[k]) @ T_wc if k in self.kf_poses else None
        self._lag_pose[index] = (k, rel, T_wc.copy())

    def _lag_now(self, index: int) -> np.ndarray:
        k, rel, T = self._lag_pose[index]
        return self.kf_poses[k] @ rel if rel is not None and k in self.kf_poses else T

    def _lag_emit(self, final: bool) -> list[PoseEstimate]:
        """The pairs whose lag has passed, oldest first (all held-back pairs when final). Not final: 0 or 1 pair,
        since each step adds one pending pair and releases at most one (FixedLagCA.push returns 0 or 1 step)."""
        out = []
        if self.out_lag_mode == 'smooth':
            new = []
            for est, _, _ in self._lag_pending[self._lag_pushed:]:   # R_wc(k-1) · t_k, rotations chained causally
                v = self._lag_R @ est.rel_increment[:3, 3]
                self._lag_R = self._lag_R @ est.rel_increment[:3, :3]
                new += self._lag_ca.push(v, est.source == 'tracked')
            self._lag_pushed = len(self._lag_pending)
            if final:
                new += self._lag_ca.flush()
            for vs in new:
                est, _, _ = self._lag_pending.pop(0)
                self._lag_pushed -= 1
                Rw = self._lag_Rout
                T = est.rel_increment.copy()
                T[:3, 3] = Rw.T @ vs
                self._lag_Rout = Rw @ T[:3, :3]
                est.rel_increment = T
                out.append(est)
        else:
            while self._lag_pending and (final or len(self._lag_pending) > self.out_lag):
                est, inc, g_k = self._lag_pending.pop(0)
                k = est.index
                then = np.linalg.inv(self._lag_pose[k - 1][2]) @ self._lag_pose[k][2]
                later = np.linalg.inv(self._lag_now(k - 1)) @ self._lag_now(k)
                T = inc @ np.linalg.inv(then) @ later
                T[:3, 3] *= np.exp(-g_k)
                est.rel_increment = T
                self._lag_pose.pop(k - 1, None)
                out.append(est)
        return out

    def flush(self) -> list[PoseEstimate]:
        """out_lag > 0: the pairs still held back at the end of a sequence (their lag shrinks to what is available)."""
        return self._lag_emit(final=True) if self.out_lag else []
