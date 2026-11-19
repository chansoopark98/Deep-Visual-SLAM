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
  g          : one global log-scale, BA units = metric · e^g. After each BA, m_k = median(log d_BA − log d_p) over
               parallax-gated landmarks; after g_min_m samples g ← g + (m_k − g) / g_tau. Earlier increments are
               tagged BOOTSTRAP. Deviation from §4.5: g is NOT inside the prior factor by default (g_in_prior=False).
               With g in the factor, the BA is pulled to d_p·e^g, m_k − g only measures the visual/prior disagreement,
               and the filter integrates it into a random walk (measured: g → −0.39 over 600 frames).
  output     : ΔT_t = inv(T_{t−1}) · T_t from one map state at t (BA units), then translation × e^{−g} once (§4.2).
               g and BA updates happen after ΔT_t is emitted. An increment that jumps > max_jump_m / max_jump_deg from
               the previous one is treated as a PnP failure (motion model, DEGRADED).

Prior: `prior` ∈ {perturbed (b = 2 %, jitter 3 %), oracle (GT depth), da3 (DA3METRIC-L), dav2_ft, ssl}. `prior_seed`
seeds the perturbed jitter. `ba=False kf_align=False` is the 0.7 sensor-depth path with a prior (baseline).
`head_inputs` (BA-unit ΔT^raw, g, BA depths of frame t−1) is exposed for the precision head (klt_head.py).
"""
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from dvslam.odometry.klt_pnp import _STATE_CODE, KLTPnPOdometry


def make_prior(name: str, K: np.ndarray, seed: int = 0, bias: float = 0.02, jitter: float = 0.03,
               ckpt: str | None = None):
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
    raise ValueError(f'prior must be perturbed|oracle|da3|dav2_ft|ssl, got {name!r}')


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
                 ba_max_turn_deg: float = 3.0, prior_ckpt: str | None = None, **klt_kw):
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
        self.kf_refine = kf_refine                     # max px change of the host->keyframe re-track (0 = off)
        self.prior = None

    # ------------------------------------------------------------------ setup
    def reset(self, camera: PinholeCamera) -> None:
        super().reset(camera)
        if self.prior is None:
            self.prior = make_prior(self.prior_name, self.K, self.prior_seed, self.prior_bias, self.prior_jitter,
                                    self.prior_ckpt)
        self.g, self.n_m, self.n_jump, self.n_ba_reject = 0.0, 0, 0, 0
        self.fail_run = 0                               # consecutive untracked frames, for the reported state only
        self.last_prior = None                         # (frame index, depth map) of the last prior evaluation
        self.head_inputs = None
        self._clear_window()
        for k in ('g', 'ba_ms', 'n_window', 'n_ba_lm'):
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
            self.last_prior = (frame.index, np.asarray(self.prior.predict(frame)[0], np.float32))
        return self.last_prior[1]

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
            uvk, zk = self._project(self.kf_poses[k], np.nan_to_num(X)) if len(u) else (None, None)
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
        oh = host[o_lm]
        n_p = 6 * (n_kf - 1)
        g_ba = self._g_ba()
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        def unpack(p):
            om = np.zeros((n_kf, 3))
            v = np.zeros((n_kf, 3))
            om[1:], v[1:] = p[:n_p].reshape(-1, 6)[:, :3], p[:n_p].reshape(-1, 6)[:, 3:]
            R = R0 @ Rotation.from_rotvec(om).as_matrix()
            return R, t0w + v, p[n_p:]

        def geom(p):
            R, t, x = unpack(p)
            ph = bear[o_lm] * np.exp(-x[o_lm])[:, None]
            pw = np.einsum('nij,nj->ni', R[oh], ph) + t[oh]
            pk = np.einsum('nji,nj->ni', R[o_kf], pw - t[o_kf])
            return R, x, ph, pk

        def fun(p):
            _, x, _, pk = geom(p)
            z = pk[:, 2]
            uv = np.c_[fx * pk[:, 0] / z + cx, fy * pk[:, 1] / z + cy]
            r_px = ((uv - o_px) / self.px_sigma).ravel()
            r_pr = (lp[pri] + g_ba + x[pri]) / sig
            return np.r_[r_px, r_pr]

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
        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        if self.prev_gray is None:
            self._start_window(gray, frame, np.eye(4))
            self.prev_gray = gray
            self.last_inc_ba = np.eye(4)
            return None

        prev_px, cur_px = self._track(gray)
        self.px = cur_px.astype(np.float32)
        n_tracks = len(self.px)
        m = np.isfinite(self.X[:, 0])
        head_prev = (prev_px[m].astype(np.float64), self._project(self.T_prev, self.X[m])[1]) if m.any() else None

        g_out = self.g                                  # the g of this output; updated only after ΔT_t
        solved = self._solve_pair(prev_px, self.T_prev, self.T_prev @ self.last_inc_ba)
        inc, source, n_inl, rms = self.last_inc_ba, 'motion_model', 0, np.nan
        if solved is not None and not self._plausible(solved[0], g_out):
            solved = None                               # degenerate PnP (e.g. a flipped solution): treat as a failure
            self.n_jump += 1
        if solved is not None:
            inc, T_cur, n_inl, rms = solved
            source, self.state, self.n_fail = 'tracked', 'OK', 0
        else:
            self.n_fail += 1
            self.state = 'DEGRADED' if self.n_fail < 3 else 'LOST'
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
        self.T_prev, self.last_inc_ba, self.prev_gray = T_cur, inc, gray
        ms = (time.perf_counter() - t0) * 1e3
        d = self.diag
        for k, v in (('index', frame.index), ('state', _STATE_CODE[state]), ('n_tracks', n_tracks),
                     ('n_mapped', int(np.isfinite(self.X[:, 0]).sum())), ('n_inliers', n_inl),
                     ('is_keyframe', is_kf), ('reproj_rms_px', rms), ('total_ms', ms), ('g', g_out),
                     ('ba_ms', self._ba_ms), ('n_window', len(self.kfs)), ('n_ba_lm', self._ba_lm)):
            d[k].append(v)
        return PoseEstimate(frame.index, frame.timestamp, np.asarray(out, np.float64), state, source, is_kf,
                            {'total': ms, 'ba': self._ba_ms})
