"""KLT + PnP keyframe-map odometry, tracker "C" (plan v3 item 0.7, E-mono-C / E2). OpenCV only.

Corners (~400, spread by min-distance, sub-pixel refined) are tracked frame to frame with full-resolution
pyramidal LK and a forward-backward check. Tracked corners carry a 3D map point once one is available:
  mode='depth_free'   : 5-point essential matrix + triangulation bootstraps the map (gauge: the first
                        baseline equals the number of bootstrap frames, i.e. one unit per frame); later
                        points are triangulated from their anchor keyframe once the parallax is large enough.
                        Scale is arbitrary; the evaluator fits it per 1-2 s window as an oracle.
  mode='sensor_depth' : keyframe corners are back-projected with the depth PNG (metres).
At every frame t, frames t-1 and t are both solved by PnP against the same map points (same correspondence
set), so the scored increment ΔT_t = inv(T_wc_{t-1}) · T_wc_t comes from one map state at time t.
New map points are added only after ΔT_t is computed.
"""
import time

import cv2
import numpy as np

from dvslam.data.redwood import load_depth
from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate

_STATE_CODE = {'BOOTSTRAP': 0, 'OK': 1, 'DEGRADED': 2, 'LOST': 3}


def _to_rt(T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_cw = np.linalg.inv(T_wc)
    return cv2.Rodrigues(T_cw[:3, :3])[0], T_cw[:3, 3].reshape(3, 1).copy()


def _from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T_cw = np.eye(4)
    T_cw[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cw[:3, 3] = tvec.ravel()
    return np.linalg.inv(T_cw)


class KLTPnPOdometry:
    def __init__(self, mode: str = 'depth_free', n_features: int = 400, min_distance: int = 20,
                 quality: float = 0.01, lk_win: int = 31, lk_levels: int = 3, fb_px: float = 0.5,
                 ransac_px: float = 2.0, min_inliers: int = 20, kf_ratio: float = 0.8, init_px: float = 10.0,
                 tri_deg: float = 1.5, retri: float = 1.5, rejoint: bool = True, ray_anchor: bool = True,
                 robust_k: float = 3.0, robust_min_px: float = 0.1, depth_min: float = 0.2, depth_max: float = 6.0,
                 depth_edge: float = 1.05, lk_iters: int = 30, lk_eps: float = 0.001):
        if mode not in ('depth_free', 'sensor_depth'):
            raise ValueError(f'mode must be depth_free or sensor_depth, got {mode}')
        self.mode = mode
        self.n_features, self.min_distance, self.quality = n_features, min_distance, quality
        self.lk = {'winSize': (lk_win, lk_win), 'maxLevel': lk_levels,
                   'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_iters, lk_eps)}
        self.fb_px, self.ransac_px, self.min_inliers = fb_px, ransac_px, min_inliers
        self.kf_ratio, self.init_px, self.tri_cos = kf_ratio, init_px, np.cos(np.radians(tri_deg))
        self.rejoint, self.ray_anchor, self.retri = rejoint, ray_anchor, retri
        self.robust_k, self.robust_min_px = robust_k, robust_min_px
        self.depth_min, self.depth_max, self.depth_edge = depth_min, depth_max, depth_edge

    # ------------------------------------------------------------------ setup
    def reset(self, camera: PinholeCamera) -> None:
        self.K = np.asarray(camera.K, dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.size = (camera.width, camera.height)
        self.prev_gray = None
        self.T_prev = np.eye(4)                       # T_wc of frame t-1 (re-solved at t when tracked)
        self.last_inc = np.eye(4)                     # last emitted increment (motion model)
        self.speed = 1.0                              # bootstrap step length in map units (gauge)
        self.recent_steps: list[float] = []           # last tracked step lengths (gauge after a reset)
        self.state = 'BOOTSTRAP'
        self.n_fail = 0
        self.kf_poses: dict[int, np.ndarray] = {}     # anchor keyframe index -> T_wc
        self._clear_tracks()
        self.diag = {k: [] for k in ('index', 'state', 'n_tracks', 'n_mapped', 'n_inliers', 'is_keyframe',
                                     'reproj_rms_px', 'total_ms')}

    def _clear_tracks(self) -> None:
        self.px = np.empty((0, 2), np.float32)        # positions in frame t-1 (then t after the step)
        self.anchor_idx = np.empty(0, np.int64)
        self.anchor_px = np.empty((0, 2), np.float64)
        self.X = np.empty((0, 3), np.float64)         # world points, NaN = not mapped yet
        self.par = np.empty(0, np.float64)            # parallax angle (rad) of the last triangulation

    def finalize(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.diag.items()}

    # ------------------------------------------------------------------ helpers
    def _keep(self, m: np.ndarray) -> None:
        self.px, self.anchor_idx, self.anchor_px, self.X = self.px[m], self.anchor_idx[m], self.anchor_px[m], self.X[m]
        self.par = self.par[m]

    def _detect(self, gray: np.ndarray, index: int, T_wc: np.ndarray) -> int:
        """Add corners away from existing tracks; anchor them at this (key)frame."""
        need = self.n_features - len(self.px)
        if need <= 0:
            return 0
        dm = getattr(self, 'detect_mask', None)       # optional allowed-region mask (EXP-011 field-of-view cap)
        mask = np.full(gray.shape, 255, np.uint8) if dm is None else dm.copy()
        for u, v in self.px:                            # int(): round(np.float32) stays np.float32 on NumPy < 2
            cv2.circle(mask, (int(round(u)), int(round(v))), self.min_distance, 0, -1)  # noqa: RUF046
        c = cv2.goodFeaturesToTrack(gray, need, self.quality, self.min_distance, mask=mask, blockSize=7)
        if c is None:
            return 0
        c = cv2.cornerSubPix(gray, c.astype(np.float32), (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001)).reshape(-1, 2)
        self.px = np.vstack([self.px, c])
        self.anchor_idx = np.concatenate([self.anchor_idx, np.full(len(c), index)])
        self.anchor_px = np.vstack([self.anchor_px, c.astype(np.float64)])
        self.X = np.vstack([self.X, np.full((len(c), 3), np.nan)])
        self.par = np.concatenate([self.par, np.zeros(len(c))])
        self.kf_poses[index] = T_wc.copy()
        return len(c)

    def _track(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """LK t-1 -> t with forward-backward check; drops failed tracks. Returns (px in t-1, px in t)."""
        if len(self.px) == 0:
            return self.px.copy(), self.px.copy()
        p1, ok = self._lk(gray)
        self._keep(ok)
        return self.px.copy(), p1[ok]

    def _lk(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """LK of every track t-1 -> t: (positions in t (n, 2) float32, passed forward-backward and bounds (n,))."""
        p0 = self.px.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.lk)
        p1, p0b = p1.reshape(-1, 2), p0b.reshape(-1, 2)
        w, h = self.size
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0b - self.px, axis=1) < self.fb_px)
        ok &= (p1[:, 0] >= 0) & (p1[:, 0] <= w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= h - 1)
        return p1, ok

    def _project(self, T_wc: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T_cw = np.linalg.inv(T_wc)
        Xc = X @ T_cw[:3, :3].T + T_cw[:3, 3]
        uv = Xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

    def _refine(self, X: np.ndarray, uv: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
        r, t = _to_rt(T_wc)
        r, t = cv2.solvePnPRefineLM(X, uv.astype(np.float64), self.K, None, r, t)
        return _from_rt(r, t)

    def _robust_refine(self, X: np.ndarray, uv: np.ndarray, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """LM, then twice: keep residuals < max(robust_k * 1.4826 * MAD, robust_min_px) and re-solve."""
        keep = np.ones(len(X), bool)
        T_wc = self._refine(X, uv, T_wc)
        for _ in range(2 if self.robust_k > 0 else 0):
            e = np.linalg.norm(self._project(T_wc, X)[0] - uv, axis=1)
            mad = 1.4826 * np.median(np.abs(e - np.median(e)))
            keep = e < max(np.median(e) + self.robust_k * mad, self.robust_min_px)
            if keep.sum() < self.min_inliers:
                break
            T_wc = self._refine(X[keep], uv[keep], T_wc)
        return T_wc, keep

    # ------------------------------------------------------------------ mapping
    def _backproject(self, depth: np.ndarray, T_wc: np.ndarray) -> None:
        """sensor_depth: give unmapped tracks a world point from this frame's depth (edge-safe 3x3 check)."""
        idx = np.flatnonzero(np.isnan(self.X[:, 0]))
        if len(idx) == 0:
            return
        h, w = depth.shape
        ui = np.clip(np.round(self.px[idx, 0]).astype(int), 1, w - 2)
        vi = np.clip(np.round(self.px[idx, 1]).astype(int), 1, h - 2)
        patch = np.stack([depth[vi + dv, ui + du] for dv in (-1, 0, 1) for du in (-1, 0, 1)], 1)
        ok = (patch.min(1) > self.depth_min) & (patch.max(1) < self.depth_max)
        ok &= patch.max(1) < self.depth_edge * np.maximum(patch.min(1), 1e-6)
        x, y = self.px[idx, 0].astype(np.float64), self.px[idx, 1].astype(np.float64)   # bilinear depth
        x0, y0 = np.clip(np.floor(x).astype(int), 0, w - 2), np.clip(np.floor(y).astype(int), 0, h - 2)
        fx, fy = x - x0, y - y0
        d = ((1 - fy) * ((1 - fx) * depth[y0, x0] + fx * depth[y0, x0 + 1])
             + fy * ((1 - fx) * depth[y0 + 1, x0] + fx * depth[y0 + 1, x0 + 1])).astype(np.float64)
        idx, d = idx[ok], d[ok]
        Xc = np.c_[self.px[idx].astype(np.float64), np.ones(len(idx))] @ self.Kinv.T * d[:, None]
        self.X[idx] = Xc @ T_wc[:3, :3].T + T_wc[:3, 3]

    def _triangulate(self, T_wc: np.ndarray) -> None:
        """depth_free: triangulate tracks from their anchor keyframe once parallax > tri_deg; re-triangulate a
        mapped track when its parallax has grown by the factor retri (0 = never), so depth improves with baseline."""
        idx = np.arange(len(self.X)) if self.retri > 0 else np.flatnonzero(np.isnan(self.X[:, 0]))
        if len(idx) == 0:
            return
        b_now = np.c_[self.px[idx].astype(np.float64), np.ones(len(idx))] @ self.Kinv.T @ T_wc[:3, :3].T
        b_anc = np.empty_like(b_now)
        for a in np.unique(self.anchor_idx[idx]):
            m = self.anchor_idx[idx] == a
            b_anc[m] = np.c_[self.anchor_px[idx[m]], np.ones(m.sum())] @ self.Kinv.T @ self.kf_poses[a][:3, :3].T
        cos = (b_now * b_anc).sum(1) / np.linalg.norm(b_now, axis=1) / np.linalg.norm(b_anc, axis=1)
        ang = np.arccos(np.clip(cos, -1, 1))
        sel = (cos < self.tri_cos) & ((self.par[idx] == 0) | (ang > self.retri * self.par[idx]))
        P1 = self.K @ np.linalg.inv(T_wc)[:3]
        for a in np.unique(self.anchor_idx[idx[sel]]):
            j = idx[sel & (self.anchor_idx[idx] == a)]
            P0 = self.K @ np.linalg.inv(self.kf_poses[a])[:3]
            Xh = cv2.triangulatePoints(P0, P1, self.anchor_px[j].T, self.px[j].astype(np.float64).T)
            X = (Xh[:3] / Xh[3]).T
            uv0, z0 = self._project(self.kf_poses[a], X)
            uv1, z1 = self._project(T_wc, X)
            ok = (z0 > 0) & (z1 > 0) & (np.linalg.norm(uv0 - self.anchor_px[j], axis=1) < self.ransac_px)
            ok &= np.linalg.norm(uv1 - self.px[j], axis=1) < self.ransac_px
            self.X[j[ok]] = X[ok]
            self.par[j[ok]] = ang[sel & (self.anchor_idx[idx] == a)][ok]

    def _bootstrap(self, prev_px: np.ndarray, T_anchor: np.ndarray, index: int) -> bool:
        """depth_free: 5-point E between the anchor keyframe and frame t, then triangulate. Sets self.T_cur."""
        if len(self.px) < 2 * self.min_inliers:
            return False
        if np.median(np.linalg.norm(self.px - self.anchor_px, axis=1)) < self.init_px:
            return False
        a, c = self.anchor_px, self.px.astype(np.float64)
        E, m = cv2.findEssentialMat(a, c, self.K, cv2.RANSAC, 0.999, self.ransac_px)
        if E is None or E.shape != (3, 3):
            return False
        n, R, t, m = cv2.recoverPose(E, a, c, self.K, mask=m)
        if n < 2 * self.min_inliers:
            return False
        steps = index - self.anchor_idx[0]
        T_ac = np.eye(4)                              # x_cur = R x_anchor + t  ->  T_anchor_cur = inv([R | t])
        T_ac[:3, :3], T_ac[:3, 3] = R, t.ravel() * self.speed * steps
        T_cur = T_anchor @ np.linalg.inv(T_ac)
        self.kf_poses = {self.anchor_idx[0]: T_anchor}
        keep = m.ravel() > 0
        self._keep(keep)
        prev_px = prev_px[keep]
        self._triangulate(T_cur)
        if np.isfinite(self.X[:, 0]).sum() < 2 * self.min_inliers:
            self.X[:], self.par[:] = np.nan, 0.0
            return False
        self.T_cur, self._boot_prev_px = T_cur, prev_px
        return True

    # ------------------------------------------------------------------ main step
    def step(self, frame: Frame) -> PoseEstimate | None:
        t0 = time.perf_counter()
        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        depth = load_depth(frame.depth_path) if self.mode == 'sensor_depth' else None
        if self.prev_gray is None:                     # first frame: keyframe 0 at the origin
            self._start_map(gray, depth, frame.index, np.eye(4))
            self.prev_gray = gray
            return None

        prev_px, cur_px = self._track(gray)
        self.px = cur_px.astype(np.float32)
        n_tracks = len(self.px)

        inc, T_cur, source, n_inl, rms, is_kf = self.last_inc, None, 'motion_model', 0, np.nan, False
        if self.state == 'BOOTSTRAP' and self.mode == 'depth_free':
            T_anchor = self.kf_poses[self.anchor_idx[0]] if len(self.anchor_idx) else self.T_prev
            if len(self.anchor_idx) and self._bootstrap(prev_px, T_anchor, frame.index):
                prev_px = self._boot_prev_px
                m = np.isfinite(self.X[:, 0])
                T_prev_guess = self._refine(self.X[m], prev_px[m], self.T_cur)
                solved = self._solve_pair(prev_px, T_prev_guess, self.T_cur)
                if solved is not None:
                    inc, T_cur, n_inl, rms = solved
                    source, self.state = 'tracked', 'OK'
            elif n_tracks < 3 * self.min_inliers:       # bootstrap tracks thinned out: restart from here
                self._clear_tracks()
                self._detect(gray, frame.index, self.T_prev @ self.last_inc)
                is_kf = True
        else:
            T_pred = self.T_prev @ self.last_inc
            solved = self._solve_pair(prev_px, self.T_prev, T_pred)
            if solved is not None:
                inc, T_cur, n_inl, rms = solved
                source, self.state, self.n_fail = 'tracked', 'OK', 0
            else:
                self.n_fail += 1
                self.state = 'DEGRADED' if self.n_fail < 3 else 'LOST'

        state = self.state
        if source == 'motion_model':
            T_cur = self.T_prev @ inc
        if state == 'BOOTSTRAP' and source == 'motion_model' and np.allclose(inc, np.eye(4)):
            inc = np.eye(4)                            # very first bootstrap: one gauge unit per frame along +z
            inc[2, 3] = self.speed
            T_cur = self.T_prev @ inc

        # map maintenance happens after ΔT_t is fixed
        if state == 'LOST' or (self.mode == 'sensor_depth' and source == 'motion_model'):
            self._start_map(gray, depth, frame.index, T_cur)
            is_kf = True
        elif state != 'BOOTSTRAP' and source == 'tracked':
            if self.mode == 'depth_free':
                self._triangulate(T_cur)
            if len(self.px) < self.kf_ratio * self.n_features:
                self._detect(gray, frame.index, T_cur)
                is_kf = True
                if self.mode == 'sensor_depth':
                    self._backproject(depth, T_cur)
            self._prune_anchors()

        self.T_prev, self.last_inc, self.prev_gray = T_cur, inc, gray
        if source == 'tracked':
            self.recent_steps = (self.recent_steps + [float(np.linalg.norm(inc[:3, 3]))])[-15:]
            self.speed = max(float(np.median(self.recent_steps)), 1e-9)
        ms = (time.perf_counter() - t0) * 1e3
        d = self.diag
        for k, v in (('index', frame.index), ('state', _STATE_CODE[state]), ('n_tracks', n_tracks),
                     ('n_mapped', int(np.isfinite(self.X[:, 0]).sum())), ('n_inliers', n_inl),
                     ('is_keyframe', is_kf), ('reproj_rms_px', rms), ('total_ms', ms)):
            d[k].append(v)
        return PoseEstimate(frame.index, frame.timestamp, np.asarray(inc, np.float64).copy(), state, source,
                            is_kf, {'total': ms})

    def _start_map(self, gray, depth, index: int, T_wc: np.ndarray) -> None:
        """New keyframe from scratch (first frame or LOST). depth_free re-bootstraps with the last step length."""
        self._clear_tracks()
        self.kf_poses = {}
        self._detect(gray, index, T_wc)
        if self.mode == 'sensor_depth':
            self._backproject(depth, T_wc)
            self.state = 'OK'
        else:
            self.state = 'BOOTSTRAP'
        self.n_fail = 0

    def _prune_anchors(self) -> None:
        live = set(self.anchor_idx.tolist())
        self.kf_poses = {k: v for k, v in self.kf_poses.items() if k in live or k == max(self.kf_poses)}

    def _solve_pair(self, prev_px, T_prev, T_pred):
        """PnP of frame t (RANSAC + LM) and, on the same inliers, LM re-solve of frame t-1 (one map state).

        ray_anchor: the scored ΔT_t is then refined frame to frame, with each inlier map point moved onto its
        observed ray in t-1 (depth kept from the map), so accumulated track drift against the map cancels.
        Returns (ΔT_t, T_wc_t, n_inliers, rms_px) or None when tracking fails; then self.fail_reason says why
        (few_mapped | pnp_error | ransac_inliers | rejoint_inliers; diagnostics only).
        """
        m = np.isfinite(self.X[:, 0])
        if m.sum() < self.min_inliers:
            self.fail_reason = 'few_mapped'
            return None
        X, uv1, uv0 = self.X[m], self.px[m].astype(np.float64), prev_px[m].astype(np.float64)
        try:
            ok, r, t, inl = cv2.solvePnPRansac(X, uv1, self.K, None, iterationsCount=200,
                                               reprojectionError=self.ransac_px, confidence=0.999,
                                               flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:                              # SQPnP asserts on degenerate minimal sets
            self.fail_reason = 'pnp_error'
            return None
        if not ok or inl is None or len(inl) < self.min_inliers:
            self.fail_reason = 'ransac_inliers'
            return None
        inl = inl.ravel()
        if self.rejoint:   # the pair uses only points consistent with frame t-1 as well
            e0 = np.linalg.norm(self._project(T_prev, X[inl])[0] - uv0[inl], axis=1)
            inl = inl[e0 < 2 * self.ransac_px]
            if len(inl) < self.min_inliers:
                self.fail_reason = 'rejoint_inliers'
                return None
        T_cur, keep = self._robust_refine(X[inl], uv1[inl], _from_rt(r, t))
        if self.rejoint:
            inl = inl[keep]
        T_prev_r = self._refine(X[inl], uv0[inl], T_prev) if self.rejoint else T_prev
        e1 = np.linalg.norm(self._project(T_cur, X)[0] - uv1, axis=1)
        rms = float(np.sqrt(np.mean(e1[inl] ** 2)))
        inc = np.linalg.inv(T_prev_r) @ T_cur
        if self.ray_anchor:
            z = self._project(T_prev_r, X[inl])[1]
            Xr = np.c_[uv0[inl], np.ones(len(inl))] @ self.Kinv.T * z[:, None]   # points in camera t-1
            inc = self._robust_refine(Xr, uv1[inl], inc)[0]
        bad = np.flatnonzero(m)[e1 > 2 * self.ransac_px]   # drop map tracks that disagree with frame t
        keep = np.ones(len(self.px), bool)
        keep[bad] = False
        self._keep(keep)
        return inc, T_cur, len(inl), rms
