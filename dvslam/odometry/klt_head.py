"""Precision head over any base tracker (plan v3 item 1.5, §4.4, A-N1, A-N6).

  KLTHeadOdometry(base='klt_window', **opts)    options: head options unprefixed, base options as 'base.<name>'
  e.g. --odometry klt_head --opt base=klt_window --opt base.window=8 --opt k=3

Per frame t, after the base has emitted ΔT_t^raw:
  1. ~n_corners Shi-Tomasi corners on frame t−1 (full resolution, cornerSubPix), pyramidal LK t−1 → t with a
     forward-backward check.
  2. Corner depth = prior_{t−1}(corner) × r_{t−1}, where r_{t−1} is the Huber location estimate (in log space, MAD
     scale) of BA depth / prior depth over the base's valid points in frame t−1. The prior thus fixes local shape only;
     a per-frame prior scale c_t cancels inside r_{t−1}, and the scale follows the base's BA units.
  3. Two-frame pose-only LM (scipy least_squares, Huber kernel) initialised at ΔT^raw in BA units.
  4. Accept only if the result lies within k·σ of ΔT^raw (translation: σ_mm·e^{g} in BA units; rotation: σ_deg),
     converged, and has ≥ min_inliers inliers; otherwise keep ΔT^raw. Rejections are logged (finalize()).
  5. Output translation × e^{−g} once (§4.1).

A base that exposes `head_inputs` = {'T_raw' (BA units), 'g', 'prev_px', 'prev_z', 'tracked'} after each step
(klt_window does) gets BA-unit refinement. Any other base is refined in its own output units with r_{t−1} = 1 and g = 0
(prior-unit corner depth): the head then contributes the prior's per-frame scale, gated by k·σ.
"""
import importlib
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate, load_increments
from dvslam.odometry.klt_window import huber_location, make_prior, sample_depth  # noqa: F401 (re-export)

BASES = {
    'klt_window': ('dvslam.odometry.klt_window', 'KLTWindowOdometry'),
    'klt_pnp': ('dvslam.odometry.klt_pnp', 'KLTPnPOdometry'),
    'dpvo_mono': ('dvslam.odometry.dpvo_mono', 'DPVOMonoOdometry'),
    'dpvo_metric': ('dvslam.odometry.dpvo_metric', 'DPVOMetricOdometry'),
    'cuvslam_rgbd': ('dvslam.odometry.cuvslam_rgbd', 'CuVSLAMRGBDOdometry'),
    'replay': ('dvslam.odometry.klt_head', 'ReplayOdometry'),
}
HEAD_OPTS = ('k', 'n_corners', 'min_distance', 'quality', 'lk_win', 'lk_levels', 'fb_px', 'huber_px', 'inlier_px',
             'min_inliers', 'min_ratio_pts', 'sigma_mm', 'sigma_deg', 'prior', 'prior_seed', 'prior_bias',
             'prior_jitter', 'enabled')


def _pose(p: np.ndarray, T0: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = T0[:3, 3] + p[3:]
    return T


class ReplayOdometry:
    """Base that re-emits a finished run's increments (<run>/<scene>.inc.npz), so the head can be evaluated on bases
    that need another environment (cuVSLAM: py3.10) or the GPU. The scene is the image's grand-parent directory."""

    def __init__(self, run: str):
        self.run = Path(run)

    def reset(self, camera: PinholeCamera) -> None:
        self.scene, self.z = None, None

    def step(self, frame: Frame) -> PoseEstimate | None:
        scene = Path(frame.image_path).parent.parent.name
        if scene != self.scene:
            self.scene, self.z = scene, load_increments(self.run / f'{scene}.inc.npz')
        if frame.index == 0:
            return None
        k = frame.index - 1
        if self.z['index'][k] != frame.index:
            raise ValueError(f'{self.run}/{scene}: increment {k} has index {self.z["index"][k]}, expected {frame.index}')
        return PoseEstimate(frame.index, frame.timestamp, self.z['rel_increment'][k].copy(),
                            str(self.z['tracking_state'][k]), str(self.z['source'][k]))


class KLTHeadOdometry:
    def __init__(self, base='klt_window', **opts):
        base_opts = {k[5:]: v for k, v in opts.items() if k.startswith('base.')}
        head = {k: v for k, v in opts.items() if not k.startswith('base.')}
        bad = sorted(set(head) - set(HEAD_OPTS))
        if bad:
            raise ValueError(f'unknown head options {bad}; base options need the prefix "base."')
        if isinstance(base, str):
            module, cls = BASES[base]
            base = getattr(importlib.import_module(module), cls)(**base_opts)
        self.base = base
        h = dict(k=3.0, n_corners=400, min_distance=12, quality=0.005, lk_win=21, lk_levels=3, fb_px=0.3,
                 huber_px=1.0, inlier_px=1.5, min_inliers=40, min_ratio_pts=20, sigma_mm=2.0, sigma_deg=0.1,
                 prior=None, prior_seed=None, prior_bias=None, prior_jitter=None, enabled=True)
        h.update(head)
        self.o = h
        self.lk = dict(winSize=(h['lk_win'], h['lk_win']), maxLevel=h['lk_levels'],
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        self.prior = None

    def reset(self, camera: PinholeCamera) -> None:
        self.base.reset(camera)
        self.K = np.asarray(camera.K, np.float64)
        self.Kinv = np.linalg.inv(self.K)
        if self.prior is None:
            self.prior = getattr(self.base, 'prior', None)
            if self.prior is None or self.o['prior'] is not None:      # own prior: head options, else the base's
                b, o = self.base, self.o
                pick = lambda k, attr, d: o[k] if o[k] is not None else getattr(b, attr, d)  # noqa: E731
                self.prior = make_prior(pick('prior', 'prior_name', 'perturbed'), self.K,
                                        pick('prior_seed', 'prior_seed', 0), pick('prior_bias', 'prior_bias', 0.02),
                                        pick('prior_jitter', 'prior_jitter', 0.03))
        self.prev = None                                  # (frame, gray, prior depth) of t−1
        self.diag = {k: [] for k in ('index', 'raw_inc', 'status', 'n_inliers', 'log_r', 'dt_mm', 'drot_deg',
                                     'head_ms')}

    def finalize(self) -> dict[str, np.ndarray]:
        out = dict(self.base.finalize()) if hasattr(self.base, 'finalize') else {}
        out.update({f'head_{k}': np.asarray(v) for k, v in self.diag.items()})
        st = np.asarray(self.diag['status'])
        out['head_accepted'] = st == 'accepted'
        return out

    def _prior_of(self, frame: Frame) -> np.ndarray:
        lp = getattr(self.base, 'last_prior', None)
        if lp is not None and lp[0] == frame.index and getattr(self.base, 'prior', None) is self.prior:
            return lp[1]
        return np.asarray(self.prior.predict(frame)[0], np.float32)

    # ------------------------------------------------------------------ core (unit-tested)
    def refine(self, gray0, gray1, prior0, T_raw, g, prev_px=None, prev_z=None):
        """ΔT in BA units from frame t−1 (gray0, prior0) to t (gray1). Returns (T or None, status, info)."""
        o = self.o
        if prev_px is not None:
            dp = sample_depth(prior0, prev_px, 0.0, np.inf)        # scale-free validity (edge ratio only)
            ok = np.isfinite(dp) & (prev_z > 0)
            if ok.sum() < o['min_ratio_pts']:
                return None, 'no_ratio', {}
            log_r = huber_location(np.log(prev_z[ok]) - np.log(dp[ok]))
        else:
            log_r = 0.0
        c = cv2.goodFeaturesToTrack(gray0, o['n_corners'], o['quality'], o['min_distance'], blockSize=7)
        if c is None or len(c) < o['min_inliers']:
            return None, 'few_corners', {'log_r': log_r}
        c = cv2.cornerSubPix(gray0, c.astype(np.float32), (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001))
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, c, None, **self.lk)
        p0b, s2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **self.lk)
        c, p1, p0b = c.reshape(-1, 2), p1.reshape(-1, 2), p0b.reshape(-1, 2)
        h, w = gray0.shape
        ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(p0b - c, axis=1) < o['fb_px'])
        ok &= (p1[:, 0] >= 0) & (p1[:, 0] <= w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= h - 1)
        d = sample_depth(prior0, c.astype(np.float64), 0.0, np.inf)
        ok &= np.isfinite(d)
        if ok.sum() < o['min_inliers']:
            return None, 'few_tracks', {'log_r': log_r}
        X = np.c_[c[ok].astype(np.float64), np.ones(ok.sum())] @ self.Kinv.T * (d[ok] * np.exp(log_r))[:, None]
        uv = p1[ok].astype(np.float64)
        K = self.K

        def fun(p):
            T = _pose(p, T_raw)
            q = (X - T[:3, 3]) @ T[:3, :3]                  # points in camera t
            return ((q[:, :2] / q[:, 2:3]) @ K[:2, :2].T + K[:2, 2] - uv).ravel()

        try:
            res = least_squares(fun, np.zeros(6), method='trf', loss='huber', f_scale=o['huber_px'],
                                xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=100, x_scale='jac')
        except (ValueError, np.linalg.LinAlgError):
            return None, 'solver', {'log_r': log_r}
        e = np.linalg.norm(res.fun.reshape(-1, 2), axis=1)
        n_inl = int((e < o['inlier_px']).sum())
        T = _pose(res.x, T_raw)
        info = {'log_r': log_r, 'n_inliers': n_inl}
        if res.status <= 0 or not np.isfinite(res.x).all():
            return None, 'no_converge', info
        if n_inl < o['min_inliers']:
            return None, 'few_inliers', info
        dt = np.linalg.norm(T[:3, 3] - T_raw[:3, 3])
        drot = np.degrees(np.linalg.norm(Rotation.from_matrix(T_raw[:3, :3].T @ T[:3, :3]).as_rotvec()))
        info.update(dt_mm=dt * np.exp(-g) * 1e3, drot_deg=drot)
        if dt > o['k'] * o['sigma_mm'] * 1e-3 * np.exp(g) or drot > o['k'] * o['sigma_deg']:
            return None, 'gate', info
        return T, 'accepted', info

    # ------------------------------------------------------------------ step
    def step(self, frame: Frame) -> PoseEstimate | None:
        t0 = time.perf_counter()
        est = self.base.step(frame)
        tb = time.perf_counter()
        gray = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        prev, self.prev = self.prev, None
        if est is None:
            self.prev = (frame, gray, self._prior_of(frame))
            return None
        hi = getattr(self.base, 'head_inputs', None)
        if hi is not None:
            T_raw, g, tracked = hi['T_raw'], hi['g'], hi['tracked']
            prev_px, prev_z = hi['prev_px'], hi['prev_z']
        else:
            T_raw, g, tracked, prev_px, prev_z = est.rel_increment, 0.0, est.source == 'tracked', None, None
        status, info, out = 'skipped', {}, est.rel_increment
        if not self.o['enabled']:
            status = 'disabled'
        elif tracked and prev is not None and prev[0].index == frame.index - 1:
            T, status, info = self.refine(prev[1], gray, prev[2], T_raw, g, prev_px, prev_z)
            if T is not None:
                out = T.copy()
                out[:3, 3] *= np.exp(-g)
        self.prev = (frame, gray, self._prior_of(frame))
        ms = (time.perf_counter() - tb) * 1e3
        d = self.diag
        for k, v in (('index', frame.index), ('raw_inc', est.rel_increment), ('status', status),
                     ('n_inliers', info.get('n_inliers', 0)), ('log_r', info.get('log_r', np.nan)),
                     ('dt_mm', info.get('dt_mm', np.nan)), ('drot_deg', info.get('drot_deg', np.nan)),
                     ('head_ms', ms)):
            d[k].append(v)
        timings = dict(est.timings_ms)
        timings['head'] = ms
        timings['total'] = (time.perf_counter() - t0) * 1e3
        return PoseEstimate(est.index, est.timestamp, np.asarray(out, np.float64), est.tracking_state, est.source,
                            est.is_keyframe, timings)
