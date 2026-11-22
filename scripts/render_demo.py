"""Render one tracker run as SLAM-paper figures: an animated GIF (camera frame with tracked features | 3D map |
top-down trajectory) and a static PNG (top-down trajectory | final 3D map).

  python scripts/render_demo.py --run runs/demo/euroc --scene MH_01_easy --out media/euroc_MH_01_easy_demo.gif \
      [--png media/euroc_MH_01_easy_geometry.png --stride 17 --fps 10 --width 1080]

Inputs: <run>/<scene>.inc.npz (increments of frames start+1..n-1, apps/run_sequence.py / scripts/demo_track.py
format); <run>/<scene>.tracks.npz (every KLT observation per frame); <run>/<scene>.map.npz (scripts/demo_track.py: the
tracker's mapped landmarks per frame in that frame's camera coordinates, metres); <run>/<scene>.extra.npz (the
tracker's per-frame log, is_keyframe); <run>/<scene>.seq.npz (scripts/demo_track.py: GT, image paths). Without seq.npz
the run is read as a TUM-VI run (rectified cam0 from the TUM-VI loader, DVSLAM_TUMVI_ROOT / DVSLAM_CACHE_ROOT); without
map.npz the 3D view has no points; without tracks.npz no features are drawn; without extra.npz every frame is a
keyframe candidate.

Trajectory: the estimate is the causal chain of the increments started at the GT pose of the run's first frame
(metres, no alignment). Display frame: z up (TUM-VI mocap and EuRoC world frames are gravity aligned; for Redwood the
up direction is the mean of the GT camera -y axes).
Camera panel (ORB-SLAM frame viewer): the grayscale frame, the features tracked in it as small green squares.
3D map (ORB-SLAM map viewer, fixed oblique orthographic view --azim / --elev): map points black, points seen in the
shown frame red, keyframes blue frustums, the current camera a larger green frustum, the estimated trajectory a thin
green line. Map point = the latest estimate of each landmark seen in at least `min_obs` frames so far, placed with the
estimated pose of the frame it was logged in; only points inside the estimated trajectory's bounding box grown by
`margin` (and above the floor) are drawn. Keyframes drawn = the tracker's keyframes, each kept only when it is at least
`kf_spacing` x the trajectory extent away from every keyframe drawn before it.
Top-down plot (evo_traj --plot_mode xy): ground truth dashed grey (left open across GT dropouts), estimate solid blue.
GIF: frames start..end every `stride`-th frame (plus the last) at `fps`, the last one held 3 s; one shared palette (the
overlay colours and their tints + `grays` grey levels), no dithering. PNG: top-down | 3D, 2:1, at `dpi`.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.eval.metrics import pair_metrics
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import causal_trajectory, load_increments

WHITE = (255, 255, 255)        # RGB
MAP_RGB = (0, 0, 0)            # map points
CUR_RGB = (255, 0, 0)          # map points seen in the shown frame
KF_RGB = (0, 0, 255)           # keyframes
CAM_RGB = (0, 200, 0)          # current camera
TRAJ_RGB = (0, 150, 0)         # estimated trajectory (3D)
FEAT_RGB = (0, 255, 0)         # tracked features on the camera frame
EST_HEX, GT_HEX = '#1f77b4', '#808080'
COLOURS = [CUR_RGB, KF_RGB, CAM_RGB, TRAJ_RGB, FEAT_RGB, (31, 119, 180)]
FONT_PT = 8


# ------------------------------------------------------------------ geometry (pure functions)
def integrate(T0: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """(n + 1, 4, 4) poses T0 · ΔT_1 · … · ΔT_t, t = 0..n (T0 for t = 0)."""
    return np.asarray(T0, np.float64) @ causal_trajectory(np.asarray(increments, np.float64))


def running_errors(T_wc: np.ndarray, index: np.ndarray, increments: np.ndarray, valid: np.ndarray):
    """Translation step error (%) and rotation error (deg) so far after each pair (t-1, t), t = index: running means
    over the pairs with valid[t-1] (pair_valid); NaN before the first scored pair."""
    pos, rpy, ok = [], [], []
    for t, dT in zip(index, increments):
        m = pair_metrics(relative_pose(T_wc[t - 1], T_wc[t]), dT)
        pos.append(m['pos_error_percent'])
        rpy.append(float(np.mean(m['rpy_abs_err_deg'])))
        ok.append(bool(valid[t - 1]))
    ok = np.array(ok)
    n = np.cumsum(ok)
    with np.errstate(invalid='ignore', divide='ignore'):
        e_t = np.cumsum(np.where(ok, pos, 0.0)) / n
        e_r = np.cumsum(np.where(ok, rpy, 0.0)) / n
    return np.where(n > 0, e_t, np.nan), np.where(n > 0, e_r, np.nan)


def ate_rmse(p_gt: np.ndarray, p_pred: np.ndarray) -> float:
    """RMSE (m) of the positions after the best rigid (SE(3), no scale) alignment of p_pred onto p_gt (Umeyama)."""
    a, b = np.asarray(p_pred, np.float64), np.asarray(p_gt, np.float64)
    ma, mb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((b - mb).T @ (a - ma))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return float(np.sqrt(np.mean(np.sum((b - (a - ma) @ R.T - mb) ** 2, 1))))


def up_rotation(T_wc: np.ndarray, z_up: bool) -> np.ndarray:
    """(3, 3) R_dw: world -> display frame with z up. z_up: the world z axis is up (identity). Otherwise up = the
    mean of the cameras' -y axes (image up), x = world x projected onto the horizontal plane."""
    if z_up:
        return np.eye(3)
    up = -np.asarray(T_wc)[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    e = np.eye(3)[int(np.argmin(np.abs(up)))]
    x = e - (e @ up) * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


def world_points(T_pred: np.ndarray, rows: np.ndarray, xc: np.ndarray) -> np.ndarray:
    """(n, 3) world points T_pred[rows[i]] · xc[i] (camera-frame points logged per frame)."""
    T = np.asarray(T_pred)[rows]
    return np.einsum('nij,nj->ni', T[:, :3, :3], np.asarray(xc, np.float64)) + T[:, :3, 3]


def thin_keyframes(pos: np.ndarray, rows: np.ndarray, spacing: float) -> np.ndarray:
    """The rows (ascending) kept for drawing: each row whose position pos[row] is at least `spacing` from every row
    kept before it (the first row is always kept)."""
    kept = []
    for r in rows:
        if not kept or np.linalg.norm(pos[kept] - pos[r], axis=1).min() >= spacing:
            kept.append(int(r))
    return np.array(kept, np.int64)


class MapState:
    """Causal map for the animation: the latest position of each landmark (gid) and how many frames saw it.
    `frames` must be non-decreasing."""

    def __init__(self, frames: np.ndarray, gids: np.ndarray, X: np.ndarray):
        self.frames, self.X = np.asarray(frames), np.asarray(X)
        u, self.idx = np.unique(gids, return_inverse=True)
        self.pos = np.full((len(u), 3), np.nan)
        self.count = np.zeros(len(u), np.int64)
        self.upto = 0                                    # rows < upto are applied

    def advance(self, f: int) -> np.ndarray:
        """Apply every row with frame <= f; returns the landmark indices seen in frame f."""
        end = int(np.searchsorted(self.frames, f, side='right'))
        if end > self.upto:
            i = self.idx[self.upto:end]
            np.add.at(self.count, i, 1)
            last = len(i) - 1 - np.unique(i[::-1], return_index=True)[1]   # last row of each landmark
            self.pos[i[last]] = self.X[self.upto:end][last]
            self.upto = end
        a = int(np.searchsorted(self.frames, f, side='left'))
        return np.unique(self.idx[a:end])


class View3D:
    """Orthographic view of display coordinates (z up) from azimuth / elevation (deg), fitted to the points `fit`."""

    def __init__(self, fit: np.ndarray, size: int, margin: int, azim: float = -60.0, elev: float = 35.0):
        a, e = np.radians(azim), np.radians(elev)
        self.d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # towards the viewer
        self.r = np.array([-np.sin(a), np.cos(a), 0.0])
        self.u = np.cross(self.r, -self.d)
        fit = np.asarray(fit, np.float64)
        q = np.c_[fit @ self.r, fit @ self.u]
        lo, hi = q.min(0), q.max(0)
        self.c = (lo + hi) / 2
        self.s = (size - 2 * margin) / np.maximum(hi - lo, 1e-3).max()
        self.size = size

    def __call__(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pixels (..., 2), depth towards the viewer (...))."""
        P = np.asarray(P, np.float64)
        x, y = (P @ self.r - self.c[0]) * self.s, (P @ self.u - self.c[1]) * self.s
        return np.stack([self.size / 2 + x, self.size / 2 - y], -1), P @ self.d


def frustum(T_wc: np.ndarray, K: np.ndarray, rect: tuple[float, float, float, float], depth: float) -> np.ndarray:
    """(5, 3) apex + the 4 image-rectangle corners (u0, v0, u1, v1) at `depth` m, in the frame of T_wc."""
    u0, v0, u1, v1 = rect
    px = np.array([[u0, v0, 1], [u1, v0, 1], [u1, v1, 1], [u0, v1, 1]], np.float64)
    rays = px @ np.linalg.inv(K).T
    pc = np.r_[np.zeros((1, 3)), rays / rays[:, 2:] * depth]
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3]


# ------------------------------------------------------------------ drawing
def palette(grays: int) -> Image.Image:
    """Shared GIF palette: the overlay colours with three tints towards white (anti-aliasing on the white panels) +
    `grays` grey levels (the camera image, black points, plot text and grid)."""
    tints = [tuple(round(255 - a * (255 - v)) for v in c) for c in COLOURS for a in (0.25, 0.5, 0.75, 1.0)]
    cols = tints + [(v, v, v) for v in np.linspace(0, 255, grays).round().astype(int).tolist()]
    pal = Image.new('P', (1, 1))
    pal.putpalette([c for rgb in cols for c in rgb] + list(WHITE) * (256 - len(cols)))
    return pal


def poly(img: np.ndarray, pts: np.ndarray, rgb, thick: int, ok: np.ndarray | None = None) -> None:
    """Anti-aliased polyline; ok (len(pts) - 1,) bool leaves out the segments i -> i + 1 where it is False."""
    runs = [pts] if ok is None else np.split(pts, np.flatnonzero(~np.asarray(ok[:len(pts) - 1], bool)) + 1)
    runs = [np.round(r * 16).astype(np.int32) for r in runs if len(r) > 1]
    if runs:
        cv2.polylines(img, runs, False, rgb, thick, cv2.LINE_AA, shift=4)


def dots(img: np.ndarray, px: np.ndarray, rgb, r: int) -> None:
    """Square r x r dots (r = 1: single pixels), clipped to the image; fast for many points."""
    h, w = img.shape[:2]
    q = np.floor(np.asarray(px)).astype(np.int64)
    for dy in range(r):
        for dx in range(r):
            x, y = q[:, 0] + dx, q[:, 1] + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img[y[ok], x[ok]] = rgb


class Scene:
    """Everything the panels need, in display coordinates (z up)."""

    def __init__(self, args):
        inc = load_increments(args.run / f'{args.scene}.inc.npz')
        meta, self.idx, rel = inc['meta'], inc['index'], inc['rel_increment']
        if np.any(np.diff(self.idx) != 1):
            raise ValueError('increments must cover a contiguous frame range')
        sp = args.run / f'{args.scene}.seq.npz'
        if sp.exists():
            z = np.load(sp)
            self.dataset = str(z['dataset'])
            self.T_wc, self.gt_valid = z['T_wc'], z['gt_valid']
            self.K, self.size = z['K'], tuple(int(v) for v in z['size'])
            self.disp_paths, self.A_disp = [Path(p) for p in z['display_paths']], z['A_disp']
            self.disp_size = tuple(int(v) for v in z['display_size'])
            mask = z['mask']
        else:                                          # TUM-VI run without seq.npz (e.g. scripts/exp003_track_log.py)
            from dvslam.data import registry
            seq = registry.load_sequence('tumvi', args.scene, None, args.data_root or registry.default_root('tumvi'),
                                         meta.get('time_offset_s', 0.0))
            self.dataset = 'tumvi'
            self.T_wc, self.gt_valid = seq.T_wc, seq.gt_valid
            self.K, self.size = seq.K, (seq.width, seq.height)
            self.disp_paths, self.A_disp, self.disp_size = list(seq.image_paths), np.eye(3)[:2], self.size
            mask = np.zeros((0, 0), np.uint8)
        if mask.size:                                  # image rectangle of the frustum: the valid canvas region
            ys, xs = np.nonzero(mask)
            self.rect = (xs.min(), ys.min(), xs.max(), ys.max())
        else:
            self.rect = (0, 0, self.size[0] - 1, self.size[1] - 1)
        self.f0 = int(self.idx[0]) - 1                 # the run's first frame
        self.T_pred = integrate(self.T_wc[self.f0], rel)   # row k = frame f0 + k
        n = len(self.T_pred)
        pair_valid = self.gt_valid[:-1] & self.gt_valid[1:]
        self.err_t, self.err_r = running_errors(self.T_wc, self.idx, rel, pair_valid)
        self.pairs_gt = int(pair_valid[self.f0:].sum())
        R = up_rotation(self.T_wc, self.dataset != 'redwood')
        self.gt = self.T_wc[self.f0:self.f0 + n, :3, 3] @ R.T
        self.pred = self.T_pred[:, :3, 3] @ R.T
        self.R = R
        gv = self.gt_valid[self.f0:self.f0 + n]
        self.ate = ate_rmse(self.gt[gv], self.pred[gv])
        self.gt_xy = np.where(gv[:, None], self.gt[:, :2], np.nan)   # NaN: the plot leaves GT dropouts open

        trk = args.run / f'{args.scene}.tracks.npz'
        self.tracks = None
        if trk.exists():
            z = np.load(trk)
            o = np.argsort(z['obs_frame'], kind='stable')
            px = z['obs_px'][o] @ self.A_disp[:, :2].T + self.A_disp[:, 2]     # tracker pixels -> display image
            self.tracks = (z['obs_frame'][o], px)
        else:
            print(f'{trk} not found: no features drawn')

        ep = args.run / f'{args.scene}.extra.npz'
        rows = np.arange(n)
        if ep.exists():
            z = np.load(ep)
            rows = np.r_[0, z['index'][z['is_keyframe']] - self.f0]
        self.kf_rows = rows[(rows >= 0) & (rows < n)]

        lo, hi = self.pred.min(0), self.pred.max(0)   # 3D box: the estimate (GT is not drawn in 3D)
        ext = float((hi - lo)[:2].max())
        self.map = None
        mp = args.run / f'{args.scene}.map.npz'
        floor = lo[2]
        if mp.exists():
            z = np.load(mp)
            f, g, xc = z['map_frame'], z['map_gid'], z['map_xc']
            keep = (f >= self.f0) & (f < self.f0 + n) & (xc[:, 2] > 0.1)
            f, g, xc = f[keep], g[keep], xc[keep]
            o = np.argsort(f, kind='stable')
            f, g, xc = f[o], g[o], xc[o]
            Xd = world_points(self.T_pred, f - self.f0, xc) @ R.T
            m = args.margin * max(ext, 1.0)
            inside = np.all((Xd[:, :2] > lo[:2] - m) & (Xd[:, :2] < hi[:2] + m), 1)
            if inside.any():
                floor = min(floor, float(np.percentile(Xd[inside, 2], 3)))
            self.map = (f, g, Xd)
        else:
            print(f'{mp} not found: the 3D view has no map points')
        m = args.margin * max(ext, 1.0)
        self.box = np.array([[lo[0] - m, lo[1] - m, floor], [hi[0] + m, hi[1] + m, hi[2] + 0.1 * m]])


class Renderer:
    """Camera panel and 3D map panel, P x P px (the camera panel cw x P)."""

    def __init__(self, sc: Scene, P: int, args):
        self.sc, self.P, self.args = sc, P, args
        fit = [sc.pred]
        if sc.map is not None:                         # map points in the box, without the outer 1 % per axis
            X = sc.map[2][np.all((sc.map[2] >= sc.box[0]) & (sc.map[2] <= sc.box[1]), 1)]
            if len(X):
                lo, hi = np.percentile(X, [1, 99], axis=0)
                fit.append(X[np.all((X >= lo) & (X <= hi), 1)])
        self.v3 = View3D(np.concatenate(fit), P, max(8, P // 25), args.azim, args.elev)
        self.pred3, _ = self.v3(sc.pred)
        ext = float(np.ptp(sc.pred[:, :2], 0).max())
        self.frustum_m = args.kf_size * ext
        self.kf = thin_keyframes(sc.pred, sc.kf_rows, args.kf_spacing * ext)
        self.kf3 = [self.v3(frustum(self._pose(k), sc.K, sc.rect, self.frustum_m))[0] for k in self.kf]
        if sc.map is not None:
            self.ms = MapState(*sc.map)

    def _pose(self, k: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.sc.R @ self.sc.T_pred[k, :3, :3], self.sc.pred[k]
        return T

    def camera(self, f: int, cw: int) -> np.ndarray:
        sc, P = self.sc, self.P
        img = cv2.imread(str(sc.disp_paths[f]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(sc.disp_paths[f])
        cam = cv2.cvtColor(cv2.resize(img, (cw, P), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
        if sc.tracks is not None:
            of, op = sc.tracks
            a, b = np.searchsorted(of, [f, f + 1])
            r = max(2, round(P / 130))
            for x, y in np.round(op[a:b] * [cw / sc.disp_size[0], P / sc.disp_size[1]]).astype(int):
                cv2.rectangle(cam, (x - r, y - r), (x + r, y + r), FEAT_RGB, 1)
        return cam

    def view3d(self, k: int, final: bool = False, r_pt: int = 1) -> np.ndarray:
        img, sc = np.full((self.P, self.P, 3), WHITE, np.uint8), self.sc
        if sc.map is not None:
            now = self.ms.advance(sc.f0 + k)
            P3 = self.ms.pos[np.flatnonzero(self.ms.count >= self.args.min_obs)]
            dots(img, self.v3(P3[np.all((P3 >= sc.box[0]) & (P3 <= sc.box[1]), 1)])[0], MAP_RGB, r_pt)
            if not final and len(now):
                P3 = self.ms.pos[now]
                dots(img, self.v3(P3[np.all((P3 >= sc.box[0]) & (P3 <= sc.box[1]), 1)])[0], CUR_RGB, r_pt + 1)
        poly(img, self.pred3[:k + 1], TRAJ_RGB, 1)
        for fr in self.kf3[:int(np.searchsorted(self.kf, k, side='right'))]:
            for i in range(1, 5):
                poly(img, fr[[0, i]], KF_RGB, 1)
            poly(img, fr[[1, 2, 3, 4, 1]], KF_RGB, 1)
        fr, _ = self.v3(frustum(self._pose(k), sc.K, sc.rect, 1.6 * self.frustum_m))   # ORB-SLAM: camera 1.6x a KF
        for i in range(1, 5):
            poly(img, fr[[0, i]], CAM_RGB, 2)
        poly(img, fr[[1, 2, 3, 4, 1]], CAM_RGB, 2)
        return img


class XYPlot:
    """Top-down trajectory plot in the style of evo_traj --plot_mode xy, rendered to an (h, w, 3) uint8 image; the
    axis limits are fixed by the full trajectories, the estimate is drawn up to row k."""

    def __init__(self, gt_xy: np.ndarray, est_xy: np.ndarray, w: int, h: int, dpi: float, marker: bool = False):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.w, self.h, self.est_xy = w, h, est_xy
        fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor='white')
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], '--', color=GT_HEX, lw=1.0, label='ground truth')
        self.est, = ax.plot(est_xy[:, 0], est_xy[:, 1], '-', color=EST_HEX, lw=1.0, label='estimate')
        self.cur = ax.plot([], [], 'o', color=EST_HEX, ms=3)[0] if marker else None
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x [m]', fontsize=FONT_PT)
        ax.set_ylabel('y [m]', fontsize=FONT_PT)
        ax.tick_params(labelsize=FONT_PT)
        ax.grid(True, color='0.88', lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=FONT_PT,
                  borderaxespad=0.2)
        fig.tight_layout(pad=0.4)
        self.canvas.draw()                             # applies the equal aspect, then the limits are frozen
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

    def __call__(self, k: int) -> np.ndarray:
        self.est.set_data(self.est_xy[:k + 1, 0], self.est_xy[:k + 1, 1])
        if self.cur is not None:
            self.cur.set_data(self.est_xy[k:k + 1, 0], self.est_xy[k:k + 1, 1])
        self.canvas.draw()
        img = np.asarray(self.canvas.buffer_rgba())[:self.h, :self.w, :3]
        out = np.full((self.h, self.w, 3), 255, np.uint8)   # the canvas can be 1 px short of w x h (rounding)
        out[:img.shape[0], :img.shape[1]] = img
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', type=Path, default=None, help='animated GIF')
    ap.add_argument('--png', type=Path, default=None, help='static PNG (top-down | final 3D map)')
    ap.add_argument('--data-root', type=Path, default=None, help='TUM-VI root for runs without seq.npz')
    ap.add_argument('--start', type=int, default=None, help='first shown frame (default: the run\'s first frame)')
    ap.add_argument('--end', type=int, default=-1, help="last shown frame (-1: the run's last)")
    ap.add_argument('--stride', type=int, default=11)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--width', type=int, default=1080, help='GIF width: camera | 3D | top-down, the last two square')
    ap.add_argument('--dpi', type=float, default=150.0, help='PNG resolution (the PNG is 9 x 4.5 in)')
    ap.add_argument('--grays', type=int, default=24, help='grey levels of the shared GIF palette')
    ap.add_argument('--azim', type=float, default=-60.0, help='3D view azimuth (deg, display frame, z up)')
    ap.add_argument('--elev', type=float, default=35.0, help='3D view elevation (deg)')
    ap.add_argument('--margin', type=float, default=0.15, help='3D box = trajectory box grown by margin · extent')
    ap.add_argument('--min-obs', type=int, default=3, help='frames a landmark must be seen in to be drawn')
    ap.add_argument('--kf-size', type=float, default=0.03, help='frustum depth as a fraction of the trajectory extent')
    ap.add_argument('--kf-spacing', type=float, default=0.15, help='minimum distance between drawn keyframes as a '
                    'fraction of the trajectory extent')
    args = ap.parse_args()
    if args.out is None and args.png is None:
        sys.exit('nothing to do: give --out and / or --png')

    sc = Scene(args)
    last = len(sc.T_pred) - 1
    print(f'{args.scene}: frames {sc.f0}..{sc.f0 + last} ({last} pairs, {sc.pairs_gt} with GT), translation step '
          f'error {sc.err_t[-1]:.2f} %, rotation error {sc.err_r[-1]:.3f} deg, ATE (SE(3)) {sc.ate:.3f} m')

    if args.png is not None:
        P = round(4.5 * args.dpi)
        rd = Renderer(sc, P, args)
        img = np.concatenate([XYPlot(sc.gt_xy, sc.pred[:, :2], P, P, args.dpi)(last),
                              rd.view3d(last, final=True, r_pt=2)], 1)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(args.png, optimize=True, dpi=(args.dpi, args.dpi))
        print(f'{args.png}: {img.shape[1]}x{img.shape[0]}, {args.png.stat().st_size / 1e6:.2f} MB')

    if args.out is None:
        return
    P = round(args.width / (sc.disp_size[0] / sc.disp_size[1] + 2))
    cw = args.width - 2 * P
    rd = Renderer(sc, P, args)
    xy = XYPlot(sc.gt_xy, sc.pred[:, :2], P, P, 100.0, marker=True)
    end = sc.f0 + last if args.end < 0 else min(args.end, sc.f0 + last)
    start = sc.f0 if args.start is None else max(args.start, sc.f0)
    shown = list(range(start, end + 1, args.stride))
    if shown and shown[-1] != end:                     # always end on the last frame
        shown.append(end)
    if not shown:
        sys.exit('no frames in the requested range')
    pal = palette(args.grays)
    q = []
    for f in shown:
        k = f - sc.f0                                  # row of T_pred
        im = Image.fromarray(np.concatenate([rd.camera(f, cw), rd.view3d(k), xy(k)], 1))
        q.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
    dur = [round(1000 / args.fps)] * len(q)
    dur[-1] = 3000                                     # hold the last frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
    print(f'{args.out}: {len(q)} frames ({start}..{end} every {args.stride}), {q[0].size[0]}x{q[0].size[1]}, '
          f'{args.fps:g} fps = {len(q) / args.fps:.1f} s, {args.out.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
