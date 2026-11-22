"""Render an animated GIF of one TUM-VI tracker run: camera frame with the tracked points | top-down trajectory.

  python scripts/render_demo.py --run runs/eval/exp020/hints --scene room6 --out media/tumvi_room6_demo.gif \
      [--start 0 --end -1 --stride 11 --fps 10 --width 640 --tail 8 --grays 32]

Inputs: <run>/<scene>.inc.npz (increments of frames start+1..n-1, apps/run_sequence.py or scripts/exp003_track_log.py
format) and, when present, <run>/<scene>.tracks.npz (exp003_track_log.py: every KLT observation per frame; the points
and their last `tail` frames are drawn). Images and mocap GT come from the TUM-VI loader (DVSLAM_TUMVI_ROOT,
rectified cam0 cache under DVSLAM_CACHE_ROOT, built on first use) with the run's GT time offset.
Trajectory: the prediction is the causal chain of the increments started at the GT pose of the run's first frame
(metres, no alignment); top-down view = world x-y (the TUM-VI mocap frame has z up); GT is not drawn across mocap
dropouts (pair_valid False); the last `trail` seconds are bright, earlier parts dimmed. Overlay: frame index, time and
items 03 / 04 so far = running means over the scored pairs up to the shown frame (pairs with measured GT at both ends,
per-pair values of dvslam.eval.metrics.pair_metrics, as apps/eval_pairs.py).
GIF: frames start..end every `stride`-th frame (plus the last) at `fps`, the last one held 3 s; one shared palette
(the overlay colours + `grays` grey levels), no dithering. A whole room at the defaults is about 7 MB.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry
from dvslam.eval.metrics import pair_metrics
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import causal_trajectory, load_increments

BG = (18, 21, 27)              # RGB
GRID = (44, 49, 58)
GT_RGB = (225, 228, 232)       # GT, last `trail` seconds
GT_OLD = (96, 101, 110)        # GT, earlier
PRED_RGB = (255, 140, 40)      # prediction, last `trail` seconds
PRED_OLD = (125, 80, 42)       # prediction, earlier
PT_RGB = (60, 230, 120)        # tracked points
TAIL_RGB = (40, 150, 90)
TEXT_RGB = (235, 237, 240)


def integrate(T0: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """(n + 1, 4, 4) poses T0 · ΔT_1 · … · ΔT_t, t = 0..n (T0 for t = 0)."""
    return np.asarray(T0, np.float64) @ causal_trajectory(np.asarray(increments, np.float64))


def running_items(T_wc: np.ndarray, index: np.ndarray, increments: np.ndarray, valid: np.ndarray):
    """Items 03 (%) and 04 (deg) so far after each pair (t-1, t), t = index: running means over the pairs with
    valid[t-1] (pair_valid); NaN before the first scored pair."""
    pos, rpy, ok = [], [], []
    for t, dT in zip(index, increments):
        m = pair_metrics(relative_pose(T_wc[t - 1], T_wc[t]), dT)
        pos.append(m['pos_error_percent'])
        rpy.append(float(np.mean(m['rpy_abs_err_deg'])))
        ok.append(bool(valid[t - 1]))
    ok = np.array(ok)
    n = np.cumsum(ok)
    with np.errstate(invalid='ignore', divide='ignore'):
        i03 = np.cumsum(np.where(ok, pos, 0.0)) / n
        i04 = np.cumsum(np.where(ok, rpy, 0.0)) / n
    return np.where(n > 0, i03, np.nan), np.where(n > 0, i04, np.nan)


def palette(grays: int) -> Image.Image:
    """Shared GIF palette: the exact overlay colours + `grays` grey levels (the camera image)."""
    cols = [BG, GRID, GT_RGB, GT_OLD, PRED_RGB, PRED_OLD, PT_RGB, TAIL_RGB, TEXT_RGB]
    cols += [(v, v, v) for v in np.linspace(0, 255, grays).round().astype(int).tolist()]
    pal = Image.new('P', (1, 1))
    pal.putpalette([c for rgb in cols for c in rgb] + list(BG) * (256 - len(cols)))
    return pal


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    except (ImportError, OSError, ValueError):     # no matplotlib / font: Pillow's own scalable default
        return ImageFont.load_default(size)


class TopDown:
    """World x-y (metres) -> panel pixels, equal aspect, fixed for the whole animation."""

    def __init__(self, xy: np.ndarray, size: int, margin: int):
        lo, hi = xy.min(0), xy.max(0)
        self.c = (lo + hi) / 2
        self.s = (size - 2 * margin) / max(float((hi - lo).max()), 1e-3)
        self.size = size

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        p = (np.asarray(xy) - self.c) * self.s
        return np.stack([self.size / 2 + p[..., 0], self.size / 2 - p[..., 1]], -1)


def poly(img: np.ndarray, pts: np.ndarray, rgb, thick: int, ok: np.ndarray | None = None) -> None:
    """Anti-aliased polyline; ok (len(pts) - 1,) bool leaves out the segments i -> i + 1 where it is False."""
    runs = [pts] if ok is None else np.split(pts, np.flatnonzero(~np.asarray(ok[:len(pts) - 1], bool)) + 1)
    runs = [np.round(r * 16).astype(np.int32) for r in runs if len(r) > 1]
    if runs:
        cv2.polylines(img, runs, False, rgb, thick, cv2.LINE_AA, shift=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=None, help='default: DVSLAM_TUMVI_ROOT')
    ap.add_argument('--start', type=int, default=None, help='first shown frame (default: the run\'s first frame)')
    ap.add_argument('--end', type=int, default=-1, help="last shown frame (-1: the run's last)")
    ap.add_argument('--stride', type=int, default=11)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--width', type=int, default=640, help='GIF width; each panel is width / 2 square')
    ap.add_argument('--tail', type=int, default=8, help='frames of track history drawn behind each point')
    ap.add_argument('--trail', type=float, default=15.0, help='seconds of trajectory drawn bright (older: dimmed)')
    ap.add_argument('--grays', type=int, default=32, help='grey levels of the shared GIF palette (plus the overlay '
                    'colours)')
    ap.add_argument('--label', default='', help='extra header text, e.g. "unseen room"')
    args = ap.parse_args()

    inc = load_increments(args.run / f'{args.scene}.inc.npz')
    meta, idx, rel = inc['meta'], inc['index'], inc['rel_increment']
    if np.any(np.diff(idx) != 1):
        raise ValueError('increments must cover a contiguous frame range')
    seq = registry.load_sequence('tumvi', args.scene, None, args.data_root or registry.default_root('tumvi'),
                                 meta.get('time_offset_s', 0.0))
    f0 = int(idx[0]) - 1                                   # the run's first frame
    T_pred = integrate(seq.T_wc[f0], rel)                  # row k = frame f0 + k
    i03, i04 = running_items(seq.T_wc, idx, rel, seq.pair_valid)
    trk = args.run / f'{args.scene}.tracks.npz'
    if trk.exists():
        z = np.load(trk)
        order = np.argsort(z['obs_frame'], kind='stable')
        of, og, op = z['obs_frame'][order], z['obs_gid'][order], z['obs_px'][order]
    else:
        print(f'{trk} not found: no tracked points drawn')
        of = None

    P = args.width // 2                                    # square panel size
    H_HEAD = max(34, P // 9)
    sc = P / seq.width
    gt_xy, pred_xy = seq.T_wc[f0:, :2, 3], T_pred[:, :2, 3]
    gt_ok = seq.pair_valid[f0:]                            # GT segments across a mocap dropout are not drawn
    view = TopDown(np.concatenate([gt_xy, pred_xy]), P, margin=max(14, P // 16))
    gt_px, pred_px = view(gt_xy), view(pred_xy)
    font, small = load_font(max(11, P // 22)), load_font(max(10, P // 27))

    # static trajectory background: grid every 0.5 m, 1 m scale bar, legend
    base = np.full((P, P, 3), BG, np.uint8)
    step = 0.5
    lo = np.floor((view.c - P / 2 / view.s) / step) * step
    for v in np.arange(lo[0], view.c[0] + P / 2 / view.s + step, step):
        x = round(float(view(np.array([v, 0.0]))[0]))
        cv2.line(base, (x, 0), (x, P - 1), GRID, 1)
    for v in np.arange(lo[1], view.c[1] + P / 2 / view.s + step, step):
        y = round(float(view(np.array([0.0, v]))[1]))
        cv2.line(base, (0, y), (P - 1, y), GRID, 1)
    bar = view.s * 1.0
    x0, y0 = 10, P - 12
    cv2.line(base, (x0, y0), (int(x0 + bar), y0), TEXT_RGB, 2, cv2.LINE_AA)
    bim = Image.fromarray(base)
    d = ImageDraw.Draw(bim)
    d.text((x0 + bar / 2, y0 - 4), '1 m', font=small, fill=TEXT_RGB, anchor='mb')
    lx, ly = P - 8, 8
    for txt, rgb in (('ground truth (mocap)', GT_RGB), ('prediction (metric, no alignment)', PRED_RGB)):
        d.text((lx, ly), txt, font=small, fill=rgb, anchor='ra')
        ly += small.size + 4
    base = np.asarray(bim).copy()

    end = int(idx[-1]) if args.end < 0 else min(args.end, int(idx[-1]))   # the run's last frame
    start = f0 if args.start is None else max(args.start, f0)
    frames = []
    shown = list(range(start, end + 1, args.stride))
    if shown and shown[-1] != end:                         # always end on the last frame (final items 03 / 04)
        shown.append(end)
    for f in shown:
        k = f - f0                                         # row of T_pred; pair index k - 1
        cam = cv2.cvtColor(cv2.imread(str(seq.image_paths[f]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
        cam = cv2.resize(cam, (P, P), interpolation=cv2.INTER_AREA)
        if of is not None:
            a, b = np.searchsorted(of, [f - args.tail, f + 1])
            g, p, fr = og[a:b], op[a:b] * sc, of[a:b]
            for gid in np.unique(g[fr == f]):
                tr = p[g == gid]
                poly(cam, tr, TAIL_RGB, 1)
            for q in p[fr == f]:
                cv2.circle(cam, (round(float(q[0]) * 16), round(float(q[1]) * 16)), 2 * 16, PT_RGB, -1, cv2.LINE_AA, 4)
        traj = base.copy()
        r = max(0, k - round(args.trail * seq.frame_rate_hz))     # first row of the bright trail
        poly(traj, gt_px[:r + 1], GT_OLD, 1, gt_ok)
        poly(traj, pred_px[:r + 1], PRED_OLD, 1)
        poly(traj, gt_px[r:k + 1], GT_RGB, 2, gt_ok[r:])
        poly(traj, pred_px[r:k + 1], PRED_RGB, 2)
        for px, rgb in ((gt_px[k], GT_RGB), (pred_px[k], PRED_RGB)):
            cv2.circle(traj, (round(float(px[0])), round(float(px[1]))), 4, rgb, -1, cv2.LINE_AA)
        canvas = np.full((H_HEAD + P, 2 * P, 3), BG, np.uint8)
        canvas[H_HEAD:, :P], canvas[H_HEAD:, P:] = cam, traj
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        title = f'TUM-VI {args.scene}' + (f'  ·  {args.label}' if args.label else '')
        d.text((8, H_HEAD / 2), title, font=font, fill=TEXT_RGB, anchor='lm')
        d.text((2 * P - 8, H_HEAD / 2), f'frame {f} / {len(seq) - 1}   {seq.timestamps[f]:5.1f} s', font=font,
               fill=TEXT_RGB, anchor='rm')
        ok = k >= 1 and np.isfinite(i03[k - 1])
        for j, txt in enumerate((f'item 03 so far  {i03[k - 1]:.2f} %' if ok else 'item 03 so far  –',
                                 f'item 04 so far  {i04[k - 1]:.3f}°' if ok else 'item 04 so far  –')):
            d.text((2 * P - 8, H_HEAD + P - 8 - (1 - j) * (font.size + 4)), txt, font=font, fill=TEXT_RGB,
                   anchor='rd')
        d.text((8, H_HEAD + P - 8), 'cam0 + tracked points', font=small, fill=TEXT_RGB, anchor='ld',
               stroke_width=2, stroke_fill=BG)
        frames.append(im)
    if not frames:
        sys.exit('no frames in the requested range')

    q = [im.quantize(palette=palette(args.grays), dither=Image.Dither.NONE) for im in frames]
    dur = [round(1000 / args.fps)] * len(q)
    dur[-1] = 3000                                         # hold the final numbers
    args.out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
    print(f'{args.out}: {len(q)} frames ({start}..{end} every {args.stride}), {q[0].size[0]}x{q[0].size[1]}, '
          f'{args.fps:g} fps = {len(q) / args.fps:.1f} s, {args.out.stat().st_size / 1e6:.2f} MB; '
          f'final item 03 {i03[-1]:.3f} %, item 04 {i04[-1]:.4f}°')


if __name__ == '__main__':
    main()
