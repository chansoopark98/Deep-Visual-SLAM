"""Plan v3.4 S5: audit of teacher depth quality on TUM-VI against the stereo hints (before E1b / E4 / E6).

  run   (server, venv ~/park/envs/geo, one GPU):
        python scripts/s5_teacher_depth_audit.py run --models pi3x vggt ... [--scenes room5] [--gaps 1 5] [--out DIR]
  score (CPU, project env):
        python scripts/s5_teacher_depth_audit.py score [--out DIR] [--json FILE]
Both run from the repository root (SSL cache and output paths are relative to it); the code itself may be a frozen
copy (runs/v34/s5_code/scripts/...), whose dvslam package is then used.

Reference: the stereo hints that `hints` training used, $DVSLAM_CACHE_ROOT/tumvi_stereo_hints/<scene>.npy (N, 512, 512)
float16 metres (runs/eval/exp010_cache_hints.py = scripts/exp009_stereo_depth.py StereoDepth: cam0 + cam1 SGBM,
left-right check, depth edges removed), indexed by ALL cam0 frames of the scene (dvslam.ssl.tumvi_data.scene_frames),
whereas the tracker sequence (dvslam.data.tumvi.load_sequence, also the index of the SSL depth caches) is cut to the
mocap range; the offset between the two is found by image file name (hint_offset). Valid reference pixels: the
training filter, hint > HINT_MIN = 0.1 m (dvslam.ssl.losses.HINT_MIN), and a finite positive candidate depth.

Frames are the tracker's: rectified cam0 512x512 pinhole K_OUT (f = 190). Every candidate is evaluated on the same
8-frame windows t, t + g, ..., t + 7g:
  tiled   (validation room5, gaps 1 and 5): blocks of 8g frames, starts b * 8g + o for o = 0..g-1, so every frame of
          the sequence (except the incomplete last block) is in exactly one window.
  sampled (train rooms room1-4, gap 1): one window of 8 consecutive frames every 160 frames (1 frame in 20).
Teachers (scripts/exp016_models.py runners, unchanged: bf16, Anchor3R fp16, K input where the runner takes it) run in
their native multi-frame mode, one window per forward pass, exactly as EXP-016. Their depth (model resolution, the
whole frame resized, never cropped) is resized back to 512x512 (cv2.INTER_LINEAR) and, like the reference and the SSL
depth, sampled on the pixel grid SUB = [2::4, 2::4] (128 x 128 = 16,384 pixels per frame) to keep the cache small.
SSL depths (`core` = runs/eval/exp003/depth, `hints` = runs/eval/exp014/depth/hints_core_last: the tracker's depth
caches, one image per forward pass) are read from their caches on the same windows.

Metrics per frame on valid pixels, d = aligned candidate depth, r = reference: AbsRel = mean |d - r| / r,
delta = fraction of max(d / r, r / d) < 1.25, log-RMSE = sqrt(mean (log d - log r)^2). Alignments:
  frame   s_k = median(r / d) per frame (shape quality only)
  window  s_w = median(r / d) over the pooled valid pixels of the window's frames (one scale per window, plan §14.3)
  none    s = 1 (metric candidates only: SSL, pi3x, mapanything, da3_nested); also the bias median log(d / r)
Scale stability: within a window, sd of log s_k over its frames (a coefficient of variation of the per-frame scale,
in %), windows with >= MIN_FRAMES valid frames; median and mean over windows. A frame is valid with >= MIN_PX valid
pixels. Reported metrics are means over valid frames (room1-4 pooled over their frames).
Verdict rule (fixed before any result, plan §14.4 S5): a teacher is better than SSL depth if, on room5 at one of its
gaps, its window-aligned AbsRel is below the best SSL depth's at that gap AND its within-window scale sd is not above
that SSL depth's. Otherwise E4 (depth backbone swap) and E6 (depth distillation) are not done.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

N = 8
SUB = (slice(2, None, 4), slice(2, None, 4))
HINT_MIN = 0.1                  # m, dvslam.ssl.losses.HINT_MIN: a hint at or below this is "no hint"
MIN_PX = 500                    # valid pixels (of 16,384) for a frame to count
MIN_FRAMES = 6                  # valid frames for a window's scale sd
TRAIN_EVERY = 160               # one 8-frame window per 160 frames on the train rooms
VAL_SCENES = ('room5',)
TRAIN_SCENES = ('room1', 'room2', 'room3', 'room4')
OUT = Path('runs/v34/s5')
SSL = {'ssl_core': 'runs/eval/exp003/depth', 'ssl_hints': 'runs/eval/exp014/depth/hints_core_last'}
METRIC = {'ssl_core', 'ssl_hints', 'pi3x', 'mapanything', 'da3_nested'}
TEACHERS = ('vggt', 'pi3x', 'mapanything', 'da3_large', 'da3_nested', 'worldmirror', 'abot_recon', 'anchor3r')


# ---------------------------------------------------------------- windows and frame mapping
def tiled_windows(n_frames: int, gap: int, n: int = N) -> np.ndarray:
    """(W, n) frame indices; every frame of the complete blocks of n * gap frames is in exactly one window."""
    blocks = n_frames // (n * gap)
    starts = (np.arange(blocks)[:, None] * n * gap + np.arange(gap)[None]).ravel()
    return starts[:, None] + gap * np.arange(n)[None]


def sampled_windows(n_frames: int, every: int = TRAIN_EVERY, n: int = N) -> np.ndarray:
    """(W, n) windows of n consecutive frames starting at 0, every, 2 every, ..."""
    starts = np.arange(0, n_frames - n + 1, every)
    return starts[:, None] + np.arange(n)[None]


def scene_windows(scene: str, n_frames: int, gap: int) -> np.ndarray:
    if scene in VAL_SCENES:
        return tiled_windows(n_frames, gap)
    if gap != 1:
        raise ValueError(f'{scene}: train rooms are sampled at gap 1 only')
    return sampled_windows(n_frames)


def hint_offset(seq_names: list[str], cam0_names: list[str]) -> int:
    """Index of the sequence's first frame in the full cam0 list (the hint cache index), checked frame by frame."""
    off = cam0_names.index(seq_names[0])
    if cam0_names[off:off + len(seq_names)] != seq_names:
        raise ValueError('tracker sequence is not a contiguous run of the cam0 frames')
    return off


def scene_index(scene: str) -> tuple[list[Path], int]:
    """Tracker image paths and the hint offset of a scene."""
    from dvslam.data import tumvi
    seq = tumvi.load_sequence(scene, load_images=False)
    _, images = tumvi.camera_frames(tumvi.sequence_dir(scene), 'cam0')
    paths = [Path(p) for p in seq.image_paths]
    return paths, hint_offset([p.name for p in paths], [Path(p).name for p in images])


# ---------------------------------------------------------------- metrics (pure numpy, unit-tested)
def valid_mask(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (ref > HINT_MIN) & np.isfinite(pred) & (pred > 0)


def median_scale(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """s = median(ref / pred) over mask (in log space; = the ratio median for an odd count)."""
    return float(np.exp(np.median(np.log(ref[mask]) - np.log(pred[mask]))))


def depth_metrics(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray, scale: float = 1.0) -> dict:
    r, d = ref[mask].astype(np.float64), scale * pred[mask].astype(np.float64)
    lr = np.log(d) - np.log(r)
    return {'absrel': float(np.mean(np.abs(d - r) / r)), 'delta': float(np.mean(np.abs(lr) < np.log(1.25))),
            'logrmse': float(np.sqrt(np.mean(lr ** 2))), 'bias': float(np.median(lr))}


def evaluate_window(ref: np.ndarray, pred: np.ndarray, min_px: int = MIN_PX) -> dict:
    """ref, pred (n, H, W) of one window. Per frame: valid flag, pixel counts, per-frame scale s_k, metrics under
    frame / window / no alignment ((n,) arrays, NaN for invalid frames); the window scale s_w; sd of log s_k."""
    n = len(ref)
    masks = [valid_mask(ref[k], pred[k]) for k in range(n)]
    npx = np.array([m.sum() for m in masks])
    ok = npx >= min_px
    out = {'ok': ok, 'npx': npx, 'nref': np.array([(ref[k] > HINT_MIN).sum() for k in range(n)]),
           's_frame': np.full(n, np.nan)}
    for a in ('frame', 'window', 'none'):
        for m in ('absrel', 'delta', 'logrmse', 'bias'):
            out[f'{a}_{m}'] = np.full(n, np.nan)
    if not ok.any():
        out['s_window'], out['log_s_sd'] = np.nan, np.nan
        return out
    lr = np.concatenate([np.log(ref[k][masks[k]]) - np.log(pred[k][masks[k]]) for k in range(n) if ok[k]])
    s_w = float(np.exp(np.median(lr)))
    for k in np.flatnonzero(ok):
        s_k = median_scale(ref[k], pred[k], masks[k])
        out['s_frame'][k] = s_k
        for a, s in (('frame', s_k), ('window', s_w), ('none', 1.0)):
            for m, v in depth_metrics(ref[k], pred[k], masks[k], s).items():
                out[f'{a}_{m}'][k] = v
    out['s_window'] = s_w
    out['log_s_sd'] = float(np.std(np.log(out['s_frame'][ok]), ddof=1)) if ok.sum() >= MIN_FRAMES else np.nan
    return out


def summarise(evals: list[dict]) -> dict:
    """Means over valid frames of every per-frame metric; scale sd over windows (median, mean); coverage."""
    cat = {k: np.concatenate([e[k] for e in evals]) for k in evals[0] if isinstance(evals[0][k], np.ndarray)}
    ok = cat['ok']
    out = {'frames': int(ok.size), 'frames_ok': int(ok.sum()),
           'coverage': float(cat['npx'][ok].sum() / max(cat['nref'][ok].sum(), 1))}
    for k, v in cat.items():
        if k.split('_')[0] in ('frame', 'window', 'none'):
            out[k] = float(np.mean(v[ok])) if ok.any() else np.nan
    sd = np.array([e['log_s_sd'] for e in evals], np.float64)
    sd = sd[np.isfinite(sd)]
    out.update(windows=len(evals), windows_sd=int(sd.size), scale_sd_median=float(np.median(sd)) if sd.size else np.nan,
               scale_sd_mean=float(np.mean(sd)) if sd.size else np.nan)
    return out


# ---------------------------------------------------------------- run (GPU, teachers)
def run(args) -> int:
    import cv2
    import torch

    from dvslam.data import tumvi
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))      # the geo venv has a site-packages `scripts` that shadows ours
    from exp016_models import MODELS, unload
    cv2.setNumThreads(4)
    for name in args.models:
        for scene in args.scenes:
            paths, _ = scene_index(scene)
            for gap in (args.gaps if scene in VAL_SCENES else [1]):
                out = args.out / name / f'{scene}_g{gap}.npz'
                if out.exists():
                    print(f'[{name}] {scene} g{gap}: exists, skipped', flush=True)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                win = scene_windows(scene, len(paths), gap)[:args.limit]
                depth = np.empty((len(win), N, 128, 128), np.float32)
                sec, wall, peak = np.empty(len(win)), np.empty(len(win)), np.empty(len(win))
                try:
                    load = [cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for i in win[0]]
                    MODELS[name](load, tumvi.K_OUT)                                      # warm-up
                    t_run = time.perf_counter()
                    for w, idx in enumerate(win):
                        imgs = [cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for i in idx]
                        t0 = time.perf_counter()
                        r = MODELS[name](imgs, tumvi.K_OUT)
                        for k, d in enumerate(np.asarray(r['depth'], np.float32)):
                            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                            depth[w, k] = cv2.resize(d, (tumvi.OUT_SIZE, tumvi.OUT_SIZE),
                                                     interpolation=cv2.INTER_LINEAR)[SUB]
                        wall[w] = time.perf_counter() - t0
                        sec[w], peak[w] = r['seconds'], r['peak_gb']
                        if w % 100 == 0:
                            print(f'[{name}] {scene} g{gap} window {w}/{len(win)} fwd {sec[w]:.3f} s '
                                  f'peak {peak[w]:.2f} GB', flush=True)
                except Exception as e:                       # noqa: BLE001  (recorded, next model / scene runs)
                    print(f'[{name}] {scene} g{gap}: FAILED {type(e).__name__}: {e}', flush=True)
                    out.with_suffix('.error.txt').write_text(f'{type(e).__name__}: {e}\n')
                    continue
                meta = {'model': name, 'scene': scene, 'gap': gap, 'metric': bool(r['metric']), 'res': list(r['res']),
                        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                        'total_s': time.perf_counter() - t_run}
                tmp = out.with_name(out.stem + '.tmp.npz')
                np.savez(tmp, windows=win, depth=depth, seconds=sec, wall=wall, peak_gb=peak, meta=json.dumps(meta))
                tmp.rename(out)
                print(f'[{name}] {scene} g{gap}: {len(win)} windows, median fwd {np.median(sec):.3f} s, '
                      f'peak {peak.max():.2f} GB, total {meta["total_s"]:.0f} s -> {out}', flush=True)
        unload()
    return 0


# ---------------------------------------------------------------- score (CPU)
def _hints(scene: str) -> np.ndarray:
    import os
    return np.load(Path(os.environ['DVSLAM_CACHE_ROOT']) / 'tumvi_stereo_hints' / f'{scene}.npy', mmap_mode='r')


def _ref(hints: np.ndarray, win: np.ndarray, off: int) -> np.ndarray:
    return np.stack([np.asarray(hints[off + i][SUB], np.float32) for i in win])


def _ssl(name: str, scene: str, paths: list[Path]) -> np.ndarray:
    d = Path(SSL[name])
    listed = (d / f'{scene}.paths.txt').read_text().split()
    if listed != [str(p) for p in paths]:
        raise ValueError(f'{name} {scene}: cache paths differ from the tracker sequence')
    return np.load(d / f'{scene}.npy', mmap_mode='r')


def offset_check(scene: str, paths: list[Path], off: int, frames: int = 20) -> dict:
    """Frame-alignment check of reference and SSL depth: frame-aligned log-RMSE of ssl_core against hints shifted by
    -1 / 0 / +1 frame on `frames` frames spread over the sequence (the true offset must be the clear minimum)."""
    hints, core = _hints(scene), _ssl('ssl_core', scene, paths)
    idx = np.linspace(2, len(paths) - 3, frames).astype(int)
    out = {}
    for sh in (-1, 0, 1):
        v = []
        for i in idx:
            r, p = np.asarray(hints[off + i + sh][SUB], np.float32), np.asarray(core[i][SUB], np.float32)
            m = valid_mask(r, p)
            v.append(depth_metrics(r, p, m, median_scale(r, p, m))['logrmse'])
        out[sh] = float(np.mean(v))
    return out


def score(args) -> int:
    res, checks = {}, {}
    for scene in VAL_SCENES + TRAIN_SCENES:
        paths, off = scene_index(scene)
        checks[scene] = {'offset': off, 'frames': len(paths), 'shift_logrmse': offset_check(scene, paths, off)}
        print(f'{scene}: hint offset {off}, frame-shift check (ssl_core frame-aligned log-RMSE, shift -1/0/+1): '
              + ' / '.join(f'{v:.4f}' for v in checks[scene]['shift_logrmse'].values()), flush=True)
        hints = _hints(scene)
        ssl = {k: _ssl(k, scene, paths) for k in SSL}
        for gap in ((1, 5) if scene in VAL_SCENES else (1,)):
            win = scene_windows(scene, len(paths), gap)
            refs = [_ref(hints, w, off) for w in win]
            for name in list(SSL) + list(TEACHERS):
                if name in SSL:
                    preds = [np.stack([np.asarray(ssl[name][i][SUB], np.float32) for i in w]) for w in win]
                else:
                    f = args.out / name / f'{scene}_g{gap}.npz'
                    if not f.exists():
                        continue
                    z = np.load(f)
                    if not np.array_equal(z['windows'], win):
                        raise ValueError(f'{f}: windows differ from {scene} g{gap}')
                    preds = list(z['depth'])
                evals = [evaluate_window(r, p) for r, p in zip(refs, preds)]
                res.setdefault(name, {}).setdefault(scene, {})[gap] = (evals, summarise(evals))
    rows = {}
    for name, sc in res.items():
        for gap in (1, 5):
            if 'room5' in sc and gap in sc['room5']:
                rows[(name, 'room5', gap)] = sc['room5'][gap][1]
        tr = [sc[s][1][0] for s in TRAIN_SCENES if s in sc and 1 in sc[s]]
        if len(tr) == len(TRAIN_SCENES):
            rows[(name, 'room1-4', 1)] = summarise([e for evals in tr for e in evals])
    _print(rows)
    _verdict(rows)
    if args.json:
        args.json.write_text(json.dumps({'checks': checks, 'rows': [dict(model=k[0], set=k[1], gap=k[2], **v)
                                                                     for k, v in rows.items()]}, indent=1))
    return 0


def _print(rows: dict) -> None:
    hdr = (f'{"model":<12} {"set":<8} {"g":>2} {"frames":>6} {"cov%":>5} | {"frame-aligned AbsRel/d<1.25/logRMSE":>36} | '
           f'{"window-aligned AbsRel/d<1.25/logRMSE":>37} | {"scale sd med/mean %":>19} | '
           f'{"unaligned AbsRel/d<1.25/logRMSE/bias%":>39}')
    print(hdr)
    for (name, s, g), v in sorted(rows.items(), key=lambda kv: (kv[0][1] != 'room5', kv[0][2], kv[1]['window_absrel'])):
        un = (f'{100 * v["none_absrel"]:8.2f} {100 * v["none_delta"]:6.2f} {100 * v["none_logrmse"]:7.2f} '
              f'{100 * v["none_bias"]:+8.2f}') if name in METRIC else f'{"-":>8} {"-":>6} {"-":>7} {"-":>8}'
        print(f'{name:<12} {s:<8} {g:>2} {v["frames_ok"]:>6} {100 * v["coverage"]:5.1f} | '
              f'{100 * v["frame_absrel"]:10.2f} {100 * v["frame_delta"]:9.2f} {100 * v["frame_logrmse"]:9.2f}      | '
              f'{100 * v["window_absrel"]:11.2f} {100 * v["window_delta"]:9.2f} {100 * v["window_logrmse"]:9.2f}      | '
              f'{100 * v["scale_sd_median"]:8.2f} {100 * v["scale_sd_mean"]:8.2f}   | {un}')


def _verdict(rows: dict) -> None:
    print('\nVerdict (room5; teacher better = window-aligned AbsRel below the best SSL AND scale sd median not above '
          'that SSL depth, at the same gap):')
    for g in (1, 5):
        ssl = {k: rows[(k, 'room5', g)] for k in SSL if (k, 'room5', g) in rows}
        if not ssl:
            continue
        best = min(ssl, key=lambda k: ssl[k]['window_absrel'])
        b = ssl[best]
        print(f'  g{g}: best SSL {best} window AbsRel {100 * b["window_absrel"]:.2f} %, scale sd '
              f'{100 * b["scale_sd_median"]:.2f} %')
        for t in TEACHERS:
            if (t, 'room5', g) not in rows:
                continue
            v = rows[(t, 'room5', g)]
            better = v['window_absrel'] < b['window_absrel'] and v['scale_sd_median'] <= b['scale_sd_median']
            print(f'    {t:<12} AbsRel {100 * v["window_absrel"]:6.2f} ({100 * (v["window_absrel"] - b["window_absrel"]):+.2f}'
                  f' pp)  scale sd {100 * v["scale_sd_median"]:6.2f} ({100 * (v["scale_sd_median"] - b["scale_sd_median"]):+.2f}'
                  f' pp)  -> {"BETTER" if better else "not better"}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--models', nargs='+', required=True, choices=TEACHERS)
    r.add_argument('--scenes', nargs='+', default=list(VAL_SCENES + TRAIN_SCENES))
    r.add_argument('--gaps', nargs='+', type=int, default=[1, 5])
    r.add_argument('--limit', type=int, default=None, help='first N windows per scene / gap (smoke runs)')
    s = sub.add_parser('score')
    s.add_argument('--json', type=Path, default=None)
    for p in (r, s):
        p.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    return run(args) if args.cmd == 'run' else score(args)


if __name__ == '__main__':
    sys.exit(main())
