"""EXP-016 (E1 part 2): zero-shot audit of the large feed-forward geometry models on TUM-VI room5 (validation only).

  run   (server, venv ~/park/envs/geo, one GPU per process):
        python scripts/exp016_audit.py run --models vggt pi3x ... [--gaps 1 2 5 10] [--limit N] [--out DIR]
  score (CPU, project env):
        python scripts/exp016_audit.py score [--out DIR] [--sanity] [--json FILE]

Windows: N = 8 frames t, t+g, ..., t+7g at frame gap g; window starts t = 0, 2g, 4g, ... (stride 2g, adjacent windows
share 6 frames). Every frame of every window is a multiple of g. A window yields 7 relative poses between
window-adjacent frames, rel[w, k] = inv(T_wc[w, k]) @ T_wc[w, k + 1] (pose of frame k + 1 in camera k), i.e. the
displacement of pair i = t + k g -> i + g. Pair i lies in the windows with start i - k g, k in 0..6 and the start a
multiple of 2g: 4 windows when i / g is even (k = 0, 2, 4, 6), 3 when odd (k = 1, 3, 5), fewer at the sequence ends.

Run output (cache, so scoring needs no GPU): DIR/<model>/g<gap>.npz with starts (W,), T_wc (W, 8, 4, 4) float64 as
returned by scripts/exp016_models.py (re-based to frame 0, model units), seconds (forward pass) / wall (whole runner
call incl. pre-processing) / peak_gb per window, depth_med (W, 8) median of the positive model depth per frame, meta.
<model> = a key of exp016_models.MODELS, or <key>_noK for the runners that take K (pi3x, mapanything, worldmirror)
run without it. K = dvslam.data.tumvi.K_OUT is passed to every runner (primary); runners that cannot take K ignore it.

Scored pairs (mocap GT for scoring / diagnosis only): pair i -> i + g with i >= MIN_I = 4 (every constituent
consecutive pair (t - 1, t) has t >= 5, the scorer's --min-index 5) and mocap valid at both ends
(seq.gt_valid[i] & seq.gt_valid[i + g]). g_t = |t_gt| of inv(T_gt[i]) @ T_gt[i + g] (m), p = |t_pred|.

Measurements (every number that uses a GT-fitted scale is an ORACLE diagnostic, not a deployable result):
  A1  each pair predicted by ONE window: the window where the pair is most central (k closest to 3; for even i / g the
      tie k = 2 / 4 goes to k = 2). One L1-optimal scale per sequence s = argmin Σ |s p − g| / g
      (exp015_scale_oracle.oracle_scale 'metric'); error |s p − g| / g: mean / median / p90 (%), by GT bins
      (< 5, 5-20, 20-40, > 40 mm).
  A2  mean of the up-to-4 window predictions of a pair, each first expressed in the scale of the A1 (central) window
      w*: c_v = median over the frames shared by v and w* of depth_med[w*, f] / depth_med[v, f]. The depth of a
      shared frame is a model output that does not contain the pair's own translation (no circular self-normalisation),
      needs no GT, and keeps A1's scale reference, so A2 − A1 is only the gain from averaging. Then one sequence scale.
  B   per-window L1-optimal scale s_w over the window's scored pairs (raw model length). Series statistics on the
      windows with all 7 pairs scored and GT path Σ g ≥ 35 mm (5 mm per pair; s_w is noise for a still camera; fixed
      before any result was seen, all-window values are printed too): sd of log s_w, and sd of log s_w minus its
      running median over windows starting within ±100 frames (10 s) = window-to-window jitter. 'per-window oracle':
      A1 with every pair scaled by its own central window's s_w (item-03 mean, lower bound if windows could be
      chained perfectly; s_w is fitted on ≤ 7 pairs, so it also fits per-pair noise).
  C   pair seen in two windows v (earlier start) and w: log(p_v / p_w) on raw model lengths, and after the A2 depth
      normalisation; mean / sd / robust sd (1.4826 MAD), pairs with GT ≥ 5 mm.
  D   rotation error (geodesic angle of R_gt^T R_pred, deg) and translation-direction error (angle between t_pred and
      t_gt in the pair's first camera, deg) of the A1 prediction, by the same bins; direction only for GT ≥ 5 mm.
  E   tracker reference: saved increments (RUN/room5.inc.npz, index t = increment (t − 1) -> t) composed over g:
      T_{i -> i+g} = Π_{t=i+1}^{i+g} ΔT_t, put into the same windows; same statistics (A2 = A1, C = 0 by construction).
  F   pairs where tracker (a) fell back to the motion model (source == 'motion_model' for any t in i+1..i+g, the
      'fallback pairs' of exp003_score): A1 / D of every model and tracker on that subset, with each system's
      sequence scale from all pairs (not refitted).
  G   seconds per window (median forward / median wall), peak GPU memory (max over the run).
Decision rule (report 63 §7, fixed in advance, applied per gap): a gap qualifies for direct use if (A1 median <
min over the (b) tracker rows at that gap, or on F pairs A1 median < 0.75 × tracker (a) and direction median < 5°),
and the model's A1 median ≤ 5 % and B jitter ≤ 3 % at that gap. A model with no qualifying gap is a teacher / depth
candidate only.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

N = 8
K_MID = 3
MIN_I = 4
GAPS = (1, 2, 5, 10)
BINS_MM = (0, 5, 20, 40, np.inf)
DIR_MIN_M = 0.005
PATH_MIN_M = 0.035
RUNMED_HALF = 100
G_EPS = 1e-6
OUT = Path('runs/eval/exp016_audit')
TRACKERS = {'a hints': 'runs/eval/exp014/a_hints', 'b core': 'runs/eval/exp009b/b_hyb_scale_rep1',
            'b hints': 'runs/eval/exp014/b_hints', 'b hints L1': 'runs/eval/exp014/b_hints_lag1'}
B_ROWS = ('b core', 'b hints', 'b hints L1')
FALLBACK_RUN = 'a hints'
# EXP-015 §3.2, room5, one 'metric' oracle scale per sequence, pairs t >= 1 (reproduction target at g = 1)
EXP015_ROOM5_SEQ = {'b core': 4.065, 'b hints': 4.014, 'b hints L1': 3.840}


# ---------------------------------------------------------------- window / pair geometry (pure numpy)
def window_starts(n_frames: int, gap: int, n: int = N) -> np.ndarray:
    """Starts t = 0, 2g, ... of every N-frame window with its last frame t + (N-1) g inside the sequence."""
    return np.arange(0, n_frames - (n - 1) * gap, 2 * gap)


def window_rel(T_wc: np.ndarray) -> np.ndarray:
    """(W, N, 4, 4) window poses -> (W, N-1, 4, 4) relative poses inv(T_k) @ T_{k+1}."""
    return np.linalg.inv(T_wc[:, :-1]) @ T_wc[:, 1:]


def pair_ids(starts: np.ndarray, gap: int, k: int = N - 1) -> np.ndarray:
    """(W, N-1) start frame i of the pair i -> i + g at window position k."""
    return starts[:, None] + np.arange(k)[None] * gap


def central_choice(pid: np.ndarray) -> dict:
    """{pair i: (w, k)} of the window where i is most central (|k − 3| minimal, tie -> smaller k)."""
    w, k = np.indices(pid.shape)
    key = 2 * np.abs(k - K_MID) + (k > K_MID)
    order = np.lexsort((key.ravel(), pid.ravel()))
    out = {}
    for j in order:
        out.setdefault(int(pid.ravel()[j]), (int(w.ravel()[j]), int(k.ravel()[j])))
    return out


def compose_increments(index: np.ndarray, inc: np.ndarray, pairs: np.ndarray, gap: int) -> np.ndarray:
    """Tracker relative pose i -> i + g = Π_{t=i+1}^{i+g} ΔT_t, ΔT_t = inc[index == t]."""
    pos = {int(t): j for j, t in enumerate(index)}
    out = np.empty((len(pairs), 4, 4))
    for n, i in enumerate(pairs):
        T = np.eye(4)
        for t in range(int(i) + 1, int(i) + gap + 1):
            T = T @ inc[pos[t]]
        out[n] = T
    return out


def rot_deg(R: np.ndarray) -> np.ndarray:
    """Geodesic angle as atan2(|axis part|, (tr − 1) / 2): arccos((tr − 1) / 2) turns the ~1e-5 non-orthonormality
    of float32 model rotations into a ~0.1° floor (θ ≈ √(2δ) near 0); the skew part ignores a symmetric deviation."""
    v = 0.5 * np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return np.degrees(np.arctan2(np.linalg.norm(v, axis=-1), (np.trace(R, axis1=-2, axis2=-1) - 1) / 2))


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between vectors, atan2(|a × b|, a · b) (well conditioned near 0); NaN if either is zero."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))
    return np.where((na > 0) & (nb > 0), ang, np.nan)


def running_median(x: np.ndarray, t: np.ndarray, half: float) -> np.ndarray:
    return np.array([np.median(x[np.abs(t - tj) <= half]) for tj in t])


# ---------------------------------------------------------------- analysis of one (system, gap)
def analyse(rel_w: np.ndarray, starts: np.ndarray, gap: int, T_gt: np.ndarray, gt_valid: np.ndarray,
            depth_med: np.ndarray | None = None, min_i: int = MIN_I) -> dict:
    """Per scored pair (sorted by i) and per window statistics for one system at one gap (see module docstring)."""
    from scripts.exp015_scale_oracle import oracle_scale
    pid = pair_ids(starts, gap, rel_w.shape[1])
    p_raw = np.linalg.norm(rel_w[..., :3, 3], axis=-1)
    cen = central_choice(pid)
    ids = np.array(sorted(i for i in cen if i >= min_i and gt_valid[i] and gt_valid[i + gap]), dtype=int)
    rel_gt = np.linalg.inv(T_gt[ids]) @ T_gt[ids + gap]
    g = np.linalg.norm(rel_gt[:, :3, 3], axis=1)
    wk = np.array([cen[i] for i in ids]).reshape(-1, 2)
    rel1 = rel_w[wk[:, 0], wk[:, 1]]
    p1 = p_raw[wk[:, 0], wk[:, 1]]
    # windows of every pair; A2 depth normalisation to the central window; C overlap log ratios
    where = {}
    for (w, k), i in np.ndenumerate(pid):
        where.setdefault(int(i), []).append((w, k))
    frames = pair_ids(starts, gap, N)
    p2, lr_raw, lr_dn, lr_g = np.empty(len(ids)), [], [], []
    for n, i in enumerate(ids):
        ws, c = where[int(i)], {}
        w0 = cen[int(i)][0]
        for w, _ in ws:
            if depth_med is None:
                c[w] = 1.0
                continue
            shared = np.intersect1d(frames[w], frames[w0])
            d0 = depth_med[w0, (shared - starts[w0]) // gap]
            dw = depth_med[w, (shared - starts[w]) // gap]
            c[w] = float(np.median(d0 / dw))
        vals = [c[w] * p_raw[w, k] for w, k in ws]
        p2[n] = np.mean(vals) if np.all(np.isfinite(vals)) else np.nan
        for (v, kv), (w, kw) in itertools.combinations(sorted(ws), 2):
            if p_raw[v, kv] > 0 and p_raw[w, kw] > 0:
                lr_raw.append(np.log(p_raw[v, kv] / p_raw[w, kw]))
                lr_dn.append(np.log(c[v] * p_raw[v, kv] / (c[w] * p_raw[w, kw])))
                lr_g.append(g[n])
    # per-window oracle scale on the window's scored pairs
    gpos = {int(i): n for n, i in enumerate(ids)}
    s_w, full, path = np.ones(len(starts)), np.zeros(len(starts), bool), np.zeros(len(starts))
    for w in range(len(starts)):
        sel = [(k, gpos[int(i)]) for k, i in enumerate(pid[w]) if int(i) in gpos]
        if sel:
            ks, ns = np.array(sel).T
            s_w[w] = oracle_scale(p_raw[w, ks], g[ns], 'metric')
            full[w], path[w] = len(sel) == pid.shape[1], g[ns].sum()
    return {'gap': gap, 'ids': ids, 'g': g, 'p1': p1, 'p2': p2, 'w1': wk[:, 0],
            'rot': rot_deg(np.swapaxes(rel_gt[:, :3, :3], 1, 2) @ rel1[:, :3, :3]),
            'dir': angle_deg(rel1[:, :3, 3], rel_gt[:, :3, 3]), 's_w': s_w, 'full': full, 'path': path,
            'starts': starts, 'lr_raw': np.array(lr_raw), 'lr_dn': np.array(lr_dn), 'lr_g': np.array(lr_g)}


def item03(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.where(g > G_EPS, np.abs(p - g) / np.maximum(g, 1e-12) * 100, 0.0)


def stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0, 'mean': np.nan, 'median': np.nan, 'p90': np.nan}
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p90': float(np.percentile(x, 90))}


def robust(x: np.ndarray) -> dict:
    if not len(x):
        return {'n': 0, 'mean': np.nan, 'sd': np.nan, 'rsd': np.nan}
    return {'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std()),
            'rsd': float(1.4826 * np.median(np.abs(x - np.median(x))))}


def summarise(a: dict, subset: np.ndarray | None = None, scales: dict | None = None) -> dict:
    """A / B / C / D numbers of one analyse() result; subset = boolean mask over pairs (F), scales reuse s."""
    from scripts.exp015_scale_oracle import oracle_scale
    g, p1, p2 = a['g'], a['p1'], a['p2']
    s1 = oracle_scale(p1, g, 'metric') if scales is None else scales['s1']
    ok2 = np.isfinite(p2)
    s2 = (oracle_scale(p2[ok2], g[ok2], 'metric') if ok2.any() else np.nan) if scales is None else scales['s2']
    m = np.ones(len(g), bool) if subset is None else subset
    e1, e2 = item03(s1 * p1, g), item03(s2 * p2, g)
    gmm = g * 1e3
    out = {'s1': float(s1), 's2': float(s2), 'A1': stats(e1[m]), 'A2': stats(e2[m]), 'rot': stats(a['rot'][m]),
           'dir': stats(a['dir'][m & (g >= DIR_MIN_M)]), 'bins': {}}
    for lo, hi in itertools.pairwise(BINS_MM):
        b = m & (gmm >= lo) & (gmm < hi)
        out['bins'][f'{lo}-{hi}'] = {'A1': stats(e1[b]), 'rot': stats(a['rot'][b]),
                                     'dir': stats(a['dir'][b]) if lo >= DIR_MIN_M * 1e3 else None}
    if subset is not None:
        return out
    # B: window scales; per-window oracle item 03 through each pair's central window
    ls = np.log(a['s_w'])
    prim = a['full'] & (a['path'] >= PATH_MIN_M)
    allw = a['path'] > 0
    for name, sel in (('B', prim), ('B_all', allw)):
        x, t = ls[sel], a['starts'][sel].astype(float)
        jit = x - running_median(x, t, RUNMED_HALF) if len(x) else x
        out[name] = {'windows': int(sel.sum()), 'sd_log_s': float(x.std()) if len(x) else np.nan,
                     'jitter': float(jit.std()) if len(x) else np.nan}
    out['B']['perwin_item03'] = float(item03(a['s_w'][a['w1']] * p1, g).mean())
    big = a['lr_g'] >= DIR_MIN_M
    out['C_raw'], out['C_dn'] = robust(a['lr_raw'][big]), robust(a['lr_dn'][big])
    return out


# ---------------------------------------------------------------- data loading for scoring
def load_gt(scene: str) -> tuple[np.ndarray, np.ndarray]:
    from dvslam.data import tumvi
    seq = tumvi.load_sequence(scene, load_images=False)
    return np.asarray(seq.T_wc, np.float64), np.asarray(seq.gt_valid, bool)


def tracker_windows(run: Path, scene: str, n_frames: int, gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, rel_w (W, 7, 4, 4) of composed increments, fallback flag per pair start frame (n_frames,))."""
    from dvslam.odometry.base import load_increments
    inc = load_increments(run / f'{scene}.inc.npz')
    starts = window_starts(n_frames, gap)
    pid = pair_ids(starts, gap)
    uniq = np.unique(pid)
    rel = compose_increments(inc['index'], inc['rel_increment'], uniq, gap)
    pos = np.searchsorted(uniq, pid)
    fb_t = np.zeros(n_frames + 1, bool)
    fb_t[inc['index']] = inc['source'] != 'tracked'
    fb = np.zeros(n_frames, bool)
    for i in uniq:
        fb[i] = fb_t[i + 1:i + gap + 1].any()
    return starts, rel[pos], fb


def load_model(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d['meta'] = json.loads(str(d['meta']))
    return d


# ---------------------------------------------------------------- run (GPU)
def run(args) -> int:
    import cv2
    import torch

    from dvslam.data import tumvi
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))      # the geo venv has a site-packages `scripts` that shadows ours
    from exp016_models import MODELS, unload
    cv2.setNumThreads(4)
    seq = tumvi.load_sequence(args.scene, load_images=False)
    frames = {}

    def frame(i: int) -> np.ndarray:
        if i not in frames:
            frames[i] = cv2.imread(str(seq.image_paths[i]), cv2.IMREAD_GRAYSCALE)
        return frames[i]
    for tag in args.models:
        name, kw = (tag[:-4], {'use_K': False}) if tag.endswith('_noK') else (tag, {})
        for gap in args.gaps:
            out = args.out / tag / f'g{gap}.npz'
            if out.exists():
                print(f'[{tag}] g{gap}: exists, skipped', flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            starts = window_starts(len(seq), gap)[:args.limit]
            W = len(starts)
            T = np.empty((W, N, 4, 4))
            sec, wall, peak, dmed = np.empty(W), np.empty(W), np.empty(W), np.empty((W, N))
            try:
                MODELS[name]([frame(int(starts[0]) + k * gap) for k in range(N)], tumvi.K_OUT, **kw)   # warm-up
                t_run = time.perf_counter()
                for w, t in enumerate(starts):
                    imgs = [frame(int(t) + k * gap) for k in range(N)]
                    t0 = time.perf_counter()
                    r = MODELS[name](imgs, tumvi.K_OUT, **kw)
                    wall[w] = time.perf_counter() - t0
                    T[w], sec[w], peak[w] = r['T_wc'], r['seconds'], r['peak_gb']
                    dmed[w] = [np.median(d[np.isfinite(d) & (d > 0)]) if (d > 0).any() else np.nan
                               for d in np.asarray(r['depth'], np.float64)]
                    if w % 200 == 0:
                        print(f'[{tag}] g{gap} window {w}/{W} fwd {sec[w]:.3f} s wall {wall[w]:.3f} s '
                              f'peak {peak[w]:.2f} GB', flush=True)
            except Exception as e:                       # noqa: BLE001  (recorded, next model / gap runs)
                print(f'[{tag}] g{gap}: FAILED {type(e).__name__}: {e}', flush=True)
                out.with_suffix('.error.txt').write_text(f'{type(e).__name__}: {e}\n')
                continue
            meta = {'model': tag, 'scene': args.scene, 'gap': gap, 'n': N, 'metric': bool(r['metric']),
                    'res': list(r['res']), 'K_input': name in ('pi3x', 'mapanything', 'worldmirror') and not kw,
                    'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                    'total_s': time.perf_counter() - t_run}
            tmp = out.with_name(out.stem + '.tmp.npz')
            np.savez(tmp, starts=starts, T_wc=T, seconds=sec, wall=wall, peak_gb=peak, depth_med=dmed,
                     meta=json.dumps(meta))
            tmp.rename(out)
            print(f'[{tag}] g{gap}: {W} windows, median fwd {np.median(sec):.3f} s, wall {np.median(wall):.3f} s, '
                  f'peak {peak.max():.2f} GB, total {meta["total_s"]:.0f} s -> {out}', flush=True)
        unload()
    return 0


# ---------------------------------------------------------------- score (CPU)
def _fmt(s: dict, key: str = 'median') -> str:
    return f'{s[key]:7.2f}' if s['n'] else '      -'


def sanity(args, T_gt: np.ndarray, valid: np.ndarray) -> None:
    """GT-as-model and model-vs-itself give zero; pair indexing equals exp003_score / the tracker's increments."""
    from dvslam.odometry.base import load_increments
    from scripts.exp003_score import gt_steps
    rel_c, _ = gt_steps(args.scene)
    for path in sorted(args.out.glob('*/g*.npz')):
        d = load_model(path)
        gap, starts = d['meta']['gap'], d['starts']
        # GT poses as the 'model' (one metric scale, so the depth normalisation is c = 1)
        T_win = np.stack([np.linalg.inv(T_gt[t]) @ T_gt[t + gap * np.arange(N)] for t in starts])
        a = analyse(window_rel(T_win), starts, gap, T_gt, valid, np.ones((len(starts), N)))
        s = summarise(a)
        z1 = max(np.abs(item03(a['p1'], a['g'])).max(), np.abs(item03(a['p2'], a['g'])).max(),
                 np.nanmax(a['rot']), np.nanmax(a['dir']))
        # model against itself: 'GT' trajectory chained from the model's own central-window poses (frames i = 0, g, ..)
        rel_m = window_rel(d['T_wc'])
        cen = central_choice(pair_ids(starts, gap))
        T_self, ok_self = np.tile(np.eye(4), (len(T_gt), 1, 1)), np.zeros(len(T_gt), bool)
        ok_self[0] = True
        for i in sorted(cen):
            T_self[i + gap], ok_self[i + gap] = T_self[i] @ rel_m[cen[i]], True
        b = summarise(analyse(rel_m, starts, gap, T_self, ok_self, d['depth_med'], min_i=0))
        z2 = max(b['A1']['p90'], abs(b['s1'] - 1), b['rot']['p90'], np.nan_to_num(b['dir']['p90']))
        # pair indexing: GT displacement over g = composition of exp003's consecutive GT steps (index t = i + 1..i + g)
        comp = np.stack([np.linalg.multi_dot([rel_c[t] for t in range(i, i + gap)]) if gap > 1 else rel_c[i]
                         for i in a['ids']])
        z3 = np.abs(np.linalg.norm(comp[:, :3, 3], axis=1) - a['g']).max()
        print(f'sanity {path.parent.name} g{gap}: {len(starts)} windows, {len(a["ids"])} pairs '
              f'(i = {a["ids"][0]}..{a["ids"][-1]}); GT-as-model max err {z1:.2e} (s1 {s["s1"]:.6f}, '
              f'jitter {s["B"]["jitter"]:.2e}, C sd {s["C_raw"]["sd"]:.2e}); model-vs-itself max(A1 p90, |s-1|, '
              f'rot p90, dir p90) {z2:.2e}; '
              f'|g - composed exp003 GT steps| max {z3:.2e} m')
    for name, run_dir in TRACKERS.items():
        inc = load_increments(Path(run_dir) / f'{args.scene}.inc.npz')
        for gap in (1, 5):
            st, rel_w, _ = tracker_windows(Path(run_dir), args.scene, len(T_gt), gap)
            pid = pair_ids(st, gap)
            if gap == 1:
                pos = np.searchsorted(inc['index'], pid + 1)
                z = np.abs(rel_w - inc['rel_increment'][pos]).max()
                print(f'sanity tracker {name} g1: window pair i == increment index i + 1, max diff {z:.1e}')
            a = analyse(rel_w, st, gap, T_gt, valid)
            i5 = a['ids'][:3]
            print(f'sanity tracker {name} g{gap}: first pairs i {i5.tolist()} GT mm '
                  f'{np.round(a["g"][:3] * 1e3, 3).tolist()} tracker mm {np.round(a["p1"][:3] * 1e3, 3).tolist()}')


def score(args) -> int:
    from scripts.exp015_scale_oracle import load_room, oracle_scale
    T_gt, valid = load_gt(args.scene)
    n_frames = len(T_gt)
    if args.sanity:
        sanity(args, T_gt, valid)
        return 0
    res, cost = {}, {}
    print('ORACLE DIAGNOSTIC: every error below uses a scale fitted to mocap GT (not a deployable result).')
    # E reproduction: g = 1, pairs t >= 1, against EXP-015 and exp015.load_room
    for name, run_dir in TRACKERS.items():
        st, rel_w, _ = tracker_windows(Path(run_dir), args.scene, n_frames, 1)
        a = analyse(rel_w, st, 1, T_gt, valid, min_i=0)
        room = load_room(Path(run_dir) / f'{args.scene}.inc.npz')
        same = np.array_equal(room['index'] - 1, a['ids']) and np.allclose(room['pred'], a['p1'], rtol=0, atol=1e-12)
        e = item03(oracle_scale(a['p1'], a['g'], 'metric') * a['p1'], a['g']).mean()
        print(f'E check g1 t>=1 {name:<11}: seq-oracle item 03 {e:.3f} % (EXP-015 recorded '
              f'{EXP015_ROOM5_SEQ.get(name, float("nan")):.3f}); pairs and lengths identical to exp015.load_room: '
              f'{same}')
    systems = {}
    for path in sorted(args.out.glob('*/g*.npz')):
        systems.setdefault(path.parent.name, {})[load_model(path)['meta']['gap']] = path
    fb_by_gap = {}
    for gap in args.gaps:
        res[gap] = {}
        trk = {}
        for name, run_dir in TRACKERS.items():
            st, rel_w, fb = tracker_windows(Path(run_dir), args.scene, n_frames, gap)
            trk[name] = analyse(rel_w, st, gap, T_gt, valid)
            if name == FALLBACK_RUN:
                fb_by_gap[gap] = fb
        for name, a in trk.items():
            res[gap][f'tracker {name}'] = (a, summarise(a))
        for tag, by_gap in systems.items():
            if gap not in by_gap:
                continue
            d = load_model(by_gap[gap])
            a = analyse(window_rel(d['T_wc']), d['starts'], gap, T_gt, valid, d['depth_med'])
            res[gap][tag] = (a, summarise(a))
            cost.setdefault(tag, []).append((gap, d))
    out = {}
    for gap in args.gaps:
        rows = res[gap]
        ids0 = next(iter(rows.values()))[0]['ids']
        for k, (a, _) in rows.items():
            if not np.array_equal(a['ids'], ids0):
                raise ValueError(f'g{gap}: {k} scores a different pair set')
        fb = fb_by_gap[gap][ids0]
        print(f'\n===== gap {gap}: {len(ids0)} scored pairs (i -> i+{gap}, i = {ids0[0]}..{ids0[-1]}, i multiple of '
              f'{gap}, i >= {MIN_I}, GT valid at both ends); fallback pairs of (a) hints: {int(fb.sum())}')
        print(f'{"system":<18}{"s_seq":>8} | A1 % mean   med   p90 | A2 % mean   med | A1 med by GT bin <5 5-20 '
              f'20-40 >40 | rot° med   p90 | dir° med (>=5mm) p90 | B sd   jit  (n) all-jit | perwin03 | '
              f'C raw sd rsd | C dn sd')
        out[gap] = {}
        for k, (a, s) in rows.items():
            bins = ' '.join(_fmt(b['A1']) for b in s['bins'].values())
            print(f'{k:<18}{s["s1"]:8.4f} | {s["A1"]["mean"]:7.2f} {s["A1"]["median"]:5.2f} {s["A1"]["p90"]:5.2f} | '
                  f'{s["A2"]["mean"]:7.2f} {s["A2"]["median"]:5.2f} | {bins} | {s["rot"]["median"]:6.3f} '
                  f'{s["rot"]["p90"]:6.3f} | {s["dir"]["median"]:7.2f} {s["dir"]["p90"]:7.2f} | '
                  f'{100 * s["B"]["sd_log_s"]:5.2f} {100 * s["B"]["jitter"]:5.2f} ({s["B"]["windows"]:4d}) '
                  f'{100 * s["B_all"]["jitter"]:6.2f} | {s["B"]["perwin_item03"]:8.2f} | '
                  f'{100 * s["C_raw"]["sd"]:6.2f} {100 * s["C_raw"]["rsd"]:5.2f} | {100 * s["C_dn"]["sd"]:6.2f}')
            out[gap][k] = s
        print('  by GT bin (A1 mean / median / p90 %, rot median °, dir median ° ; n):')
        for k, (a, s) in rows.items():
            print(f'  {k:<16}' + '  '.join(
                f'{lh}: {b["A1"]["mean"]:.2f}/{b["A1"]["median"]:.2f}/{b["A1"]["p90"]:.2f} '
                f'{b["rot"]["median"]:.3f} {"-" if b["dir"] is None else format(b["dir"]["median"], ".2f")} '
                f'n{b["A1"]["n"]}' for lh, b in s['bins'].items()))
        print(f'  F: fallback pairs of tracker (a) hints at gap {gap} (n = {int(fb.sum())}, '
              f'{int((fb & (rows[next(iter(rows))][0]["g"] >= DIR_MIN_M)).sum())} with GT >= 5 mm):')
        for k, (a, s) in rows.items():
            f = summarise(a, subset=fb, scales=s)
            out[gap][k]['F'] = f
            print(f'  {k:<16} A1 mean {f["A1"]["mean"]:7.2f} med {f["A1"]["median"]:6.2f} p90 {f["A1"]["p90"]:7.2f} '
                  f'| A2 med {f["A2"]["median"]:6.2f} | rot med {f["rot"]["median"]:6.3f} | dir med '
                  f'{f["dir"]["median"]:6.2f} (n {f["dir"]["n"]})')
    # G: cost
    print('\n===== G cost (forward seconds per window median / wall median, peak GPU GB max, total GPU s over gaps)')
    for tag, lst in cost.items():
        sec = np.concatenate([d['seconds'] for _, d in lst])
        wall = np.concatenate([d['wall'] for _, d in lst])
        peak = max(float(d['peak_gb'].max()) for _, d in lst)
        tot = sum(d['meta']['total_s'] for _, d in lst)
        out.setdefault('G', {})[tag] = {'sec_median': float(np.median(sec)), 'wall_median': float(np.median(wall)),
                                        'peak_gb': peak, 'total_s': tot, 'windows': len(sec),
                                        'metric': lst[0][1]['meta']['metric'], 'gpu': lst[0][1]['meta']['gpu']}
        print(f'  {tag:<16} {np.median(sec):.3f} / {np.median(wall):.3f} s   {peak:5.2f} GB   {tot:6.0f} s   '
              f'{len(sec)} windows   metric={lst[0][1]["meta"]["metric"]}')
    # decision rule
    print('\n===== decision rule (report 63 §7, fixed in advance)')
    verdict = {}
    for tag in systems:
        qual = []
        for gap in args.gaps:
            if tag not in out[gap]:
                continue
            s = out[gap][tag]
            ref = min(out[gap][f'tracker {r}']['A1']['median'] for r in B_ROWS)
            fa = out[gap][f'tracker {FALLBACK_RUN}']['F']
            beats = s['A1']['median'] < ref
            fb_ok = s['F']['A1']['median'] < 0.75 * fa['A1']['median'] and s['F']['dir']['median'] < 5.0
            ok = (beats or fb_ok) and s['A1']['median'] <= 5.0 and s['B']['jitter'] <= 0.03
            if ok:
                qual.append(gap)
            print(f'  {tag:<16} g{gap:<2}: A1 med {s["A1"]["median"]:6.2f} vs (b) best {ref:5.2f} -> beats={beats}; '
                  f'F A1 med {s["F"]["A1"]["median"]:6.2f} vs (a) {fa["A1"]["median"]:6.2f}, F dir med '
                  f'{s["F"]["dir"]["median"]:5.2f} -> fallback={fb_ok}; jitter {100 * s["B"]["jitter"]:.2f} % '
                  f'-> qualifies={ok}')
        verdict[tag] = {'qualifying_gaps': qual, 'verdict': 'direct-use candidate' if qual else 'teacher / depth only'}
        print(f'  => {tag}: {verdict[tag]["verdict"]} {qual if qual else ""}')
    out['verdict'] = verdict
    if args.json:
        args.json.write_text(json.dumps({str(k): v for k, v in out.items()}, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--models', nargs='+', required=True)
    s = sub.add_parser('score')
    s.add_argument('--sanity', action='store_true')
    s.add_argument('--json', type=Path, default=None)
    for p in (r, s):
        p.add_argument('--scene', default='room5')
        p.add_argument('--gaps', nargs='+', type=int, default=list(GAPS))
        p.add_argument('--out', type=Path, default=OUT)
    r.add_argument('--limit', type=int, default=None, help='first N windows per gap (sanity runs)')
    args = ap.parse_args()
    if args.scene != 'room5':
        raise SystemExit('EXP-016 audits room5 only (validation room)')
    return run(args) if args.cmd == 'run' else score(args)


if __name__ == '__main__':
    sys.exit(main())
