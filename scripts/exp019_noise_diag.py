"""EXP-019: where does the per-pair length noise of the best admissible tracker come from? (diagnostic, CPU only)

  python scripts/exp019_noise_diag.py RUN [RUN ...] [--rooms room1 room2 room3 room4 room5] [--json out.json]

Reads only saved outputs: RUN/room<N>.{inc,extra,tracks}.npz, the depth cache the run used (its meta 'prior_ckpt',
<dir>/<room>.npy, median depth at the tracked points), the rectified cam0 images (gradient energy, mean intensity) and
dso/cam0/times.txt (exposure). Mocap GT is used for diagnosis only. room6 is read only when listed in --rooms.

Conventions
  pair t         : (t−1, t) with the filter of scripts/exp003_score.pair_table (verified at load). Vectors are in camera
                   t−1 (x right, y down, z = optical axis): t_pred = the run's increment translation (output scale),
                   t_gt = GT. g = ‖t_gt‖, p = ‖t_pred‖, d = p − g (signed length error), item 03 = |d| / g · 100.
                   Bins: GT step < 5, 5-20, 20-40, > 40 mm.
  decomposition  : e = t_pred − t_gt, û = t_gt / g; along a = e·û, transverse e⊥ = e − a û, b = ‖e⊥‖. Exactly
                   p² = (g + a)² + b². 'transverse perfect' predicts |g + a|, 'along perfect' predicts √(g² + b²).
                   Robust σ: along 1.4826·MAD(a); transverse per component median(b) / √(2 ln 2) (2-D Gaussian).
  GT noise       : per-axis σ of the 120 Hz mocap camera centre (exp003_gt_floor.noise_sigma, mean over x, y, z of the
                   3rd-difference σ), linearly interpolated to camera time: s_end(t) = σ √(w² + (1 − w)²), pair sd
                   s_pair = √(s_end(t−1)² + s_end(t)²) per axis (as exp003_gt_floor.floor).
  budget (Q1)    : per bin, sd of d (classic and robust) against s = √mean(s_pair²) (GT noise along û); tracker-own
                   sd = √(sd² − s²). Monte Carlo item 03 of an estimator with k × the tracker's own error: ε̂ = e with
                   its along part shrunk by √(1 − s²/σ_along²) and its transverse part by √(1 − s²/σ_trans²) (robust σ
                   of the bin), p_k = ‖t_gt + k ε̂‖ scored against ‖t_gt + n‖, n ~ N(0, s_pair² I), 20 draws, seed 0.
                   k = 1 checks the model against the deployed value, k = 0 is the GT white-noise floor (EXP-003);
                   'k = 1, GT exact' scores ‖t_gt + ε̂‖ against ‖t_gt‖ (what perfect GT would report).
  predictors (Q3): per pair from extra.npz at frame t (and t + 1: the lagged output is emitted after frame t + 1),
                   tracks.npz (gids seen at t−1 and t: count, median flow px, median parallax px after removing the
                   run's own rotation with K R K⁻¹, median cached depth at the t−1 points), image t (mean squared
                   Sobel gradient, mean intensity), exposure t (ms), blur = exposure · flow / 50 ms (px), GT motion
                   (step, rotation, ‖Δv‖; diagnostic only). Spearman ρ with |d| (all pairs) and with item 03 (pairs
                   < 5 mm, pairs ≥ 20 mm), Pearson r of d² (all), item 03 (≥ 20 mm) in the lowest / highest decile
                   (two groups for 0/1 flags), and a 5-fold cross-validated (contiguous folds) R² of a linear model
                   on the ranks of all predictors, target rank of item 03 (≥ 20 mm) or of |d| (all).
  ACF (Q4)       : centred x over consecutive pairs that are both scored, ρ_k = mean(x_t x_{t+k}) / mean(x²), k = 1-10,
                   for d (mm), d / g (g ≥ 5 mm at both ends) and the world-frame error vector (dot products). The GT
                   noise of the shared frame t adds −s_end(t)² û_t·û_{t+1} to the lag-1 autocovariance of d and
                   −3 s_end(t)² to that of the vector. Model x = correlated (ρ_k = A φ^k, fitted on lags 2-10) +
                   per-frame position noise u (only lag 1: −E‖u‖²) + white w: E‖u‖² = −(ρ1 − Aφ)·var, tracker part
                   = that + GT autocov_1 (GT part is negative); variance shares A / 2E‖u‖²/var / rest. 'clipped'
                   repeats with x winsorised at its 1st / 99th percentile (vector: norm clipped at its 99th), 'rank'
                   with normal scores of x (typical pairs, not the tails). Short windows: ORACLE scale per W = 2-20
                   frames, real against the time-shuffled control of exp015 (shuffled − real = correlated part, pp).
                   Future g (GT-free, non-causal): output rescaled by e^(g_t − g_{t+k}) with the run's own g state.
  GT lever arm   : δ (camera frame) minimising the Huber norm of e − (R_rel − I) δ over scored tracked pairs: the GT
                   step error a camera-centre (lever-arm) calibration error δ would cause. Fitted per room, on room1-4
                   (applied to room5) and on room5 (applied to room1-4); item 03 against the corrected GT. Diagnosis of
                   the GT only; never used to correct an output.
                   --shared: d / g (g ≥ 5 mm, winsorised 1 / 99 %) of other runs split into a centred 5-frame mean
                   (low-pass) and the rest; Pearson r of each part between runs that differ in one input (depth
                   checkpoint, flow front-end) tells which input the correlated part follows.
  gap (Q5 i)     : non-overlapping blocks of g pairs (t0+1 .. t0+g, t0 a multiple of g, all scored): composed run and
                   GT displacement in camera t0, relative length error. 'Perfect redistribution': every pair of the
                   block gets its GT step plus e_block / g (rotated into the pair's camera with GT rotations), scored
                   with the Monte Carlo GT noise above (e_block keeps its GT end noise, √2 s / g per axis).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import norm, pearsonr, rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry, tumvi
from dvslam.odometry.base import load_increments
from scripts.exp003_gt_floor import mocap_centre, noise_sigma
from scripts.exp003_score import BINS_MM, gt_steps, pair_table
from scripts.exp015_scale_oracle import item03, scale_groups

ROOMS = ('room1', 'room2', 'room3', 'room4', 'room5')
SETS = {'room5': ('room5',), 'room1-4': ('room1', 'room2', 'room3', 'room4')}
DRAWS = 20
GAPS = (2, 5, 10)
LAGS = 10
KS = (1.0, 0.5, 0.25, 0.0)
RICE = np.sqrt(2 * np.log(2))           # median of ‖N(0, σ² I₂)‖ / σ


def rsd(x: np.ndarray) -> float:
    """Robust sd: 1.4826 · MAD."""
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def decompose(t_pred: np.ndarray, t_gt: np.ndarray) -> dict:
    """Along-track / transverse split of the increment error (see module docstring); (n, 3) inputs in metres."""
    g, p = np.linalg.norm(t_gt, axis=1), np.linalg.norm(t_pred, axis=1)
    u = t_gt / np.maximum(g, 1e-12)[:, None]
    e = t_pred - t_gt
    a = (e * u).sum(1)
    perp = e - a[:, None] * u
    b = np.linalg.norm(perp, axis=1)
    return {'g': g, 'p': p, 'd': p - g, 'e': e, 'u': u, 'along': a, 'perp': perp, 'trans': b,
            'p_trans_perfect': np.abs(g + a), 'p_along_perfect': np.sqrt(g ** 2 + b ** 2)}


def shrink_factors(along: np.ndarray, trans: np.ndarray, s: float) -> tuple[float, float]:
    """√(1 − s² / σ²) for the along and the transverse component (robust σ); 0 when GT noise explains all of it."""
    s_par, s_perp = rsd(along), float(np.median(trans)) / RICE
    f = (lambda sig: float(np.sqrt(max(0.0, 1 - s ** 2 / sig ** 2))) if sig > 0 else 0.0)
    return f(s_par), f(s_perp)


def mc_item03(t_gt: np.ndarray, eps: np.ndarray, s_pair: np.ndarray, k: float, rng: np.random.Generator,
              gt_noise: bool = True, draws: int = DRAWS) -> np.ndarray:
    """Per-pair item 03 (%) of ‖t_gt + k·eps‖ against ‖t_gt + n‖, n ~ N(0, s_pair² I), mean over draws."""
    p = np.linalg.norm(t_gt + k * eps, axis=1)
    if not gt_noise:
        return item03(p, np.linalg.norm(t_gt, axis=1))
    out = np.zeros(len(t_gt))
    for _ in range(draws):
        out += item03(p, np.linalg.norm(t_gt + rng.normal(size=t_gt.shape) * s_pair[:, None], axis=1))
    return out / draws


def acf(x: np.ndarray, ok: np.ndarray, lags: int = LAGS) -> list[float]:
    """ρ_k = mean(x_t·x_{t+k}) / mean(x·x) over t with ok[t] & ok[t+k]; x (n,) or (n, 3), centred over ok."""
    x = x - x[ok].mean(0)
    dot = (lambda a, b: a * b if a.ndim == 1 else (a * b).sum(1))
    v = dot(x[ok], x[ok]).mean()
    return [float(dot(x[:-k][ok[:-k] & ok[k:]], x[k:][ok[:-k] & ok[k:]]).mean() / v) for k in range(1, lags + 1)]


def bin_masks(g: np.ndarray) -> dict:
    gmm = g * 1e3
    out = {f'{lo}-{hi}': (gmm >= lo) & (gmm < hi) for lo, hi in itertools.pairwise(BINS_MM)}
    out['all'] = np.ones(len(g), bool)
    return out


def rot_deg(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    s = np.linalg.norm(np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1), axis=1)
    return np.degrees(np.arctan2(s / 2, c))


# ------------------------------------------------------------------ loading
def room_context(scene: str) -> dict:
    """Per-frame quantities that do not depend on the run: GT noise, images, exposure, GT poses."""
    t_ns, c = mocap_centre(scene)
    sigma = float(np.mean(noise_sigma(t_ns, c)['sigma_d3_mm'])) * 1e-3
    seq = registry.load_sequence('tumvi', scene, None, registry.default_root('tumvi'), 0.0, load_images=False)
    j = np.clip(np.searchsorted(t_ns, seq.timestamps_ns), 1, len(t_ns) - 1)
    w = (seq.timestamps_ns - t_ns[j - 1]) / np.maximum(t_ns[j] - t_ns[j - 1], 1)
    times = np.loadtxt(tumvi.sequence_dir(scene, registry.default_root('tumvi')) / 'dso/cam0/times.txt',
                       comments='#', dtype=str)
    expo = dict(zip(times[:, 0].astype(np.int64).tolist(), times[:, 2].astype(float).tolist()))
    grad, inten = np.full(len(seq), np.nan), np.full(len(seq), np.nan)
    for i, path in enumerate(seq.image_paths):
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        f = im.astype(np.float32)
        grad[i] = float(np.mean(cv2.Sobel(f, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(f, cv2.CV_32F, 0, 1) ** 2))
        inten[i] = float(f.mean())
    return {'sigma': sigma, 's_end': sigma * np.sqrt(w ** 2 + (1 - w) ** 2), 'T_wc': seq.T_wc, 'K': seq.K,
            't_ns': seq.timestamps_ns,
            'paths': [str(p) for p in seq.image_paths], 'grad': grad, 'intensity': inten,
            'exposure_ms': np.array([expo.get(int(t), np.nan) for t in seq.timestamps_ns])}


def track_features(tracks: Path, idx: np.ndarray, R_pred: np.ndarray, K: np.ndarray, depth: np.ndarray) -> dict:
    """Per pair t: tracks seen at t−1 and t, median flow, rotation-compensated parallax (px), median depth (m)."""
    with np.load(tracks) as z:
        f, gid, px = z['obs_frame'], z['obs_gid'], z['obs_px']
    o = np.argsort(f, kind='stable')
    f, gid, px = f[o], gid[o], px[o]
    st = np.searchsorted(f, np.arange(int(f.max()) + 2))
    Kinv = np.linalg.inv(K)
    out = {k: np.full(len(idx), np.nan) for k in ('n_common', 'flow_px', 'parallax_px', 'depth_m')}
    for k, t in enumerate(idx):
        if t + 1 >= len(st):
            continue
        ga, ia = np.unique(gid[st[t - 1]:st[t]], return_index=True)
        gb, ib = np.unique(gid[st[t]:st[t + 1]], return_index=True)
        _, ja, jb = np.intersect1d(ga, gb, assume_unique=True, return_indices=True)
        out['n_common'][k] = len(ja)
        if not len(ja):
            continue
        xa, xb = px[st[t - 1]:st[t]][ia[ja]], px[st[t]:st[t + 1]][ib[jb]]
        out['flow_px'][k] = np.median(np.linalg.norm(xb - xa, axis=1))
        h = np.c_[xb, np.ones(len(xb))] @ (K @ R_pred[k] @ Kinv).T
        out['parallax_px'][k] = np.median(np.linalg.norm(h[:, :2] / h[:, 2:] - xa, axis=1))
        r = np.clip(np.rint(xa).astype(int), 0, np.array(depth.shape[2:0:-1]) - 1)
        out['depth_m'][k] = np.median(np.asarray(depth[t - 1][r[:, 1], r[:, 0]]))
    return out


def load_room(run: Path, scene: str, ctx: dict) -> dict:
    """All pairs of the run (contiguous index), the scored mask, decomposition, GT noise and predictors."""
    inc_path = run / f'{scene}.inc.npz'
    inc = load_increments(inc_path)
    rel, valid = gt_steps(scene, inc['meta'].get('time_offset_s', 0.0))
    idx = inc['index']
    if np.any(np.diff(idx) != 1):
        raise ValueError(f'{inc_path}: increments must be contiguous')
    T, G = inc['rel_increment'], rel[idx - 1]
    dec = decompose(T[:, :3, 3], G[:, :3, 3])
    keep = valid[idx - 1] & (idx >= 1)
    ref = pair_table(inc_path)
    if not (np.array_equal(ref['gt'], dec['g'][keep]) and np.array_equal(ref['pred'], dec['p'][keep])):
        raise ValueError(f'{inc_path}: pair filter differs from exp003_score.pair_table')
    s_end = ctx['s_end']
    R_prev = ctx['T_wc'][idx - 1, :3, :3]
    room = {'scene': scene, 'index': idx, 'keep': keep, 'T_pred': T, 'T_gt': G, **dec,
            's_pair': np.sqrt(s_end[idx - 1] ** 2 + s_end[idx] ** 2), 's_shared': s_end[idx],
            'e_world': np.einsum('nij,nj->ni', np.nan_to_num(R_prev), dec['e']),
            'tracked': inc['source'] == 'tracked'}
    with np.load(run / f'{scene}.extra.npz') as z:
        row = {int(i): k for k, i in enumerate(z['index'])}
        ex = {k: z[k] for k in z.files if z[k].ndim == 1 and len(z[k]) == len(row)}

    def at(key: str, shift: int = 0) -> np.ndarray:
        r = np.array([row.get(int(t) + shift, -1) for t in idx])
        v = ex[key][np.maximum(r, 0)].astype(float)
        return np.where(r >= 0, v, np.nan)

    pr = {'n_tracks': at('n_tracks'), 'n_mapped': at('n_mapped'), 'n_inliers': at('n_inliers'),
          'reproj_rms_px': at('reproj_rms_px'), 'keyframe_t': at('is_keyframe'), 'keyframe_t+1': at('is_keyframe', 1),
          'ba_t': (at('ba_ms') > 0).astype(float), 'ba_t+1': (at('ba_ms', 1) > 0).astype(float),
          'n_window': at('n_window'), 'n_ba_lm': at('n_ba_lm'), 'photo_refined': (at('photo') == 1).astype(float),
          'flow_used': at('flow_used'), 'fallback': (~room['tracked']).astype(float),
          'abs_dg_state': np.abs(at('g') - at('g', -1))}
    room['g_state'] = at('g')
    dmap = Path(inc['meta']['opts']['prior_ckpt'])
    dmap = dmap if dmap.is_absolute() else REPO_ROOT / dmap
    lines = (dmap / f'{scene}.paths.txt').read_text().splitlines()
    if lines != ctx['paths']:
        raise ValueError(f'{dmap}: depth cache rows differ from the sequence frames of {scene}')
    pr.update(track_features(run / f'{scene}.tracks.npz', idx, T[:, :3, :3], ctx['K'],
                             np.load(dmap / f'{scene}.npy', mmap_mode='r')))
    pr['grad_energy'], pr['intensity'] = ctx['grad'][idx], ctx['intensity'][idx]
    pr['exposure_ms'] = ctx['exposure_ms'][idx]
    pr['blur_px'] = pr['exposure_ms'] * pr['flow_px'] / 50.0
    pr['rot_pred_deg'] = rot_deg(T[:, :3, :3])
    v = np.einsum('nij,nj->ni', np.nan_to_num(R_prev), G[:, :3, 3])          # GT step in world (m)
    pr['gt_step_mm'] = dec['g'] * 1e3
    pr['gt_rot_deg'] = rot_deg(G[:, :3, :3])
    pr['gt_dv_mm'] = np.r_[np.nan, np.linalg.norm(np.diff(v, axis=0), axis=1) * 1e3]
    room['pred'] = pr
    room['depth_path'], room['t_ns'] = dmap / f'{scene}.npy', ctx['t_ns']
    return room


# ------------------------------------------------------------------ analyses
def pool(rooms: list[dict], key: str, sub: str | None = None) -> np.ndarray:
    return np.concatenate([(r[sub][key] if sub else r[key])[r['keep']] for r in rooms])


def budget(rooms: list[dict], seed: int = 0) -> dict:
    """Q1 (+ Q5 ii): error budget per bin against the GT noise, and the Monte Carlo item 03 for k × own noise."""
    g, d, a, b = (pool(rooms, k) for k in ('g', 'd', 'along', 'trans'))
    u, perp, t_gt, s_pair = pool(rooms, 'u'), pool(rooms, 'perp'), g[:, None] * pool(rooms, 'u'), pool(rooms, 's_pair')
    p, pt, pa = pool(rooms, 'p'), pool(rooms, 'p_trans_perfect'), pool(rooms, 'p_along_perfect')
    e_ax = pool(rooms, 'e')
    masks = bin_masks(g)
    eps = np.zeros_like(t_gt)
    out = {'bins': {}, 'mc': {}}
    n = len(g)
    for name, m in masks.items():
        s = float(np.sqrt(np.mean(s_pair[m] ** 2)))
        sd, rs = float(np.std(d[m])), rsd(d[m])
        f_par, f_perp = shrink_factors(a[m], b[m], s)
        if name != 'all':
            eps[m] = f_par * a[m, None] * u[m] + f_perp * perp[m]
        out['bins'][name] = {
            'n': int(m.sum()), 'item03': float(item03(p[m], g[m]).mean()),
            'contrib_pp': float(item03(p[m], g[m]).sum() / n),
            'sd_d_mm': sd * 1e3, 'rsd_d_mm': rs * 1e3, 'gt_sd_mm': s * 1e3,
            'own_sd_mm': float(np.sqrt(max(0.0, sd ** 2 - s ** 2))) * 1e3,
            'own_rsd_mm': float(np.sqrt(max(0.0, rs ** 2 - s ** 2))) * 1e3,
            'gt_var_share': s ** 2 / sd ** 2, 'gt_var_share_robust': s ** 2 / rs ** 2,
            'shrink_along': f_par, 'shrink_trans': f_perp,
            # Q2: components (mm)
            'rsd_along_mm': rsd(a[m]) * 1e3, 'rms_along_mm': float(np.sqrt(np.mean(a[m] ** 2))) * 1e3,
            'mean_along_mm': float(np.mean(a[m])) * 1e3, 'median_along_mm': float(np.median(a[m])) * 1e3,
            'rsd_trans_mm': float(np.median(b[m])) / RICE * 1e3,
            'rms_trans_mm': float(np.sqrt(np.mean(b[m] ** 2) / 2)) * 1e3,
            'rician_mm': float(np.mean(p[m] - pt[m])) * 1e3,
            'rsd_cam_xyz_mm': [rsd(e_ax[m, i]) * 1e3 for i in range(3)],
            'rms_cam_xyz_mm': [float(np.sqrt(np.mean(e_ax[m, i] ** 2))) * 1e3 for i in range(3)],
            # Q5 ii
            'item03_trans_perfect': float(item03(pt[m], g[m]).mean()),
            'item03_along_perfect': float(item03(pa[m], g[m]).mean()),
            'contrib_trans_perfect_pp': float(item03(pt[m], g[m]).sum() / n),
            'contrib_along_perfect_pp': float(item03(pa[m], g[m]).sum() / n)}
    rng = np.random.default_rng(seed)
    for k in KS:
        v = mc_item03(t_gt, eps, s_pair, k, rng)
        out['mc'][f'k={k:g}'] = {name: float(v[m].mean()) for name, m in masks.items()}
        out['mc'][f'k={k:g}']['contrib_pp'] = {name: float(v[m].sum() / n) for name, m in masks.items()}
    v = mc_item03(t_gt, eps, s_pair, 1.0, rng, gt_noise=False)
    out['mc']['k=1 GT exact'] = {name: float(v[m].mean()) for name, m in masks.items()}
    # Q2: direction of motion against the optical axis (pairs ≥ 20 mm)
    uz = np.abs(u[:, 2])
    big = g >= 0.020
    out['direction'] = {}
    for name, m in (('|u_z| >= 0.8 (optical axis)', uz >= 0.8), ('0.5-0.8', (uz >= 0.5) & (uz < 0.8)),
                    ('|u_z| < 0.5 (lateral)', uz < 0.5)):
        m = m & big
        out['direction'][name] = {
            'n': int(m.sum()), 'item03': float(item03(p[m], g[m]).mean()) if m.any() else float('nan'),
            'median_item03': float(np.median(item03(p[m], g[m]))) if m.any() else float('nan'),
            'rsd_along_rel_pct': rsd(a[m] / g[m]) * 100, 'mean_along_rel_pct': float(np.mean(a[m] / g[m])) * 100,
            'rsd_trans_rel_pct': float(np.median(b[m] / g[m])) / RICE * 100,
            'median_g_mm': float(np.median(g[m])) * 1e3 if m.any() else float('nan')}
    return out


def cv_r2(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    """Cross-validated R² of least squares with intercept, contiguous folds (no time leakage inside a fold)."""
    n = len(y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    pred = np.empty(n)
    for lo, hi in itertools.pairwise(edges):
        tr = np.ones(n, bool)
        tr[lo:hi] = False
        coef = np.linalg.lstsq(np.c_[np.ones(tr.sum()), X[tr]], y[tr], rcond=None)[0]
        pred[lo:hi] = np.c_[np.ones(hi - lo), X[lo:hi]] @ coef
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


GT_PREDICTORS = ('gt_step_mm', 'gt_rot_deg', 'gt_dv_mm')


def predictors(rooms: list[dict]) -> dict:
    """Q3: rank correlations of each logged quantity with the per-pair error, deciles, and a joint CV R²."""
    g, d, p = pool(rooms, 'g'), pool(rooms, 'd'), pool(rooms, 'p')
    pct, ad = item03(p, g), np.abs(d)
    small, big = g < 0.005, g >= 0.020
    names = list(rooms[0]['pred'])
    out = {'per_predictor': {}, 'ref_item03_big': float(pct[big].mean()), 'n_big': int(big.sum())}
    ranks = {}
    for k in names:
        x = pool(rooms, k, 'pred')
        ok = np.isfinite(x)
        ranks[k] = np.where(ok, rankdata(np.where(ok, x, 0)) / len(x), 0.5)

        def rho(m, y, x=x, ok=ok):
            mm = m & ok
            return float(spearmanr(x[mm], y[mm])[0]) if mm.sum() > 10 and np.ptp(x[mm]) > 0 else float('nan')

        mb = big & ok
        row = {'n_finite': int(ok.sum()), 'rho_absd_all': rho(np.ones(len(x), bool), ad),
               'rho_item03_small': rho(small, pct), 'rho_item03_big': rho(big, pct),
               'r_d2_all': float(pearsonr(x[ok], d[ok] ** 2)[0]) if np.ptp(x[ok]) > 0 else float('nan')}
        xv = x[mb]
        if len(np.unique(xv)) <= 2:
            lo_m, hi_m = mb & (x == xv.min()), mb & (x == xv.max())
            row.update(kind='flag', share_hi=float(hi_m.sum() / mb.sum()))
        else:
            q10, q90 = np.quantile(xv, [0.1, 0.9])
            lo_m, hi_m = mb & (x <= q10), mb & (x >= q90)
            row.update(kind='decile', q10=float(q10), q90=float(q90))
        row.update(item03_lo=float(pct[lo_m].mean()) if lo_m.any() else float('nan'),
                   item03_hi=float(pct[hi_m].mean()) if hi_m.any() else float('nan'))
        out['per_predictor'][k] = row
    deploy = [k for k in names if k not in GT_PREDICTORS]
    for label, cols in (('deployable', deploy), ('with GT motion', names)):
        X = np.stack([ranks[k] for k in cols], 1)
        out[f'cv_r2_big {label}'] = cv_r2(X[big], rankdata(pct[big]) / big.sum())
        out[f'cv_r2_absd_all {label}'] = cv_r2(X, rankdata(ad) / len(ad))
    return out


def padded(arrs: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-room series with LAGS zero rows between rooms, so no lag pairs two rooms."""
    pad = np.zeros((LAGS, *arrs[0].shape[1:]), arrs[0].dtype)
    return np.concatenate([x for a in arrs for x in (np.nan_to_num(a), pad)][:-1])


def ar_fit(rho: list[float]) -> dict:
    """ρ_k ≈ A φ^k on lags 2-10 (grid φ, least-squares A clipped to [0, 1]); lag-1 excess ρ1 − Aφ. Among fits within
    0.1 % of the least SSE the smallest Aφ is kept (lags ≥ 2 near 0 do not fix A for small φ). Variance shares:
    correlated A, per-frame position noise 2·max(0, −excess), the rest white."""
    r = np.asarray(rho)
    k = np.arange(2, len(r) + 1)
    fits = []
    for phi in np.linspace(0.01, 0.99, 99):
        A = float(np.clip((r[1:] * phi ** k).sum() / (phi ** (2 * k)).sum(), 0, 1))
        fits.append((float(((r[1:] - A * phi ** k) ** 2).sum()), A * phi, A, phi))
    tol = min(f[0] for f in fits) * 1.001 + 1e-12
    _, _, A, phi = min((f for f in fits if f[0] <= tol), key=lambda f: f[1])
    ex = float(r[0] - A * phi)
    pos = 2 * max(0.0, -ex)
    return {'A': A, 'phi': float(phi), 'tau_frames': float(-1 / np.log(phi)), 'lag1_excess': ex,
            'share_correlated': A, 'share_position': pos, 'share_white': 1 - A - pos}


def temporal(rooms: list[dict]) -> dict:
    """Q4: autocorrelation of the signed length error and of the error vector; GT and position-noise parts."""
    out = {}
    series = {'d_mm': lambda r: r['d'] * 1e3, 'rel_pct': lambda r: r['d'] / np.maximum(r['g'], 1e-12) * 100,
              'vec_mm': lambda r: r['e_world'] * 1e3}
    u, s2 = padded([r['u'] for r in rooms]), padded([r['s_shared'] for r in rooms]) ** 2 * 1e6
    for name, fn in series.items():
        ok = padded([r['keep'] & (r['g'] >= 0.005) if name == 'rel_pct' else r['keep'] for r in rooms])
        for mode in ('raw', 'clipped', 'rank'):
            if name == 'vec_mm' and mode == 'rank':
                continue
            x = padded([fn(r) for r in rooms])
            if mode == 'clipped' and name == 'vec_mm':
                c = np.quantile(np.linalg.norm(x[ok], axis=1), 0.99)
                x = x * np.minimum(1, c / np.maximum(np.linalg.norm(x, axis=1), 1e-12))[:, None]
            elif mode == 'clipped':
                x = np.clip(x, *np.quantile(x[ok], [0.01, 0.99]))
            elif mode == 'rank':                                    # normal scores: typical pairs, not the tails
                x = x.copy()
                x[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / ok.sum())
                x[~ok] = 0.0
            row = {'acf': acf(x, ok)}
            row['fit'] = ar_fit(row['acf'])
            xc = x - x[ok].mean(0)
            dot = (lambda a, b: a * b if a.ndim == 1 else (a * b).sum(1))
            both = ok[:-1] & ok[1:]
            var, ac1 = float(dot(xc[ok], xc[ok]).mean()), float(dot(xc[:-1][both], xc[1:][both]).mean())
            row.update(var=var, autocov1=ac1)
            if name in ('d_mm', 'vec_mm') and mode != 'rank':
                gt = (-(s2[:-1] * (u[:-1] * u[1:]).sum(1))[both].mean() if name == 'd_mm'
                      else -3 * s2[:-1][both].mean())
                pos = max(0.0, -row['fit']['lag1_excess']) * var
                row.update(gt_autocov1=float(gt), gt_rho1=float(gt / var), pos_noise=pos,
                           pos_noise_tracker=pos + float(gt), pos_share=2 * pos / var,
                           pos_share_tracker=2 * max(0.0, pos + float(gt)) / var)
            out[name if mode == 'raw' else f'{name} {mode}'] = row
    return out


WINDOWS_SHORT = (2, 3, 5, 10, 20)
FUTURE_G = (1, 2, 3, 5, 10)


def short_windows(rooms: list[dict]) -> dict:
    """Q4 on the item-03 scale: ORACLE scale per W-frame window (exp015 'metric') against the shuffled control
    (seed 0 per room, as exp015); shuffled − real = what time correlation inside W frames is worth."""
    out = {'deployed': float(item03(pool(rooms, 'p'), pool(rooms, 'g')).mean())}
    for w in WINDOWS_SHORT:
        real, shuf, gs = [], [], []
        for r in rooms:
            p, g, idx = r['p'][r['keep']], r['g'][r['keep']], r['index'][r['keep']]
            perm = np.random.default_rng(0).permutation(len(p))
            real.append(scale_groups(p, g, (idx - 1) // w, 'metric')[0])
            s = np.empty_like(p)
            s[perm] = scale_groups(p[perm], g[perm], (idx - 1) // w, 'metric')[0]
            shuf.append(s)
            gs.append(g)
        g = np.concatenate(gs)
        out[f'W={w}'] = {'real': float(item03(np.concatenate(real), g).mean()),
                         'shuffled': float(item03(np.concatenate(shuf), g).mean())}
        out[f'W={w}']['correlated_pp'] = out[f'W={w}']['shuffled'] - out[f'W={w}']['real']
    return out


def future_g(rooms: list[dict]) -> dict:
    """GT-free (non-causal) check of the scale-state lag: rescale each output by e^(g_t − g_{t+k}), i.e. apply the
    scale state k frames later (the tracker's own extra.npz g, no GT). Also |Δg| in the 5 frames after t."""
    out = {}
    for k in (0, *FUTURE_G):
        p, g = [], []
        for r in rooms:
            gs = r['g_state']
            fut = np.r_[gs[k:], np.full(k, np.nan)] if k else gs
            f = np.where(np.isfinite(fut) & np.isfinite(gs), np.exp(gs - fut), 1.0)
            p.append((r['p'] * f)[r['keep']])
            g.append(r['g'][r['keep']])
        out[f'k={k}'] = float(item03(np.concatenate(p), np.concatenate(g)).mean())
    return out


def gap_analysis(rooms: list[dict], seed: int = 0) -> dict:
    """Q5 i: g-frame displacement error and the item 03 of a perfect redistribution of it over the g pairs."""
    rng = np.random.default_rng(seed)
    out = {}
    for gap in (1, *GAPS):
        rel, eq_mm, d1, tg, tn, sp, pdep = [], [], [], [], [], [], []
        for r in rooms:
            idx, keep = r['index'], r['keep']
            for t0 in range(((idx[0] - 1 + gap - 1) // gap) * gap, idx[-1] - gap + 1, gap):
                ks = np.arange(t0 + 1 - idx[0], t0 + 1 - idx[0] + gap)
                if ks[0] < 0 or ks[-1] >= len(idx) or not keep[ks].all():
                    continue
                P, G = np.eye(4), np.eye(4)
                for k in ks:
                    P, G = P @ r['T_pred'][k], G @ r['T_gt'][k]
                lp, lg = np.linalg.norm(P[:3, 3]), np.linalg.norm(G[:3, 3])
                if lg <= 1e-6:
                    continue
                rel.append(abs(lp - lg) / lg * 100)
                eq_mm.append(abs(lp - lg) / gap * 1e3)
                eb, Rc = P[:3, 3] - G[:3, 3], np.eye(3)
                for k in ks:
                    t_gt = r['T_gt'][k][:3, 3]
                    tg.append(t_gt)
                    tn.append(t_gt + Rc.T @ eb / gap)
                    sp.append(r['s_pair'][k])
                    pdep.append(r['p'][k])
                    d1.append(abs(r['d'][k]) * 1e3)
                    Rc = Rc @ r['T_gt'][k][:3, :3]
        tg, tn, sp, pdep = np.array(tg), np.array(tn), np.array(sp), np.array(pdep)
        gl = np.linalg.norm(tg, axis=1)
        row = {'blocks': len(rel), 'pairs': len(tg), 'median_rel_pct': float(np.median(rel)),
               'mean_rel_pct': float(np.mean(rel)), 'median_eq_mm': float(np.median(eq_mm)),
               'median_pair_absd_mm': float(np.median(d1)), 'deployed_item03_same_pairs': float(item03(pdep, gl).mean())}
        if gap > 1:
            row['redistributed_item03'] = float(mc_item03(tg, tn - tg, sp, 1.0, rng).mean())
            row['redistributed_item03_gt_exact'] = float(mc_item03(tg, tn - tg, sp, 1.0, rng, gt_noise=False).mean())
            row['floor_same_pairs'] = float(mc_item03(tg, np.zeros_like(tg), sp, 0.0, rng).mean())
        out[f'g={gap}'] = row
    return out


def lever_fit(rooms: list[dict], iters: int = 10) -> np.ndarray:
    """δ (camera frame, m) minimising Σ ρ_Huber(‖e − (R_rel − I) δ‖) over scored tracked pairs (IRLS, c = 1.345 ·
    robust σ of the residual norm). A lever-arm error δ in the camera centre of the GT makes the GT step wrong by
    (R_rel − I) δ (rotation-dependent, the same for every tracker)."""
    A = np.concatenate([(r['T_gt'][:, :3, :3] - np.eye(3))[r['keep'] & r['tracked']] for r in rooms])
    e = np.concatenate([r['e'][r['keep'] & r['tracked']] for r in rooms])
    w, delta = np.ones(len(e)), np.zeros(3)
    for _ in range(iters):
        sw = np.sqrt(w)[:, None]
        delta = np.linalg.lstsq((A * sw[:, :, None]).reshape(-1, 3), (e * sw).reshape(-1), rcond=None)[0]
        res = np.linalg.norm(e - A @ delta, axis=1)
        c = 1.345 * 1.4826 * np.median(res)
        w = np.where(res <= c, 1.0, c / np.maximum(res, 1e-15))
    return delta


def lever_item03(rooms: list[dict], delta: np.ndarray) -> float:
    """item 03 against the GT step corrected by (R_rel − I) δ."""
    p, g = [], []
    for r in rooms:
        t = r['T_gt'][:, :3, 3] + (r['T_gt'][:, :3, :3] - np.eye(3)) @ delta
        p.append(r['p'][r['keep']])
        g.append(np.linalg.norm(t, axis=1)[r['keep']])
    return float(item03(np.concatenate(p), np.concatenate(g)).mean())


def lever_arm(loaded: dict) -> dict:
    """GT lever-arm diagnosis: δ per room and pooled; δ fitted on room1-4 applied to room5 and vice versa."""
    out = {'per_room_mm': {s: (lever_fit([r]) * 1e3).tolist() for s, r in loaded.items()}}
    for fit_set, app_set in (('room1-4', 'room5'), ('room5', 'room1-4')):
        if not all(s in loaded for s in SETS[fit_set] + SETS[app_set]):
            continue
        delta = lever_fit([loaded[s] for s in SETS[fit_set]])
        out[f'fit {fit_set}'] = {'delta_mm': (delta * 1e3).tolist()}
        for name in (fit_set, app_set):
            rr = [loaded[s] for s in SETS[name]]
            out[f'fit {fit_set}'][name] = {'before': lever_item03(rr, np.zeros(3)), 'after': lever_item03(rr, delta)}
    return out


SLOW_MED = 41        # frames (2 s): the slow running median a scale series is compared with
SMOOTH = 10          # frames (0.5 s): the smoothing of the follow-up gain estimate
G_TAU = 5            # the runs' g_tau: g ← g + (m − g) / g_tau at a BA, so m = g_t + g_tau (g_{t+1} − g_t)


def running(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Centred running fn (np.nanmedian / np.nanmean) over w frames; NaN where the window has no finite value."""
    h = w // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        v = x[max(0, i - h):i + h + 1]
        if np.isfinite(v).any():
            out[i] = fn(v)
    return out


def r2_segments(x: np.ndarray, seg: np.ndarray) -> float:
    """Share of the variance of x (finite entries) explained by one constant per segment id."""
    ok = np.isfinite(x)
    x, seg = x[ok], seg[ok]
    _, inv = np.unique(seg, return_inverse=True)
    mean = np.bincount(inv, x) / np.bincount(inv)
    return float(1 - ((x - mean[inv]) ** 2).sum() / ((x - x.mean()) ** 2).sum())


def ssl_stereo_logratio(depth: np.ndarray, hint: np.ndarray, rows: np.ndarray, step: int = 4) -> np.ndarray:
    """Per frame f: median over pixels with a stereo hint of log(SSL depth / stereo depth) (NaN: < 500 pixels)."""
    out = np.full(len(rows), np.nan)
    for f, r in enumerate(rows):
        if r < 0:
            continue
        s, h = np.asarray(depth[f, ::step, ::step], np.float32), np.asarray(hint[r, ::step, ::step], np.float32)
        ok = (h > 0.1) & (h < 20) & np.isfinite(s) & (s > 0.1)
        if ok.sum() >= 500:
            out[f] = float(np.median(np.log(s[ok] / h[ok])))
    return out


def scale_sources(rooms: list[dict], hints_dir: Path | None, seed: int = 0) -> dict:
    """Follow-up of Q4: is the persistent signed length error a keyframe-interval or depth-scale effect?
    (a) variance of the relative error (g ≥ 5 mm, winsorised 1 / 99 %) explained by one constant per host-keyframe
    interval against contiguous random segments of the same lengths in random order (50 draws). (b) Pearson /
    Spearman r of the error of pair t with the deviation of a per-frame scale series at frame t + L from its SLOW_MED
    running median, L = −3..3: the global scale state g (logged), its per-BA measurement m_k (recovered from g,
    held until the next BA) and, with stereo hints, log(SSL depth / stereo depth) per frame. (c) ESTIMATE of the
    item-03 change if a series were smoothed over SMOOTH frames: for g directly (output × e^(g − ĝ)); for the
    others p × (1 − β Δq) with β fitted on the same pairs (uses GT, optimistic)."""
    rng = np.random.default_rng(seed)
    xs, segs, ctrl_lengths, series = [], [], [], {'g': [], 'm_k': [], 'ssl/stereo': []}
    offset = 0
    for r in rooms:
        idx, n = r['index'], int(r['index'][-1]) + 2
        x = np.full(n, np.nan)
        ok = r['keep'] & (r['g'] >= 0.005)
        x[idx[ok]] = r['d'][ok] / r['g'][ok] * 100
        xs.append(x)
        kf = np.zeros(n, bool)
        kf[idx] = np.nan_to_num(r['pred']['keyframe_t']) > 0
        kf[0] = True
        seg = np.r_[0, np.cumsum(kf)[:-1]]                  # host keyframe of pair t = last keyframe at ≤ t − 1
        segs.append(seg + offset)
        offset += seg.max() + 1
        ctrl_lengths.append(np.bincount(seg))
        g = np.full(n, np.nan)
        g[idx] = r['g_state']
        m = np.full(n, np.nan)
        dg = np.r_[g[1:] - g[:-1], np.nan]
        at = np.isfinite(dg) & (np.abs(dg) > 0)
        m[at] = g[at] + G_TAU * dg[at]
        last = np.nan
        for i in range(n):                                  # hold the last measurement
            last = m[i] if np.isfinite(m[i]) else last
            m[i] = last
        series['g'].append(g)
        series['m_k'].append(m)
        q = np.full(n, np.nan)
        if hints_dir is not None and (hints_dir / f'{r["scene"]}.npy').exists():
            t_hint, _ = tumvi.camera_frames(tumvi.sequence_dir(r['scene'], registry.default_root('tumvi')), 'cam0')
            j = np.clip(np.searchsorted(t_hint, r['t_ns']), 0, len(t_hint) - 1)
            rows = np.where(t_hint[j] == r['t_ns'], j, -1)
            q[:len(rows)] = ssl_stereo_logratio(np.load(r['depth_path'], mmap_mode='r'),
                                                np.load(hints_dir / f'{r["scene"]}.npy', mmap_mode='r'), rows)
        series['ssl/stereo'].append(q)
    lo, hi = np.nanquantile(np.concatenate(xs), [0.01, 0.99])
    xw = [np.clip(x, lo, hi) for x in xs]
    X, S = np.concatenate(xw), np.concatenate(segs)
    ctrl = []
    for _ in range(50):
        cs, off = [], 0
        for lens in ctrl_lengths:
            perm = rng.permutation(lens)
            cs.append(np.repeat(np.arange(len(perm)) + off, perm))
            off += len(perm)
        ctrl.append(r2_segments(X, np.concatenate(cs)))
    allk = np.concatenate([np.bincount(s - s.min()) for s in segs])
    out = {'kf_intervals': {'r2': r2_segments(X, S), 'r2_control_mean': float(np.mean(ctrl)),
                            'r2_control_sd': float(np.std(ctrl)), 'mean_len_frames': float(allk.mean()),
                            'median_len_frames': float(np.median(allk)), 'n_intervals': len(allk)}}
    out['corr'], out['gain'] = {}, {}
    for name, ser in series.items():
        dev = [s - running(s, SLOW_MED, np.nanmedian) for s in ser]
        out['corr'][name] = {}
        for L in range(-3, 4):
            a, b = [], []
            for x, d in zip(xw, dev):
                dd = np.r_[d[L:], np.full(L, np.nan)] if L >= 0 else np.r_[np.full(-L, np.nan), d[:L]]
                ok = np.isfinite(x) & np.isfinite(dd)
                a.append(x[ok])
                b.append(dd[ok])
            a, b = np.concatenate(a), np.concatenate(b)
            out['corr'][name][L] = ({'n': len(a), 'pearson': float(pearsonr(a, b)[0]),
                                     'spearman': float(spearmanr(a, b)[0]), 'sd_dev': float(np.std(b))}
                                    if len(a) > 100 and np.std(b) > 0 else {'n': len(a)})
    for name, ser in series.items():                        # (c) estimate per room set, at lag 0 and the best lag
        best = 0 if name == 'g' else max((L for L in out['corr'][name] if 'pearson' in out['corr'][name][L]),
                                         key=lambda L: abs(out['corr'][name][L]['pearson']), default=None)
        if best is None:
            continue
        p_new, p_old, gg, dq_all, x_all = [], [], [], [], []
        for r, s, x in zip(rooms, ser, xw):
            dq = s - running(s, SMOOTH, np.nanmean)
            dq = np.r_[dq[best:], np.full(best, np.nan)] if best >= 0 else np.r_[np.full(-best, np.nan), dq[:best]]
            dq_all.append(dq[r['index']])
            x_all.append(x[r['index']])
        D, Xa = np.concatenate(dq_all), np.concatenate(x_all) / 100
        okf = np.isfinite(D) & np.isfinite(Xa)
        beta = float((D[okf] * Xa[okf]).sum() / (D[okf] ** 2).sum()) if name != 'g' else 1.0
        for r, dq in zip(rooms, dq_all):
            k = r['keep']
            f = np.exp(np.nan_to_num(dq)) if name == 'g' else 1 - beta * np.nan_to_num(dq)
            p_old.append(r['p'][k])
            p_new.append((r['p'] * f)[k])
            gg.append(r['g'][k])
        gg = np.concatenate(gg)
        out['gain'][name] = {'lag': int(best), 'beta': beta, 'item03_before': float(item03(np.concatenate(p_old),
                                                                                              gg).mean()),
                             'item03_after': float(item03(np.concatenate(p_new), gg).mean())}
    return out


LOWPASS = 5


def rel_series(run: Path, scene: str) -> np.ndarray:
    """Relative signed length error d / g (%) per frame index of a run (NaN where not scored or g < 5 mm); split
    into a centred LOWPASS-frame mean of the available values (low) and the rest (high)."""
    inc = load_increments(run / f'{scene}.inc.npz')
    rel, valid = gt_steps(scene, inc['meta'].get('time_offset_s', 0.0))
    idx = inc['index']
    g = np.linalg.norm(rel[idx - 1, :3, 3], axis=1)
    p = np.linalg.norm(inc['rel_increment'][:, :3, 3], axis=1)
    out = np.full(len(rel) + 1, np.nan)
    ok = valid[idx - 1] & (g >= 0.005)
    out[idx[ok]] = (p[ok] - g[ok]) / g[ok] * 100
    return out


def shared_parts(runs: list[Path], rooms: tuple) -> dict:
    """Pearson r of the low-pass (centred LOWPASS-frame mean) and high-pass parts of d / g between runs, on the
    pairs scored in both; winsorised at 1 / 99 % per run. Says which tracker inputs the correlated part follows."""
    lo_hi = {}
    for run in runs:
        parts = []
        for s in rooms:
            x = rel_series(run, s)
            fin = np.isfinite(x)
            q = np.nanquantile(x, [0.01, 0.99])
            xc = np.where(fin, np.clip(x, *q), 0.0)
            k = np.ones(LOWPASS)
            low = np.convolve(xc, k, 'same') / np.maximum(np.convolve(fin.astype(float), k, 'same'), 1)
            parts.append((np.where(fin, low, np.nan), np.where(fin, xc - low, np.nan)))
        lo_hi[run] = (np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
    out = {}
    for a, b in itertools.combinations(runs, 2):
        m = np.isfinite(lo_hi[a][0]) & np.isfinite(lo_hi[b][0])
        out[f'{a} vs {b}'] = {'n': int(m.sum()), 'r_low': float(np.corrcoef(lo_hi[a][0][m], lo_hi[b][0][m])[0, 1]),
                              'r_high': float(np.corrcoef(lo_hi[a][1][m], lo_hi[b][1][m])[0, 1]),
                              'var_low_share_a': float(np.var(lo_hi[a][0][m]) / np.var(lo_hi[a][0][m] +
                                                                                         lo_hi[a][1][m]))}
    return out


def compare_runs(a: list[dict], b: list[dict]) -> dict:
    """Same pairs of two runs: correlation of d, variance of each and of the difference, ACF of the difference."""
    da, db = np.concatenate([r['d'] for r in a]) * 1e3, np.concatenate([r['d'] for r in b]) * 1e3
    ok = np.concatenate([r['keep'] & s['keep'] for r, s in zip(a, b)])
    return {'corr_d': float(np.corrcoef(da[ok], db[ok])[0, 1]), 'var_a': float(np.var(da[ok])),
            'var_b': float(np.var(db[ok])), 'var_diff': float(np.var(da[ok] - db[ok])),
            'acf_diff': acf(da - db, ok)}


# ------------------------------------------------------------------ report
def report(name: str, res: dict) -> None:
    print(f'\n######## {name}')
    for sname, r in res.items():
        if not isinstance(r, dict) or 'budget' not in r:
            continue
        B, P, A, Q = r['budget'], r['predictors'], r['temporal'], r['gap']
        print(f'\n== {sname}: deployed item 03 per room ' + '  '.join(f'{k} {v:.3f}' for k, v in r['rooms'].items()))
        print('Q1/Q2 per GT-step bin: n | item03 contrib | sd(d) rsd(d) GT-sd own-sd own-rsd (mm) | GT var share '
              'classic/robust | along rsd/rms/median, trans rsd/rms, rician (mm) | cam x/y/z rsd (mm) | '
              'item03 trans-perfect / along-perfect')
        for b, v in B['bins'].items():
            print(f'  {b:>7} {v["n"]:5d} | {v["item03"]:6.2f} {v["contrib_pp"]:.3f} | {v["sd_d_mm"]:.3f} '
                  f'{v["rsd_d_mm"]:.3f} {v["gt_sd_mm"]:.3f} {v["own_sd_mm"]:.3f} {v["own_rsd_mm"]:.3f} | '
                  f'{v["gt_var_share"]:.3f}/{v["gt_var_share_robust"]:.3f} | {v["rsd_along_mm"]:.3f}/'
                  f'{v["rms_along_mm"]:.3f}/{v["median_along_mm"]:+.3f} {v["rsd_trans_mm"]:.3f}/{v["rms_trans_mm"]:.3f}'
                  f' {v["rician_mm"]:+.3f} | ' + '/'.join(f'{x:.3f}' for x in v['rsd_cam_xyz_mm']) +
                  f' | {v["item03_trans_perfect"]:.2f} / {v["item03_along_perfect"]:.2f}')
        print('Q1 Monte Carlo item 03 (all / bins):')
        for k, v in B['mc'].items():
            print(f'  {k:<13} ' + '  '.join(f'{b} {v[b]:.3f}' for b in ('all', *[x for x in v if x not in
                                                                               ('all', 'contrib_pp')])))
        print('Q2 direction (pairs >= 20 mm): n item03 median | along rel rsd % mean % | trans rel rsd %')
        for k, v in B['direction'].items():
            print(f'  {k:<28} {v["n"]:5d} {v["item03"]:.2f} {v["median_item03"]:.2f} | {v["rsd_along_rel_pct"]:.2f}'
                  f' {v["mean_along_rel_pct"]:+.2f} | {v["rsd_trans_rel_pct"]:.2f}   (median g {v["median_g_mm"]:.1f})')
        print(f'Q3 predictors: rho(|d|, all) rho(item03, <5) rho(item03, >=20) r(d², all) | item03 (>=20 mm, '
              f'mean {P["ref_item03_big"]:.2f}, n {P["n_big"]}) low / high decile or flag 0 / 1')
        for k, v in sorted(P['per_predictor'].items(), key=lambda kv: -abs(np.nan_to_num(kv[1]['rho_item03_big']))):
            extra = (f'q10 {v["q10"]:.4g} q90 {v["q90"]:.4g}' if v['kind'] == 'decile' else f'share 1: {v["share_hi"]:.3f}')
            print(f'  {k:<16} {v["rho_absd_all"]:+.3f} {v["rho_item03_small"]:+.3f} {v["rho_item03_big"]:+.3f} '
                  f'{v["r_d2_all"]:+.3f} | {v["item03_lo"]:.2f} / {v["item03_hi"]:.2f}  ({extra})')
        print('  CV R² ' + '  '.join(f'{k[6:]} {v:.3f}' for k, v in P.items() if k.startswith('cv_r2')))
        print('Q4 ACF lags 1-10 | var autocov1 | fit A phi tau lag1-excess, shares corr/pos/white | GT autocov1 '
              '(rho) | position noise E|u|² (tracker part)')
        for k, v in A.items():
            f = v['fit']
            tail = (f' | GT {v["gt_autocov1"]:+.4f} ({v["gt_rho1"]:+.3f}) | u {v["pos_noise"]:.4f} '
                    f'({v["pos_noise_tracker"]:+.4f})' if 'gt_autocov1' in v else '')
            print(f'  {k:<16} ' + ' '.join(f'{x:+.3f}' for x in v['acf']) +
                  f' | {v["var"]:.4f} {v["autocov1"]:+.4f} | {f["A"]:.3f} {f["phi"]:.2f} {f["tau_frames"]:.1f} '
                  f'{f["lag1_excess"]:+.3f}, {f["share_correlated"]:.3f}/{f["share_position"]:.3f}/'
                  f'{f["share_white"]:.3f}{tail}')
        S = r['short_windows']
        print(f'Q4 short-window ORACLE scale (item 03 %, deployed {S["deployed"]:.3f}): W real / shuffled / '
              'correlated pp')
        print('  ' + '  '.join(f'{k} {v["real"]:.3f}/{v["shuffled"]:.3f}/{v["correlated_pp"]:.3f}'
                               for k, v in S.items() if k != 'deployed'))
        print('Q4 future g (GT-free, non-causal) item 03: ' + '  '.join(f'{k} {v:.3f}' for k, v in r['future_g'].items()))
        print('Q5 gap: blocks pairs | g-frame rel err median mean % | eq. per-pair mm vs |d| median | deployed '
              'same pairs | redistributed (GT exact) | floor')
        for k, v in Q.items():
            red = (f' | {v["redistributed_item03"]:.3f} ({v["redistributed_item03_gt_exact"]:.3f}) | '
                   f'{v["floor_same_pairs"]:.3f}' if 'redistributed_item03' in v else '')
            print(f'  {k:<5} {v["blocks"]:5d} {v["pairs"]:5d} | {v["median_rel_pct"]:.2f} {v["mean_rel_pct"]:.2f} | '
                  f'{v["median_eq_mm"]:.3f} vs {v["median_pair_absd_mm"]:.3f} | {v["deployed_item03_same_pairs"]:.3f}'
                  + red)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--rooms', nargs='+', default=list(ROOMS))
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--sources', type=Path, default=None, metavar='HINTS_DIR',
                    help='follow-up: keyframe intervals and scale series (stereo hints <dir>/<room>.npy, if present)')
    ap.add_argument('--shared', nargs='+', type=Path, default=[],
                    help='runs (inc.npz only) whose low- / high-pass relative errors are correlated pairwise')
    args = ap.parse_args()
    ctx = {s: room_context(s) for s in args.rooms}
    print('GT sigma per room (mm): ' + '  '.join(f'{s} {c["sigma"] * 1e3:.3f}' for s, c in ctx.items()))
    allres, loaded = {}, {}
    for run in args.runs:
        loaded[run] = {s: load_room(run, s, ctx[s]) for s in args.rooms}
        res = {}
        for sname, rooms in SETS.items():
            if not all(s in loaded[run] for s in rooms):
                continue
            rr = [loaded[run][s] for s in rooms]
            res[sname] = {'rooms': {r['scene']: float(item03(r['p'][r['keep']], r['g'][r['keep']]).mean())
                                    for r in rr},
                          'budget': budget(rr), 'predictors': predictors(rr), 'temporal': temporal(rr),
                          'short_windows': short_windows(rr), 'future_g': future_g(rr), 'gap': gap_analysis(rr)}
        res['lever_arm'] = lever_arm(loaded[run])
        if args.sources is not None:
            for sname, rooms in SETS.items():
                if all(s in loaded[run] for s in rooms):
                    res[sname]['sources'] = src = scale_sources([loaded[run][s] for s in rooms], args.sources)
                    k = src['kf_intervals']
                    print(f'\n== {run} {sname} follow-up: keyframe intervals n {k["n_intervals"]} mean / median '
                          f'length {k["mean_len_frames"]:.2f} / {k["median_len_frames"]:.0f} frames; R² of d/g '
                          f'{k["r2"]:.3f} vs random segments {k["r2_control_mean"]:.3f} ± {k["r2_control_sd"]:.3f}')
                    for name, c in src['corr'].items():
                        print(f'  {name:<11} r(d/g_t, dev_t+L) L=-3..3 pearson ' + ' '.join(
                            f'{v["pearson"]:+.3f}' if 'pearson' in v else '  nan ' for v in c.values()) +
                            ' | spearman ' + ' '.join(f'{v["spearman"]:+.3f}' if 'pearson' in v else '  nan '
                                                      for v in c.values()) +
                            f' | sd(dev) {c[0].get("sd_dev", float("nan")):.4f} n {c[0]["n"]}')
                    for name, v in src['gain'].items():
                        print(f'  ESTIMATE smooth {name} over {SMOOTH} frames (lag {v["lag"]:+d}, beta {v["beta"]:+.3f}):'
                              f' item 03 {v["item03_before"]:.3f} -> {v["item03_after"]:.3f}')
        allres[str(run)] = res
        report(str(run), res)
        L = res['lever_arm']
        print('GT lever arm δ (camera frame, mm) per room: ' + '  '.join(
            f'{s} ' + '/'.join(f'{x:+.2f}' for x in v) for s, v in L['per_room_mm'].items()))
        for k, v in L.items():
            if k.startswith('fit'):
                print(f'  {k}: δ ' + '/'.join(f'{x:+.2f}' for x in v['delta_mm']) + ' mm; item 03 ' + '  '.join(
                    f'{s} {w["before"]:.3f} -> {w["after"]:.3f}' for s, w in v.items() if s != 'delta_mm'))
    for a, b in itertools.combinations(args.runs, 2):
        for sname, rooms in SETS.items():
            if all(s in loaded[a] for s in rooms):
                c = compare_runs([loaded[a][s] for s in rooms], [loaded[b][s] for s in rooms])
                allres[f'compare {a} vs {b} {sname}'] = c
                print(f'\n== d of {a} vs {b} ({sname}): corr {c["corr_d"]:.3f} var {c["var_a"]:.4f} / '
                      f'{c["var_b"]:.4f} var(diff) {c["var_diff"]:.4f} ACF(diff) ' +
                      ' '.join(f'{x:+.3f}' for x in c['acf_diff']))
    if args.shared:
        print(f'\n== shared parts of d / g (g >= 5 mm, winsorised): r of the {LOWPASS}-frame low-pass / high-pass '
              'parts between runs (n, low-pass variance share of the first run)')
        for sname, rooms in SETS.items():
            if all(s in args.rooms for s in rooms):
                res = shared_parts(args.shared, rooms)
                allres[f'shared {sname}'] = res
                for k, v in res.items():
                    print(f'  {sname:<8} {k}: r_low {v["r_low"]:+.3f} r_high {v["r_high"]:+.3f} (n {v["n"]}, '
                          f'low share {v["var_low_share_a"]:.3f})')
    if args.json:
        args.json.write_text(json.dumps(allres, indent=1))


if __name__ == '__main__':
    main()
