"""Diagnostics for item 03 (plan v3 item 0.3, §2.2).

Inputs are per-pair translation arrays in meters: gt_t, pred_t of shape (n, 3) for one scene.
Scene aggregation is always "mean of scene means", like the official score.
"""
import numpy as np

from dvslam.eval.metrics import ZERO_MOTION_M
from dvslam.geometry.se3 import relative_pose, rotation_angle_deg, umeyama


def _pct_errors(gt_t: np.ndarray, pred_t: np.ndarray) -> np.ndarray:
    g, p = np.linalg.norm(gt_t, axis=1), np.linalg.norm(pred_t, axis=1)
    out = np.zeros_like(g)
    ok = g > ZERO_MOTION_M
    out[ok] = np.abs(p[ok] - g[ok]) / g[ok] * 100
    return out


def median_step_ratio(gt_t: np.ndarray, pred_t: np.ndarray) -> float:
    """Median of ‖pred‖/‖gt‖ over pairs with GT motion."""
    g, p = np.linalg.norm(gt_t, axis=1), np.linalg.norm(pred_t, axis=1)
    ok = g > ZERO_MOTION_M
    return float(np.median(p[ok] / g[ok]))


def best_rescale(gt_t: np.ndarray, pred_t: np.ndarray) -> tuple[float, float]:
    """Global scale s minimising mean |s‖p‖ − ‖g‖| / ‖g‖ (exact: weighted median of ‖g‖/‖p‖, weights ‖p‖/‖g‖).

    Returns (s, MAPE % after rescale). Zero-motion pairs keep their 0 %.
    """
    g, p = np.linalg.norm(gt_t, axis=1), np.linalg.norm(pred_t, axis=1)
    ok = (g > ZERO_MOTION_M) & (p > 0)
    r, w = g[ok] / p[ok], p[ok] / g[ok]
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    s = float(r[order][np.searchsorted(cw, 0.5 * cw[-1])])
    return s, float(_pct_errors(gt_t, s * pred_t).mean())


def stride_mape(T_gt: np.ndarray, increments: np.ndarray, k: int) -> float:
    """G1' variant of item 03 on pairs (t-k, t): increments composed over k frames vs GT (plan NO-GO branch, Q4)."""
    gt, pred = [], []
    for t in range(k, len(T_gt)):
        gt.append(relative_pose(T_gt[t - k], T_gt[t])[:3, 3])
        P = np.eye(4)
        for dT in increments[t - k:t]:
            P = P @ dT
        pred.append(P[:3, 3])
    return float(_pct_errors(np.array(gt), np.array(pred)).mean())


def min_motion_mape(gt_t: np.ndarray, pred_t: np.ndarray, min_m: float = 2e-3) -> float:
    """G1' variant of item 03 that scores only pairs whose GT step is at least min_m (plan Q4)."""
    m = np.linalg.norm(gt_t, axis=1) >= min_m
    return float(_pct_errors(gt_t[m], pred_t[m]).mean())


def windowed_rescale_mape(gt_t: np.ndarray, pred_t: np.ndarray, window: int) -> float:
    """Oracle G0 diagnostic: item-03 MAPE after fitting one best scale per window of `window` consecutive pairs.

    Uses GT, so it isolates the per-pair noise term from scale drift; never a submittable number.
    """
    pct = np.empty(len(gt_t))
    for s in range(0, len(gt_t), window):
        g, p = gt_t[s:s + window], pred_t[s:s + window]
        if (np.linalg.norm(p, axis=1) > 0).any() and (np.linalg.norm(g, axis=1) > ZERO_MOTION_M).any():
            scale, _ = best_rescale(g, p)
        else:
            scale = 1.0
        pct[s:s + window] = _pct_errors(g, scale * p)
    return float(pct.mean())


def sigma_parallel(gt_t: np.ndarray, pred_t: np.ndarray) -> dict:
    """Error along the GT motion direction (σ∥) and orthogonal to it, in mm. Robust σ = 1.4826·MAD."""
    g = np.linalg.norm(gt_t, axis=1)
    ok = g > ZERO_MOTION_M
    u = gt_t[ok] / g[ok, None]
    e = pred_t[ok] - gt_t[ok]
    e_par = (e * u).sum(1)
    e_perp = np.linalg.norm(e - e_par[:, None] * u, axis=1)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    return {'sigma_par_mm': float(e_par.std() * 1e3), 'sigma_par_robust_mm': float(mad(e_par) * 1e3),
            'bias_par_mm': float(e_par.mean() * 1e3), 'rms_perp_mm': float(np.sqrt((e_perp ** 2).mean()) * 1e3)}


def bin_contribution(gt_t: np.ndarray, pred_t: np.ndarray,
                     edges_mm=(0, 1, 2, 5, 10, np.inf)) -> list[dict]:
    """Share of pairs and share of the scene's total item-03 error per GT step-length bin."""
    g = np.linalg.norm(gt_t, axis=1) * 1e3
    pct = _pct_errors(gt_t, pred_t)
    total = pct.sum()
    out = []
    for lo, hi in zip(edges_mm[:-1], edges_mm[1:]):
        m = (g >= lo) & (g < hi)
        out.append({'bin_mm': f'[{lo}, {hi})', 'pair_share': float(m.mean()),
                    'error_share': float(pct[m].sum() / total) if total > 0 else 0.0,
                    'mape_in_bin': float(pct[m].mean()) if m.any() else float('nan')})
    return out


def ate_rmse(T_gt: np.ndarray, T_est: np.ndarray, sim3: bool) -> tuple[float, float]:
    """ATE RMSE (m) of camera centres after SE(3) or Sim(3) Umeyama alignment. Returns (rmse, scale)."""
    p_gt, p_est = T_gt[:, :3, 3], T_est[:, :3, 3]
    s, R, t = umeyama(p_est, p_gt, with_scale=sim3)
    err = p_gt - (s * (R @ p_est.T).T + t)
    return float(np.sqrt((err ** 2).sum(1).mean())), s


def rpe(T_gt: np.ndarray, T_est: np.ndarray, delta: int) -> dict:
    """RPE over a fixed frame delta: translation RMSE (m) and mean rotation error (deg)."""
    te, re = [], []
    for i in range(len(T_gt) - delta):
        E = np.linalg.inv(relative_pose(T_gt[i], T_gt[i + delta])) @ relative_pose(T_est[i], T_est[i + delta])
        te.append(np.linalg.norm(E[:3, 3]))
        re.append(rotation_angle_deg(E[:3, :3]))
    return {'delta': delta, 'trans_rmse_m': float(np.sqrt(np.mean(np.square(te)))), 'rot_mean_deg': float(np.mean(re))}


def block_bootstrap_ci(per_scene_values: list[np.ndarray], block: int = 30, reps: int = 1000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Moving-block bootstrap CI of the mean of scene means (blocks of `block` consecutive pairs ≈ 1 s)."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([v.mean() for v in per_scene_values]))
    stats = np.empty(reps)
    for k in range(reps):
        means = []
        for v in per_scene_values:
            n = len(v)
            b = min(block, n)          # short scenes (smoke runs) fall back to one block of the whole scene
            nb = -(-n // b)
            starts = rng.integers(0, n - b + 1, size=nb)
            idx = (starts[:, None] + np.arange(b)).ravel()[:n]
            means.append(v[idx].mean())
        stats[k] = np.mean(means)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def simulate_mape(gt_steps: list[np.ndarray], sigma_m: float = 0.0, bias: float = 0.0, jitter_logsd: float = 0.0,
                  noise: str = 'isotropic', reps: int = 20, seed: int = 0) -> tuple[float, float]:
    """Monte-Carlo item-03 MAPE on real GT steps: pred = g·(1+b)·exp(jitter·N) + noise.

    noise='isotropic': N(0, σ²) on each axis; noise='parallel': N(0, σ²) along the GT direction only (σ∥).
    Returns (mean over reps of the mean of scene means, std over reps), both in %.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        scene_means = []
        for g in gt_steps:
            n = len(g)
            scale = (1 + bias) * np.exp(jitter_logsd * rng.standard_normal(n))
            pred = g * scale[:, None]
            if noise == 'isotropic':
                pred = pred + sigma_m * rng.standard_normal((n, 3))
            elif noise == 'parallel':
                u = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
                pred = pred + sigma_m * rng.standard_normal(n)[:, None] * u
            else:
                raise ValueError(noise)
            scene_means.append(_pct_errors(g, pred).mean())
        vals.append(np.mean(scene_means))
    return float(np.mean(vals)), float(np.std(vals))
