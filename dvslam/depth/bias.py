"""E1 depth-scale statistics (plan v3 item 0.10, §2.2 budget terms b and jitter).

Per frame t: r_t = median over valid pixels of pred / gt, at GT resolution. Valid = GT in [d_min, d_max] (Redwood:
uint16 mm, 0 = invalid, 0.1-10 m) and pred finite and > 0.
Summary over frames: b = median_t r_t - 1 (and mean_t r_t - 1), jitter = std_t log r_t,
frame-to-frame jitter = std over consecutive pairs of (log r_{t+1} - log r_t).
"""
import numpy as np

D_MIN, D_MAX = 0.1, 10.0
MIN_VALID_PX = 1000


def frame_stats(pred: np.ndarray, gt: np.ndarray, d_min: float = D_MIN, d_max: float = D_MAX) -> dict:
    """pred, gt: HxW metres at the same (GT) resolution."""
    if pred.shape != gt.shape:
        raise ValueError(f'pred {pred.shape} and gt {gt.shape} differ; resample pred to GT resolution first')
    valid = (gt >= d_min) & (gt <= d_max) & np.isfinite(pred) & (pred > 0)
    n = int(valid.sum())
    if n < MIN_VALID_PX:
        return dict(r=np.nan, absrel=np.nan, absrel_med=np.nan, n_valid=n)
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    r = float(np.median(p / g))
    return dict(r=r,
                absrel=float(np.mean(np.abs(p - g) / g)),
                absrel_med=float(np.mean(np.abs(p / r - g) / g)),  # after per-frame median scaling (structure only)
                n_valid=n)


def summarize(r: np.ndarray, pairs: np.ndarray, absrel: np.ndarray | None = None) -> dict:
    """r: (T,) per-frame ratios (NaN = frame skipped). pairs: (P, 2) indices into r of consecutive frames (t, t+1)."""
    r = np.asarray(r, np.float64)
    ok = np.isfinite(r) & (r > 0)
    log_r = np.log(np.where(ok, r, np.nan))
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    d = log_r[pairs[:, 1]] - log_r[pairs[:, 0]]
    d = d[np.isfinite(d)]
    out = dict(n_frames=int(ok.sum()), n_pairs=int(d.size),
               b_median=float(np.median(r[ok]) - 1) if ok.any() else np.nan,
               b_mean=float(np.mean(r[ok]) - 1) if ok.any() else np.nan,
               jitter=float(np.std(log_r[ok], ddof=1)) if ok.sum() > 1 else np.nan,
               f2f_jitter=float(np.std(d, ddof=1)) if d.size > 1 else np.nan)
    if absrel is not None:
        out['absrel'] = float(np.nanmean(absrel))
    return out
