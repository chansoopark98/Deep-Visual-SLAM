"""Metric (a): GT-noise-corrected, Redwood-equivalent item-03 MAPE (plan v3.2 §12.2, task T.4).

Source pairs i (e.g. TUM-VI), all in metres (§12.2.1):
    e_i = ‖t_pred‖ − ‖t_gt‖,  s_i = ‖t_gt‖,  w_i = z̄_i / f   (z̄_i: median depth of tracked patches, f in px)
Error model:
    e_i = b·s_i + j_i·s_i + w_i·σ_px·n_i + ε_gt,i,   sd(j_i) = j, n_i ~ N(0, 1), Var(ε_gt) = V measured separately (T.3)

Estimator (method of moments; choices documented because the plan fixes only the model form):
  1. b  = Σ e·s / Σ s²                                   (least squares through the origin)
  2. r_i = e_i − b·s_i;  regress (r_i² − (1+b)²·V) on [s_i², w_i²] without intercept → (j², σ_px²)  (deconvolution)
     If w_i is constant the w² column is the intercept, so this is literally "residual variance − V". The (1+b)²
     factor is the one deviation from the literal model: s_i = ‖t_gt‖ itself contains ε_gt,i, so the residual
     e_i − b·s_i carries −(1+b)·ε_gt,i (at b = 1 % ignoring it biases σ_px up by ~2 % in the T.4 synthetic check).
     b itself has a regression-dilution bias of about −(1+b)·V/E[s²] because s contains ε_gt (≈ −1e-4 at TUM-VI
     scales: V ≈ 6.4e-7 m², E[s²] ≈ 5.8e-3 m²); it is not corrected.
  3. Clip j², σ_px² at 0 and transfer per Redwood scene k: σ∥,RW,k = σ_px·z̄_RW,k/f_RW (z̄_RW is one value for all
     scenes or one per scene, plan §12.2 step 3 "test scene depth distribution"). `diagnostics.simulate_mape` takes
     one σ, so it is called once per scene (seed + k) with bias b and jitter_logsd = j/(1+b) (simulate_mape scales by
     (1+b)·exp(jitter·N), whose per-pair sd is (1+b)·jitter), and the scene MAPEs are averaged (mean of scene means,
     identical in expectation to one call on all scenes). A scalar z̄_RW takes the same per-scene path, so a list of
     identical values reproduces it exactly. No GT noise is added on the Redwood side: the result is tracker-only.
CI and verdict (§12.2 items 4–5): moving-block bootstrap within each scene (pooled fit over scenes, same block rule as
`diagnostics.block_bootstrap_ci`). V's uncertainty is propagated by the conservative bound rule "upper CI of residual
variance − lower CI of V": every replicate is evaluated at V_lo, V_point and V_hi, the upper bound is the largest of the
three (1 − α/2) quantiles and the lower bound the smallest of the three α/2 quantiles.
Why not just V_lo for the upper bound: (j², σ_px²) = A⁻¹(y − (1+b)²V·[Σs², Σw²]) is linear in V with
∂j²/∂V = −(1+b)²·(Σw⁴·Σs² − Σs²w²·Σw²)/det and ∂σ_px²/∂V = −(1+b)²·(Σs⁴·Σw² − Σs²w²·Σs²)/det. The first is exactly 0
when w is constant (the w² column is then the intercept) and otherwise has no fixed sign: when far scenes (large w)
also have large steps, j² rises with V while σ_px² falls, so the MAPE need not be monotone in V (verified, test
`test_bounds_not_monotone_in_var_gt`). Each pair's error variance is linear in V and the MAPE is roughly concave in
it, so an interior extremum is possible in principle; V_point is added to the grid as a cheap guard, not a proof.
Bootstrap replicates are refitted without the full-data conditioning check; a replicate whose normal matrix is
near-singular (1 − c²/(a·d) < 1e-6, possible only if the resample keeps pairs with proportional s² and w²) would give
inf/NaN or arbitrary (j², σ_px²), so it is dropped from the quantiles and counted in 'ill_conditioned_reps'.
The hyper-parameters (block, outlier_k, z̄ estimate) are fixed on the selection split (room1–2) by the caller; b, j,
σ_px are always re-estimated on the split being judged.

Well-posedness: j² and σ_px² are separated only through the spread of s² relative to w². If s_i and w_i are
proportional across pairs (e.g. a camera that moves faster exactly when the scene is farther away) the two columns are
collinear and the split is arbitrary; `fit_error_model` raises when the normal matrix is near-singular. Any GT error
proportional to s (timestamp jitter) is absorbed into j, and any GT error not in V is absorbed into σ_px.
"""
from dataclasses import dataclass

import numpy as np

from dvslam.eval.diagnostics import _pct_errors, block_bootstrap_ci, sigma_parallel, simulate_mape
from dvslam.eval.metrics import POS_MAPE_TARGET_PCT

REDWOOD_STEP_RANGE_M = (4.5e-3, 9.1e-3)    # §12.2 (b): Redwood scene median GT steps 4.48–9.07 mm
DEFAULT_BLOCK = 20                         # ≈ 1 s at 20 Hz (TUM-VI); fix on room1–2
# Plausibility limits that catch millimetre inputs (§12.2.1): a 0.5 m median step per frame pair or a 50 m median
# patch depth is not an indoor hand-held sequence.
_MAX_MEDIAN_STEP_M, _MAX_MEDIAN_DEPTH_M = 0.5, 50.0
# A GT step noise sd above 10 % of the median step is not a usable GT, and this catches a standard deviation passed as
# the variance at TUM-VI scales (σ = 0.8 mm passed as 8e-4 > (0.1 · 0.053 m)² = 2.8e-5). (median step)² itself would
# not catch it (2.8e-3).
_MAX_GT_SD_PER_STEP = 0.1


@dataclass
class ErrorModelFit:
    b: float            # common scale bias (slope)
    j: float            # per-pair scale jitter sd, clipped at 0
    sigma_px: float     # tracker noise along the motion in px, clipped at 0
    j2: float           # unclipped j² (negative = no jitter detectable)
    sigma_px2: float    # unclipped σ_px² (negative = GT noise V explains all of the non-scale residual)
    var_gt: float       # V used for the deconvolution (m²)
    n_pairs: int


def _check_metres(s: list[np.ndarray], z: list[np.ndarray], var_gt: tuple[float, ...]) -> None:
    med_s = float(np.median(np.concatenate(s)))
    if med_s > _MAX_MEDIAN_STEP_M:
        raise ValueError(f'median GT step {med_s:g} > {_MAX_MEDIAN_STEP_M} m: steps must be in metres, not mm')
    med_z = float(np.median(np.concatenate(z)))
    if med_z > _MAX_MEDIAN_DEPTH_M:
        raise ValueError(f'median depth {med_z:g} > {_MAX_MEDIAN_DEPTH_M} m: depth must be in metres, not mm')
    lim = (_MAX_GT_SD_PER_STEP * med_s) ** 2
    if max(var_gt) > lim or min(var_gt) < 0:
        raise ValueError(f'var_gt_m2 {var_gt} outside [0, {lim:.3g}] m² (= (0.1·median step)²): it must be a variance in '
                         'm², e.g. σ = 0.8 mm → 6.4e-7, not the standard deviation')


def _prepare(e_par, s, z_med, f_px, var_gt, outlier_k):
    """Per-scene float64 arrays (e, s, w) after the outlier rule; w = z̄/f broadcast to every pair."""
    s = [np.asarray(x, dtype=np.float64) for x in s]
    e = [np.asarray(x, dtype=np.float64) for x in e_par]
    z = [np.broadcast_to(np.asarray(x, dtype=np.float64), si.shape) for x, si in zip(z_med, s)]
    if not (len(e) == len(s) == len(z)) or any(a.shape != b.shape for a, b in zip(e, s)):
        raise ValueError('e_par, s, z_med must be per-scene lists of equal-length 1-D arrays')
    _check_metres(s, z, var_gt)
    if not f_px > 0:
        raise ValueError('f_px must be > 0 (focal length in px)')
    w = [zi / f_px for zi in z]
    if outlier_k is not None:
        es, ss = np.concatenate(e), np.concatenate(s)
        b0 = (es @ ss) / (ss @ ss)
        med = np.median(es - b0 * ss)
        lim = outlier_k * 1.4826 * np.median(np.abs(es - b0 * ss - med))
        keep = [np.abs(ei - b0 * si - med) <= lim for ei, si in zip(e, s)]
        e, s, w = ([a[k] for a, k in zip(arr, keep)] for arr in (e, s, w))
    return e, s, w


def _features(e: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-pair terms whose sums are sufficient for the fit. Columns:
    0 e·s, 1 s², 2 s⁴, 3 s²w², 4 w⁴, 5 s²e², 6 e·s³, 7 w²e², 8 e·s·w², 9 w²."""
    s2, w2 = s * s, w * w
    return np.stack([e * s, s2, s2 * s2, s2 * w2, w2 * w2, s2 * e * e, e * s * s2, w2 * e * e, e * s * w2, w2], 1)


def _solve(S: np.ndarray, var_gt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S: (..., 10) feature sums. Returns (b, j², σ_px²) with the same leading shape."""
    b = S[..., 0] / S[..., 1]
    s2r2 = S[..., 5] - 2 * b * S[..., 6] + b * b * S[..., 2]             # Σ s²·r²
    w2r2 = S[..., 7] - 2 * b * S[..., 8] + b * b * S[..., 3]             # Σ w²·r²
    v = (1 + b) ** 2 * var_gt          # ε_gt is also inside s = ‖t_gt‖, so r = e − b·s carries −(1+b)·ε_gt
    y0, y1 = s2r2 - v * S[..., 1], w2r2 - v * S[..., 9]                  # Σ x·(r² − (1+b)²V)
    a, c, d = S[..., 2], S[..., 3], S[..., 4]
    det = a * d - c * c
    return b, (d * y0 - c * y1) / det, (a * y1 - c * y0) / det


def _ill_conditioned(S: np.ndarray) -> np.ndarray:
    """True where the 2×2 normal matrix [[Σs⁴, Σs²w²], [Σs²w², Σw⁴]] is near-singular (NaN counts as singular)."""
    a, c, d = S[..., 2], S[..., 3], S[..., 4]
    with np.errstate(divide='ignore', invalid='ignore'):
        return ~(1 - c * c / (a * d) >= 1e-6)


def _check_conditioning(S: np.ndarray) -> None:
    if _ill_conditioned(S):
        raise ValueError('s² and (z̄/f)² are collinear over the pairs: j and σ_px are not identifiable')


def fit_error_model(e_par: list[np.ndarray], s: list[np.ndarray], z_med: list, f_px: float, var_gt_m2: float,
                    outlier_k: float | None = None) -> ErrorModelFit:
    """Fit (b, j, σ_px) on per-scene arrays (metres; z_med per pair or one value per scene; f_px in px).

    var_gt_m2: Var(ε_gt), the VARIANCE of the GT step noise along the motion in m² (σ = 0.8 mm → 6.4e-7), not its sd.

    outlier_k: drop pairs whose residual e − b·s is more than k robust sd (1.4826·MAD) from its median; None keeps all.
    """
    e, s, w = _prepare(e_par, s, z_med, f_px, (var_gt_m2,), outlier_k)
    S = _features(np.concatenate(e), np.concatenate(s), np.concatenate(w)).sum(0)
    _check_conditioning(S)
    b, j2, p2 = (float(x) for x in _solve(S, var_gt_m2))
    return ErrorModelFit(b, float(np.sqrt(max(j2, 0.0))), float(np.sqrt(max(p2, 0.0))), j2, p2, var_gt_m2,
                         int(sum(len(x) for x in s)))


def _check_redwood(redwood_steps: list[np.ndarray], z_rw, f_rw: float) -> np.ndarray:
    """Validate the Redwood side; returns z̄_RW broadcast to one value per scene."""
    if float(np.median(np.concatenate([np.linalg.norm(g, axis=1) for g in redwood_steps]))) > _MAX_MEDIAN_STEP_M:
        raise ValueError('Redwood GT steps must be in metres, not mm')
    z = np.asarray(z_rw, dtype=np.float64)
    if z.ndim > 1 or (z.ndim == 1 and len(z) != len(redwood_steps)):
        raise ValueError('z_rw must be a scalar or one value per Redwood scene (aligned with redwood_steps)')
    z = np.broadcast_to(z, (len(redwood_steps),))
    if (z > _MAX_MEDIAN_DEPTH_M).any() or not (z > 0).all():
        raise ValueError('z_rw must be a positive depth in metres, not mm')
    if not f_rw > 0:
        raise ValueError('f_rw must be > 0 (focal length in px)')
    return z


def _simulate(b: float, j: float, sigma_px: float, redwood_steps: list[np.ndarray], z: np.ndarray, f_rw: float,
              sim_reps: int, seed: int) -> tuple[float, float]:
    sig = sigma_px * z / f_rw
    m = [simulate_mape([g], sk, b, j / (1 + b), noise='parallel', reps=sim_reps, seed=seed + k)[0]
         for k, (g, sk) in enumerate(zip(redwood_steps, sig))]
    return float(np.mean(m)), float(sig.mean())


def redwood_equivalent(b: float, j: float, sigma_px: float, redwood_steps: list[np.ndarray], z_rw,
                       f_rw: float, sim_reps: int = 10, seed: int = 0) -> dict:
    """Transfer (b, j, σ_px) to Redwood: σ∥,RW,k = σ_px·z̄_RW,k/f_RW per scene, then the §2.2 simulation on the Redwood
    GT steps (per-scene (n, 3) arrays in metres). z_rw: one depth (m) or one per scene, aligned with redwood_steps.
    Returns the item-03 MAPE in % (mean of scene means) and σ∥,RW in m (mean over scenes; mm for display)."""
    z = _check_redwood(redwood_steps, z_rw, f_rw)
    mape, sigma_rw = _simulate(b, j, sigma_px, redwood_steps, z, f_rw, sim_reps, seed)
    return {'mape_pct': mape, 'sigma_par_rw_m': sigma_rw, 'sigma_par_rw_mm': sigma_rw * 1e3,
            'sigma_par_rw_scene_m': (sigma_px * z / f_rw).tolist()}


def decide(lower: float, upper: float, threshold: float) -> str:
    """§12.2 item 5: 'pass' if upper ≤ threshold, 'fail' if lower > threshold, otherwise 'undetermined'."""
    if upper <= threshold:
        return 'pass'
    if lower > threshold:
        return 'fail'
    return 'undetermined'


def _block_counts(lengths: list[int], block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, Σn) multiplicity of each pair in a moving-block resample drawn independently per scene."""
    out = []
    for n in lengths:
        bl = min(block, n)             # short scenes fall back to one block of the whole scene
        starts = rng.integers(0, n - bl + 1, size=(reps, -(-n // bl)))
        idx = (starts[:, :, None] + np.arange(bl)).reshape(reps, -1)[:, :n] + n * np.arange(reps)[:, None]
        out.append(np.bincount(idx.ravel(), minlength=reps * n).reshape(reps, n))
    return np.concatenate(out, 1).astype(np.float64)


def metric_a(e_par: list[np.ndarray], s: list[np.ndarray], z_med: list, f_px: float,
             var_gt_m2: tuple[float, float, float], redwood_steps: list[np.ndarray], z_rw, f_rw: float,
             threshold: float = POS_MAPE_TARGET_PCT, block: int = DEFAULT_BLOCK, reps: int = 1000,
             alpha: float = 0.05, sim_reps: int = 10, seed: int = 0, outlier_k: float | None = None) -> dict:
    """Redwood-equivalent MAPE with conservative CI and verdict (§12.2 items 1–5).

    var_gt_m2 = (point, lo, hi) of Var(ε_gt), a VARIANCE in m² (σ = 0.8 mm → 6.4e-7), from the T.3 probe (for white GT
    position noise of sd σ_g per axis, the step noise along the motion has V = 2·σ_g²). z_rw: scalar or per scene. Every bootstrap replicate is refitted and transferred with the
    same simulation seed (common random numbers), so the spread reflects the fit, not Monte-Carlo noise.
    Returned keys: point estimates (fit at V point, full data), *_lo / *_hi bounds, 'verdict' on the MAPE, and
    'sigma_px2_lo' (unclipped; ≤ 0 means "the σ_px² CI reaches 0").
    """
    v_pt, v_lo, v_hi = var_gt_m2
    if not v_lo <= v_pt <= v_hi:
        raise ValueError('var_gt_m2 must be (point, lo, hi) with lo <= point <= hi')
    e, s, w = _prepare(e_par, s, z_med, f_px, var_gt_m2, outlier_k)
    z_rw = _check_redwood(redwood_steps, z_rw, f_rw)
    F = _features(np.concatenate(e), np.concatenate(s), np.concatenate(w))
    _check_conditioning(F.sum(0))
    rng = np.random.default_rng(seed)
    S_boot = np.concatenate([_block_counts([len(x) for x in s], block, m, rng) @ F
                             for m in np.diff(np.r_[0:reps:200, reps])])          # chunks bound memory to 200 × Σn
    bad = _ill_conditioned(S_boot)
    S_boot = S_boot[~bad]
    if len(S_boot) == 0:
        raise ValueError('every bootstrap replicate is ill-conditioned')

    def transfer(S, v):
        """Per replicate: (MAPE %, mean σ∥,RW m, b, j, σ_px, unclipped σ_px²)."""
        b, j2, p2 = (np.atleast_1d(x) for x in _solve(S, v))
        j, px = np.sqrt(np.maximum(j2, 0)), np.sqrt(np.maximum(p2, 0))
        sims = np.array([_simulate(bi, ji, pi, redwood_steps, z_rw, f_rw, sim_reps, seed)
                         for bi, ji, pi in zip(b, j, px)])
        return np.stack([sims[:, 0], sims[:, 1], b, j, px, p2])

    pt = transfer(F.sum(0)[None], v_pt)[:, 0]
    grid = [transfer(S_boot, v) for v in (v_lo, v_pt, v_hi)]                  # (3 V values) × (6 stats, reps)
    lo = np.min([np.quantile(g, alpha / 2, axis=1) for g in grid], axis=0)
    hi = np.max([np.quantile(g, 1 - alpha / 2, axis=1) for g in grid], axis=0)
    pt, lo, hi = pt.tolist(), lo.tolist(), hi.tolist()
    return {
        'mape_pct': pt[0], 'mape_lo_pct': lo[0], 'mape_hi_pct': hi[0],
        'sigma_par_rw_m': pt[1], 'sigma_par_rw_lo_m': lo[1], 'sigma_par_rw_hi_m': hi[1],
        'sigma_par_rw_mm': pt[1] * 1e3, 'sigma_par_rw_lo_mm': lo[1] * 1e3, 'sigma_par_rw_hi_mm': hi[1] * 1e3,
        'b': pt[2], 'b_lo': lo[2], 'b_hi': hi[2],
        'j': pt[3], 'j_lo': lo[3], 'j_hi': hi[3],
        'sigma_px': pt[4], 'sigma_px_lo': lo[4], 'sigma_px_hi': hi[4],
        'sigma_px2': pt[5], 'sigma_px2_lo': lo[5], 'sigma_px2_hi': hi[5],
        'verdict': decide(lo[0], hi[0], threshold), 'threshold_pct': threshold,
        'var_gt_m2': var_gt_m2, 'n_pairs': int(len(F)), 'block': block, 'reps': reps, 'alpha': alpha,
        'ill_conditioned_reps': int(bad.sum()),
        # per-V bootstrap quantiles of the MAPE, to see which V end sets each bound
        'mape_q_by_var_pct': {'v_lo': [float(np.quantile(grid[0][0], q)) for q in (alpha / 2, 1 - alpha / 2)],
                              'v_point': [float(np.quantile(grid[1][0], q)) for q in (alpha / 2, 1 - alpha / 2)],
                              'v_hi': [float(np.quantile(grid[2][0], q)) for q in (alpha / 2, 1 - alpha / 2)]},
    }


def diagnostics(gt_t: list[np.ndarray], pred_t: list[np.ndarray]) -> dict:
    """§12.2 reported-only diagnostics on per-scene (n, 3) translations in metres.

    (b) matched-interval MAPE: raw item 03 over pairs with GT step in [4.5, 9.1] mm, mean of the scene means of the
        scenes that have at least one such pair (NaN if none), with the pair count.
    (c) raw item 03 on all pairs (mean of scene means).
    σ∥: `diagnostics.sigma_parallel` on all pairs pooled over scenes (std and robust, mm).
    """
    lo, hi = REDWOOD_STEP_RANGE_M
    matched, n_matched = [], 0
    for g, p in zip(gt_t, pred_t):
        sl = np.linalg.norm(g, axis=1)
        m = (sl >= lo) & (sl <= hi)
        n_matched += int(m.sum())
        if m.any():
            matched.append(_pct_errors(g[m], p[m]).mean())
    sp = sigma_parallel(np.concatenate(gt_t), np.concatenate(pred_t))
    return {'matched_mape_pct': float(np.mean(matched)) if matched else float('nan'), 'matched_pairs': n_matched,
            'raw_mape_pct': float(np.mean([_pct_errors(g, p).mean() for g, p in zip(gt_t, pred_t)])),
            'sigma_par_mm': sp['sigma_par_mm'], 'sigma_par_robust_mm': sp['sigma_par_robust_mm']}


def paired_difference(err_a: list[np.ndarray], err_b: list[np.ndarray], block: int = DEFAULT_BLOCK,
                      reps: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """§12.2 item 7: paired block-bootstrap CI of raw item-03 MAPE(A) − MAPE(B) (mean of scene means, %).

    err_a, err_b: per-scene per-pair item-03 errors (%) of two trackers on the SAME pairs. Resampling the per-pair
    difference with one set of blocks is the paired bootstrap, because the mean of scene means is linear.
    """
    if len(err_a) != len(err_b) or any(len(x) != len(y) for x, y in zip(err_a, err_b)):
        raise ValueError('paired comparison needs the same scenes and pairs for both trackers')
    d = [np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) for a, b in zip(err_a, err_b)]
    pt, lo, hi = block_bootstrap_ci(d, block=block, reps=reps, alpha=alpha, seed=seed)
    return {'diff_pct': pt, 'diff_lo_pct': lo, 'diff_hi_pct': hi, 'excludes_zero': bool(lo > 0 or hi < 0)}
