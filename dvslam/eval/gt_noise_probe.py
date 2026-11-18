"""GT-noise probe (plan v3 item 0.9).

Model: GT positions p_gt(t) = p(t) + n(t), n ~ N(0, σ_g² I) white. Two depth-free trackers A, B give per-pair steps.
Stride-k residual of a tracker against GT: r^(k)(t) = Σ_{t-k<j<=t} e(j) − (n(t) − n(t−k)), so per axis

    Cov(r_A^(k), r_B^(k)) = 2 σ_g² + C_shared(k)
    Cov(r_A^(1)(t), r_B^(1)(t+1)) = −σ_g² + C_shared,lag1

Estimators (σ_g in metres, isotropic, averaged over axes):
  - 'k1'        : sqrt(C_1 / 2)                      — biased up by any error the trackers share
  - 'intercept' : sqrt(a / 2), a from C_k = a + b·k, k = 1..3 — removes shared errors that are white in time
  - 'lag1'      : sqrt(−L_1)                          — biased by shared errors correlated across adjacent pairs
Do not use a sensor-depth tracker as A or B: it can share error with the GT reconstruction (plan C-N1).
Tracker increments are in the camera frame; rotate them into the world with the GT orientation of frame t-1
(step_world = R_wc,gt(t-1) · t_inc) before calling probe_scene.
"""
import numpy as np

STRIDES = (1, 2, 3)


def _stride_residuals(track_steps: np.ndarray, gt_pos: np.ndarray, k: int) -> np.ndarray:
    """track_steps: (n, 3) per-pair steps t-1 -> t in the world frame; gt_pos: (n+1, 3). Returns (n-k+1, 3)."""
    csum = np.vstack([np.zeros(3), np.cumsum(track_steps, 0)])
    return (csum[k:] - csum[:-k]) - (gt_pos[k:] - gt_pos[:-k])


def _xcov(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - a.mean(0)) * (b - b.mean(0))).mean(0)


def probe_scene(steps_a: np.ndarray, steps_b: np.ndarray, gt_pos: np.ndarray) -> dict:
    """Per-axis cross-covariances for one sequence (all inputs in metres, world frame)."""
    C = {k: _xcov(_stride_residuals(steps_a, gt_pos, k), _stride_residuals(steps_b, gt_pos, k)) for k in STRIDES}
    ra, rb = _stride_residuals(steps_a, gt_pos, 1), _stride_residuals(steps_b, gt_pos, 1)
    L1 = _xcov(ra[:-1], rb[1:])
    return {'C': C, 'L1': L1, 'n': len(steps_a)}


def estimate_sigma_g(scenes: list[dict]) -> dict:
    """Pool probe_scene() outputs (pair-count weighted) and return σ_g estimates in metres."""
    w = np.array([s['n'] for s in scenes], dtype=float)
    w /= w.sum()
    C = np.array([sum(wi * s['C'][k] for wi, s in zip(w, scenes)) for k in STRIDES])   # (len(STRIDES), 3)
    L1 = sum(wi * s['L1'] for wi, s in zip(w, scenes))
    c_mean = C.mean(1)                                   # isotropic: average the 3 axes
    b, a = np.polyfit(np.array(STRIDES, dtype=float), c_mean, 1)
    root = lambda v: float(np.sqrt(max(v, 0.0)))
    return {'k1': root(c_mean[0] / 2), 'intercept': root(a / 2), 'lag1': root(-L1.mean()),
            'C_by_stride': c_mean.tolist(), 'shared_slope': float(b), 'C1_per_axis': C[0].tolist()}
