"""Unit tests for dvslam.eval.diagnostics (plan v3 item 0.3)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.eval import diagnostics as D
from dvslam.geometry.se3 import umeyama


def random_steps(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 4e-3, (n, 3)) + np.array([5e-3, 0, 0])


def random_traj(n=200, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.3, (n, 3))).as_matrix()
    T[:, :3, 3] = np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)
    return T


def test_best_rescale_recovers_global_scale():
    g = random_steps()
    s, mape = D.best_rescale(g, 0.8 * g)
    assert s == pytest.approx(1.25, rel=1e-12) and mape == pytest.approx(0, abs=1e-10)


def test_best_rescale_is_the_minimiser():
    rng = np.random.default_rng(1)
    g = random_steps()
    p = g * np.exp(rng.normal(0, 0.2, len(g)))[:, None]
    s, m = D.best_rescale(g, p)
    for ds in (0.99, 1.01):
        assert m <= D._pct_errors(g, s * ds * p).mean() + 1e-12


def test_median_step_ratio():
    g = random_steps()
    assert D.median_step_ratio(g, 1.1 * g) == pytest.approx(1.1)


def test_sigma_parallel_recovers_injected_noise():
    rng = np.random.default_rng(2)
    g = random_steps(20000)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    sp = D.sigma_parallel(g, g + 1e-4 * rng.standard_normal(len(g))[:, None] * u)
    assert sp['sigma_par_mm'] == pytest.approx(0.1, rel=0.03)
    assert sp['sigma_par_robust_mm'] == pytest.approx(0.1, rel=0.05)
    assert sp['rms_perp_mm'] < 1e-9


def test_bin_contribution_shares_sum_to_one():
    rng = np.random.default_rng(3)
    g = random_steps()
    bins = D.bin_contribution(g, g + rng.normal(0, 1e-4, g.shape))
    assert sum(b['pair_share'] for b in bins) == pytest.approx(1)
    assert sum(b['error_share'] for b in bins) == pytest.approx(1)


@pytest.mark.parametrize('sim3', [False, True])
def test_ate_zero_after_known_transform(sim3):
    T = random_traj()
    A = np.eye(4)
    A[:3, :3] = Rotation.from_rotvec([0.3, -1.0, 2.0]).as_matrix()
    A[:3, 3] = [1, 2, 3]
    s = 2.5 if sim3 else 1.0
    T_est = T.copy()
    T_est[:, :3, 3] = s * T[:, :3, 3]
    T_est = np.einsum('ij,njk->nik', A, T_est)
    rmse, scale = D.ate_rmse(T, T_est, sim3=sim3)
    assert rmse < 1e-9 and scale == pytest.approx(1 / s if sim3 else 1.0)


def test_umeyama_never_returns_reflection():
    T = random_traj(50, seed=4)
    T_mirror = T.copy()
    T_mirror[:, 2, 3] *= -1
    _, R, _ = umeyama(T_mirror[:, :3, 3], T[:, :3, 3], with_scale=True)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rpe_zero_for_identical_and_invariant_to_global_transform():
    T = random_traj()
    A = np.eye(4)
    A[:3, :3] = Rotation.from_rotvec([1, 0, 0]).as_matrix()
    r = D.rpe(T, np.einsum('ij,njk->nik', A, T), delta=5)
    assert r['trans_rmse_m'] < 1e-9 and r['rot_mean_deg'] < 1e-5


def test_block_bootstrap_ci_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(5)
    vals = [rng.exponential(2.0, n) for n in (300, 400, 500)]
    pt, lo, hi = D.block_bootstrap_ci(vals)
    assert lo < pt < hi
    assert D.block_bootstrap_ci(vals) == (pt, lo, hi)
