"""EXP-019 noise diagnosis: along / transverse decomposition, GT-noise deconvolution, ACF and gap analysis on
synthetic data with known tracker and GT noise."""
import numpy as np
import pytest

from scripts.exp015_scale_oracle import item03
from scripts.exp019_noise_diag import (
    acf,
    ar_fit,
    budget,
    decompose,
    gap_analysis,
    lever_fit,
    lever_item03,
    mc_item03,
    shrink_factors,
    temporal,
)


def test_decompose_known_vector_and_exact_identity():
    dec = decompose(np.array([[0.001, 0.0, 0.012]]), np.array([[0.0, 0.0, 0.010]]))
    assert dec['along'][0] == pytest.approx(0.002)
    assert dec['trans'][0] == pytest.approx(0.001)
    assert dec['d'][0] == pytest.approx(np.sqrt(0.001 ** 2 + 0.012 ** 2) - 0.010)
    assert dec['p_trans_perfect'][0] == pytest.approx(0.012)
    assert dec['p_along_perfect'][0] == pytest.approx(np.sqrt(0.010 ** 2 + 0.001 ** 2))
    rng = np.random.default_rng(0)
    t_gt, t_pred = rng.normal(size=(500, 3)) * 0.02, rng.normal(size=(500, 3)) * 0.02
    d = decompose(t_pred, t_gt)
    np.testing.assert_allclose(d['p'] ** 2, (d['g'] + d['along']) ** 2 + d['trans'] ** 2, rtol=1e-10)
    np.testing.assert_allclose((d['perp'] * d['u']).sum(1), 0, atol=1e-15)
    np.testing.assert_allclose(d['along'][:, None] * d['u'] + d['perp'], d['e'], atol=1e-15)


def test_pure_along_or_pure_transverse_error_is_removed_by_the_matching_variant():
    rng = np.random.default_rng(1)
    t_gt = rng.normal(size=(300, 3)) * 0.03
    u = t_gt / np.linalg.norm(t_gt, axis=1, keepdims=True)
    along = decompose(t_gt + 0.002 * rng.normal(size=(300, 1)) * u, t_gt)
    np.testing.assert_allclose(along['p_along_perfect'], along['g'], rtol=1e-12)
    perp = np.cross(u, rng.normal(size=(300, 3)))
    trans = decompose(t_gt + 0.002 * perp, t_gt)
    np.testing.assert_allclose(trans['p_trans_perfect'], trans['g'], rtol=1e-12)
    assert np.all(trans['d'] >= -1e-15)                  # transverse error only lengthens (Rician)


def _synthetic_room(n=20000, sig_tr=0.0006, sig_gt=0.00015, pos_tr=0.0, seed=0, scene='room5'):
    """Tracker error = white step noise sig_tr (+ per-frame position noise pos_tr); GT = per-frame noise sig_gt."""
    rng = np.random.default_rng(seed)
    t_true = rng.normal(size=(n, 3)) * 0.02
    n_gt, n_pos = rng.normal(size=(n + 1, 3)) * sig_gt, rng.normal(size=(n + 1, 3)) * pos_tr
    t_gt = t_true + n_gt[1:] - n_gt[:-1]
    t_pred = t_true + rng.normal(size=(n, 3)) * sig_tr + n_pos[1:] - n_pos[:-1]
    dec = decompose(t_pred, t_gt)
    eye = np.broadcast_to(np.eye(4), (n, 4, 4))
    T_pred, T_gt = eye.copy(), eye.copy()
    T_pred[:, :3, 3], T_gt[:, :3, 3] = t_pred, t_gt
    return {'scene': scene, 'index': np.arange(1, n + 1), 'keep': np.ones(n, bool), **dec,
            's_pair': np.full(n, np.sqrt(2) * sig_gt), 's_shared': np.full(n, sig_gt), 'e_world': dec['e'],
            'T_pred': T_pred, 'T_gt': T_gt, 't_true': t_true, 't_pred': t_pred}


def test_shrink_factor_recovers_the_tracker_share():
    rng = np.random.default_rng(2)
    a = rng.normal(size=200000) * 0.0006 + rng.normal(size=200000) * 0.0002
    b = np.linalg.norm(rng.normal(size=(200000, 2)) * np.hypot(0.0006, 0.0002), axis=1)
    f_par, f_perp = shrink_factors(a, b, 0.0002)
    assert f_par == pytest.approx(0.0006 / np.hypot(0.0006, 0.0002), rel=0.01)
    assert f_perp == pytest.approx(0.0006 / np.hypot(0.0006, 0.0002), rel=0.01)
    assert shrink_factors(a * 0.1, b * 0.1, 0.0002) == (0.0, 0.0)


def test_budget_separates_tracker_and_gt_noise_and_mc_reproduces_deployed():
    room = _synthetic_room()
    out = budget([room])
    allb = out['bins']['all']
    assert out['bins']['20-40']['own_rsd_mm'] == pytest.approx(0.6, rel=0.06)
    assert out['bins']['20-40']['gt_sd_mm'] == pytest.approx(np.sqrt(2) * 0.15, rel=1e-6)
    assert out['mc']['k=1']['all'] == pytest.approx(allb['item03'], rel=0.05)
    # k = 0: the floor of a perfect estimator against the same GT noise; k = 1, GT exact: the tracker against truth
    rng = np.random.default_rng(3)
    floor = mc_item03(room['g'][:, None] * room['u'], np.zeros((len(room['g']), 3)), room['s_pair'], 0.0, rng).mean()
    assert out['mc']['k=0']['all'] == pytest.approx(floor, rel=0.03)
    truth = item03(np.linalg.norm(room['t_pred'], axis=1), np.linalg.norm(room['t_true'], axis=1)).mean()
    assert out['mc']['k=1 GT exact']['all'] == pytest.approx(truth, rel=0.06)
    assert out['mc']['k=0']['all'] < out['mc']['k=0.25']['all'] < out['mc']['k=0.5']['all'] < out['mc']['k=1']['all']


def test_acf_of_position_noise_and_white_noise():
    rng = np.random.default_rng(4)
    u = rng.normal(size=50001)
    ok = np.ones(50000, bool)
    rho = acf(u[1:] - u[:-1], ok, lags=3)
    assert rho[0] == pytest.approx(-0.5, abs=0.02) and abs(rho[1]) < 0.02
    assert max(abs(r) for r in acf(rng.normal(size=50000), ok, lags=3)) < 0.02
    v = rng.normal(size=(50001, 3))
    assert acf(v[1:] - v[:-1], ok, lags=1)[0] == pytest.approx(-0.5, abs=0.02)


def test_temporal_attributes_lag1_anticorrelation_to_gt_or_tracker():
    gt_only = temporal([_synthetic_room(n=100000, sig_tr=0.0, sig_gt=0.0003)])['vec_mm']
    assert gt_only['autocov1'] == pytest.approx(gt_only['gt_autocov1'], rel=0.05)
    assert abs(gt_only['pos_noise_tracker']) < 0.1 * gt_only['pos_noise']
    trk = temporal([_synthetic_room(n=100000, sig_tr=0.0003, sig_gt=0.0, pos_tr=0.0004)])['vec_mm']
    assert trk['pos_noise_tracker'] == pytest.approx(3 * 0.4 ** 2, rel=0.05)      # E‖u‖², mm²
    assert trk['pos_share'] == pytest.approx(2 * 3 * 0.16 / (2 * 3 * 0.16 + 3 * 0.09), rel=0.05)
    # two rooms: no lag pairs across the boundary (the result equals the pooled per-room statistics)
    two = temporal([_synthetic_room(n=5000, seed=5), _synthetic_room(n=5000, seed=6)])['d_mm']
    assert len(two['acf']) == 10


def test_gap_analysis_perfect_tracker_and_position_noise():
    room = _synthetic_room(n=2000, sig_tr=0.0, sig_gt=0.0)
    room['T_pred'] = room['T_gt'].copy()
    room['p'], room['d'] = room['g'].copy(), np.zeros(2000)
    out = gap_analysis([room])
    assert out['g=5']['median_rel_pct'] == pytest.approx(0, abs=1e-9)
    assert out['g=5']['redistributed_item03_gt_exact'] == pytest.approx(0, abs=1e-9)
    assert out['g=5']['blocks'] == 400 and out['g=2']['pairs'] == 2000
    # per-frame position noise only: the g-frame error does not grow with g, so redistribution divides it by g
    pos = gap_analysis([_synthetic_room(n=4000, sig_tr=0.0, sig_gt=0.0, pos_tr=0.0005)])
    assert pos['g=10']['median_eq_mm'] < 0.2 * pos['g=1']['median_eq_mm']
    assert pos['g=10']['redistributed_item03_gt_exact'] < 0.2 * pos['g=1']['deployed_item03_same_pairs']


def test_ar_fit_separates_correlated_position_and_white_parts():
    rng = np.random.default_rng(7)
    n, phi = 400000, 0.8
    z = rng.normal(size=n) * np.sqrt(0.5 * (1 - phi ** 2))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + z[t]
    u = rng.normal(size=n + 1) * np.sqrt(0.15)                         # position noise: 2·0.15 = 0.3 of the variance
    x = ar + u[1:] - u[:-1] + rng.normal(size=n) * np.sqrt(0.2)
    fit = ar_fit(acf(x, np.ones(n, bool)))
    assert fit['phi'] == pytest.approx(phi, abs=0.03)
    assert fit['share_correlated'] == pytest.approx(0.5, abs=0.04)
    assert fit['share_position'] == pytest.approx(0.3, abs=0.04)
    assert fit['share_white'] == pytest.approx(0.2, abs=0.05)


def test_lever_fit_recovers_an_injected_gt_lever_arm():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(8)
    n, delta = 5000, np.array([0.004, -0.002, 0.006])
    R = Rotation.from_rotvec(rng.normal(size=(n, 3)) * np.radians(4)).as_matrix()
    t_true = rng.normal(size=(n, 3)) * 0.02
    t_gt = t_true - (R - np.eye(3)) @ delta                     # GT built with a wrong camera centre
    t_pred = t_true + rng.normal(size=(n, 3)) * 0.0002
    T_gt = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    T_gt[:, :3, :3], T_gt[:, :3, 3] = R, t_gt
    room = {'keep': np.ones(n, bool), 'tracked': np.ones(n, bool), 'T_gt': T_gt, **decompose(t_pred, t_gt)}
    est = lever_fit([room])
    np.testing.assert_allclose(est, delta, atol=2e-4)
    assert lever_item03([room], est) < 0.5 * lever_item03([room], np.zeros(3))


def test_segment_r2_and_running_median():
    from scripts.exp019_noise_diag import r2_segments, running
    seg = np.repeat(np.arange(50), 4)
    x = np.repeat(np.random.default_rng(9).normal(size=50), 4)
    assert r2_segments(x, seg) == pytest.approx(1.0)
    x2 = x + np.random.default_rng(10).normal(size=200) * 1e6
    assert r2_segments(x2, seg) < 0.5
    y = np.r_[np.zeros(20), np.ones(20)]
    y[5] = 9.0                                              # an outlier does not move the running median
    med = running(y, 5, np.nanmedian)
    assert med[5] == 0.0 and med[30] == 1.0
