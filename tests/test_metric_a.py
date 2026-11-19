"""Synthetic verification of metric (a) (plan v3.2 §12.2, T.4). No GPU, no data files (one optional real-GT test).

Source pairs mimic TUM-VI room (20 Hz): GT step log-normal, median 53 mm, log-sd 0.6, AR(1) φ = 0.97 (autocorrelated, so
the block bootstrap matters); patch depth 1–3 m, AR(1) φ = 0.99; f = 190 px; 6 scenes × 2,500 pairs. GT position noise
is white, so the step noise ε_gt is MA(1) with Var = V. Errors follow the simulate_mape form
pred = s_true·(1+b)·exp(j·N) + (z̄/f)·σ_px·n.

Tolerances come from 20 independent seeds of `_source` with the defaults below (fit spread, measured 2026-11-19):
sd(b) = 7.9e-5, sd(j) = 2.4e-4, sd(σ_px) = 1.2e-3 px, sd(Δ MAPE vs truth) = 0.06 %p (max |Δ| 0.16 %p). The asserted
tolerances are ≈ 4–5 sd: b ± 4e-4, j ± 1e-3, σ_px ± 5e-3 px; the transfer criterion ±0.2 %p is the plan's.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from dvslam.eval import metric_a as M
from dvslam.eval.diagnostics import simulate_mape
from dvslam.eval.metrics import position_error_pct

B, J, PX, V = 0.01, 0.005, 0.03, (0.4e-3) ** 2      # bias, jitter sd, σ_px (px), Var(ε_gt) (m²)
F_SRC, Z_RW, F_RW = 190.0, 2.0, 525.0
BOOT = dict(reps=200, sim_reps=4)                  # reduced for test time; production default is 1000 / 10


def _ar1(rng, n, mu, sd, phi):
    x = np.empty(n)
    x[0] = rng.normal(mu, sd)
    q = sd * np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + q * rng.standard_normal()
    return x


def _source(b=B, j=J, px=PX, v=V, n_scenes=6, n=2500, seed=0):
    """Per-scene (e∥, s, z̄) in metres, TUM-VI-like."""
    rng = np.random.default_rng(seed)
    E, S, Z = [], [], []
    for _ in range(n_scenes):
        s_true = np.exp(_ar1(rng, n, np.log(0.053), 0.6, 0.97))
        z = np.clip(np.exp(_ar1(rng, n, np.log(1.8), 0.3, 0.99)), 1.0, 3.0)
        pred = s_true * (1 + b) * np.exp(j * rng.standard_normal(n)) + z / F_SRC * px * rng.standard_normal(n)
        m = rng.normal(0, np.sqrt(v / 2), n + 1)       # white GT position noise along the motion → MA(1) step noise
        s_gt = s_true + (m[1:] - m[:-1])
        E.append(pred - s_gt)
        S.append(s_gt)
        Z.append(z)
    return E, S, Z


def _redwood_like(seed=1):
    """5 scenes × 1,234 steps (≈ Redwood test 6,170 pairs), scene medians 4.5–9.1 mm, overall median ≈ 6.8 mm."""
    rng = np.random.default_rng(seed)
    out = []
    for med in (4.5e-3, 5.5e-3, 6.8e-3, 8.0e-3, 9.1e-3):
        u = rng.normal(size=(1234, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        out.append(med * np.exp(0.7 * rng.standard_normal(1234))[:, None] * u)
    return out


@pytest.fixture(scope='module')
def src():
    return _source()


@pytest.fixture(scope='module')
def rw():
    return _redwood_like()


def _transfer_delta(src, steps):
    fit = M.fit_error_model(*src, F_SRC, V)
    est = M.redwood_equivalent(fit.b, fit.j, fit.sigma_px, steps, Z_RW, F_RW)['mape_pct']
    true, _ = simulate_mape(steps, PX * Z_RW / F_RW, B, J, noise='parallel', reps=10, seed=0)
    return est, true


def test_parameter_recovery(src, rw):
    fit = M.fit_error_model(*src, F_SRC, V)
    print(f'\nfit: b={fit.b:.5f} j={fit.j:.5f} σ_px={fit.sigma_px:.5f} (true {B}, {J}, {PX})')
    assert fit.b == pytest.approx(B, abs=4e-4)
    assert fit.j == pytest.approx(J, abs=1e-3)
    assert fit.sigma_px == pytest.approx(PX, abs=5e-3)
    assert fit.n_pairs == 15000
    r = M.metric_a(*src, F_SRC, (V, 0.8 * V, 1.2 * V), rw, Z_RW, F_RW, **BOOT)
    assert r['b_lo'] <= B <= r['b_hi'] and r['j_lo'] <= J <= r['j_hi'] and r['sigma_px_lo'] <= PX <= r['sigma_px_hi']
    assert r['mape_lo_pct'] < r['mape_pct'] < r['mape_hi_pct']


def test_transfer_matches_direct_simulation(src, rw):
    est, true = _transfer_delta(src, rw)
    print(f'\ntransfer (synthetic Redwood): fitted {est:.3f} % vs true-parameter simulation {true:.3f} %')
    assert abs(est - true) <= 0.2


@lru_cache(maxsize=1)
def _real_redwood_steps():
    root = os.environ.get('DVSLAM_REDWOOD_ROOT')
    if not root or not (Path(root) / 'test').is_dir():
        return None
    from dvslam.data.redwood import SCENES, load_poses
    from dvslam.geometry.se3 import relative_pose
    out = []
    for sc in SCENES:
        T = load_poses(Path(root) / 'test' / sc / f'{sc}.json')
        out.append(np.array([relative_pose(T[i - 1], T[i])[:3, 3] for i in range(1, len(T))]))
    return out


@pytest.mark.skipif(_real_redwood_steps() is None, reason='DVSLAM_REDWOOD_ROOT/test not available')
def test_transfer_on_real_redwood_steps(src):
    steps = _real_redwood_steps()
    assert sum(len(g) for g in steps) == 6170
    est, true = _transfer_delta(src, steps)
    print(f'\ntransfer (real Redwood test GT steps): fitted {est:.3f} % vs true-parameter simulation {true:.3f} %')
    assert abs(est - true) <= 0.2


def test_sensitivity_to_gt_noise_variance(src, rw):
    mape = {}
    for k in (0.7, 1.0, 1.3):
        fit = M.fit_error_model(*src, F_SRC, k * V)
        mape[k] = M.redwood_equivalent(fit.b, fit.j, fit.sigma_px, rw, Z_RW, F_RW)['mape_pct']
    print(f'\nsensitivity: Var(ε_gt) −30 % → {mape[0.7]:.3f} % ({mape[0.7] - mape[1.0]:+.3f} %p), '
          f'nominal {mape[1.0]:.3f} %, +30 % → {mape[1.3]:.3f} % ({mape[1.3] - mape[1.0]:+.3f} %p)')
    # over-estimating GT noise removes tracker noise, so the tracker looks better
    assert mape[1.3] < mape[1.0] < mape[0.7]


@pytest.mark.parametrize('b, j, px, v_ci, verdict', [
    (B, J, PX, (0.8, 1.2), 'pass'),
    (0.005, 0.002, 0.002, (0.8, 1.2), 'pass'),           # good tracker buried in GT noise: σ_px² CI reaches 0
    (0.03, J, 0.08, (0.8, 1.2), 'fail'),
    (B, J, 0.042, (0.5, 1.5), 'undetermined'),            # point ≈ 2.87 %, CI straddles 2.96 %
])
def test_decision_rule(rw, b, j, px, v_ci, verdict):
    r = M.metric_a(*_source(b, j, px), F_SRC, (V, v_ci[0] * V, v_ci[1] * V), rw, Z_RW, F_RW, **BOOT)
    print(f"\n{verdict}: [{r['mape_lo_pct']:.3f}, {r['mape_hi_pct']:.3f}] %, σ_px² lo {r['sigma_px2_lo']:.2e}")
    assert r['verdict'] == verdict
    if px == 0.002:
        assert r['sigma_px2_lo'] <= 0


def test_decide_boundaries():
    assert M.decide(1.0, 2.96, 2.96) == 'pass'
    assert M.decide(2.97, 3.5, 2.96) == 'fail'
    assert M.decide(2.96, 3.5, 2.96) == 'undetermined'


def test_paired_difference():
    rng = np.random.default_rng(3)
    a, b0, bz = [], [], []
    for _ in range(4):
        shared = 3 * np.exp(_ar1(rng, 2000, 0, 0.8, 0.9))    # GT-noise part common to both trackers, autocorrelated
        a.append(shared + 0.3 + rng.normal(0, 0.5, 2000))
        b0.append(shared + rng.normal(0, 0.5, 2000))
        bz.append(shared + 0.3 + rng.normal(0, 0.5, 2000))
    d = M.paired_difference(a, b0, reps=500)
    assert d['diff_lo_pct'] <= 0.3 <= d['diff_hi_pct'] and d['excludes_zero']
    z = M.paired_difference(a, bz, reps=500)
    assert z['diff_lo_pct'] <= 0 <= z['diff_hi_pct'] and not z['excludes_zero']
    with pytest.raises(ValueError, match='same scenes'):
        M.paired_difference(a, b0[:3])                               # one scene missing
    with pytest.raises(ValueError, match='same scenes'):
        M.paired_difference(a, [b0[0][:-1]] + b0[1:])                # one pair missing


def test_diagnostics_matched_and_raw():
    rng = np.random.default_rng(4)
    gt = [rng.normal(0, 8e-3, (500, 3)), rng.normal(0, 30e-3, (500, 3))]
    pred = [1.02 * g for g in gt]
    out = M.diagnostics(gt, pred)
    n_in = sum(int(((np.linalg.norm(g, axis=1) >= 4.5e-3) & (np.linalg.norm(g, axis=1) <= 9.1e-3)).sum()) for g in gt)
    assert out['matched_pairs'] == n_in > 0
    assert out['matched_mape_pct'] == pytest.approx(2.0) and out['raw_mape_pct'] == pytest.approx(2.0)
    raw = np.mean([np.mean([position_error_pct(x, y) for x, y in zip(g, p)]) for g, p in zip(gt, pred)])
    assert out['raw_mape_pct'] == pytest.approx(raw)
    assert out['sigma_par_mm'] > 0


def test_units_mm_inputs_are_rejected(src, rw):
    E, S, Z = src
    mm = lambda xs: [1e3 * x for x in xs]
    with pytest.raises(ValueError, match='metres'):
        M.fit_error_model(mm(E), mm(S), Z, F_SRC, V)                 # steps in mm
    with pytest.raises(ValueError, match='metres'):
        M.fit_error_model(E, S, mm(Z), F_SRC, V)                     # depth in mm
    with pytest.raises(ValueError, match='m²'):
        M.fit_error_model(E, S, Z, F_SRC, V * 1e6)                   # variance in mm²
    with pytest.raises(ValueError, match='metres'):
        M.redwood_equivalent(B, J, PX, mm(rw), Z_RW, F_RW)           # Redwood steps in mm
    with pytest.raises(ValueError, match='metres'):
        M.metric_a(mm(E), mm(S), Z, F_SRC, (V, V, V), rw, Z_RW, F_RW, **BOOT)
    with pytest.raises(ValueError, match='variance'):
        M.fit_error_model(E, S, Z, F_SRC, 8e-4)                      # σ = 0.8 mm passed as the variance
    with pytest.raises(ValueError, match='f_px'):
        M.fit_error_model(E, S, Z, 0.0, V)
    with pytest.raises(ValueError, match='f_rw'):
        M.redwood_equivalent(B, J, PX, rw, Z_RW, 0.0)


def test_units_mm_display_fields(src, rw):
    r = M.metric_a(*src, F_SRC, (V, 0.8 * V, 1.2 * V), rw, Z_RW, F_RW, reps=50, sim_reps=2)
    for k in ('sigma_par_rw', 'sigma_par_rw_lo', 'sigma_par_rw_hi'):
        assert r[f'{k}_mm'] == pytest.approx(1e3 * r[f'{k}_m'])
    assert r['sigma_par_rw_m'] == pytest.approx(r['sigma_px'] * Z_RW / F_RW)
    assert 5e-5 < r['sigma_par_rw_m'] < 5e-4                         # ≈ 0.11 mm stored as metres


def test_collinear_design_is_rejected():
    rng = np.random.default_rng(5)
    s = [np.full(1000, 0.05)]
    with pytest.raises(ValueError, match='identifiable'):
        M.fit_error_model([rng.normal(0, 1e-3, 1000)], s, [2.0], F_SRC, V)



def _two_depth_source(n=2500, seed=0):
    """Near scenes (z̄ = 1 m, steps ≈ 30 mm) alternating with far scenes (z̄ = 4 m, steps ≈ 100 mm), narrow step spread.
    Far scenes have the larger steps, so ∂j²/∂V < 0 and ∂σ_px²/∂V > 0: the MAPE RISES with Var(ε_gt)."""
    rng = np.random.default_rng(seed)
    E, S, Z = [], [], []
    for z0, med in [(1.0, 0.03), (4.0, 0.10)] * 3:
        s_true = np.exp(_ar1(rng, n, np.log(med), 0.05, 0.97))
        pred = s_true * (1 + B) * np.exp(J * rng.standard_normal(n)) + z0 / F_SRC * PX * rng.standard_normal(n)
        m = rng.normal(0, np.sqrt(V / 2), n + 1)
        s_gt = s_true + (m[1:] - m[:-1])
        E.append(pred - s_gt)
        S.append(s_gt)
        Z.append(z0)
    return E, S, Z


def test_bounds_not_monotone_in_var_gt(rw):
    src2 = _two_depth_source()
    point = {}
    for k in (0.3, 1.0, 1.7):
        f = M.fit_error_model(*src2, F_SRC, k * V)
        point[k] = M.redwood_equivalent(f.b, f.j, f.sigma_px, rw, Z_RW, F_RW, sim_reps=4)['mape_pct']
    print(f'\nnon-monotone design: MAPE at 0.3 V / V / 1.7 V = {point[0.3]:.3f} / {point[1.0]:.3f} / {point[1.7]:.3f} %')
    assert point[0.3] < point[1.0] < point[1.7]          # opposite of the TUM-VI-like design (test_sensitivity...)
    r = M.metric_a(*src2, F_SRC, (V, 0.3 * V, 1.7 * V), rw, Z_RW, F_RW, **BOOT)
    q = r['mape_q_by_var_pct']
    print(f"bootstrap quantiles by V: {q}; bounds [{r['mape_lo_pct']:.3f}, {r['mape_hi_pct']:.3f}]")
    assert q['v_hi'][1] > q['v_lo'][1]                  # the old rule (upper bound from V_lo) would be too low
    assert r['mape_hi_pct'] == max(v[1] for v in q.values()) and r['mape_lo_pct'] == min(v[0] for v in q.values())
    assert r['mape_lo_pct'] <= min(point.values()) and max(point.values()) <= r['mape_hi_pct']
    assert r['ill_conditioned_reps'] == 0


def test_per_scene_redwood_depth(rw):
    scalar = M.redwood_equivalent(B, J, PX, rw, Z_RW, F_RW)
    same = M.redwood_equivalent(B, J, PX, rw, [Z_RW] * len(rw), F_RW)
    assert same['mape_pct'] == scalar['mape_pct']                      # identical path, identical seeds
    direct, _ = simulate_mape(rw, PX * Z_RW / F_RW, B, J / (1 + B), noise='parallel', reps=10, seed=0)
    assert scalar['mape_pct'] == pytest.approx(direct, abs=0.05)       # per-scene calls = one call, within MC noise
    zs = [1.0, 1.5, 2.0, 3.0, 4.0]
    per = M.redwood_equivalent(B, J, PX, rw, zs, F_RW)
    manual = np.mean([simulate_mape([g], PX * z / F_RW, B, J / (1 + B), noise='parallel', reps=10, seed=k)[0]
                      for k, (g, z) in enumerate(zip(rw, zs))])
    assert per['mape_pct'] == pytest.approx(manual)
    assert per['sigma_par_rw_scene_m'] == pytest.approx([PX * z / F_RW for z in zs])
    with pytest.raises(ValueError, match='per Redwood scene'):
        M.redwood_equivalent(B, J, PX, rw, zs[:3], F_RW)
