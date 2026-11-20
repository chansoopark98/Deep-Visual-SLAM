"""EXP-015 oracle-scale solver: the metric-optimal scale (weighted median) and the per-window grouping."""
import numpy as np
import pytest

from scripts.exp015_scale_oracle import item03, oracle_scale, scale_groups, templates
from scripts.exp015_scale_oracle import weighted_median as wmedian


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.001, 0.06, n)                        # 1-60 mm steps
    p = 0.9 * g * (1 + 0.05 * rng.standard_normal(n))      # predictions = GT x 0.9 with 5 % noise
    return np.abs(p), g


def test_metric_scale_recovers_inverse_bias_and_beats_median_ratio():
    p, g = _synthetic()
    s_metric, s_median = oracle_scale(p, g, 'metric'), oracle_scale(p, g, 'median')
    assert s_metric == pytest.approx(1 / 0.9, rel=0.01)
    assert s_median == pytest.approx(1 / 0.9, rel=0.01)
    assert item03(s_metric * p, g).mean() <= item03(s_median * p, g).mean() + 1e-12
    assert item03(s_metric * p, g).mean() < item03(p, g).mean()


def test_metric_scale_is_the_minimum_on_a_grid():
    p, g = _synthetic(n=500, seed=1)
    p[::7] *= 1.6                                          # skewed errors: weighted median and median ratio differ
    s = oracle_scale(p, g, 'metric')
    grid = np.linspace(0.5 * s, 1.5 * s, 4001)
    best = min(item03(c * p, g).mean() for c in grid)
    assert item03(s * p, g).mean() <= best + 1e-9


def test_weighted_median_and_degenerate_groups():
    assert wmedian(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0
    assert oracle_scale(np.zeros(3), np.ones(3), 'metric') == 1.0     # no usable pair: s stays 1
    assert oracle_scale(np.ones(3), np.zeros(3), 'median') == 1.0


def test_one_scale_per_window():
    p, g = _synthetic(n=400, seed=2)
    p[200:] *= 1.25                                        # second window has a different scale bias
    out, s = scale_groups(p, g, np.arange(400) // 200, 'metric')
    assert s[0] == pytest.approx(1 / 0.9, rel=0.02)
    assert s[1] == pytest.approx(1 / (0.9 * 1.25), rel=0.02)
    assert item03(out, g).mean() < item03(scale_groups(p, g, np.zeros(400, int), 'metric')[0], g).mean()


def test_templates():
    t1, t2 = templates(np.array([1.0, 2.0, 3.0, 6.0]))
    assert np.allclose(t1, [1.0, 1.0, 2.0, 3.0])
    assert np.allclose(t2, [1.5, 2.0, 11 / 3, 4.5])
