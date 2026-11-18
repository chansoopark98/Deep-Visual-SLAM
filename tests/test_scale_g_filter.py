"""Phase 2 items 2.2/2.3: robust g filter and the depth warp used by the asynchronous-prior simulation."""
import numpy as np
import pytest

from dvslam.scale.g_filter import RobustGFilter, apply_g, g_series
from scripts.phase2_run import warp_depth

K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])


def test_g_bootstrap_then_median_init():
    g = g_series(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5]), min_m=5)
    assert np.all(g[:4] == 0.0)
    assert g[4] == pytest.approx(0.3)
    assert g[5] > 0.3


def test_g_converges_and_clips_outliers():
    rng = np.random.default_rng(0)
    m = 0.05 + 0.01 * rng.standard_normal(2000)
    m[500] = 5.0                                           # one wild measurement
    g = g_series(m, tau=50)
    assert g[-1] == pytest.approx(0.05, abs=0.005)
    assert abs(g[500] - g[499]) < 0.01                     # clipped step
    assert RobustGFilter().update(np.nan) == 0.0            # NaN measurement leaves g unchanged


def test_apply_g_is_causal_translation_scaling():
    inc = np.tile(np.eye(4), (3, 1, 1))
    inc[:, :3, 3] = 1.0
    out = apply_g(inc, np.array([0.0, np.log(2.0), 0.0, 9.0]))
    assert np.allclose(out[:, 0, 3], [1.0, 0.5, 1.0])
    assert np.array_equal(out[:, :3, :3], inc[:, :3, :3])


def test_warp_identity_and_forward_motion():
    d = np.full((480, 640), 2.0, np.float32)
    assert np.allclose(warp_depth(d, K, np.eye(4), np.eye(4))[1:-1, 1:-1], 2.0)
    T = np.eye(4)
    T[2, 3] = 0.1                                          # destination 10 cm forward: a fronto-parallel wall at 1.9 m
    w = warp_depth(d, K, np.eye(4), T)
    c = w[200:280, 280:360]
    assert np.allclose(c, 1.9, atol=1e-4)
