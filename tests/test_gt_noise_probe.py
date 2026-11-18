"""Plan v3 item 0.9: the GT-noise probe recovers injected GT noise when tracker residuals are independent."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from gt_noise_probe_validation import run_cell, smooth_truth  # noqa: E402

from dvslam.data import redwood  # noqa: E402

pytestmark = pytest.mark.skipif(not (redwood.DEFAULT_ROOT / 'test/apartment/apartment.json').exists(),
                                reason='Redwood data not mounted')


@pytest.fixture(scope='module')
def truths():
    return smooth_truth()


@pytest.mark.parametrize('sigma_g_mm', [0.0, 0.1, 0.2, 0.3])
@pytest.mark.parametrize('estimator', ['intercept', 'k1', 'lag1'])
def test_probe_recovers_sigma_g_with_independent_trackers(truths, sigma_g_mm, estimator):
    for seed in range(3):
        est = run_cell(truths, sigma_g_mm, 'independent', seed)[estimator] * 1e3
        assert abs(est - sigma_g_mm) <= 0.05, (seed, est)


def test_k1_is_biased_by_shared_error_but_intercept_is_not(truths):
    r = run_cell(truths, 0.0, 'shared_white', 0)
    assert r['k1'] * 1e3 > 0.04 and r['intercept'] * 1e3 < 0.02
    assert np.isfinite(r['shared_slope']) and r['shared_slope'] > 0
