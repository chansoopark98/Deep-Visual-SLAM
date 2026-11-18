"""Plan v3 item 1.0: perturbed prior statistics within ±0.1 % of the settings."""
import numpy as np
import pytest

from dvslam.depth.perturbed import PerturbedGTDepth


@pytest.mark.parametrize('bias,jitter', [(0.02, 0.03), (0.0, 0.0), (-0.01, 0.10)])
def test_generated_scale_statistics_match_settings(bias, jitter):
    p = PerturbedGTDepth(bias, jitter, seed=1)
    logs = np.log([p.scale(i) for i in range(200_000)])
    assert np.median(logs) == pytest.approx(np.log1p(bias), abs=1e-3)
    assert logs.std() == pytest.approx(jitter, abs=1e-3)


def test_scale_is_deterministic_per_frame():
    a, b = PerturbedGTDepth(seed=3), PerturbedGTDepth(seed=3)
    assert a.scale(17) == b.scale(17) != PerturbedGTDepth(seed=4).scale(17)
