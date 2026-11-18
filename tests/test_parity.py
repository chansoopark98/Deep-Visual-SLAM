"""Plan v3 item 5.1: parity harness between two runs of the same tracker."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.eval.parity import compare_increments, compare_runs
from dvslam.odometry.base import PoseEstimate, save_increments


def incs(n=50, seed=0):
    rng = np.random.default_rng(seed)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.01, (n, 3))).as_matrix()
    out[:, :3, 3] = rng.normal(0, 5e-3, (n, 3))
    return out


def test_identical_increments_have_zero_differences():
    a = incs()
    r = compare_increments(a, a.copy(), gt_t=a[:, :3, 3])
    assert r['max_dt_mm'] == 0 and r['max_drot_deg'] == 0 and r['max_drift_mm'] == 0 and r['item03_diff_pp'] == 0


def test_known_translation_offset_is_reported():
    a = incs()
    b = a.copy()
    b[10, 0, 3] += 1e-4                               # 0.1 mm on one pair
    r = compare_increments(a, b)
    assert r['max_dt_mm'] == pytest.approx(0.1, rel=1e-6)
    assert r['end_drift_mm'] > 0


def test_compare_runs_rejects_misaligned_indices(tmp_path):
    a = incs(5)
    for name, idx in (('A', range(1, 6)), ('B', range(2, 7))):
        (tmp_path / name).mkdir()
        save_increments(tmp_path / name / 'x.inc.npz',
                        [PoseEstimate(i, 0.0, T) for i, T in zip(idx, a)], {'scene': 'x'})
    with pytest.raises(ValueError, match='indices differ'):
        compare_runs(tmp_path / 'A', tmp_path / 'B')
