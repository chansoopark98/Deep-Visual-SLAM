"""Plan v3 item 0.8: RGBDICPOdometry on one real Redwood pair agrees with GT to a few mm (skips without open3d/data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('open3d')
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera  # noqa: E402
from dvslam.odometry.rgbd_icp import RGBDICPOdometry  # noqa: E402

pytestmark = pytest.mark.skipif(not (redwood.DEFAULT_ROOT / 'validation/loft/loft.json').exists(),
                                reason='Redwood data not mounted')


def test_pair_matches_gt_increment():
    seq = redwood.load_sequence('loft', 'validation')
    odo = RGBDICPOdometry()
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    frames = [Frame(i, float(seq.timestamps[i]), seq.image_paths[i], seq.depth_paths[i]) for i in (100, 101)]
    assert odo.step(frames[0]) is None
    est = odo.step(frames[1])
    gt = relative_pose(seq.T_wc[100], seq.T_wc[101])
    assert est.index == 101 and est.rel_increment.dtype == np.float64 and est.source == 'tracked'
    assert np.abs(est.rel_increment[:3, 3] - gt[:3, 3]).max() < 5e-3      # metres
    assert np.abs(est.rel_increment[:3, :3] - gt[:3, :3]).max() < 2e-3
