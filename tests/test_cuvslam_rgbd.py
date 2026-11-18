"""Plan v3 item 1.4: CuVSLAMRGBDOdometry depth-prior options (skips without cuvslam / data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('cuvslam')
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.odometry.base import Frame  # noqa: E402
from dvslam.odometry.cuvslam_rgbd import CuVSLAMRGBDOdometry  # noqa: E402

needs_data = pytest.mark.skipif(not (redwood.DEFAULT_ROOT / 'validation/loft/loft.json').exists(),
                                reason='Redwood data not mounted')


def _frame(seq, i):
    return Frame(i, float(seq.timestamps[i]), seq.image_paths[i], seq.depth_paths[i])


def test_bad_prior_rejected():
    with pytest.raises(ValueError):
        CuVSLAMRGBDOdometry(prior='nope')


@needs_data
def test_prior_depth_maps():
    seq = redwood.load_sequence('loft', 'validation')
    f = _frame(seq, 100)
    raw = CuVSLAMRGBDOdometry()._depth(f)
    assert raw.dtype == np.uint16 and (raw == np.array(redwood.load_depth(f.depth_path) * 1000).round().astype(np.uint16)).all()
    assert (CuVSLAMRGBDOdometry(prior='oracle')._depth(f) == raw).all()
    p0 = CuVSLAMRGBDOdometry(prior='perturbed', prior_seed=0)._depth(f)
    assert (p0 == CuVSLAMRGBDOdometry(prior='perturbed', prior_seed=0)._depth(f)).all()      # deterministic
    assert (p0 == 0).sum() == (raw == 0).sum() and not (p0 == raw).all()                    # invalid stays invalid
    assert not (p0 == CuVSLAMRGBDOdometry(prior='perturbed', prior_seed=1)._depth(f)).all()


@needs_data
def test_cache_prior(tmp_path):
    seq = redwood.load_sequence('loft', 'validation')
    f = _frame(seq, 100)
    (tmp_path / 'loft').mkdir()
    np.save(tmp_path / 'loft' / f.depth_path.stem, redwood.load_depth(f.depth_path))
    raw = CuVSLAMRGBDOdometry()._depth(f)
    assert (CuVSLAMRGBDOdometry(prior=f'cache:{tmp_path}')._depth(f) == raw).all()
