"""Plan v3 item 0.4: loader, increment I/O and the chaining convention."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.data import redwood
from dvslam.geometry.se3 import chain_increments, relative_pose
from dvslam.odometry.base import (PoseEstimate, causal_trajectory, load_increments, save_increments,
                                  write_tum)

REPO_ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (redwood.DEFAULT_ROOT / 'test/apartment/apartment.json').exists()


def random_traj(n=100, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.5, (n, 3))).as_matrix()
    T[:, :3, 3] = np.cumsum(rng.normal(0, 0.01, (n, 3)), 0)
    return T


def test_chaining_gt_increments_recovers_T_wc():
    T = random_traj()
    inc = np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, len(T))])
    np.testing.assert_allclose(chain_increments(T[0], inc), T, atol=1e-12)


@pytest.mark.skipif(not HAVE_DATA, reason='Redwood data not mounted')
def test_chaining_real_gt_recovers_T_wc():
    seq = redwood.load_sequence('apartment', 'test')
    inc = np.stack([relative_pose(seq.T_wc[i - 1], seq.T_wc[i]) for i in range(1, len(seq))])
    np.testing.assert_allclose(chain_increments(seq.T_wc[0], inc), seq.T_wc, atol=1e-9)


def test_increment_file_round_trip(tmp_path):
    T = random_traj(20)
    est = [PoseEstimate(i, i / 30, relative_pose(T[i - 1], T[i]), 'OK' if i % 2 else 'BOOTSTRAP',
                        'tracked' if i % 3 else 'motion_model') for i in range(1, 20)]
    save_increments(tmp_path / 'x.inc.npz', est, {'scene': 'x', 'n': 20})
    z = load_increments(tmp_path / 'x.inc.npz')
    assert z['meta'] == {'scene': 'x', 'n': 20}
    np.testing.assert_array_equal(z['index'], np.arange(1, 20))
    np.testing.assert_array_equal(z['rel_increment'], np.stack([e.rel_increment for e in est]))
    assert z['rel_increment'].dtype == np.float64
    assert list(z['tracking_state']) == [e.tracking_state for e in est]
    assert list(z['source']) == [e.source for e in est]


def test_tum_output_round_trip(tmp_path):
    T = random_traj(10)
    T_out = causal_trajectory(np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, 10)]))
    write_tum(tmp_path / 'x.tum', np.arange(10) / 30, T_out)
    rows = np.loadtxt(tmp_path / 'x.tum')
    np.testing.assert_allclose(rows[:, 1:4], T_out[:, :3, 3], atol=1e-8)
    np.testing.assert_allclose(Rotation.from_quat(rows[:, 4:]).as_matrix(), T_out[:, :3, :3], atol=1e-8)


@pytest.mark.skipif(not HAVE_DATA, reason='Redwood data not mounted')
def test_loader_on_test_split():
    seq = redwood.load_sequence('apartment', 'test')
    assert len(seq) == 1596 and seq.T_wc.dtype == np.float64 and seq.T_wc.shape == (1596, 4, 4)
    assert np.all(np.diff(seq.frame_ids) == 1)
    assert seq.K[0, 0] == 525 and redwood.FRAME_RATE_VERIFIED is False
    d = redwood.load_depth(seq.depth_paths[0])
    assert d.shape == (480, 640) and 0 < d[d > 0].mean() < 10


@pytest.mark.skipif(not HAVE_DATA, reason='Redwood data not mounted')
def test_loader_rejects_image_pose_mismatch(tmp_path):
    src = redwood.DEFAULT_ROOT / 'test/lobby'
    dst = tmp_path / 'test/lobby'
    (dst / 'image').mkdir(parents=True)
    (dst / 'depth').mkdir()
    shutil.copy(src / 'lobby.json', dst)
    shutil.copy(redwood.DEFAULT_ROOT / 'intrinsic.npy', tmp_path)
    for p in sorted((src / 'image').glob('*.jpg'))[:5]:
        (dst / 'image' / p.name).touch()
    with pytest.raises(ValueError, match='images but'):
        redwood.load_sequence('lobby', 'test', tmp_path)
