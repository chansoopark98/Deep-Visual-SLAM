"""Unit tests for dvslam.data.base (validation stage V0): GT interpolation and the rectification cache."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.data import base


def gt_track(n=50, dt_ns=10_000_000):
    """Constant linear velocity and constant angular velocity about z, sampled at 100 Hz."""
    t = np.arange(n, dtype=np.int64) * dt_ns + 1_700_000_000_000_000_000   # realistic epoch-sized ns stamps
    s = (t - t[0]) * 1e-9
    pos = np.stack([0.3 * s, -0.1 * s, 0.05 * s], 1)
    quat = Rotation.from_rotvec(np.outer(s, [0, 0, 0.8])).as_quat()
    return t, pos, quat


def test_interpolation_is_exact_for_constant_velocity():
    t, pos, quat = gt_track()
    tq = t[:-1] + 3_300_000                       # 3.3 ms after each sample
    T, valid = base.interpolate_poses(tq, t, pos, quat, max_gap_s=0.02)
    s = (tq - t[0]) * 1e-9
    assert valid.all()
    np.testing.assert_allclose(T[:, :3, 3], np.stack([0.3 * s, -0.1 * s, 0.05 * s], 1), atol=1e-12)
    np.testing.assert_allclose(T[:, :3, :3], Rotation.from_rotvec(np.outer(s, [0, 0, 0.8])).as_matrix(), atol=1e-12)
    np.testing.assert_array_equal(T[:, 3], np.tile([0, 0, 0, 1.0], (len(tq), 1)))


def test_queries_outside_the_gt_span_are_invalid():
    t, pos, quat = gt_track()
    tq = np.array([t[0] - 1, t[0], t[-1], t[-1] + 1])
    _, valid = base.interpolate_poses(tq, t, pos, quat, max_gap_s=0.02)
    assert valid.tolist() == [False, True, True, False]


def test_dropout_is_flagged_not_hidden():
    t, pos, quat = gt_track()
    keep = np.r_[0:20, 30:50]                      # 100 ms hole between samples 19 and 30
    tq = np.array([t[10] + 1, t[19] + 1, t[25], t[30] - 1, t[30] + 1])
    _, valid = base.interpolate_poses(tq, t[keep], pos[keep], quat[keep], max_gap_s=0.02)
    assert valid.tolist() == [True, False, False, False, True]


def test_non_increasing_gt_timestamps_are_rejected():
    t, pos, quat = gt_track()
    t[5] = t[4]
    with pytest.raises(ValueError):
        base.interpolate_poses(t[:3], t, pos, quat, max_gap_s=0.02)


def test_rectify_cache_writes_once_and_rejects_changed_parameters(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    img = (np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) % 251).astype(np.uint8)
    paths = []
    for k in range(3):
        p = src / f'{k:04d}.png'
        cv2.imwrite(str(p), img + k)
        paths.append(p)
    mx, my = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(32, dtype=np.float32))   # identity map
    dst = tmp_path / 'cache'
    out = base.rectify_to_cache(paths, dst, mx, my, {'K': np.eye(3).tolist(), 'width': 48, 'height': 32})
    assert [p.name for p in out] == ['0000.png', '0001.png', '0002.png']
    np.testing.assert_array_equal(cv2.imread(str(out[1]), cv2.IMREAD_UNCHANGED), img + 1)
    mtime = out[0].stat().st_mtime_ns
    base.rectify_to_cache(paths, dst, mx, my, {'K': np.eye(3).tolist(), 'width': 48, 'height': 32})
    assert out[0].stat().st_mtime_ns == mtime      # second call reuses the files
    with pytest.raises(ValueError):
        base.rectify_to_cache(paths, dst, mx, my, {'K': (2 * np.eye(3)).tolist(), 'width': 48, 'height': 32})


def test_rectify_cache_stores_16_bit_sources_as_8_bit_and_restores_opencv_threads(tmp_path):
    src = tmp_path / 'a.png'
    img16 = (np.arange(32 * 48, dtype=np.uint32).reshape(32, 48) * 40 % 65536).astype(np.uint16)
    cv2.imwrite(str(src), img16)
    mx, my = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(32, dtype=np.float32))
    before = cv2.getNumThreads()
    out = base.rectify_to_cache([src], tmp_path / 'cache', mx, my, {'K': np.eye(3).tolist()})
    assert cv2.getNumThreads() == before
    got = cv2.imread(str(out[0]), cv2.IMREAD_UNCHANGED)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, (img16 >> 8).astype(np.uint8))
    assert sorted(p.name for p in (tmp_path / 'cache').iterdir()) == ['a.png', 'meta.json']   # no temp files left


def test_rectify_cache_meta_is_written_before_the_images(tmp_path):
    mx, my = np.meshgrid(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError):                      # unreadable source: the build stops ...
        base.rectify_to_cache([tmp_path / 'missing.png'], tmp_path / 'cache', mx, my, {'K': [[1.0]]})
    assert (tmp_path / 'cache' / 'meta.json').exists()   # ... but the parameters are already on record
    with pytest.raises(ValueError, match='different parameters'):
        base.rectify_to_cache([tmp_path / 'missing.png'], tmp_path / 'cache', mx, my, {'K': [[2.0]]})


def test_pair_valid_needs_gt_at_both_ends():
    seq = base.Sequence('x', 's', 'validation', np.arange(4), [], np.tile(np.eye(4), (4, 1, 1)), np.eye(3), 4, 4,
                        np.arange(4.0), np.arange(4), np.array([True, False, True, True]), 20.0)
    assert seq.pair_valid.tolist() == [False, False, True]
    assert len(seq) == 4
