"""Plan v3.2 item T.1: TUM-VI loader (rectification, GT composition, metres, chaining)."""
import json

import cv2
import numpy as np
import pytest
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from dvslam.data import base, tumvi
from dvslam.eval.diagnostics import _pct_errors
from dvslam.geometry.se3 import chain_increments, relative_pose

INTRINSICS = np.array([190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504])
DIST = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])
ROOMS_ON_DISK = [s for s in tumvi.SCENES if tumvi.sequence_dir(s).is_dir()]


def equidistant_project(X, intrinsics, dist):
    """Kannala-Brandt / Kalibr 'equidistant': theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8), r_px = f theta_d."""
    fx, fy, cx, cy = intrinsics
    r = np.hypot(X[:, 0], X[:, 1])
    theta = np.arctan2(r, X[:, 2])
    th2 = theta ** 2
    theta_d = theta * (1 + dist[0] * th2 + dist[1] * th2 ** 2 + dist[2] * th2 ** 3 + dist[3] * th2 ** 4)
    s = np.where(r > 0, theta_d / np.maximum(r, 1e-300), 0.0)
    return np.stack([fx * s * X[:, 0] + cx, fy * s * X[:, 1] + cy], 1)


def test_rectification_map_matches_equidistant_model():
    map_x, map_y = tumvi.rectify_maps(INTRINSICS, DIST)
    assert map_x.shape == map_y.shape == (tumvi.OUT_SIZE, tumvi.OUT_SIZE)
    rng = np.random.default_rng(0)
    n = tumvi.OUT_SIZE
    uv = np.concatenate([rng.integers(0, n, (2000, 2)), [[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1]]])
    # 3D points in front of the camera whose pinhole projection K_OUT X / Z lands on those output pixels
    X = (np.linalg.inv(tumvi.K_OUT) @ np.c_[uv, np.ones(len(uv))].T).T * rng.uniform(0.3, 8.0, (len(uv), 1))
    p = (tumvi.K_OUT @ X.T).T
    np.testing.assert_allclose(p[:, :2] / p[:, 2:], uv, atol=1e-9)
    src = np.stack([map_x[uv[:, 1], uv[:, 0]], map_y[uv[:, 1], uv[:, 0]]], 1)
    err = np.linalg.norm(src - equidistant_project(X, INTRINSICS, DIST), axis=1)
    assert err.max() < 0.1, err.max()
    # the whole pinhole view samples inside the source image
    assert map_x.min() >= 0 and map_y.min() >= 0 and map_x.max() <= n - 1 and map_y.max() <= n - 1
    fov_h = 2 * np.degrees(np.arctan(n / 2 / tumvi.K_OUT[0, 0]))
    print(f'\nrectification: max reprojection error {err.max():.2e} px, K_OUT f {tumvi.K_OUT[0, 0]}, '
          f'FoV {fov_h:.1f} deg h/v, {2 * np.degrees(np.arctan(n / 2 * np.sqrt(2) / tumvi.K_OUT[0, 0])):.1f} deg diag')


T_CAM_IMU = np.eye(4)
T_CAM_IMU[:3, :3] = Rotation.from_euler('xyz', [-90, 2, 178], degrees=True).as_matrix()
T_CAM_IMU[:3, 3] = [0.047, -0.047, -0.068]
T0_NS = 1_520_000_000_000_000_000


def imu_pose(t_s):
    """Synthetic IMU trajectory: constant velocity, constant-rate rotation (both exact under lerp / slerp)."""
    T = np.tile(np.eye(4), (len(t_s), 1, 1))
    T[:, :3, :3] = (Rotation.from_rotvec([0.1, -0.2, 0.3]) * Rotation.from_rotvec(np.outer(t_s, [0.2, 0.5, -0.4])))\
        .as_matrix()
    T[:, :3, 3] = np.array([1.0, -2.0, 0.5]) + np.outer(t_s, [0.6, -0.3, 0.2])
    return T


def write_fixture(root):
    d = tumvi.sequence_dir('room1', root)
    (d / 'mav0/cam0/data').mkdir(parents=True)
    (d / 'mav0/mocap0').mkdir(parents=True)
    (d / 'dso').mkdir()
    t_cam = T0_NS + np.arange(12) * 50_000_000                       # 20 Hz, frames at 0.00 .. 0.55 s
    img = np.zeros((8, 8), np.uint8)
    with open(d / 'mav0/cam0/data.csv', 'w') as f:
        f.write('#timestamp [ns],filename\n')
        for t in t_cam:
            f.write(f'{t},{t}.png\n')
            cv2.imwrite(str(d / f'mav0/cam0/data/{t}.png'), img)
    # mocap 120 Hz from 0.07 s to 0.478 s (frames 0, 1 and 10, 11 outside), no samples between 0.22 and 0.28 s
    t_s = np.arange(0.07, 0.48, 1 / 120)
    t_s = t_s[(t_s < 0.22) | (t_s > 0.28)]
    T = imu_pose(t_s)
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    with open(d / 'mav0/mocap0/data.csv', 'w') as f:
        f.write('#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], '
                'q_RS_z []\n')
        for t, p, qq in zip(T0_NS + np.round(t_s * 1e9).astype(np.int64), T[:, :3, 3], q):
            f.write(f'{t},{p[0]:.12f},{p[1]:.12f},{p[2]:.12f},{qq[3]:.12f},{qq[0]:.12f},{qq[1]:.12f},{qq[2]:.12f}\n')
    cam = {'cam0': {'T_cam_imu': T_CAM_IMU.tolist(), 'distortion_model': 'equidistant',
                    'distortion_coeffs': DIST.tolist(), 'intrinsics': INTRINSICS.tolist(), 'resolution': [512, 512]}}
    (d / 'dso/camchain.yaml').write_text(yaml.safe_dump(cam))
    return t_cam


@pytest.mark.parametrize('offset_s', [0.0, 0.004])
def test_gt_composition_trimming_and_dropout(tmp_path, monkeypatch, offset_s):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    t_cam = write_fixture(tmp_path / 'data')
    seq = tumvi.load_sequence('room1', root=tmp_path / 'data', time_offset_s=offset_s)
    assert seq.dataset == 'tumvi' and seq.split == 'train' and len(seq) == 8          # frames 2..9 kept
    np.testing.assert_array_equal(seq.timestamps_ns, t_cam[2:10])
    np.testing.assert_allclose(seq.timestamps, np.arange(8) * 0.05, atol=1e-12)
    # only the 0.25 s frame (index 3 after trimming) falls in the dropout (mocap gap about 0.21-0.29 s), also when
    # queried at +4 ms
    np.testing.assert_array_equal(seq.gt_valid, [True, True, True, False, True, True, True, True])
    np.testing.assert_array_equal(seq.pair_valid, [True, True, False, False, True, True, True])
    expect = imu_pose((t_cam[2:10] - T0_NS) * 1e-9 + offset_s) @ np.linalg.inv(T_CAM_IMU)
    np.testing.assert_allclose(seq.T_wc[seq.gt_valid], expect[seq.gt_valid], atol=1e-6)
    assert seq.T_wc.dtype == np.float64
    np.testing.assert_array_equal(seq.K, tumvi.K_OUT)
    assert all(p.exists() and p.parent == tmp_path / 'cache/tumvi/room1' / tumvi.CACHE_TAG for p in seq.image_paths)


def test_cache_independent_of_time_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    write_fixture(tmp_path / 'data')
    a = tumvi.load_sequence('room1', root=tmp_path / 'data', time_offset_s=0.0)
    b = tumvi.load_sequence('room1', root=tmp_path / 'data', time_offset_s=-0.03)   # moves the trimmed range
    assert len(a) != len(b) and a.image_paths[0].parent == b.image_paths[0].parent
    meta = json.loads((a.image_paths[0].parent / 'meta.json').read_text())
    assert meta['n_frames'] == 12 and meta['bit_depth'] == 8


def test_load_images_false_does_not_touch_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    write_fixture(tmp_path / 'data')
    seq = tumvi.load_sequence('room1', root=tmp_path / 'data', load_images=False)
    assert not (tmp_path / 'cache').exists()
    full = tumvi.load_sequence('room1', root=tmp_path / 'data')
    assert seq.image_paths == full.image_paths
    np.testing.assert_array_equal(seq.T_wc, full.T_wc)


def test_split_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    write_fixture(tmp_path / 'data')
    assert tumvi.load_sequence('room1', root=tmp_path / 'data', load_images=False).split == 'train'
    assert tumvi.load_sequence('room1', 'train', tmp_path / 'data', load_images=False).split == 'train'
    for split in ('test', 'validation', 'all'):
        with pytest.raises(ValueError):
            tumvi.load_sequence('room1', split, tmp_path / 'data', load_images=False)


def test_splits_are_whole_sequences():
    # user decision 2026-11-19: train room1-4, validation room5, test room6; every room in exactly one split
    assert tumvi.SPLITS == {'train': ('room1', 'room2', 'room3', 'room4'), 'validation': ('room5',), 'test': ('room6',)}
    assert sorted(sc for v in tumvi.SPLITS.values() for sc in v) == list(tumvi.SCENES)
    assert set(tumvi.available_scenes('test', '/nonexistent')) == set()


_SEQS = {}


def load(scene):
    if scene not in _SEQS:
        _SEQS[scene] = tumvi.load_sequence(scene)
    return _SEQS[scene]


def steps(seq):
    return np.linalg.norm(np.diff(seq.T_wc[:, :3, 3], axis=0), axis=1)


@pytest.mark.skipif(not ROOMS_ON_DISK, reason='TUM-VI data not found under DVSLAM_TUMVI_ROOT')
@pytest.mark.parametrize('scene', ROOMS_ON_DISK)
def test_real_sequence_consistency_and_chaining(scene):
    seq = load(scene)
    n_src = sum(1 for _ in (tumvi.sequence_dir(scene) / 'mav0/cam0/data').glob('*.png'))
    assert len(seq) == len(seq.image_paths) == len(seq.T_wc) == len(seq.timestamps) == len(seq.gt_valid)
    assert len(seq) <= n_src and all(p.exists() for p in seq.image_paths)
    assert [p.stem for p in seq.image_paths] == [str(t) for t in seq.timestamps_ns]
    img = cv2.imread(str(seq.image_paths[len(seq) // 2]), cv2.IMREAD_UNCHANGED)
    assert img.shape == (seq.height, seq.width) == (512, 512) and img.dtype == np.uint8
    assert seq.gt_valid[0] and seq.gt_valid[-1]
    assert np.all(np.diff(seq.timestamps_ns) > 0) and seq.timestamps[0] == 0
    inc = relative_pose(seq.T_wc[:-1], seq.T_wc[1:])
    np.testing.assert_allclose(chain_increments(seq.T_wc[0], inc), seq.T_wc, atol=1e-9)
    s = steps(seq)[seq.pair_valid]
    print(f'\n{scene}: {len(seq)} frames kept of {n_src}, {int((~seq.gt_valid).sum())} frames bridging a mocap '
          f'dropout, {int((~seq.pair_valid).sum())} invalid pairs, median step {np.median(s) * 1e3:.1f} mm')


@pytest.mark.skipif('room1' not in ROOMS_ON_DISK, reason='TUM-VI room1 not found under DVSLAM_TUMVI_ROOT')
def test_room1_metres_and_gt_roughness():
    seq = load('room1')
    extent = np.ptp(seq.T_wc[:, :3, 3], axis=0)
    assert np.all((extent > 0.5) & (extent < 10)), extent
    ok = seq.pair_valid
    s = steps(seq)
    med = np.median(s[ok])
    assert abs(med / 0.053 - 1) < 0.15, med
    both = ok[:-1] & ok[1:]
    print(f'\nroom1: extent {np.round(extent, 3)} m')
    # camera centre (what the loader returns) and, for comparison with plan §11.2 (53 mm, 3.7 %, 1.3 %), the IMU
    # origin that the mocap tracks: the lever arm of the camera adds rotation-induced motion to its steps
    T_cam_imu = tumvi.load_cam0_calib(tumvi.sequence_dir('room1'))[0]
    for frame, T in (('camera', seq.T_wc), ('imu', seq.T_wc @ T_cam_imu)):
        p = T[:, :3, 3]
        st = np.linalg.norm(np.diff(p, axis=0), axis=1)
        roughness = np.median(np.abs(np.diff(st))[both] / st[:-1][both]) * 100
        raw = np.einsum('nji,nj->ni', T[:-1, :3, :3], np.diff(p, axis=0))
        smooth = np.einsum('nji,nj->ni', T[:-1, :3, :3], np.diff(savgol_filter(p, 15, 3, axis=0), axis=0))
        mape = _pct_errors(raw[ok], smooth[ok]).mean()
        print(f'room1 {frame}: median step {np.median(st[ok]) * 1e3:.2f} mm, roughness {roughness:.2f} %, '
              f'GT vs SG(15,3) item-03 MAPE {mape:.2f} %')
        # SG(15,3) at 20 Hz spans 0.75 s and also removes real hand motion, so this MAPE is an upper bound on the GT
        # noise floor; measured 4.4 % (camera) / 3.7 % (IMU), above the plan's 1.3 %, hence the bound of 6 %
        assert 1 < roughness < 8 and 0.3 < mape < 6
