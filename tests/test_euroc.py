"""Tests for dvslam.data.euroc (validation stage V0). Real-data tests skip when a sequence is not on disk."""
import numpy as np
import pytest
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from dvslam.data import base, euroc
from dvslam.eval.diagnostics import _pct_errors
from dvslam.geometry.se3 import chain_increments, relative_pose

K = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1]])
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])   # EuRoC cam0
T_BS = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                 [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                 [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                 [0, 0, 0, 1]])


def radtan_project(X, K, dist):
    """Independent radial-tangential projection (camera frame points -> pixels)."""
    k1, k2, p1, p2 = dist
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)


def test_rectify_maps_match_radtan_projection():
    map_x, map_y = euroc.rectify_maps(K, DIST, (752, 480))
    rng = np.random.default_rng(0)
    # 3D points whose pinhole projection K X / Z lands on integer pixels of the output image
    u = rng.integers(40, 712, 500)
    v = rng.integers(30, 450, 500)
    Z = rng.uniform(0.5, 20, 500)
    X = (np.linalg.inv(K) @ np.stack([u, v, np.ones(500)])).T * Z[:, None]
    uv_pin = (K @ X.T).T
    assert np.allclose(uv_pin[:, :2] / uv_pin[:, 2:], np.stack([u, v], 1))
    expected = radtan_project(X, K, DIST)
    got = np.stack([map_x[v, u], map_y[v, u]], 1)
    assert np.abs(got - expected).max() < 0.1


def write_fixture(root, scene, cam_t_ns, gt_t_ns, w=16, h=12):
    import cv2
    d = root / 'machine_hall' / scene / 'mav0'
    (d / 'cam0' / 'data').mkdir(parents=True)
    (d / 'state_groundtruth_estimate0').mkdir(parents=True)
    lines = ['#timestamp [ns],filename']
    for t in cam_t_ns:
        cv2.imwrite(str(d / 'cam0' / 'data' / f'{t}.png'), np.full((h, w), 128, np.uint8))
        lines.append(f'{t},{t}.png')
    (d / 'cam0' / 'data.csv').write_text('\n'.join(lines) + '\n')
    (d / 'cam0' / 'sensor.yaml').write_text(
        'sensor_type: camera\nT_BS:\n  cols: 4\n  rows: 4\n'
        '  data: [' + ', '.join(repr(float(x)) for x in T_BS.ravel()) + ']\n'
        f'rate_hz: 20\nresolution: [{w}, {h}]\ncamera_model: pinhole\n'
        f'intrinsics: [10.0, 10.0, {w / 2}, {h / 2}] #fu, fv, cu, cv\n'
        'distortion_model: radial-tangential\ndistortion_coefficients: [-0.1, 0.01, 0.001, 0.0005]\n')
    rows = ['#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []']
    for t in gt_t_ns:
        p, q = gt_pose(t)
        rows.append(f'{t},' + ','.join(f'{x:.9f}' for x in (*p, q[3], q[0], q[1], q[2])) + ',0,0,0,0,0,0,0,0,0')
    (d / 'state_groundtruth_estimate0' / 'data.csv').write_text('\n'.join(rows) + '\n')


T0 = 1_403_636_579_763_555_584


def fixture_times():
    return T0 + np.arange(40) * 50_000_000, T0 + 120_000_000 + np.arange(330) * 5_000_000


def gt_pose(t_ns):
    s = (t_ns - T0) * 1e-9
    return np.array([1.0 + 0.5 * s, -2.0 + 0.1 * s, 0.8]), Rotation.from_rotvec([0, 0, 0.2 * s]).as_quat()


def test_gt_composition_trim_and_dropout(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    cam, gt = fixture_times()                                   # cam 20 Hz, 2 s; GT 200 Hz, 0.12 s to 1.765 s
    gt = gt[(gt < T0 + 1_000_000_000) | (gt > T0 + 1_100_000_000)]   # dropout: 0.995 s -> 1.105 s
    write_fixture(tmp_path, 'MH_01_easy', cam, gt)
    seq = euroc.load_sequence('MH_01_easy', root=tmp_path)
    # first frame covered: 0.15 s (index 3); last: 1.75 s (index 35)
    assert seq.timestamps_ns[0] == cam[3] and seq.timestamps_ns[-1] == cam[35] and len(seq) == 33
    assert seq.split == 'validation' and seq.dataset == 'euroc' and seq.frame_rate_hz == 20.0
    np.testing.assert_allclose(seq.timestamps, (cam[3:36] - cam[3]) * 1e-9)
    # frames at 1.0, 1.05, 1.1 s fall inside the 0.11 s GT gap
    invalid = {int((t - T0) // 50_000_000) for t in seq.timestamps_ns[~seq.gt_valid]}
    assert invalid == {20, 21, 22}
    for i in np.flatnonzero(seq.gt_valid):
        p, q = gt_pose(seq.timestamps_ns[i])
        T_wb = np.eye(4)
        T_wb[:3, :3], T_wb[:3, 3] = Rotation.from_quat(q).as_matrix(), p
        np.testing.assert_allclose(seq.T_wc[i], T_wb @ T_BS, atol=1e-8)
    assert seq.T_wc.dtype == np.float64 and len(seq.image_paths) == len(seq) and seq.K[0, 0] == 10.0
    assert all(p.exists() for p in seq.image_paths)


def test_split_is_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    write_fixture(tmp_path, 'MH_01_easy', *fixture_times())
    assert euroc.load_sequence('MH_01_easy', root=tmp_path).split == 'validation'
    assert euroc.load_sequence('MH_01_easy', 'validation', root=tmp_path).split == 'validation'
    for bad in ('test', 'train'):
        with pytest.raises(ValueError):
            euroc.load_sequence('MH_01_easy', bad, root=tmp_path)


def test_load_images_false_leaves_cache_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    write_fixture(tmp_path, 'MH_01_easy', *fixture_times())
    seq = euroc.load_sequence('MH_01_easy', root=tmp_path, load_images=False)
    assert not (tmp_path / 'cache').exists()
    dst = base.cache_dir('euroc', 'MH_01_easy', euroc.CACHE_TAG)
    assert seq.image_paths == [dst / f'{t}.png' for t in seq.timestamps_ns]
    full = euroc.load_sequence('MH_01_easy', root=tmp_path)
    assert seq.image_paths == full.image_paths
    np.testing.assert_array_equal(seq.T_wc, full.T_wc)


def test_two_time_offsets_share_one_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'CACHE_ROOT', tmp_path / 'cache')
    write_fixture(tmp_path, 'MH_01_easy', *fixture_times())
    a = euroc.load_sequence('MH_01_easy', root=tmp_path, time_offset_s=0.0)
    b = euroc.load_sequence('MH_01_easy', root=tmp_path, time_offset_s=0.1)
    assert b.timestamps_ns[0] < a.timestamps_ns[0] and a.image_paths[0].parent == b.image_paths[0].parent
    assert all(q.exists() for q in a.image_paths + b.image_paths)


def test_missing_scene_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        euroc.load_sequence('MH_03_medium', root=tmp_path)
    assert euroc.available_scenes('test', root=tmp_path) == []


ON_DISK = euroc.available_scenes('validation') + euroc.available_scenes('test')
ALL = euroc.SPLITS['validation'] + euroc.SPLITS['test']


@pytest.fixture(scope='module', params=ALL)
def seq(request):
    if request.param not in ON_DISK:
        pytest.skip(f'EuRoC {request.param} not on disk under {euroc.DEFAULT_ROOT}')
    return euroc.load_sequence(request.param)


def test_real_lengths(seq):
    t_src = euroc._read_cam_csv(euroc._scene_dir(seq.scene, None))[0]
    cached = base.cache_dir('euroc', seq.scene, euroc.CACHE_TAG)
    assert len(seq) == len(seq.image_paths) == len(seq.T_wc) == len(seq.timestamps) == len(seq.gt_valid)
    assert len(seq) + trimmed_count(seq, t_src) == len(t_src)
    assert all(p.exists() and p.parent == cached for p in seq.image_paths)
    assert [p.stem for p in seq.image_paths] == [str(t) for t in seq.timestamps_ns]
    assert all((cached / f'{t}.png').exists() for t in t_src)   # every cam0/data.csv row is in the cache
    assert (seq.width, seq.height) == (752, 480)


def trimmed_count(seq, t_src):
    """Source frames before the first and after the last kept frame."""
    return int((t_src < seq.timestamps_ns[0]).sum() + (t_src > seq.timestamps_ns[-1]).sum())


def test_real_chaining(seq):
    inc = np.stack([relative_pose(a, b) for a, b in zip(seq.T_wc[:-1], seq.T_wc[1:])])
    np.testing.assert_allclose(chain_increments(seq.T_wc[0], inc), seq.T_wc, atol=1e-9)


def test_real_metres_and_gt_floor(seq):
    p = seq.T_wc[:, :3, 3]
    ext = p.max(0) - p.min(0)
    assert np.all((ext[:2] > 1) & (ext[:2] < 40)), ext
    ok = seq.pair_valid
    steps = np.diff(p, axis=0)[ok]
    n = np.linalg.norm(steps, axis=1)
    assert 0.005 < np.median(n) < 0.15

    n_src = len(euroc._read_cam_csv(euroc._scene_dir(seq.scene, None))[0])
    both = ok[:-1] & ok[1:]
    nn = np.linalg.norm(np.diff(p, axis=0), axis=1)
    rough = np.abs(np.diff(nn))[both] / nn[:-1][both]
    smooth = np.diff(savgol_filter(p, 15, 3, axis=0), axis=0)[ok]
    mape = _pct_errors(steps, smooth).mean()
    stats = {'kept': len(seq), 'trimmed': n_src - len(seq), 'invalid': int((~seq.gt_valid).sum()),
             'step_med_mm': 1e3 * np.median(n), 'step_p10_mm': 1e3 * np.percentile(n, 10),
             'step_p90_mm': 1e3 * np.percentile(n, 90), 'path_m': nn.sum(),
             'roughness_med': np.median(rough), 'mape_raw_vs_sg15_3_pct': mape}
    print(f'\n[euroc {seq.scene}] ' + ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                                            for k, v in stats.items()))
    assert all(np.isfinite(v) for v in stats.values())
