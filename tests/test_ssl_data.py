"""dvslam.ssl.tumvi_data: gyro integration and its frame conventions, stereo convention, the training sample dict."""
import json

import cv2
import numpy as np
import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation
from test_tumvi import equidistant_project

from dvslam.data import base, tumvi
from dvslam.ssl import tumvi_data

ON_DISK = {s for s in ('room1', 'room5') if tumvi.sequence_dir(s).is_dir()}
T0_NS = 1_520_000_000_000_000_000


def test_gyro_integration_matches_closed_form():
    t_imu = T0_NS + np.arange(0, 400) * 5_000_000                     # 200 Hz, 0 .. 1.995 s
    t0, t1 = T0_NS + 101_234_567, T0_NS + 152_345_678                  # partial IMU intervals at both ends
    w = np.array([0.3, -1.2, 0.7])
    R, ok = tumvi_data.integrate_gyro(t_imu, np.tile(w, (len(t_imu), 1)), t0, t1)
    assert ok
    np.testing.assert_allclose(R, Rotation.from_rotvec(w * (t1 - t0) * 1e-9).as_matrix(), atol=1e-12)
    # a rate linear in time about a fixed axis: angle = integral of the rate, exact under linear interpolation
    axis = np.array([2.0, -1.0, 2.0]) / 3
    s = (t_imu - T0_NS) * 1e-9
    R, ok = tumvi_data.integrate_gyro(t_imu, np.outer(0.5 + 3.0 * s, axis), t0, t1)
    a, b = (t0 - T0_NS) * 1e-9, (t1 - T0_NS) * 1e-9
    np.testing.assert_allclose(R, Rotation.from_rotvec(axis * (0.5 * (b - a) + 1.5 * (b ** 2 - a ** 2))).as_matrix(),
                               atol=1e-12)
    # not covered, or a 20 ms gap inside the interval
    assert not tumvi_data.integrate_gyro(t_imu, np.zeros((400, 3)), T0_NS - 1, t1)[1]
    assert not tumvi_data.integrate_gyro(t_imu, np.zeros((400, 3)), t0, T0_NS + 1_995_000_001)[1]
    keep = (t_imu < T0_NS + 115_000_000) | (t_imu > T0_NS + 130_000_000)
    assert not tumvi_data.integrate_gyro(t_imu[keep], np.zeros((keep.sum(), 3)), t0, t1)[1]


@pytest.mark.skipif('room1' not in ON_DISK, reason='TUM-VI room1 not found under DVSLAM_TUMVI_ROOT')
def test_room1_gyro_matches_mocap_relative_rotation():
    seq = tumvi.load_sequence('room1', load_images=False)
    R, ok = tumvi_data.gyro_rotations(tumvi.sequence_dir('room1'), seq.timestamps_ns)
    m = seq.pair_valid & ok
    R_gt = (np.linalg.inv(seq.T_wc[:-1]) @ seq.T_wc[1:])[m, :3, :3]          # R_{cam_t-1 <- cam_t}
    ang = np.degrees(Rotation.from_matrix(np.swapaxes(R_gt, 1, 2) @ R[m]).magnitude())
    step = np.degrees(Rotation.from_matrix(R_gt).magnitude())
    print(f'\nroom1 gyro vs mocap: {m.sum()} pairs, angle of R_gt^T R_gyro median {np.median(ang):.3f} deg, '
          f'p90 {np.percentile(ang, 90):.3f} deg (median GT step {np.median(step):.2f} deg)')
    assert ok.mean() > 0.99
    assert np.median(ang) < 0.25 and np.percentile(ang, 90) < 0.7     # measured 0.124 / 0.346 deg
    # the transposed convention would be off by about twice the step
    assert np.median(np.degrees(Rotation.from_matrix(R_gt @ R[m]).magnitude())) > 3 * np.median(ang)


def test_photometric_formula():
    gray = np.array([[0, 100, 200, 250, 255]], np.uint8)
    V = np.array([[1.0, 0.5, 0.8, 1.0, 0.2]], np.float32)
    pho, valid = tumvi_data.photometric(gray, V, 1.5, v_min=0.25, sat_level=250)
    np.testing.assert_allclose(pho, [[0, 100 / 255 / 0.5 * 1.5, 200 / 255 / 0.8 * 1.5, 250 / 255 * 1.5, 0]], rtol=1e-6)
    np.testing.assert_array_equal(valid, [[1, 1, 1, 0, 0]])
    assert pho.dtype == valid.dtype == np.float32


@pytest.mark.skipif('room5' not in ON_DISK, reason='TUM-VI room5 not found under DVSLAM_TUMVI_ROOT')
def test_stereo_projection_consistent_with_T_stereo():
    d = tumvi.sequence_dir('room5')
    T = np.array(yaml.safe_load((d / 'dso/camchain.yaml').read_text())['cam1']['T_cn_cnm1'])
    _, intr1, dist1 = tumvi.load_cam_calib(d, 'cam1')
    map_x, map_y = tumvi.rectify_maps(intr1, dist1)
    rng = np.random.default_rng(0)
    uv0 = rng.uniform(40, 470, (2000, 2))
    X0 = (np.linalg.inv(tumvi.K_OUT) @ np.c_[uv0, np.ones(len(uv0))].T).T * rng.uniform(1.0, 6.0, (len(uv0), 1))
    X1 = X0 @ T[:3, :3].T + T[:3, 3]
    p = X1 @ tumvi.K_OUT.T
    uv1 = p[:, :2] / p[:, 2:]
    assert np.all((uv1 >= 0) & (uv1 <= tumvi.OUT_SIZE - 1))
    x0, y0 = np.floor(uv1).astype(int).T
    ax, ay = uv1[:, 0] - x0, uv1[:, 1] - y0

    def bilinear(m):
        m = m.astype(np.float64)
        return (m[y0, x0] * (1 - ax) * (1 - ay) + m[y0, x0 + 1] * ax * (1 - ay) + m[y0 + 1, x0] * (1 - ax) * ay
                + m[y0 + 1, x0 + 1] * ax * ay)

    src = np.stack([bilinear(map_x), bilinear(map_y)], 1)
    err = np.linalg.norm(src - equidistant_project(X1, intr1, dist1), axis=1)
    assert err.max() < 1e-3, err.max()                                      # measured 6.1e-4 px


@pytest.fixture(scope='module')
def room5():
    return tumvi_data.TumviSSLDataset(['room5'], train=False)


@pytest.mark.skipif('room5' not in ON_DISK, reason='TUM-VI room5 not found under DVSLAM_TUMVI_ROOT')
def test_room5_sample_contract(room5):
    f = room5.data[0]
    n = len(f['t_ns'])
    assert 0.95 * n < len(room5) <= n - 2
    item = room5[0]
    assert set(item) == {'net', 'pho', 'valid', 'K', 'T_stereo', 'R_gyro', 'gyro_ok', 'scene', 'index'}
    shapes = {'net': (3, 3, 512, 512), 'pho': (4, 1, 512, 512), 'valid': (4, 1, 512, 512), 'K': (3, 3),
              'T_stereo': (4, 4), 'R_gyro': (2, 3, 3)}
    for k, s in shapes.items():
        assert item[k].shape == s and item[k].dtype == torch.float32, k
    assert item['gyro_ok'].shape == () and item['gyro_ok'].dtype == torch.bool
    assert item['scene'] == 'room5' and isinstance(item['index'], int)
    t = item['index']
    assert np.all(np.abs(np.diff(f['t_ns'][t - 1:t + 2]) - 50_000_000) <= 5_000_000)
    net = item['net']
    assert net.min() >= 0 and net.max() <= 1 and torch.equal(net[:, 0], net[:, 2])
    assert set(item['valid'].unique().tolist()) <= {0.0, 1.0} and item['valid'].mean() > 0.5
    np.testing.assert_array_equal(item['K'], tumvi.K_OUT.astype(np.float32))
    assert abs(np.linalg.norm(item['T_stereo'][:3, 3]) - 0.10109) < 5e-5
    R = item['R_gyro'].double()
    np.testing.assert_allclose(R @ R.transpose(1, 2), np.tile(np.eye(3), (2, 1, 1)), atol=1e-6)
    np.testing.assert_allclose(R[1], f['R_next'][t].T, atol=1e-6)
    # pho of the target frame = un-augmented input / V (exposure ratio 1) on valid pixels
    V0 = torch.from_numpy(f['V']['cam0'])
    ok = item['valid'][1, 0] > 0
    torch.testing.assert_close(item['pho'][1, 0][ok], (net[1, 0] / V0)[ok], rtol=1e-5, atol=1e-6)
    # saturated pixels are invalid
    raw = cv2.imread(f['cam0'][t], cv2.IMREAD_GRAYSCALE)
    assert np.all(item['valid'][1, 0].numpy()[raw >= room5.sat_level] == 0)


@pytest.mark.skipif('room5' not in ON_DISK, reason='TUM-VI room5 not found under DVSLAM_TUMVI_ROOT')
def test_room5_exposure_normalisation_and_gyro_vs_mocap(room5):
    f = room5.data[0]
    e = f['exposure_ms']
    idx = next(i for i, (_, t) in enumerate(room5.samples) if e[t - 1] != e[t])
    item = room5[idx]
    t = item['index']
    V0 = torch.from_numpy(f['V']['cam0'])
    ok = item['valid'][0, 0] > 0
    torch.testing.assert_close(item['pho'][0, 0][ok], (item['net'][0, 0] / V0 * (e[t] / e[t - 1]))[ok].float(),
                               rtol=1e-5, atol=1e-6)
    print(f'\nroom5 sample {idx} (frame {t}): exposure t-1 {e[t - 1]:.3f} ms, t {e[t]:.3f} ms')
    # R_gyro against mocap for both sources: X_src = R X_tgt
    seq = tumvi.load_sequence('room5', load_images=False)
    j = int(np.searchsorted(seq.timestamps_ns, f['t_ns'][len(f['t_ns']) // 2]))
    for _ in range(20):
        if seq.gt_valid[j - 1:j + 2].all():
            break
        j += 1
    t = int(np.searchsorted(f['t_ns'], seq.timestamps_ns[j]))
    item = room5[room5.samples.index((0, t))]
    assert item['gyro_ok']
    for s, R in zip((j - 1, j + 1), item['R_gyro'].double().numpy()):
        R_gt = (np.linalg.inv(seq.T_wc[s]) @ seq.T_wc[j])[:3, :3]
        assert np.degrees(Rotation.from_matrix(R_gt.T @ R).magnitude()) < 1.0


@pytest.mark.skipif('room5' not in ON_DISK, reason='TUM-VI room5 not found under DVSLAM_TUMVI_ROOT')
def test_room5_caches_and_cam0_meta_unchanged(room5):
    f = room5.data[0]
    cam1_dir = base.cache_dir('tumvi', 'room5', 'cam1_pinhole_512_f190')
    meta = json.loads((cam1_dir / 'meta.json').read_text())
    assert meta['K'] == tumvi.K_OUT.tolist() and meta['n_frames'] == len(f['t_ns'])
    assert all(p.startswith(str(cam1_dir)) for p in f['cam1'])
    # load_sequence re-checks the cam0 meta against its own and raises if the dataset had changed it
    seq = tumvi.load_sequence('room5')
    assert {str(p) for p in seq.image_paths} <= set(f['cam0'])
    assert f['V']['cam1'].max() <= 1 and f['V']['cam1'].dtype == np.float32
