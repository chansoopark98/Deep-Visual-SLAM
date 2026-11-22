"""Plan v3 item 0.7: KLTPnPOdometry on a rendered synthetic point cloud (known motion, exact depth)."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import Frame, PinholeCamera
from dvslam.odometry.klt_pnp import KLTPnPOdometry

W, H, N_FRAMES = 320, 240, 30          # half resolution keeps the rendering fast
K = np.array([[262.5, 0, 159.5], [0, 262.5, 119.5], [0, 0, 1]])


def _trajectory():
    T = np.tile(np.eye(4), (N_FRAMES, 1, 1))
    k = np.arange(N_FRAMES)
    T[:, :3, 3] = np.c_[0.006 * k, 0.001 * np.sin(k / 5), 0.002 * k]      # ~6.3 mm / frame
    T[:, :3, :3] = Rotation.from_euler('xyz', np.c_[0.001 * k, 0.002 * k, 0.0005 * k]).as_matrix()
    return T


def _render(T_wc, pts):
    """Gaussian blobs (far to near, nearest wins) and a z-buffered depth splat around each blob (mm, 0 = none)."""
    T_cw = np.linalg.inv(T_wc)
    Xc = pts @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv = Xc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    img, depth = np.full((H, W), 40.0), np.zeros((H, W), np.uint16)
    yy, xx = np.mgrid[-4:5, -4:5]
    for k in np.argsort(-Xc[:, 2]):
        (u, v), z = uv[k], Xc[k, 2]
        if z <= 0 or not (5 <= u < W - 5 and 5 <= v < H - 5):
            continue
        iu, iv = int(u), int(v)
        blob = 40 + 180 * np.exp(-((xx + iu - u) ** 2 + (yy + iv - v) ** 2) / (2 * 1.2 ** 2))
        win = img[iv - 4:iv + 5, iu - 4:iu + 5]
        np.copyto(win, blob, where=blob > 41)
        depth[iv - 3:iv + 4, iu - 3:iu + 4] = round(z * 1000)
    return np.clip(img, 0, 255).astype(np.uint8), depth


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1.5, -1.2, 1.0], [1.7, 1.2, 4.0], (260, 3))     # a cloud with real depth spread
    T = _trajectory()
    root = tmp_path_factory.mktemp('synthetic_cloud')
    frames = []
    for i, T_wc in enumerate(T):
        img, depth = _render(T_wc, pts)
        cv2.imwrite(str(root / f'{i:06d}.png'), img)
        cv2.imwrite(str(root / f'{i:06d}_d.png'), depth)
        frames.append(Frame(i, i / 30, root / f'{i:06d}.png', root / f'{i:06d}_d.png'))
    gt = np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, N_FRAMES)])
    return frames, gt


def _run(frames, **opts):
    odo = KLTPnPOdometry(min_distance=10, lk_win=15, **opts)
    odo.reset(PinholeCamera(K, W, H))
    out = [odo.step(f) for f in frames]
    assert out[0] is None and all(e is not None for e in out[1:])
    assert [e.index for e in out[1:]] == list(range(1, len(frames)))
    return out[1:], odo.finalize()


def test_sensor_depth_recovers_metric_increments(scene):
    frames, gt = scene
    est, diag = _run(frames, mode='sensor_depth')
    assert all(e.source == 'tracked' and e.tracking_state == 'OK' for e in est)
    inc = np.stack([e.rel_increment for e in est])
    assert inc.dtype == np.float64
    err_mm = np.linalg.norm(inc[:, :3, 3] - gt[:, :3, 3], axis=1) * 1e3
    assert err_mm.max() < 0.3, err_mm.max()                     # steps are ~6.3 mm
    rot_err = [np.degrees(np.linalg.norm(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()))
               for a, b in zip(inc, gt)]
    assert max(rot_err) < 0.01
    assert len(diag['total_ms']) == len(est)


def test_depth_free_recovers_increments_up_to_scale(scene):
    frames, gt = scene
    est, _ = _run(frames, mode='depth_free', init_px=8.0, tri_deg=0.5)
    tracked = np.array([e.source == 'tracked' for e in est])
    boot = np.array([e.tracking_state == 'BOOTSTRAP' for e in est])
    assert tracked.sum() >= len(est) - 10 and (~tracked == boot).all()   # only the bootstrap is motion_model
    p, g = np.stack([e.rel_increment[:3, 3] for e in est])[tracked], gt[tracked, :3, 3]
    s = (p * g).sum() / (p * p).sum()                         # one scale for the whole tracked run
    rel = np.linalg.norm(s * p - g, axis=1) / np.linalg.norm(g, axis=1)
    assert np.median(rel) < 0.03 and rel.max() < 0.06, (np.median(rel), rel.max())
