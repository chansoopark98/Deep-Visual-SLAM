"""Plan v3 item 1.3: KLTWindowOdometry (window BA + prior factor + g) on a rendered synthetic point cloud."""
import cv2
import numpy as np
import pytest
from scipy.optimize import least_squares as _real_least_squares
from scipy.spatial.transform import Rotation

import dvslam.odometry.klt_window as kw
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import Frame, PinholeCamera
from dvslam.odometry.klt_window import KLTWindowOdometry

W, H, N_FRAMES = 320, 240, 40
K = np.array([[262.5, 0, 159.5], [0, 262.5, 119.5], [0, 0, 1]])


def _trajectory():
    T = np.tile(np.eye(4), (N_FRAMES, 1, 1))
    k = np.arange(N_FRAMES)
    T[:, :3, 3] = np.c_[0.006 * k, 0.001 * np.sin(k / 5), 0.002 * k]
    T[:, :3, :3] = Rotation.from_euler('xyz', np.c_[0.001 * k, 0.002 * k, 0.0005 * k]).as_matrix()
    return T


def _render(T_wc, pts):
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
        np.copyto(img[iv - 4:iv + 5, iu - 4:iu + 5], blob, where=blob > 41)
        depth[iv - 3:iv + 4, iu - 3:iu + 4] = round(z * 1000)
    return np.clip(img, 0, 255).astype(np.uint8), depth


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1.5, -1.2, 1.0], [1.7, 1.2, 4.0], (260, 3))
    T = _trajectory()
    root = tmp_path_factory.mktemp('synthetic_window')
    frames = []
    for i, T_wc in enumerate(T):
        img, depth = _render(T_wc, pts)
        cv2.imwrite(str(root / f'{i:06d}.png'), img)
        cv2.imwrite(str(root / f'{i:06d}_d.png'), depth)
        frames.append(Frame(i, i / 30, root / f'{i:06d}.png', root / f'{i:06d}_d.png'))
    gt = np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, N_FRAMES)])
    return frames, gt


def _run(frames, **opts):
    odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150, **opts)
    odo.reset(PinholeCamera(K, W, H))
    out = [odo.step(f) for f in frames]
    assert out[0] is None and [e.index for e in out[1:]] == list(range(1, len(frames)))
    return out[1:], odo


def test_oracle_prior_recovers_metric_increments(scene):
    frames, gt = scene
    est, odo = _run(frames, prior='oracle')
    assert all(e.source == 'tracked' for e in est)
    assert sum(e.is_keyframe for e in est) >= 5 and len(odo.kfs) == 6       # the window slid
    inc = np.stack([e.rel_increment for e in est])
    err_mm = np.linalg.norm(inc[:, :3, 3] - gt[:, :3, 3], axis=1) * 1e3
    assert err_mm.max() < 0.3, err_mm.max()
    assert abs(odo.g) < 0.01


def test_prior_bias_sets_metric_scale_and_window_averages_jitter(scene):
    frames, gt = scene
    g = np.linalg.norm(gt[:, :3, 3], axis=1)

    def ratios(**o):
        est, _ = _run(frames, prior='perturbed', **o)
        p = np.linalg.norm(np.stack([e.rel_increment[:3, 3] for e in est]), axis=1)
        return p / g

    r = ratios(prior_bias=0.1, prior_jitter=0.0) / ratios(prior_bias=0.0, prior_jitter=0.0)
    assert abs(np.median(r) - 1.1) < 0.005, np.median(r)      # the prior defines the metric scale
    # per-frame prior jitter (5 %): the window BA beats the 0.7 path (prior back-projection per keyframe) on both
    # the per-pair scale spread and the mean scale error, averaged over 3 prior seeds
    def stats(**o):
        lr = [np.log(ratios(prior_bias=0.0, prior_jitter=0.05, prior_seed=s, **o)) for s in (1, 2, 3)]
        return np.mean([x.std() for x in lr]), np.mean([abs(x.mean()) for x in lr])
    ba, base = stats(), stats(ba=False, kf_align=False)
    assert ba[0] < base[0] and ba[1] < base[1], (ba, base)


def test_ba_jacobian_matches_finite_differences(scene, monkeypatch):
    frames, _ = scene
    calls = []

    def spy(fun, x0, jac, **kw_):
        calls.append((fun, jac, x0.copy()))
        return _real_least_squares(fun, x0, jac=jac, **kw_)

    monkeypatch.setattr(kw, 'least_squares', spy)
    _run(frames[:16], prior='perturbed', prior_jitter=0.05)
    fun, jac, x0 = calls[-1]
    x = x0 + np.random.default_rng(0).normal(0, 1e-3, len(x0))
    J = jac(x).toarray()
    eps = 1e-7
    Jn = np.stack([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(len(x))], 1)
    # rotation columns use the small-angle (right-Jacobian ≈ I) approximation
    np.testing.assert_allclose(J, Jn, atol=2e-2 * np.abs(Jn).max(), rtol=0)
    np.testing.assert_allclose(J[:, -1], Jn[:, -1], rtol=1e-5, atol=1e-6)


def test_g_scales_output_translation_only(scene):
    frames, _ = scene
    est, odo = _run(frames[:12], prior='oracle')
    hi = odo.head_inputs
    np.testing.assert_allclose(est[-1].rel_increment[:3, 3], hi['T_raw'][:3, 3] * np.exp(-hi['g']), rtol=1e-12)
    np.testing.assert_allclose(est[-1].rel_increment[:3, :3], hi['T_raw'][:3, :3], atol=1e-15)
