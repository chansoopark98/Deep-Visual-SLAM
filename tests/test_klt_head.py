"""Plan v3 item 1.5 / §4.4: precision head invariances on a rendered synthetic scene.

- prior × c (c ∈ {0.5, 2}) leaves ΔT_t unchanged (rel. 1e-6);
- a per-frame constant c_t leaves ΔT_t unchanged;
- g → g + Δ scales the translation of ΔT_t by exactly e^{−Δ}.
"""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.data.redwood import load_depth
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from dvslam.odometry.klt_head import KLTHeadOdometry, huber_location

W, H, N_FRAMES = 320, 240, 8
K = np.array([[262.5, 0, 159.5], [0, 262.5, 119.5], [0, 0, 1]])
S_BA = 1.3                       # BA units = metric × S_BA (arbitrary internal gauge of the fake base)


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
    pts = rng.uniform([-1.5, -1.2, 1.0], [1.7, 1.2, 4.0], (300, 3))
    k = np.arange(N_FRAMES)
    T = np.tile(np.eye(4), (N_FRAMES, 1, 1))
    T[:, :3, 3] = np.c_[0.006 * k, 0.001 * np.sin(k / 5), 0.002 * k]
    T[:, :3, :3] = Rotation.from_euler('xyz', np.c_[0.001 * k, 0.002 * k, 0.0005 * k]).as_matrix()
    root = tmp_path_factory.mktemp('synthetic_head')
    frames = []
    for i, T_wc in enumerate(T):
        img, depth = _render(T_wc, pts)
        cv2.imwrite(str(root / f'{i:06d}.png'), img)
        cv2.imwrite(str(root / f'{i:06d}_d.png'), depth)
        frames.append(Frame(i, i / 30, root / f'{i:06d}.png', root / f'{i:06d}_d.png'))
    return frames, np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, N_FRAMES)])


class ScaledPrior:
    """GT depth × c(frame index)."""
    def __init__(self, c):
        self.c = c

    def predict(self, frame):
        return load_depth(frame.depth_path) * np.float32(self.c(frame.index)), None


class FakeBase:
    """Emits a perturbed GT increment in BA units (metric × S_BA) and BA depths of t−1, like klt_window."""
    def __init__(self, frames, gt, prior, g=0.0):
        self.frames, self.gt, self.prior, self.g = frames, gt, prior, g

    def reset(self, camera):
        self.t = 0

    def step(self, frame):
        i = frame.index
        if i == 0:
            return None
        T_raw = self.gt[i - 1].copy()
        T_raw[:3, 3] = T_raw[:3, 3] * S_BA * 1.15 + 1e-3        # a raw increment that is ~15 % + 1 mm off
        T_raw[:3, :3] = T_raw[:3, :3] @ Rotation.from_rotvec([2e-4, -1e-4, 1e-4]).as_matrix()
        g0 = cv2.imread(str(self.frames[i - 1].image_path), cv2.IMREAD_GRAYSCALE)
        px = cv2.goodFeaturesToTrack(g0, 200, 0.01, 8).reshape(-1, 2).astype(np.float64)
        z = load_depth(self.frames[i - 1].depth_path)[np.round(px[:, 1]).astype(int), np.round(px[:, 0]).astype(int)]
        ok = z > 0
        self.head_inputs = {'T_raw': T_raw, 'g': self.g, 'tracked': True, 'prev_px': px[ok],
                            'prev_z': z[ok].astype(np.float64) * S_BA}
        out = T_raw.copy()
        out[:3, 3] *= np.exp(-self.g)
        return PoseEstimate(i, frame.timestamp, out)


def _run(scene, c=lambda i: 1.0, g=np.log(S_BA)):     # consistent base: BA units = metric · e^g
    frames, gt = scene
    head = KLTHeadOdometry(base=FakeBase(frames, gt, ScaledPrior(c), g), min_distance=8, lk_win=9, min_inliers=20,
                           sigma_mm=3.0)
    head.reset(PinholeCamera(K, W, H))
    out = [head.step(f) for f in frames]
    assert out[0] is None
    st = head.finalize()['head_status']
    return np.stack([e.rel_increment for e in out[1:]]), st


def _assert_same_translation(inc, ref, rel=1e-6):
    """Relative tolerance on each translation vector: ‖t − t_ref‖ ≤ rel · ‖t_ref‖ (§4.4, 1e-6)."""
    d = np.linalg.norm(inc[:, :3, 3] - ref[:, :3, 3], axis=1) / np.linalg.norm(ref[:, :3, 3], axis=1)
    assert d.max() <= rel, d.max()


def test_head_accepts_and_improves_on_raw(scene):
    _, gt = scene
    inc, st = _run(scene)
    assert (st == 'accepted').all(), st
    raw_err = np.linalg.norm(gt[:, :3, 3] * 0.15 + 1e-3 / S_BA, axis=1)
    err = np.linalg.norm(inc[:, :3, 3] - gt[:, :3, 3], axis=1)
    assert err.max() < 0.7e-3 and err.mean() < 0.3 * raw_err.mean(), (err, raw_err)   # steps ~6.3 mm


@pytest.mark.parametrize('c', [0.5, 2.0])
def test_prior_constant_scale_leaves_increment_unchanged(scene, c):
    ref, _ = _run(scene)
    inc, st = _run(scene, c=lambda i: c)
    assert (st == 'accepted').all()
    _assert_same_translation(inc, ref)
    np.testing.assert_allclose(inc[:, :3, :3], ref[:, :3, :3], rtol=0, atol=1e-9)


def test_prior_per_frame_scale_leaves_increment_unchanged(scene):
    ref, _ = _run(scene)
    rng = np.random.default_rng(5)
    ct = np.exp(rng.normal(0, 0.2, N_FRAMES))
    inc, st = _run(scene, c=lambda i: ct[i])
    assert (st == 'accepted').all()
    _assert_same_translation(inc, ref)
    np.testing.assert_allclose(inc[:, :3, :3], ref[:, :3, :3], rtol=0, atol=1e-9)


@pytest.mark.parametrize('delta', [-0.3, 0.05, 0.4])
def test_g_change_scales_translation_by_exp_minus_delta(scene, delta):
    g0 = np.log(S_BA) + 0.1
    ref, _ = _run(scene, g=g0)
    inc, st = _run(scene, g=g0 + delta)
    assert (st == 'accepted').all()
    np.testing.assert_allclose(inc[:, :3, 3], ref[:, :3, 3] * np.exp(-delta), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(inc[:, :3, :3], ref[:, :3, :3])


def test_gate_rejects_far_result_and_keeps_raw(scene):
    frames, gt = scene
    head = KLTHeadOdometry(base=FakeBase(frames, gt, ScaledPrior(lambda i: 1.0), np.log(S_BA)), min_distance=8, lk_win=9,
                           min_inliers=20, sigma_mm=0.05, sigma_deg=1e-4)
    head.reset(PinholeCamera(K, W, H))
    out = [head.step(f) for f in frames][1:]
    d = head.finalize()
    assert (d['head_status'] == 'gate').all() and not d['head_accepted'].any()
    np.testing.assert_array_equal(np.stack([e.rel_increment for e in out]), d['head_raw_inc'])


def test_huber_location_is_shift_equivariant():
    x = np.random.default_rng(0).standard_t(3, 200)
    assert abs(huber_location(x + 0.7) - huber_location(x) - 0.7) < 1e-12


def test_base_options_need_prefix():
    with pytest.raises(ValueError, match='base.'):
        KLTHeadOdometry(base='klt_window', window=8)
