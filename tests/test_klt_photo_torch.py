"""Plan v3.4 S1: the PyTorch photometric refinement (klt_photo_torch.py) against the numpy one (klt_photo.py)."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_klt_window import H, K, W, _UnitNorm, scene  # noqa: F401  (scene is a fixture)

from dvslam.odometry import klt_photo as kp
from dvslam.odometry import klt_photo_torch as kpt
from dvslam.odometry.base import PinholeCamera
from dvslam.odometry.klt_window import KLTWindowOdometry

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _plane(n=256, f=190.0):
    """Textured plane Z_w = 2 m seen from the origin (frame 0) and from X1 = R1 X0 + t1 (frame 1):
    (I0, I1, depth 0, depth 1, K, R1, t1)."""
    c = (n - 1) / 2
    Kp = np.array([[f, 0, c], [0, f, c], [0, 0, 1]])
    uu, vv = np.meshgrid(np.arange(n), np.arange(n))
    ray = np.stack([(uu - c) / f, (vv - c) / f, np.ones((n, n))], -1)

    def render(R, t):                                   # camera X_c = R X_w + t
        rw, cw = ray @ R, -R.T @ t
        d = (2.0 - cw[2]) / rw[..., 2]
        P = cw + d[..., None] * rw
        img = 0.5 + 0.2 * np.sin(9 * P[..., 0]) * np.cos(7 * P[..., 1]) + 0.1 * np.sin(23 * P[..., 0] + 11 * P[..., 1])
        return img.astype(np.float32), d

    R1 = Rotation.from_euler('xyz', [0.3, -0.4, 0.2], degrees=True).as_matrix()
    t1 = np.array([0.02, -0.01, 0.03])
    I0, D0 = render(np.eye(3), np.zeros(3))
    I1, D1 = render(R1, t1)
    return I0, I1, D0, D1, Kp, R1, t1


@pytest.mark.parametrize('levels', [1, 3])
def test_torch_refine_matches_numpy(levels):
    """refine_prepared_torch reproduces refine_prepared on a rendered plane: float64 to 1e-9, float32 to 2e-6 m;
    prepare_torch reproduces cv2 prepare (Sobel / 8, pyrDown, reflect-101 borders)."""
    I0, I1, D0, _, Kp, _, t1 = _plane()
    M = np.ones(I0.shape, bool)
    M[:, -10:] = False                                  # an invalid band in frame t
    pts = kp.select_points(I0, np.ones_like(M), 2, 0.002)
    z = D0[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    P0, P1 = kp.prepare(1.1 * I0 + 0.05, np.ones_like(M), levels), kp.prepare(I1, M, levels)
    args = (0.97, pts, z, Kp, np.eye(3), np.zeros(3), levels, 20, 0.03, 0.0)
    R_n, t_n, i_n = kp.refine_prepared(P0, P1, *args)
    assert np.linalg.norm(t_n - t1) < 5e-4 and i_n['cost'] < i_n['cost0']
    R_t, t_t, i_t = kpt.refine_prepared_torch(P0, P1, *args, device=DEV)
    assert np.abs(t_t - t_n).max() < 1e-9 and np.abs(R_t - R_n).max() < 1e-9, (t_t - t_n)
    assert i_t['n'] == i_n['n'] and abs(i_t['cost'] - i_n['cost']) < 1e-9 * i_n['cost0']
    R_f, t_f, _ = kpt.refine_prepared_torch(P0, P1, *args, dtype=torch.float32, device=DEV)
    assert np.abs(t_f - t_n).max() < 2e-6 and np.abs(R_f - R_n).max() < 2e-6, (t_f - t_n)
    for lv, (Pn, Pt) in enumerate(zip(P1, kpt.prepare_torch(torch.from_numpy(I1), torch.from_numpy(M), levels))):
        assert Pt.shape == Pn.shape and np.abs(Pt.numpy() - Pn).max() < 1e-5, lv


def test_gradients_reach_features_and_weights():
    """A loss on the refined pose back-propagates to a 2-channel feature map (through prepare_torch) and to per-point
    weights; autograd matches central finite differences of the whole unrolled LM (float64, tiny image)."""
    rng = np.random.default_rng(1)
    n, f, c = 40, 40.0, 19.5
    Kp = np.array([[f, 0, c], [0, f, c], [0, 0, 1]])
    yy, xx = np.mgrid[:n, :n] / n
    base = np.stack([np.sin(7 * xx + 3 * yy), np.cos(5 * yy - 4 * xx) * xx])
    feat1 = torch.tensor(base + 0.01 * rng.standard_normal(base.shape), dtype=torch.float64)
    feat0 = torch.tensor(np.roll(base, 1, axis=2), dtype=torch.float64, requires_grad=True)
    valid = torch.ones(n, n, dtype=torch.bool)
    v, u = np.mgrid[6:n - 6:3, 6:n - 6:3]
    pts = torch.tensor(np.c_[u.ravel(), v.ravel()], dtype=torch.float64)
    z = torch.full((len(pts),), 1.5, dtype=torch.float64)
    wgt = torch.tensor(rng.uniform(0.5, 1.5, len(pts)), requires_grad=True)
    P1 = kpt.prepare_torch(feat1, valid, 1)
    t0 = np.array([0.01, 0.0, 0.0])
    proj = torch.tensor([0.3, -1.2, 0.7], dtype=torch.float64)

    def loss(F0, w):
        R, t, info = kpt.refine_torch(kpt.prepare_torch(F0, valid, 1), P1, 1.0, pts, z, Kp, np.eye(3), t0, 1, 3, 0.03,
                                      0.0, w)
        return (t * proj).sum() + R[0, 2] + R[1, 2], info

    L, info = loss(feat0, wgt)
    assert info['cost'] < info['cost0']
    L.backward()
    g_f, g_w = feat0.grad, wgt.grad
    assert g_f.abs().sum() > 0 and g_w.abs().sum() > 0
    eps = 1e-6
    with torch.no_grad():
        for k in rng.choice(len(pts), 4, replace=False):
            e = torch.zeros_like(wgt)
            e[k] = eps
            fd = (loss(feat0, wgt + e)[0] - loss(feat0, wgt - e)[0]) / (2 * eps)
            assert abs(float(fd) - float(g_w[k])) < 1e-5 * max(1.0, abs(float(fd))), (k, float(fd), float(g_w[k]))
        big = torch.argsort(g_f.abs().ravel(), descending=True)[:6]
        for k in big.tolist():
            e = torch.zeros_like(feat0).ravel()
            e[k] = eps
            e = e.view_as(feat0)
            fd = (loss(feat0 + e, wgt)[0] - loss(feat0 - e, wgt)[0]) / (2 * eps)
            gk = float(g_f.ravel()[k])
            assert abs(float(fd) - gk) < 1e-4 * max(1e-3, abs(gk)) + 1e-8, (k, float(fd), gk)


def test_tracker_photo_backend_torch(scene):  # noqa: F811
    """photo_backend=torch (float64) in the tracker (photo=all, photo_bidir) gives the numpy output to < 1e-6 m;
    torch32 to < 1e-4 m; an unknown backend is a ValueError."""
    frames, _ = scene

    def run(**o):
        odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150,
                                prior='oracle', photo='all', photo_cell=2, photo_grad=0.002, photo_bidir=True, **o)
        odo.photo_norm = _UnitNorm()
        odo.reset(PinholeCamera(K, W, H))
        est = [odo.step(f) for f in frames[:20]][1:]
        return np.stack([e.rel_increment for e in est]), np.array(odo.diag['photo'])

    ref, ph = run()
    assert (ph == 1).sum() >= 10
    for backend, tol in (('torch', 1e-6), ('torch32', 1e-4)):
        out, ph_t = run(photo_backend=backend)
        assert np.array_equal(ph_t, ph) and np.abs(out - ref)[:, :3, 3].max() < tol, (backend, np.abs(out - ref).max())
    with pytest.raises(ValueError, match='photo_backend'):
        KLTWindowOdometry(prior='oracle', photo_backend='jax')
