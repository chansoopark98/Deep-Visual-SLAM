"""Plan v3.4 S2: learned photometric features / weights (photo_learned.py), their training losses and window pass
(apps/train_photo_s2.py), and the tracker hook photo_feat."""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_klt_photo_torch import _plane
from test_klt_window import H, K, W, _UnitNorm, scene  # noqa: F401  (scene is a fixture)

from dvslam.odometry import klt_photo as kp
from dvslam.odometry import klt_photo_torch as kpt
from dvslam.odometry import photo_learned as pl
from dvslam.odometry.base import PinholeCamera
from dvslam.odometry.klt_window import KLTWindowOdometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps'))
import train_photo_s2 as tr

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _ckpt(tmp_path, mode, width=4):
    torch.manual_seed(0)
    p = tmp_path / f'{mode}.pt'
    pl.save_net(p, pl.PhotoNet(mode, width))
    return p


@pytest.mark.parametrize('mode', pl.MODES)
def test_untrained_net_reproduces_s1_refinement(tmp_path, mode):
    """Zero-initialised heads: weights mode is bit-identical to refine_prepared_torch; feature modes match it to the
    prepare_torch vs cv2 difference (< 1e-6 m)."""
    I0, I1, D0, _, Kp, _, t1 = _plane(128, 95.0)
    M = np.ones(I0.shape, bool)
    M[:, -8:] = False
    pts = kp.select_points(I0, np.ones_like(M), 2, 0.002)
    z = D0[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    P0, P1 = kp.prepare(I0, np.ones_like(M), 1), kp.prepare(I1, M, 1)
    args = (0.97, pts, z, Kp, np.eye(3), np.zeros(3), 1, 10, 0.03, 0.01)
    R_r, t_r, i_r = kpt.refine_prepared_torch(P0, P1, *args, device=DEV)
    lp = pl.LearnedPhoto(_ckpt(tmp_path, mode), DEV)
    R_l, t_l, i_l = lp(P0, P1, *args)
    assert np.linalg.norm(t_r - t1) < 2e-3
    if mode == 'w':
        assert np.array_equal(R_l, R_r) and np.array_equal(t_l, t_r) and i_l == i_r
    else:
        assert np.abs(t_l - t_r).max() < 1e-6 and np.abs(R_l - R_r).max() < 1e-6 and i_l['n'] == i_r['n']
    assert lp(P0, P1, *args)[1].tolist() == t_l.tolist()      # cached frame outputs give the same result


def test_losses_on_synthetic_data():
    """reproj_loss is 0 at the true pose and grows with a pose error (Cauchy-bounded); pose_mid is the tracker's
    midpoint; point_weights have mean 1 (exactly 1 for a zero map); photo_cost is lower at the true pose."""
    I0, I1, D0, _, Kp, R1, t1 = _plane(128, 95.0)
    rng = np.random.default_rng(0)
    p0 = rng.uniform(10, 118, (60, 2))
    z0 = 2.0 / ((np.c_[p0, np.ones(60)] @ np.linalg.inv(Kp).T)[:, 2])
    X1 = (np.c_[p0, np.ones(60)] @ np.linalg.inv(Kp).T * z0[:, None]) @ R1.T + t1
    p1 = X1[:, :2] / X1[:, 2:] * Kp[[0, 1], [0, 1]] + Kp[:2, 2]
    T = pl.to_T(torch.tensor(R1), torch.tensor(t1))
    args = (torch.tensor(p0), torch.tensor(z0), torch.tensor(p1), Kp)
    assert float(pl.reproj_loss(T, *args)) < 1e-20
    dT = pl.to_T(torch.eye(3, dtype=torch.float64), torch.tensor([0.01, 0.0, 0.0]))
    l1, l2 = float(pl.reproj_loss(dT @ T, *args)), float(pl.reproj_loss(dT @ dT @ T, *args))
    assert 0 < l1 < l2 < 4 * l1
    e2 = torch.tensor([1e-4, 1.0, 1e4])
    rho = pl.cauchy(e2, 0.5)
    assert abs(float(rho[0]) - 0.5e-4) < 1e-8 and float(rho[2]) < 0.25 * 0.5 * float(e2[2]) ** 0.5
    Ra = Rotation.from_rotvec([0.01, -0.02, 0.005]).as_matrix()
    Rb = Ra @ Rotation.from_rotvec([1e-3, 2e-4, -5e-4]).as_matrix()
    mid = pl.pose_mid(pl.to_T(torch.tensor(Ra), torch.zeros(3, dtype=torch.float64)),
                      pl.to_T(torch.tensor(Rb), torch.ones(3, dtype=torch.float64)))
    ref = Ra @ Rotation.from_rotvec(0.5 * Rotation.from_matrix(Ra.T @ Rb).as_rotvec()).as_matrix()
    assert np.abs(mid[:3, :3].numpy() - ref).max() < 1e-12 and np.allclose(mid[:3, 3].numpy(), 0.5)
    pts = torch.tensor([[3.0, 4.0], [5.0, 6.0], [7.0, 1.0]])
    assert torch.equal(pl.point_weights(torch.zeros(10, 10), pts), torch.ones(3))
    big = torch.cartesian_prod(torch.arange(40.0), torch.arange(40.0))[:1413].to(DEV)   # n where CUDA's x / n
    assert torch.equal(pl.point_weights(torch.zeros(40, 40, device=DEV), big), torch.ones(1413, device=DEV))
    lw = torch.randn(10, 10)
    assert abs(float(pl.point_weights(lw, pts).mean()) - 1) < 1e-6
    S0 = [torch.from_numpy(s) for s in kp.prepare(I0, np.ones(I0.shape, bool), 1)]
    S1 = [torch.from_numpy(s) for s in kp.prepare(I1, np.ones(I1.shape, bool), 1)]
    pp = torch.from_numpy(kp.select_points(I0, np.ones(I0.shape, bool), 2, 0.002))
    zz = torch.from_numpy(D0[pp[:, 1].long(), pp[:, 0].long()])
    c_true = pl.photo_cost(S0, S1, 1.0, pp, zz, Kp, torch.tensor(R1), torch.tensor(t1))
    c_off = pl.photo_cost(S0, S1, 1.0, pp, zz, Kp, torch.tensor(R1), torch.tensor(t1 + [0.01, 0, 0]))
    assert float(c_true) < 0.5 * float(c_off)


def _sample(n=96):
    """A train_photo_s2 window (L = 1) built from the rendered plane, the tracker's start increment perturbed."""
    I0, I1, D0, D1, Kp, R1, t1 = _plane(n, 0.75 * n)
    M = np.ones(I0.shape, bool)
    P = [kp.prepare(I, M, 1)[0] for I in (I0, I1)]
    pts = [kp.select_points_g(p[..., 1], p[..., 2], M, 2, 0.002)[0] for p in P]
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R1, t1
    inc = np.linalg.inv(A)
    inc0 = inc.copy()
    inc0[:3, 3] += [0.004, -0.003, 0.002]
    zf = D0[pts[0][:, 1].astype(int), pts[0][:, 0].astype(int)]
    zb = D1[pts[1][:, 1].astype(int), pts[1][:, 0].astype(int)]
    rng = np.random.default_rng(1)
    pa = rng.uniform(8, n - 8, (40, 2))
    za = D0[pa[:, 1].astype(int), pa[:, 0].astype(int)]          # (nearest-pixel depth: a small reprojection error)
    X = (np.c_[pa, np.ones(40)] @ np.linalg.inv(Kp).T * za[:, None]) @ R1.T + t1
    pb = X[:, :2] / X[:, 2:] * Kp[[0, 1], [0, 1]] + Kp[:2, 2]
    pair = {'index': 1, 's': 1.0, 'g': 0.0, 'inc': inc0, 'out': inc, 'T_f': inc, 'T_b': inc, 'ok_f': True, 'ok_b': True,
            'ratio_f': 1.0, 'ratio_b': 1.0, 'vf': np.ones(len(zf), bool), 'vb': np.ones(len(zb), bool), 'zf': zf, 'zb': zb}
    return {'room': 'synthetic', 'frames': [0, 1], 'P': P, 'pts': pts, 'pairs': [pair],
            'corr': [(1, 0, pa, za, pb)], 'K': Kp}


def _args(**kw):
    a = {'iters': 10, 'huber': 0.03, 'tol': 0.0, 'w_bi': 1.0, 'w_ph': 1.0, 'w_r1': 1.0, 'w_rk': 1.0, 'gaps': [1],
         'c_px': 0.5, 'c_bi': 0.5, 'bi_min_mm': 5.0, 'min_tracks': 10}
    a.update(kw)
    return Namespace(**a)


def test_window_pass_gradient_reaches_the_net():
    """run_window refines both directions with the net and sums R1 + B + P; the loss back-propagates through the
    unrolled LM into every PhotoNet layer, and a head weight's autograd gradient matches central finite differences
    of the whole pass (float64)."""
    torch.manual_seed(0)
    smp, a = _sample(), _args()
    net = pl.PhotoNet('fw', 4).double().to(DEV)
    with torch.no_grad():
        net.head.weight.normal_(0, 0.05)
    terms, st = tr.run_window(net, smp, a, DEV, torch.float64)
    assert set(terms) == {'R1', 'B', 'P'} and st['n_acc'] == 2
    loss = tr.total(terms, a)
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    g = net.head.weight.grad.clone()
    k = tuple(int(x) for x in np.unravel_index(int(g.abs().argmax()), g.shape))
    eps = 1e-5
    vals = []
    with torch.no_grad():
        for s in (1, -1):
            net.head.weight[k] += s * eps
            vals.append(float(tr.total(tr.run_window(net, smp, a, DEV, torch.float64)[0], a)))
            net.head.weight[k] -= s * eps
    fd = (vals[0] - vals[1]) / (2 * eps)
    assert abs(fd - float(g[k])) < 2e-3 * abs(float(g[k])) + 1e-9, (fd, float(g[k]))


def test_window_pass_untrained_matches_plain_refinement():
    """With a zero-initialised weights net the window's refined poses are those of plain refine_torch (float64)."""
    smp, a = _sample(), _args(w_bi=0.0, w_ph=0.0)
    pr = smp['pairs'][0]
    terms, _ = tr.run_window(pl.PhotoNet('w', 4).double().to(DEV), smp, a, DEV, torch.float64)
    A0 = np.linalg.inv(pr['inc'])
    S = [[torch.from_numpy(p).to(DEV).double()] for p in smp['P']]
    R, t, _ = kpt.refine_torch(S[0], S[1], 1.0, torch.from_numpy(smp['pts'][0]).to(DEV),
                               torch.from_numpy(pr['zf']).to(DEV), smp['K'], A0[:3, :3], A0[:3, 3], 1, 10, 0.03, 0.0)
    Tf = np.linalg.inv(np.r_[np.c_[R.numpy(), t.numpy()], [[0, 0, 0, 1]]])
    pr2 = dict(pr, T_f=Tf)
    _, st2 = tr.run_window(pl.PhotoNet('w', 4).double().to(DEV), dict(smp, pairs=[pr2]), a, DEV, torch.float64)
    assert st2['dt_f_mm'][0] < 1e-9 and set(terms) == {'R1'} and float(terms['R1'].detach()) < 0.05


def test_window_pass_combines_only_the_accepted_directions():
    """As the tracker: midpoint of both accepted directions, the accepted one alone otherwise, and a pair with neither
    accepted gives no loss term (R1, B and P all skipped)."""
    smp, a = _sample(), _args()
    net = pl.PhotoNet('w', 4).double().to(DEV)
    pr = smp['pairs'][0]
    S = [[torch.from_numpy(p).to(DEV).double()] for p in smp['P']]
    pts = [torch.from_numpy(x).to(DEV) for x in smp['pts']]
    A0 = np.linalg.inv(pr['inc'])
    Rf, tf, _ = kpt.refine_torch(S[0], S[1], 1.0, pts[0], torch.from_numpy(pr['zf']).to(DEV), smp['K'], A0[:3, :3],
                                 A0[:3, 3], 1, 10, 0.03, 0.0, torch.ones(len(pts[0]), dtype=torch.float64, device=DEV))
    Rb, tb, _ = kpt.refine_torch(S[1], S[0], 1.0, pts[1], torch.from_numpy(pr['zb']).to(DEV), smp['K'],
                                 pr['inc'][:3, :3], pr['inc'][:3, 3], 1, 10, 0.03, 0.0,
                                 torch.ones(len(pts[1]), dtype=torch.float64, device=DEV))
    Tf, Tb = pl.inv_T(pl.to_T(Rf, tf)).cpu(), pl.to_T(Rb, tb).cpu()
    assert not torch.equal(Tf, Tb)                                   # the three cases below are distinct poses
    pa, za, pb = smp['corr'][0][2:]
    K = smp['K']
    for ok_f, ok_b, T in ((True, True, pl.pose_mid(Tf, Tb)), (True, False, Tf), (False, True, Tb)):
        terms, _ = tr.run_window(net, dict(smp, pairs=[dict(pr, ok_f=ok_f, ok_b=ok_b)]), a, DEV, torch.float64)
        expect = pl.reproj_loss(pl.inv_T(T), torch.from_numpy(pa), torch.from_numpy(za), torch.from_numpy(pb), K, a.c_px)
        assert set(terms) == {'R1', 'B', 'P'}
        assert float(terms['R1'].detach()) == pytest.approx(float(expect), rel=1e-9)
    terms, st = tr.run_window(net, dict(smp, pairs=[dict(pr, ok_f=False, ok_b=False)]), a, DEV, torch.float64)
    assert terms == {} and len(st['dt_f_mm']) == 1                    # diagnostics still recorded


def test_dump_option_record_round_trip(tmp_path, monkeypatch):
    """s2_dump_pairs records the tracker's refinement options next to the pairs; the trainer accepts a dump made
    with its own options, rejects other ones, warns on an old dump without the record; no pairs is a clear error."""
    from types import SimpleNamespace

    import scripts.s2_dump_pairs as dp
    a = _args(cell=1, grad=0.002, tol=0.01)
    exp = tr.dump_options(a)
    odo = SimpleNamespace(**exp, _prev_frame=SimpleNamespace(index=4), g=0.1, _photo_s=1.5)
    monkeypatch.setattr(dp, '_refine0', lambda *args: None)
    monkeypatch.setattr(dp.mod, 'main', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['s2_dump_pairs.py', '--scene', 'roomX', '--out', str(tmp_path)])
    monkeypatch.setattr(dp, 'REC', [])
    monkeypatch.setattr(dp, 'OPT', {})
    with pytest.raises(SystemExit, match='no pair'):
        dp.main()
    assert not (tmp_path / 'roomX.s2pairs.npz').exists()

    def fake_refine0(self, frame, gray, inc, head_prev):
        for T in (inc, inc):
            dp._ok(self, {'cost0': 2.0, 'cost': 1.0, 'n': 60}, T, inc)
        return inc
    monkeypatch.setattr(dp, '_refine0', fake_refine0)
    monkeypatch.setattr(dp, '_ok0', lambda self, info, T, inc: True)
    dp._refine(odo, SimpleNamespace(index=5), None, np.eye(4), None)
    dp.main()
    d = dict(np.load(tmp_path / 'roomX.s2pairs.npz'))
    assert d['index'].tolist() == [5] and d['ok_f'].tolist() == [True] and 'T_b' in d
    assert {f'opt_{k}' for k in dp.OPT_KEYS} <= set(d) and set(exp) == set(dp.OPT_KEYS)
    tr.check_dump_options(d, exp, 'roomX')
    with pytest.raises(ValueError, match='photo_levels'):
        tr.check_dump_options(dict(d, opt_photo_levels=np.array(2)), exp, 'roomX')
    with pytest.raises(ValueError, match='photo_huber'):
        tr.check_dump_options(d, dict(exp, photo_huber=0.05), 'roomX')
    with pytest.warns(UserWarning, match='no tracker-option record'):
        tr.check_dump_options({k: v for k, v in d.items() if not k.startswith('opt_')}, exp, 'roomX')


def test_correspondences_intersect_track_ids():
    fr = np.array([3, 3, 3, 4, 4, 5])
    gid = np.array([7, 8, 9, 9, 7, 7])
    px = np.arange(12.0).reshape(6, 2)
    pa, pb = tr.correspondences(fr, gid, px, 3, 4)
    assert pa.tolist() == [[0.0, 1.0], [4.0, 5.0]] and pb.tolist() == [[8.0, 9.0], [6.0, 7.0]]
    pa, pb = tr.correspondences(fr, gid, px, 3, 5)
    assert pa.tolist() == [[0.0, 1.0]] and pb.tolist() == [[10.0, 11.0]]


def test_tracker_photo_feat_off_by_default_and_untrained_identical(scene, tmp_path):  # noqa: F811
    """photo_feat is off by default; an untrained weights net in the tracker gives the photo_backend=torch output bit
    for bit; photo_feat with the numpy backend is a ValueError."""
    frames, _ = scene

    def run(**o):
        odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150,
                                prior='oracle', photo='all', photo_cell=2, photo_grad=0.002, photo_bidir=True,
                                photo_backend='torch', photo_levels=1, **o)
        odo.photo_norm = _UnitNorm()
        odo.reset(PinholeCamera(K, W, H))
        est = [odo.step(f) for f in frames[:12]][1:]
        return np.stack([e.rel_increment for e in est]), np.array(odo.diag['photo'])

    assert KLTWindowOdometry(prior='oracle').photo_feat == ''
    ref, ph = run()
    assert (ph == 1).sum() >= 5
    out, ph_l = run(photo_feat=str(_ckpt(tmp_path, 'w')))
    assert np.array_equal(out, ref) and np.array_equal(ph_l, ph)
    with pytest.raises(ValueError, match='photo_feat'):
        KLTWindowOdometry(prior='oracle', photo_feat='x.pt')
