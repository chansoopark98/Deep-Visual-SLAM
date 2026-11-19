"""dvslam.ssl.losses on a synthetic three-view + stereo plane scene (CPU, 64x64, no downloads).

The scene: frames t-1, t, t+1 of cam0 and cam1 at t (X_cam1 = T_stereo X_cam0, baseline 0.101 m) rendered
analytically. A fake model returns the true depth / poses (optionally scaled) so each photometric term can be
compared at the truth and at perturbations; a tiny real SSLModel checks that weight 0 skips a term's forward passes.
"""
import numpy as np
import pytest
import torch
from test_ssl_geometry import K_NP, H, W, pose, render_plane

from dvslam.geometry.se3 import relative_pose
from dvslam.ssl import geometry, losses, models
from dvslam.ssl.losses import SSLLoss
from dvslam.ssl.models import SSLModel

MIN_D, MAX_D = 0.1, 20.0
ONLY = {k: 0.0 for k in losses.TERMS}
T_STEREO = pose([0, 0, 0], [-0.101, 0, 0])            # cam1 0.101 m to the right of cam0
T_WC = [pose([0.01, -0.03, 0.0], [-0.10, 0.02, -0.04]), pose([0, 0, 0], [0, 0, 0]),
        pose([-0.02, 0.03, 0.01], [0.10, -0.02, 0.05])]


def t32(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@pytest.fixture(scope='module')
def scene():
    views = [render_plane(T) for T in T_WC] + [render_plane(T_WC[1] @ np.linalg.inv(T_STEREO))]
    img = t32(np.stack([v[0] for v in views]))                    # (4, H, W)
    rel = {(0, 1): relative_pose(T_WC[0], T_WC[1]), (1, 2): relative_pose(T_WC[1], T_WC[2]),
           (1, 0): relative_pose(T_WC[1], T_WC[0]), (2, 1): relative_pose(T_WC[2], T_WC[1]),
           (0, 2): relative_pose(T_WC[0], T_WC[2])}
    batch = {'net': img[:3, None].expand(3, 3, H, W)[None].contiguous(), 'pho': img[None, :, None].clone(),
             'valid': torch.ones(1, 4, 1, H, W), 'K': t32(K_NP)[None], 'T_stereo': t32(T_STEREO)[None],
             'R_gyro': t32(np.stack([rel[(0, 1)][:3, :3], np.linalg.inv(rel[(1, 2)])[:3, :3]]))[None],
             'gyro_ok': torch.tensor([True])}
    depth = {i: t32(views[i][1])[None, None] for i in range(3)}
    return batch, depth, rel


class FakeModel:
    """Model interface of SSLModel returning the scene truth; depth x depth_scale, translations x t_scale."""
    pose_iters = 1
    uncertainty = False

    def __init__(self, net: torch.Tensor, depth: dict, rel: dict, depth_scale=1.0, t_scale=1.0):
        self.frame = {net[:, i].data_ptr(): i for i in range(3)}
        self.depth_, self.rel, self.ds, self.ts = depth, rel, depth_scale, t_scale

    def disp_to_depth(self, disp):
        return 1 / (1 / MAX_D + (1 / MIN_D - 1 / MAX_D) * disp)

    def depth(self, img):
        d = self.depth_[self.frame[img.data_ptr()]] * self.ds
        return [(1 / d - 1 / MAX_D) / (1 / MIN_D - 1 / MAX_D)] * 4, None

    def depth_m(self, img):
        return self.disp_to_depth(self.depth(img)[0][0])

    def pose_net(self, a, b):
        T = self.rel[(self.frame[a.data_ptr()], self.frame[b.data_ptr()])].copy()
        T[:3, 3] *= self.ts
        return t32(T)[None]

    def pose(self, a, b, K, depth_b=None):
        T = self.pose_net(a, b)
        return T, T


def loss_of(name: str, scene, **fake) -> float:
    batch, depth, rel = scene
    torch.manual_seed(0)
    total, terms = SSLLoss({**ONLY, name: 1.0})(FakeModel(batch['net'], depth, rel, **fake), batch)
    assert set(terms) == {name}
    return float(total)


def test_temporal_minimal_at_true_pose_and_depth(scene):
    true = loss_of('temporal', scene)
    assert true < 0.01
    for kw in ({'t_scale': 0.9}, {'t_scale': 1.1}, {'depth_scale': 0.9}, {'depth_scale': 1.1}):
        assert loss_of('temporal', scene, **kw) > true * 1.5, kw


def test_stereo_minimum_is_at_the_true_metric_depth_scale(scene):
    scales = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25]
    vals = [loss_of('stereo', scene, depth_scale=s) for s in scales]
    assert int(np.argmin(vals)) == scales.index(1.0), vals
    assert loss_of('stereo', scene, t_scale=1.3) == pytest.approx(vals[3])      # the temporal pose plays no part


def test_geometric_and_flow_zero_for_consistent_truth(scene):
    true = loss_of('geometric', scene)
    assert true < 1e-4
    assert loss_of('geometric', scene, t_scale=1.2) > 10 * true
    assert loss_of('geometric', scene, depth_scale=1.1) > 10 * true      # target and source depths both scaled
    batch, depth, rel = scene

    class Teacher(torch.nn.Module):                     # flow target -> source from the true geometry
        def forward(self, tgt, src):
            i = 0 if src.data_ptr() == batch['net'][:, 0].data_ptr() else 2
            T = rel[(0, 1)] if i == 0 else np.linalg.inv(rel[(1, 2)])
            return geometry.rigid_flow(geometry.reproject(depth[1], t32(T)[None], batch['K'])[0])

    fake = FakeModel(batch['net'], depth, rel)
    total, _ = SSLLoss({**ONLY, 'flow': 1.0}, teacher=Teacher())(fake, batch)
    assert float(total) < 1e-6
    fake.ts = 1.2
    assert float(SSLLoss({**ONLY, 'flow': 1.0}, teacher=Teacher())(fake, batch)[0]) > 1e-3


def test_saturated_region_changes_nothing(scene):
    batch, depth, rel = scene
    torch.manual_seed(0)
    base = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    for img in (1, 0, 3):                      # target, temporal source, stereo source
        base['valid'][0, img, 0, 20:30, 25:40] = 0
    weights = {**ONLY, 'temporal': 1.0, 'stereo': 1.0}
    sat = {**base, 'pho': base['pho'].clone()}
    for img in (1, 0, 3):
        sat['pho'][0, img, 0, 20:30, 25:40] = 1.7          # garbage where valid = 0
    out = []
    for b in (base, sat):
        torch.manual_seed(0)
        out.append(SSLLoss(weights)(FakeModel(b['net'], depth, rel), b)[1])
    for k in ('temporal', 'stereo'):
        assert float(out[0][k]) == pytest.approx(float(out[1][k]), abs=1e-7), k


def test_gyro_and_pose_consistency_terms():
    R = geometry.pose_vec_to_mat(torch.tensor([[0.1, -0.2, 0.05], [0.0, 0.3, 0.0]]), torch.zeros(2, 3))[:, :3, :3]
    R_gyro = torch.stack([R, R], 1)[None].reshape(2, 2, 3, 3)
    ok = torch.tensor([True, True])
    assert float(losses.gyro_term(R_gyro, R_gyro, ok)) < 1e-5
    delta = geometry.pose_vec_to_mat(torch.tensor([[0.0, 0.0, 0.04]]), torch.zeros(1, 3))[:, :3, :3]
    R_pred = R_gyro @ delta[None]                       # R_pred^T R_gyro has angle 0.04
    assert float(losses.gyro_term(R_pred, R_gyro, ok)) == pytest.approx(0.04, abs=1e-5)
    bad = R_pred.clone()
    bad[1] = R_gyro[1] @ geometry.pose_vec_to_mat(torch.tensor([[0.5, 0, 0]]), torch.zeros(1, 3))[:, :3, :3]
    assert float(losses.gyro_term(bad, R_gyro, torch.tensor([True, False]))) == pytest.approx(0.04, abs=1e-5)

    T = geometry.pose_vec_to_mat(torch.tensor([[0.05, -0.02, 0.03]]), torch.tensor([[0.04, 0.01, -0.02]]))
    assert float(losses.pose_consistency_term(T, geometry.se3_inverse(T))) < 1e-4
    T2 = geometry.pose_vec_to_mat(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.03]]))
    assert float(losses.pose_consistency_term(T2, torch.eye(4)[None])) == pytest.approx(0.01 / 0.05, rel=1e-4)


def test_photometric_uncertainty_and_masked_mean():
    a = torch.rand(1, 1, 16, 16)
    assert float(losses.photometric(a, a).max()) < 1e-6
    r, sigma = torch.full((1, 1, 4, 4), 0.2), torch.full((1, 1, 4, 4), 0.5)
    assert torch.allclose(losses.with_uncertainty(r, sigma), r / 0.5 + np.log(0.5))
    m = torch.zeros(1, 1, 4, 4)
    m[..., :2, :] = 1
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    assert float(losses.masked_mean(x, m)) == pytest.approx(3.5)
    smooth = losses.smooth_term(torch.ones(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    assert float(smooth) == 0.0


@pytest.fixture(scope='module')
def tiny():
    torch.manual_seed(0)
    b = 1
    batch = {'net': torch.rand(b, 3, 3, H, W), 'pho': torch.rand(b, 4, 1, H, W), 'valid': torch.ones(b, 4, 1, H, W),
             'K': t32(K_NP)[None], 'T_stereo': t32(T_STEREO)[None], 'R_gyro': torch.eye(3).expand(b, 2, 3, 3),
             'gyro_ok': torch.ones(b, dtype=torch.bool)}
    return batch


def count_calls(model: SSLModel) -> dict:
    n = {'depth': 0, 'pose_net': 0, 'residual': 0}
    depth, pose_fwd = model.depth, model.pose_net.forward
    res_fwd = model.residual_net.forward if model.residual_net is not None else None

    def wrap(key, fn):
        def f(*a):
            n[key] += 1
            return fn(*a)
        return f

    model.depth = wrap('depth', depth)
    model.pose_net.forward = wrap('pose_net', pose_fwd)
    if res_fwd is not None:
        model.residual_net.forward = wrap('residual', res_fwd)
    return n


@pytest.mark.parametrize('weights, pose_iters, expect', [
    ({'stereo': 1.0, 'smooth': 1e-3}, 2, {'depth': 1, 'pose_net': 0, 'residual': 0}),
    ({'temporal': 1.0}, 1, {'depth': 1, 'pose_net': 2, 'residual': 0}),
    ({'temporal': 1.0}, 2, {'depth': 2, 'pose_net': 2, 'residual': 2}),     # t+1 depth (no grad) for refinement
    ({'temporal': 1.0, 'geometric': 0.5}, 2, {'depth': 3, 'pose_net': 2, 'residual': 2}),
    ({'gyro': 5.0, 'pose_consistency': 0.05}, 1, {'depth': 1, 'pose_net': 4, 'residual': 0}),
])
def test_zero_weight_skips_term_and_forward_passes(tiny, weights, pose_iters, expect):
    model = SSLModel(MIN_D, MAX_D, True, 0.01, pose_iters, pretrained=False)
    n = count_calls(model)
    loss = SSLLoss({**ONLY, **weights})
    assert loss.teacher is None                          # flow weight 0: RAFT is never built
    total, terms = loss(model, tiny)
    assert set(terms) == set(weights) and torch.isfinite(total)
    assert n == expect
    total.backward()


def test_gap2_terms_zero_at_truth_and_sensitive_to_the_gap2_pose(scene):
    batch, depth, rel = scene

    class Teacher(torch.nn.Module):                     # true flow target -> source for the three (tgt, src) pairs
        def forward(self, tgt, src):
            frame = {batch['net'][:, i].data_ptr(): i for i in range(3)}
            T = rel[(frame[src.data_ptr()], frame[tgt.data_ptr()])] if (frame[src.data_ptr()], frame[
                tgt.data_ptr()]) in rel else np.linalg.inv(rel[(frame[tgt.data_ptr()], frame[src.data_ptr()])])
            return geometry.rigid_flow(geometry.reproject(depth[frame[tgt.data_ptr()]], t32(T)[None],
                                                          batch['K'])[0])

    weights = {**ONLY, 'temporal': 1.0, 'flow': 1.0, 'gyro': 1.0, 'pose_consistency': 1.0, 'gap2': 0.5}
    gap2 = ('gap2_comp', 'gap2_photo', 'gap2_flow', 'gap2_gyro')

    def run(fake):
        torch.manual_seed(0)
        loss = SSLLoss(weights, teacher=Teacher())
        total, terms = loss(fake, batch)
        assert all(loss.w[k] == 0.5 * loss.w[b] for k, b in losses.GAP2_TERMS.items())
        assert total == pytest.approx(sum(float(loss.w[k] * v) for k, v in terms.items()), rel=1e-5)
        return {k: float(terms[k]) for k in gap2}

    truth = run(FakeModel(batch['net'], depth, rel))
    assert truth['gap2_comp'] < 1e-4 and truth['gap2_flow'] < 1e-5 and truth['gap2_gyro'] < 1e-4, truth
    assert truth['gap2_photo'] < 0.01, truth
    wrong = {**rel, (0, 2): rel[(0, 2)] @ pose([0.0, 0.02, 0.0], [0.02, 0.0, 0.0])}     # only the gap-2 pose is off
    bad = run(FakeModel(batch['net'], depth, wrong))
    assert bad['gap2_comp'] > 0.1 and bad['gap2_flow'] > 1e-2 and bad['gap2_gyro'] > 0.01, bad
    assert bad['gap2_photo'] > 1.5 * truth['gap2_photo'], (bad, truth)


def test_gap2_off_by_default_and_its_forward_passes(tiny):
    for pose_iters, expect in ((2, {'depth': 2, 'pose_net': 3, 'residual': 3}),
                               (1, {'depth': 2, 'pose_net': 3, 'residual': 0})):     # t+1 depth without grad
        model = SSLModel(MIN_D, MAX_D, False, 0.01, pose_iters, pretrained=False)
        n = count_calls(model)
        total, terms = SSLLoss({**ONLY, 'temporal': 1.0, 'gap2': 1.0})(model, tiny)
        assert set(terms) == {'temporal', 'gap2_photo'} and n == expect
        total.backward()
    assert set(SSLLoss({**ONLY, 'temporal': 1.0})(SSLModel(MIN_D, MAX_D, False, 0.01, 1, False), tiny)[1]) == {
        'temporal'}


def test_zero_motion_term_and_static_copy(tiny):
    T = torch.eye(4).repeat(2, 1, 1)
    assert float(losses.zero_motion_term(T)) == 0.0
    T[0, 0, 3] = 0.005                                   # 5 mm on one of 6 translation components
    assert float(losses.zero_motion_term(T)) == pytest.approx(0.5 / 3, rel=1e-5)
    torch.manual_seed(0)
    img = tiny['net'][:, 1, :1].expand(-1, 3, -1, -1)    # gray replicated, like the dataset's 'net' images
    copy = losses.static_copy(img)
    assert copy.shape == img.shape and 0 <= float(copy.min()) and float(copy.max()) <= 1
    assert 0.003 < float((copy - img).abs().mean()) < 0.1
    assert torch.equal(copy[:, 0], copy[:, 2])           # still a gray image replicated to 3 channels

    model = SSLModel(MIN_D, MAX_D, False, 0.01, 2, pretrained=False)
    n = count_calls(model)
    total, terms = SSLLoss({**ONLY, 'zero_motion': 0.01})(model, tiny)
    assert set(terms) == {'zero_motion'} and n == {'depth': 1, 'pose_net': 1, 'residual': 1}
    assert float(total) == pytest.approx(0.01 * float(terms['zero_motion']))
    total.backward()
    assert model.pose_net.head[-1].weight.grad.abs().sum() > 0


def test_stereo_hint_gated_log_depth(scene):
    """True hint: zero at the true depth, log(1.1) when the network depth is 10 % long (the hint wins everywhere).
    A wrong hint (x 1.5) loses the reprojection comparison to the true network depth, so it is (mostly) ignored."""
    batch, depth, rel = scene

    def run(hint, **fake):
        b = {**batch, 'hint': hint}
        torch.manual_seed(0)
        total, terms = SSLLoss({**ONLY, 'stereo_hint': 2.0})(FakeModel(batch['net'], depth, rel, **fake), b)
        assert set(terms) == {'stereo_hint'} and float(total) == pytest.approx(2 * float(terms['stereo_hint']))
        return float(terms['stereo_hint'])

    true = depth[1].clone()
    assert run(true) < 1e-5
    assert run(true, depth_scale=1.1) == pytest.approx(np.log(1.1), rel=0.05)
    assert run(true * 1.5) < 0.1 * np.log(1.5)
    holes = true.clone()
    holes[..., :, : W // 2] = 0                          # no hint on the left half: those pixels never count
    assert run(holes, depth_scale=1.1) == pytest.approx(np.log(1.1), rel=0.05)
    assert run(torch.zeros_like(true), depth_scale=1.1) == 0.0


def test_stereo_hint_trains_depth_only(tiny):
    model = SSLModel(MIN_D, MAX_D, False, 0.01, 2, pretrained=False)
    n = count_calls(model)
    b = {**tiny, 'hint': torch.full((1, 1, H, W), 2.0)}
    total, terms = SSLLoss({**ONLY, 'stereo_hint': 1.0})(model, b)
    assert set(terms) == {'stereo_hint'} and n == {'depth': 1, 'pose_net': 0, 'residual': 0}
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.depth_net.parameters())
    assert all(p.grad is None for p in model.pose_net.parameters())


def test_amp_covers_the_networks_only(tiny):
    """bfloat16 autocast on CPU: convolutions run in low precision, every output, pose composition and loss is float32,
    and autocast is never left enabled outside a network forward."""
    model = SSLModel(MIN_D, MAX_D, False, 0.01, 2, pretrained=False)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    model.set_amp(torch.bfloat16)
    seen = {'conv': set(), 'leak': []}
    convs = [model.depth_net.encoder.encoder.conv1, model.pose_net.encoder.encoder.conv1,
             model.residual_net.encoder.encoder.conv1]
    for c in convs:
        c.register_forward_hook(lambda m, i, o: seen['conv'].add(o.dtype))
    for net in (model.pose_net, model.residual_net):     # autocast is entered inside PoseNet.forward
        net.register_forward_hook(lambda m, i, o: seen['leak'].append(torch.is_autocast_enabled('cpu')))
    depth = model.depth

    def depth_hook(img):                                 # ... and inside SSLModel.depth around depth_net
        out = depth(img)
        seen['leak'].append(torch.is_autocast_enabled('cpu'))
        return out
    model.depth = depth_hook

    class Teacher(torch.nn.Module):
        amp_dtype = None

        def forward(self, tgt, src):
            with models.autocast(tgt, self.amp_dtype):
                out = torch.nn.functional.conv2d(tgt, torch.zeros(2, 3, 1, 1))
            return out.float()

    teacher = Teacher()
    teacher.amp_dtype = torch.bfloat16
    weights = {**{k: 1.0 for k in losses.TERMS}, 'gap2': 1.0, 'zero_motion': 0.01}
    total, terms = SSLLoss(weights, teacher=teacher)(model, tiny)
    assert seen['conv'] == {torch.bfloat16} and not any(seen['leak'])
    assert total.dtype == torch.float32 and all(v.dtype == torch.float32 for v in terms.values()), terms
    disps, _ = model.depth(tiny['net'][:, 1])
    T, T0 = model.pose(tiny['net'][:, 0], tiny['net'][:, 1], tiny['K'])
    assert all(d.dtype == torch.float32 for d in disps) and T.dtype == T0.dtype == torch.float32
    assert not torch.is_autocast_enabled('cpu')
    total.backward()
    assert all(p.grad.dtype == torch.float32 for p in model.parameters() if p.grad is not None)
    assert set(model.state_dict()) == set(ref)          # no new state: checkpoints are unchanged in layout
    model.set_amp(None)
    seen['conv'].clear()
    model.depth(tiny['net'][:, 1])
    assert seen['conv'] == {torch.float32}


def test_flow_teacher_called_only_with_positive_weight(tiny):
    calls = []

    class Teacher(torch.nn.Module):
        def forward(self, tgt, src):
            calls.append(1)
            return torch.zeros(tgt.shape[0], 2, *tgt.shape[-2:])

    model = SSLModel(MIN_D, MAX_D, False, 0.01, 1, pretrained=False)
    total, terms = SSLLoss({**ONLY, 'flow': 0.02}, teacher=Teacher())(model, tiny)
    assert set(terms) == {'flow'} and len(calls) == 2 and torch.isfinite(total)
