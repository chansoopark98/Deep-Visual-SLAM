"""dvslam.ssl.adapter (plan v3.4 S3): LoRA adapter on a frozen SSL depth net, frozen scale anchor, fixed pseudo-label
poses and the calibration loss (CPU, 64x64 synthetic plane scene from test_ssl_geometry, no downloads)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ssl_geometry import K_NP, H, W, pose, render_plane

from dvslam.geometry.se3 import relative_pose
from dvslam.ssl import adapter as ada
from dvslam.ssl.models import SSLModel
from dvslam.ssl.tumvi_data import photometric

T_WC = [pose([0.01, -0.03, 0.0], [-0.10, 0.02, -0.04]), pose([0, 0, 0], [0, 0, 0]),
        pose([-0.02, 0.03, 0.01], [0.10, -0.02, 0.05])]


def tiny_model() -> SSLModel:
    torch.manual_seed(0)
    return SSLModel(0.1, 20.0, False, 0.01, 1, pretrained=False)


def base_state(net: ada.AdaptedDepth) -> dict[str, torch.Tensor]:
    """Every parameter and buffer that is not a low-rank branch (BatchNorm statistics included)."""
    return {k: v.detach().clone() for k, v in net.state_dict().items()
            if not k.endswith(('.down.weight', '.up.weight'))}


def test_param_count_is_a_small_fraction_and_matches_the_formula():
    net = ada.AdaptedDepth(tiny_model(), rank=8, alpha=8.0)
    n_ada, n_base = ada.count_params(net)
    convs = [m for m in net.modules() if isinstance(m, ada.LoRAConv2d)]
    assert len(convs) == net.n_wrapped == 14                       # 10 decoder upconvs + 4 dispconvs
    expected = sum(8 * (c.base.in_channels * c.base.kernel_size[0] * c.base.kernel_size[1] + c.base.out_channels)
                   for c in convs)
    assert n_ada == expected
    assert 0 < n_ada / n_base < 0.02                             # depth + pose nets of the checkpoint as the base
    assert {n for n, p in net.named_parameters() if p.requires_grad} == {
        n for n, _ in net.named_parameters() if n.endswith(('.down.weight', '.up.weight'))}


def test_fresh_adapter_reproduces_the_frozen_depth_exactly():
    model = tiny_model().eval()
    img = torch.rand(2, 3, H, W)
    with torch.no_grad():
        ref = model.depth_m(img)
    net = ada.AdaptedDepth(model, rank=4, alpha=4.0)
    assert torch.equal(net.depth(img), ref)
    assert torch.equal(net.frozen_depth(img), ref)


def scene_batch():
    views = [render_plane(T) for T in T_WC]
    img = torch.as_tensor(np.stack([v[0] for v in views]))[:, None]           # (3, 1, H, W): t-1, t, t+1
    T_src = torch.as_tensor(np.stack([relative_pose(T_WC[0], T_WC[1]), relative_pose(T_WC[2], T_WC[1])]))[None]
    return img, T_src


def test_calibration_changes_only_the_adapter():
    net = ada.AdaptedDepth(tiny_model(), rank=4, alpha=4.0)
    before = base_state(net)
    img, T_src = scene_batch()
    params = [p for p in net.parameters() if p.requires_grad]
    up0 = [m.up.weight.detach().clone() for m in net.modules() if isinstance(m, ada.LoRAConv2d)]
    opt = torch.optim.Adam(params, lr=1e-2)
    ones = torch.ones(1, 1, H, W)
    for _ in range(3):
        net.train()                                                # stays in eval mode (BatchNorm frozen)
        total, terms = ada.calib_loss(net, img[1:2].expand(1, 3, H, W), img[1:2], ones, [img[0:1], img[2:3]],
                                      [ones, ones], [img[0:1].expand(1, 3, H, W), img[2:3].expand(1, 3, H, W)],
                                      T_src, torch.ones(1, 2, dtype=torch.bool), torch.as_tensor(K_NP).float()[None],
                                      w_geo=0.5)
        opt.zero_grad()
        total.backward()
        opt.step()
        assert torch.isfinite(total) and {'temporal', 'smooth', 'geometric', 'anchor_c2'} <= set(terms)
    assert not net.training and not net.model.training
    after = base_state(net)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    up1 = [m.up.weight for m in net.modules() if isinstance(m, ada.LoRAConv2d)]
    assert any(not torch.equal(a, b) for a, b in zip(up0, up1))


def test_scale_anchor_removes_scale_and_keeps_shape():
    torch.manual_seed(1)
    d_f = 1.0 + torch.rand(3, 1, 16, 16) * 2
    shape = torch.exp(0.1 * torch.randn(3, 1, 16, 16))
    shape = shape / shape.flatten(1).median(1).values[:, None, None, None]   # pure shape change: median 1
    pure = d_f * shape
    assert ada.scale_anchor_loss(pure, d_f) < 1e-12
    scaled = pure * torch.tensor([1.3, 0.8, 1.0])[:, None, None, None]
    c = ada.median_log_ratio(scaled, d_f)
    assert torch.allclose(c, torch.log(torch.tensor([1.3, 0.8, 1.0])), atol=1e-6)
    assert torch.isclose(ada.scale_anchor_loss(scaled, d_f), (c ** 2).mean())
    anchored = scaled * ada.anchor_scale(scaled, d_f)
    assert ada.median_log_ratio(anchored, d_f).abs().max() < 1e-6
    assert torch.allclose(anchored, pure, rtol=1e-5)                       # shape kept, scale back to frozen


def test_anchored_adapter_output_has_the_frozen_median_scale():
    net = ada.AdaptedDepth(tiny_model(), rank=4, alpha=4.0)
    for m in net.modules():
        if isinstance(m, ada.LoRAConv2d):
            torch.nn.init.normal_(m.up.weight, std=0.5)
    img = torch.rand(2, 3, H, W)
    d = net.depth(img)
    d_f = net.frozen_depth(img)
    with torch.no_grad():
        raw = net.model.depth_m(img)
    assert ada.median_log_ratio(raw, d_f).abs().max() > 1e-3            # the adapter does change the raw scale
    assert ada.median_log_ratio(d, d_f).abs().max() < 1e-6


def test_pseudo_label_poses_compose_and_respect_untracked_pairs():
    rng = np.random.default_rng(0)
    n = 8
    inc = np.tile(np.eye(4), (n, 1, 1))
    for k in range(1, n):
        inc[k] = pose(rng.normal(0, 0.02, 3), rng.normal(0, 0.03, 3))
    ok = np.ones(n, bool)
    ok[5] = False                                                     # pair (4, 5) not tracked
    C, bad = ada.pose_chain(torch.from_numpy(inc), torch.from_numpy(ok))
    T, use = ada.source_poses(C, bad, torch.tensor([2, 4, 6, 0]), (-1, 2))
    np.testing.assert_allclose(T[0, 0].numpy(), inc[2], atol=1e-12)                       # T_{1<-2}
    np.testing.assert_allclose(T[0, 1].numpy(), np.linalg.inv(inc[3] @ inc[4]), atol=1e-12)  # T_{4<-2}
    assert use[0].tolist() == [True, True]
    assert use[1].tolist() == [True, False]                          # 4 -> 6 crosses the untracked pair
    assert use[2].tolist() == [True, False]                          # 6 -> 5 uses only pair (5, 6); 8 outside
    assert use[3].tolist() == [False, True]                          # -1 outside


def test_loss_images_match_the_training_photometry():
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (2, H, W), dtype=np.uint8)
    V = rng.uniform(0.1, 1.0, (H, W)).astype(np.float32)
    e, e_t = np.array([4.0, 5.0]), np.array([5.0, 5.0])
    pho, valid = ada.loss_images(torch.from_numpy(gray), torch.from_numpy(V), torch.from_numpy(e),
                                 torch.from_numpy(e_t))
    for i in range(2):
        p_ref, v_ref = photometric(gray[i], V, e_t[i] / e[i], 0.25, 250)
        np.testing.assert_allclose(pho[i, 0].numpy(), p_ref, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(valid[i, 0].numpy(), v_ref)


def test_anchored_depth_has_no_gradient_along_the_scale_direction():
    torch.manual_seed(3)
    d_f = 1.0 + torch.rand(2, 1, 16, 16)
    d_a = (d_f * torch.exp(0.1 * torch.randn(2, 1, 16, 16))).requires_grad_(True)
    lam = torch.ones(2, 1, 1, 1, requires_grad=True)
    raw = d_a * lam
    anchored = raw * ada.anchor_scale(raw, d_f)
    ((anchored - 2.0) ** 2).sum().backward()          # any loss of the anchored depth
    assert lam.grad.abs().max() < 1e-4                # rescaling the raw output cannot lower it
    assert d_a.grad.abs().max() > 1e-2                # the shape still gets a gradient


def test_no_target_is_an_error_and_leaves_the_model_trainable():
    model = tiny_model()
    n0 = sum(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match='no nn.Conv2d'):
        ada.add_lora(model, 4, 4.0, ('no.such.prefix',))
    assert sum(p.requires_grad for p in model.parameters()) == n0 > 0
    assert not any(isinstance(m, ada.LoRAConv2d) for m in model.modules())


class _Shared(torch.nn.Module):
    """One conv registered under two names (a weight-tied decoder), plus an unshared one."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 5, 3, padding=1)
        self.b = self.a
        self.c = torch.nn.Conv2d(5, 2, 3, stride=2, padding=2, dilation=2, padding_mode='reflect')

    def forward(self, x):
        return self.c(self.b(self.a(x)[:, :3]))


def test_shared_conv_gets_one_wrapper_under_every_name():
    torch.manual_seed(0)
    m = _Shared()
    assert ada.add_lora(m, 2, 2.0, ('',)) == 2
    assert isinstance(m.a, ada.LoRAConv2d) and m.b is m.a and isinstance(m.c, ada.LoRAConv2d)
    st = ada.adapter_state(m)
    assert set(st) == {'a.down.weight', 'a.up.weight', 'c.down.weight', 'c.up.weight'}
    assert set(st) <= set(m.state_dict())                            # keys load with load_state_dict(strict=False)
    n_ada, n_base = ada.count_params(m)
    assert n_ada == 2 * (3 * 9 + 5) + 2 * (5 * 9 + 2)
    assert n_base == (5 * 3 * 9 + 5) + (2 * 5 * 9 + 2)


def test_adapter_state_keys_are_the_lora_tensors_of_the_state_dict():
    """Saved adapters stay loadable: adapter_state keeps the state_dict keys (and values) of the low-rank tensors."""
    net = ada.AdaptedDepth(tiny_model(), rank=4, alpha=4.0)
    sd = net.state_dict()
    st = ada.adapter_state(net)
    assert list(st) == [k for k in sd if k.endswith(('.down.weight', '.up.weight'))]
    assert all(torch.equal(st[k], sd[k]) for k in st)
    net2 = ada.AdaptedDepth(tiny_model(), rank=4, alpha=4.0)
    for m in net.modules():
        if isinstance(m, ada.LoRAConv2d):
            torch.nn.init.normal_(m.up.weight, std=0.1)
    res = net2.load_state_dict(ada.adapter_state(net), strict=False)
    assert not res.unexpected_keys
    img = torch.rand(1, 3, H, W)
    assert torch.equal(net2.depth(img), net.depth(img))


def test_merge_lora_folds_the_branch_exactly():
    """W' = W + s · up ∘ down: the merged plain conv gives the adapted output (float64: to rounding), shared convs
    included, and the module has no LoRAConv2d left."""
    torch.manual_seed(0)
    m = _Shared()
    ada.add_lora(m, 2, 3.0, ('',))
    m.double()
    for w in (m.a, m.c):
        torch.nn.init.normal_(w.up.weight, std=0.3)
    x = torch.rand(2, 3, 17, 19, dtype=torch.float64)
    with torch.no_grad():
        ref = m(x)
    assert ada.merge_lora(m) == 2
    assert type(m.a) is torch.nn.Conv2d and m.b is m.a and type(m.c) is torch.nn.Conv2d
    with torch.no_grad():
        out = m(x)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    net = ada.AdaptedDepth(tiny_model(), rank=4, alpha=4.0)
    for w in net.modules():
        if isinstance(w, ada.LoRAConv2d):
            torch.nn.init.normal_(w.up.weight, std=0.1)
    img = torch.rand(1, 3, H, W)
    with torch.no_grad():
        ref = net.model.depth_m(img)
        assert ada.merge_lora(net.model) == 14
        assert not any(isinstance(w, ada.LoRAConv2d) for w in net.modules())
        torch.testing.assert_close(net.model.depth_m(img), ref, rtol=1e-4, atol=1e-5)


def test_saved_adapter_loads_with_weights_only(tmp_path):
    """apps/calib_s3.py saves only tensors and plain types, so torch.load(weights_only=True) reads it."""
    net = ada.AdaptedDepth(tiny_model(), rank=4, alpha=4.0)
    n_ada, n_base = ada.count_params(net)
    obj = {'adapter': ada.adapter_state(net), 'rank': 4, 'alpha': 4.0, 'prefixes': ['depth_net.decoder'],
           'ckpt': '/x.pt', 'ckpt_sha256': 'ab', 'labels': '/l.npz', 'labels_sha256': 'cd', 'scene': 'room5',
           'n_params': n_ada, 'n_base': n_base, 'n_depth_net': 1,
           'args': {'cmd': 'train', 'rank': 4, 'alpha': 4.0, 'prefixes': ['depth_net.decoder'], 'offsets': [-1, 1],
                    'out': '/o', 'device': 'cpu', 'w_geo': 0.0}}
    torch.save(obj, tmp_path / 'adapter.pt')
    a = torch.load(tmp_path / 'adapter.pt', map_location='cpu', weights_only=True)
    assert set(a['adapter']) == set(obj['adapter']) and a['args'] == obj['args']


def test_calib_labels_need_a_start0_run_of_the_scene(tmp_path):
    """apps/calib_s3.py load_labels: inc[k] = T_{k-1<-k} (inc[0] = I), a 'motion_model' pair is not tracked; another
    scene or a run that does not cover pairs 1..n-1 is a ValueError."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps'))
    import calib_s3

    n = 5
    rel = np.stack([pose([0, 0, 0.01 * k], [0.01 * k, 0, 0]) for k in range(1, n)])
    src = np.array(['tracked', 'motion_model', 'tracked', 'tracked'])

    def save(name, index, scene='room5'):
        np.savez(tmp_path / name, index=np.asarray(index), rel_increment=rel[:len(index)], source=src[:len(index)],
                 meta=np.array(f'{{"scene": "{scene}"}}'))
        return tmp_path / name

    inc, ok = calib_s3.load_labels(save('a.npz', np.arange(1, n)), 'room5', n)
    np.testing.assert_array_equal(inc[0], np.eye(4))
    np.testing.assert_array_equal(inc[1:], rel)
    assert ok.tolist() == [True, True, False, True, True]
    with pytest.raises(ValueError, match='scene'):
        calib_s3.load_labels(save('b.npz', np.arange(1, n), 'room4'), 'room5', n)
    with pytest.raises(ValueError, match='pairs 1..4'):
        calib_s3.load_labels(save('c.npz', np.arange(2, n)), 'room5', n)
