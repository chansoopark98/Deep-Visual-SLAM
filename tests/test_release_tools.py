"""Release tooling: checkpoint stripper (scripts/export_release_weights.py) and the demo renderer's trajectory
integration / running errors (scripts/render_demo.py). Synthetic inputs only (CPU, no data)."""
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import scripts.export_release_weights as E
import scripts.render_demo as R
from dvslam.geometry.se3 import relative_pose

MODEL_CFG = {'pretrained': False, 'min_depth': 0.1, 'max_depth': 20.0, 'uncertainty': False, 'sigma_min': 0.01,
             'pose_iters': 2}


def _training_ckpt(size=(128, 128)) -> dict:
    """A checkpoint in dvslam/ssl/trainer.py's format around a randomly initialised SSLModel."""
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    from dvslam.ssl.predict import build_model
    torch.manual_seed(0)
    model = build_model(MODEL_CFG, pretrained=False)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)
    with torch.no_grad():                                   # make EMA differ from the raw weights
        for p in ema.module.parameters():
            p.add_(0.01 * torch.randn_like(p))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    K = np.array([[50.0, 0, 63.5], [0, 50.0, 63.5], [0, 0, 1]])
    return {'config': {'model': MODEL_CFG}, 'model': model.state_dict(), 'ema': ema.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': {'last_epoch': 3}, 'epoch': 3, 'step': 120, 'best': 4.2,
            'git_commit': 'abc', 'K': K, 'size': size, 'scaler': {}}


def test_strip_ssl_keeps_inference_fields_and_loads_weights_only(tmp_path):
    ck = _training_ckpt()
    dst = tmp_path / 'w.pt'
    torch.save(E.strip_ssl(ck, 'f' * 64), dst)
    out = torch.load(dst, map_location='cpu', weights_only=True)
    assert set(out) == {'config', 'ema', 'epoch', 'step', 'best', 'git_commit', 'K', 'size', 'source_sha256',
                        'format'}
    assert out['K'] == ck['K'].tolist() and out['size'] == (128, 128) and out['source_sha256'] == 'f' * 64
    assert out['ema'].keys() == ck['ema'].keys()
    assert all(torch.equal(out['ema'][k], v) for k, v in ck['ema'].items())


def test_stripped_checkpoint_gives_bit_identical_depth(tmp_path):
    from dvslam.ssl.predict import Predictor
    src, dst = tmp_path / 'train.pt', tmp_path / 'release.pt'
    torch.save(_training_ckpt(), src)
    sys_argv = sys.argv
    try:
        sys.argv = ['export_release_weights.py', str(src), str(dst)]
        E.main()
    finally:
        sys.argv = sys_argv
    img = np.random.default_rng(0).integers(0, 256, (2, 128, 128), dtype=np.uint8)
    a = Predictor.from_checkpoint(src, device='cpu').depth(img)
    b = Predictor.from_checkpoint(dst, device='cpu').depth(img)
    assert np.array_equal(a, b)
    assert dst.stat().st_size < 0.6 * src.stat().st_size            # raw weights dropped (optimizer state is empty)


def test_photonet_is_copied_unchanged(tmp_path):
    from dvslam.odometry.photo_learned import PhotoNet, save_net
    src, dst = tmp_path / 'net.pt', tmp_path / 'out' / 'net.pt'
    save_net(src, PhotoNet('w', 4), {'step': 1})
    sys_argv = sys.argv
    try:
        sys.argv = ['export_release_weights.py', str(src), str(dst)]
        E.main()
    finally:
        sys.argv = sys_argv
    assert dst.read_bytes() == src.read_bytes()


def _gt_poses(n=40, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)).as_matrix()
    T[:, :3, 3] = np.cumsum(rng.normal(0, 0.02, (n, 3)), 0) + [1.0, -2.0, 1.5]
    return T


def test_integrate_reproduces_gt_from_its_increments():
    T = _gt_poses()
    inc = np.stack([relative_pose(T[t - 1], T[t]) for t in range(1, len(T))])
    out = R.integrate(T[0], inc)
    assert out.shape == T.shape
    np.testing.assert_allclose(out, T, atol=1e-9)
    np.testing.assert_allclose(R.integrate(T[5], inc[5:]), T[5:], atol=1e-9)     # a run starting at frame 5


def test_running_errors_match_scaled_translation_and_skip_invalid_pairs():
    T = _gt_poses()
    idx = np.arange(1, len(T))
    inc = np.stack([relative_pose(T[t - 1], T[t]) for t in idx])
    inc[:, :3, 3] *= 1.10                               # every step 10 % too long, rotation exact
    inc[3, :3, 3] *= 3.0                                # a bad pair ...
    valid = np.ones(len(T) - 1, bool)
    valid[3] = False                                    # ... without measured GT: not scored
    valid[0] = False                                    # nothing scored after the first pair
    e_t, e_r = R.running_errors(T, idx, inc, valid)
    assert np.isnan(e_t[0]) and np.isnan(e_r[0])
    np.testing.assert_allclose(e_t[1:], 10.0, rtol=1e-9)
    np.testing.assert_allclose(e_r[1:], 0.0, atol=1e-9)


def test_poly_skips_segments_without_gt():
    img = np.zeros((40, 40, 3), np.uint8)
    pts = np.array([[5.0, 5.0], [35.0, 5.0], [35.0, 35.0]])
    R.poly(img, pts, (255, 255, 255), 1, np.array([True, False]))
    assert img[5, 20].any() and not img[20, 35].any()
