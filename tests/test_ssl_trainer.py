"""dvslam.ssl trainer / predictor / depth prior on a synthetic batch (CPU, 64x64, batch 2, no
pretrained weights, flow teacher off, no dataset)."""
import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from dvslam.depth.ssl import SSLDepth
from dvslam.odometry.base import Frame
from dvslam.ssl.predict import Predictor
from dvslam.ssl.trainer import Trainer, apply_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]
H = W = 64
K = np.array([[40.0, 0, 31.5], [0, 40.0, 31.5], [0, 0, 1]])


def config() -> dict:
    with open(REPO_ROOT / 'configs/ssl_tumvi.yaml') as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, ['train.device=cpu', 'model.pretrained=false', 'loss.flow=0', 'train.num_workers=0',
                                 'train.warmup_steps=1'])


def synthetic_batch(b: int = 2) -> dict:
    g = torch.Generator().manual_seed(0)
    net = torch.rand(b, 3, 3, H, W, generator=g)
    T_stereo = torch.eye(4).repeat(b, 1, 1)
    T_stereo[:, 0, 3] = -0.101
    return {'net': net, 'pho': torch.cat([net[:, :, :1], net[:, 1:2, :1]], 1), 'valid': torch.ones(b, 4, 1, H, W),
            'K': torch.as_tensor(K, dtype=torch.float32).repeat(b, 1, 1), 'T_stereo': T_stereo,
            'R_gyro': torch.eye(3).repeat(b, 2, 1, 1), 'gyro_ok': torch.ones(b, dtype=torch.bool),
            'scene': ['synthetic'] * b, 'index': torch.arange(b)}


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp('run')
    batch = synthetic_batch()
    tr = Trainer(config(), out, [batch])
    init = copy.deepcopy(tr.model.state_dict())
    logs = [tr.train_step(batch) for _ in range(2)]
    tr.save(out / 'last.pt')
    return tr, init, logs, out


def test_two_steps_finite_and_weights_change(trained):
    tr, init, logs, _ = trained
    assert tr.step == 2
    for log in logs:
        assert {'loss', 'temporal', 'stereo', 'smooth', 'geometric', 'gyro', 'pose_consistency'} == set(log)
        assert all(np.isfinite(v) for v in log.values()), log
    changed = [k for k, v in tr.model.state_dict().items() if v.is_floating_point() and not torch.equal(v, init[k])]
    assert any(k.startswith('depth_net') for k in changed) and any(k.startswith('pose_net') for k in changed)
    assert any(k.startswith('residual_net') for k in changed)
    # EMA: first update copies, second moves by (1 - 0.999): differs from both the initial and the current weights
    ema = dict(tr.ema.module.named_parameters())
    w = 'depth_net.encoder.encoder.conv1.weight'
    assert int(tr.ema.n_averaged) == 2
    assert not torch.equal(ema[w], init[w]) and not torch.equal(ema[w], tr.model.state_dict()[w])


def test_checkpoint_roundtrip_predictor_identical(trained):
    tr, _, _, out = trained
    ckpt = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    assert {'config', 'model', 'ema', 'optimizer', 'scheduler', 'epoch', 'step', 'best', 'git_commit', 'K',
            'size'} <= set(ckpt)
    assert ckpt['size'] == (H, W) and np.allclose(ckpt['K'], K)
    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (2, H, W), dtype=np.uint8) for _ in range(2))
    live = Predictor(tr.ema.module, tr.K, tr.size, 'cpu')
    loaded = Predictor.from_checkpoint(out / 'last.pt', 'cpu')
    np.testing.assert_array_equal(live.increment(a, b), loaded.increment(a, b))
    np.testing.assert_array_equal(live.depth(a[0]), loaded.depth(a[0]))
    assert loaded.increment(a[0], b[0]).shape == (4, 4) and loaded.increment(a, b).shape == (2, 4, 4)
    assert loaded.depth(a[0]).shape == (H, W) and loaded.depth(a[0]).dtype == np.float32
    raw = Predictor.from_checkpoint(out / 'last.pt', 'cpu', ema=False)
    np.testing.assert_array_equal(Predictor(tr.model, tr.K, tr.size, 'cpu').increment(a, b),
                                  raw.increment(a, b))
    # resume restores the counters and the optimiser state
    tr2 = Trainer(config(), out, [synthetic_batch()])
    tr2.load(out / 'last.pt')
    assert tr2.step == 2 and tr2.sched.last_epoch == 2 and int(tr2.ema.n_averaged) == 2


def test_depth_prior(trained, tmp_path):
    _, _, _, out = trained
    rng = np.random.default_rng(1)
    paths = []
    for i in range(4):
        p = tmp_path / f'{i:03d}.png'
        cv2.imwrite(str(p), rng.integers(0, 256, (H, W), dtype=np.uint8))
        paths.append(p)
    depth, conf = SSLDepth(str(out / 'last.pt'), K, device='cpu').predict(Frame(0, 0.0, paths[0]))
    assert conf is None and depth.shape == (H, W) and (depth >= 0.1 - 1e-4).all() and (depth <= 20.0 + 1e-3).all()


def test_amp_option(tmp_path):
    """train.amp on CPU: bfloat16 autocast on the training model's networks only; the EMA copy (validated and saved)
    stays float32, no grad scaler on CPU, the step is finite and the checkpoint carries no extra model state."""
    cfg = config()
    assert cfg['train']['amp'] is False and Trainer(cfg, tmp_path, [synthetic_batch()]).model.amp_dtype is None
    tr = Trainer(apply_overrides(config(), ['train.amp=true']), tmp_path, [synthetic_batch()])
    assert tr.model.amp_dtype == tr.model.pose_net.amp_dtype == tr.model.residual_net.amp_dtype == torch.bfloat16
    assert tr.ema.module.amp_dtype is None and tr.ema.module.pose_net.amp_dtype is None
    assert not tr.scaler.is_enabled()
    log = tr.train_step(synthetic_batch())
    assert all(np.isfinite(v) for v in log.values()), log
    tr.save(tmp_path / 'last.pt')
    ckpt = torch.load(tmp_path / 'last.pt', map_location='cpu', weights_only=False)
    assert set(ckpt['model']) == set(Trainer(config(), tmp_path, [synthetic_batch()]).model.state_dict())
    assert Predictor.from_checkpoint(tmp_path / 'last.pt', 'cpu').model.amp_dtype is None


def test_set_overrides():
    cfg = apply_overrides(config(), ['loss.flow=0.5', 'train.epochs=3', 'data.train_scenes=[room1, room2]',
                                     'model.uncertainty=false', 'data.root=/x/y'])
    assert cfg['loss']['flow'] == 0.5 and cfg['train']['epochs'] == 3 and cfg['data']['root'] == '/x/y'
    assert cfg['data']['train_scenes'] == ['room1', 'room2'] and cfg['model']['uncertainty'] is False
    with pytest.raises(KeyError):
        apply_overrides(config(), ['loss.flwo=1'])
