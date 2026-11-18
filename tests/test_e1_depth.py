"""E1 depth bias / jitter metrics (plan v3 item 0.10). Model tests are marked gpu."""
import numpy as np
import pytest

from dvslam.depth.bias import frame_stats, summarize


def _gt(seed=0, shape=(48, 64)):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.5, 6.0, shape).astype(np.float32)
    gt[:4] = 0.0          # invalid (0 = no reading)
    gt[4:6] = 12.0        # beyond 10 m
    return gt


def test_known_scale_gives_known_bias():
    gt = _gt()
    s = 1.05
    st = frame_stats(s * gt, gt)
    assert st['r'] == pytest.approx(s, rel=1e-6)
    assert st['absrel'] == pytest.approx(s - 1, rel=1e-5)
    assert st['absrel_med'] == pytest.approx(0.0, abs=1e-6)
    assert st['n_valid'] == 42 * 64

    r = np.full(10, s)
    out = summarize(r, np.array([[0, 1], [2, 3], [4, 5]]))
    assert out['b_median'] == pytest.approx(0.05)
    assert out['b_mean'] == pytest.approx(0.05)
    assert out['jitter'] == pytest.approx(0.0, abs=1e-12)
    assert out['f2f_jitter'] == pytest.approx(0.0, abs=1e-12)


def test_invalid_pred_and_gt_are_excluded():
    gt = _gt()
    pred = 0.7 * gt
    pred[10:20] = np.inf
    pred[20:22] = -1.0
    st = frame_stats(pred, gt)
    assert st['r'] == pytest.approx(0.7, rel=1e-6)
    assert st['n_valid'] == (42 - 12) * 64
    assert np.isnan(frame_stats(pred, np.zeros_like(gt))['r'])
    with pytest.raises(ValueError):
        frame_stats(pred[:-1], gt)


def test_jitter_recovers_lognormal_sigma():
    rng = np.random.default_rng(1)
    sigma, n = 0.02, 20000
    log_r = np.log(0.9) + sigma * rng.standard_normal(n)
    pairs = np.stack([np.arange(0, n - 1, 2), np.arange(1, n, 2)], 1)
    out = summarize(np.exp(log_r), pairs)
    assert out['b_median'] == pytest.approx(-0.1, abs=2e-3)
    assert out['jitter'] == pytest.approx(sigma, rel=0.03)
    assert out['f2f_jitter'] == pytest.approx(sigma * np.sqrt(2), rel=0.03)   # iid frames


def test_f2f_jitter_ignores_slow_drift():
    t = np.arange(1000)
    r = np.exp(0.1 * np.sin(t / 200.0))            # slow scale drift, tiny consecutive change
    pairs = np.stack([t[:-1:2], t[1::2]], 1)
    out = summarize(r, pairs)
    assert out['jitter'] > 0.05
    assert out['f2f_jitter'] < 1e-3


def test_nan_frames_are_skipped():
    r = np.array([1.0, np.nan, 1.0, 1.0])
    out = summarize(r, np.array([[0, 1], [2, 3]]), absrel=np.array([0.1, np.nan, 0.3, 0.2]))
    assert out['n_frames'] == 3 and out['n_pairs'] == 1
    assert out['absrel'] == pytest.approx(0.2)


@pytest.mark.gpu
@pytest.mark.parametrize('name', ['dav2s', 'da3m', 'moge2s'])
def test_model_predicts_metric_depth_at_input_resolution(name):
    from dvslam.data.redwood import load_depth, load_sequence
    from dvslam.depth import build, read_bgr
    from dvslam.odometry.base import Frame
    seq = load_sequence('bedroom', 'validation')
    model = build(name, seq.K, fp16=False)
    depth, conf = model.predict(Frame(0, 0.0, seq.image_paths[0]))
    assert depth.shape == (480, 640) and depth.dtype == np.float32
    assert conf is None or conf.shape == depth.shape
    st = frame_stats(depth, load_depth(seq.depth_paths[0]))
    assert 0.3 < st['r'] < 3.0          # metric, not disparity or a relative map
    model.fp16 = True                    # same network, autocast fp16 (one copy on the shared GPU)
    d16, _ = model.predict_image(read_bgr(seq.image_paths[0]))
    assert frame_stats(d16, load_depth(seq.depth_paths[0]))['r'] == pytest.approx(st['r'], rel=0.02)
