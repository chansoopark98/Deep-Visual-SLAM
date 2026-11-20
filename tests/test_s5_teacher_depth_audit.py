"""S5 teacher depth audit: window tiling, hint offset, alignment scales and depth metrics on synthetic depth."""
import numpy as np
import pytest

from scripts.s5_teacher_depth_audit import (
    HINT_MIN,
    N,
    depth_metrics,
    evaluate_window,
    hint_offset,
    median_scale,
    sampled_windows,
    summarise,
    tiled_windows,
    valid_mask,
)


@pytest.mark.parametrize('gap', [1, 5])
def test_tiled_windows_cover_every_frame_once(gap):
    n_frames = 2846
    win = tiled_windows(n_frames, gap)
    assert win.shape[1] == N
    assert np.all(np.diff(win, axis=1) == gap)
    flat = np.sort(win.ravel())
    full = n_frames // (N * gap) * N * gap
    np.testing.assert_array_equal(flat, np.arange(full))


def test_sampled_windows():
    win = sampled_windows(2821, every=160)
    np.testing.assert_array_equal(win[:, 0], np.arange(0, 2821 - N + 1, 160))
    assert np.all(np.diff(win, axis=1) == 1) and win.max() < 2821


def test_hint_offset():
    cam0 = [f'{i}.png' for i in range(10)]
    assert hint_offset(cam0[1:9], cam0) == 1
    assert hint_offset(cam0[:9], cam0) == 0
    with pytest.raises(ValueError):
        hint_offset(['1.png', '3.png'], cam0)


def _scene(rng, n=N, h=32, w=32):
    ref = rng.uniform(0.5, 4.0, (n, h, w)).astype(np.float32)
    ref[:, :4] = 0.0                             # no hint
    return ref


def test_metrics_exact_values():
    ref = np.array([1.0, 2.0, 4.0, 0.05])
    pred = np.array([1.1, 2.0, 2.0, 1.0])
    m = valid_mask(ref, pred)
    np.testing.assert_array_equal(m, [True, True, True, False])          # hint <= HINT_MIN is not a reference
    r = depth_metrics(ref, pred, m)
    assert r['absrel'] == pytest.approx((0.1 + 0 + 0.5) / 3)
    assert r['delta'] == pytest.approx(2 / 3)
    assert r['logrmse'] == pytest.approx(np.sqrt((np.log(1.1) ** 2 + 0 + np.log(0.5) ** 2) / 3))
    assert r['bias'] == pytest.approx(0.0)
    assert median_scale(ref, pred, m) == pytest.approx(1.0)
    assert HINT_MIN == 0.1


def test_median_scale_recovers_scale_with_outliers():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.5, 4.0, 1001)
    pred = ref / 3.7
    pred[:300] *= rng.uniform(0.2, 5.0, 300)     # 30 % outliers do not move the median
    m = np.ones_like(ref, bool)
    s = median_scale(ref, pred, m)
    assert 3.3 < s < 4.1
    assert median_scale(ref, ref / 3.7, m) == pytest.approx(3.7)


def test_window_scaled_copy_is_perfect_and_frame_scales_measure_jitter():
    rng = np.random.default_rng(1)
    ref = _scene(rng)
    s_k = np.exp(rng.normal(0, 0.05, N))         # per-frame scale jitter, sd of log 5 %
    pred = ref / (2.5 * s_k[:, None, None])      # unknown global unit 1 / 2.5 and per-frame jitter
    pred[:, :4] = rng.uniform(0.1, 1, (N, 4, 32))  # predictions where there is no hint are ignored
    e = evaluate_window(ref, pred, min_px=10)
    assert e['ok'].all()
    np.testing.assert_allclose(e['s_frame'], 2.5 * s_k, rtol=1e-6)
    np.testing.assert_allclose(e['frame_absrel'], 0, atol=1e-6)            # shape is perfect
    np.testing.assert_allclose(e['frame_delta'], 1)
    assert e['log_s_sd'] == pytest.approx(np.std(np.log(s_k), ddof=1), rel=1e-5)
    assert e['window_absrel'].mean() > 0.01                               # one scale cannot absorb the jitter
    np.testing.assert_allclose(e['none_bias'], -np.log(2.5 * s_k), rtol=1e-5)

    e2 = evaluate_window(ref, ref / 2.5, min_px=10)                       # constant unit: window alignment exact
    np.testing.assert_allclose(e2['window_absrel'], 0, atol=1e-6)
    np.testing.assert_allclose(e2['window_logrmse'], 0, atol=1e-6)
    assert e2['s_window'] == pytest.approx(2.5)
    assert e2['log_s_sd'] == pytest.approx(0, abs=1e-6)


def test_invalid_frames_and_summary():
    rng = np.random.default_rng(2)
    ref = _scene(rng)
    pred = ref.copy()
    pred[0] = np.nan                             # whole frame invalid
    pred[1, 10:] = -1.0                          # mostly invalid -> below min_px
    e = evaluate_window(ref, pred, min_px=200)
    assert not e['ok'][0] and not e['ok'][1] and e['ok'][2:].all()
    assert np.isnan(e['frame_absrel'][:2]).all()
    assert e['log_s_sd'] == pytest.approx(0, abs=1e-6)                    # 6 valid frames >= MIN_FRAMES
    s = summarise([e, evaluate_window(ref, ref * 1.1, min_px=200)])
    assert s['frames'] == 2 * N and s['frames_ok'] == 2 * N - 2 and s['windows_sd'] == 2
    assert s['none_absrel'] == pytest.approx(0.1 * N / (2 * N - 2), rel=1e-5)
    assert s['window_absrel'] == pytest.approx(0, abs=1e-6)
    assert 0 < s['coverage'] <= 1
    e3 = evaluate_window(ref, pred[:, ::-1] * 0 - 1, min_px=200)           # nothing valid
    assert not e3['ok'].any() and np.isnan(e3['log_s_sd'])
