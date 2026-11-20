"""exp003_score additions: mean length error per pair (mm/frame) and the copy-previous-step baseline."""
import numpy as np
import pytest

import scripts.exp003_score as S


def _steps(lengths_m: np.ndarray) -> np.ndarray:
    """Relative poses (n, 4, 4) translating by the given lengths along x."""
    T = np.tile(np.eye(4), (len(lengths_m), 1, 1))
    T[:, 0, 3] = lengths_m
    return T


@pytest.fixture
def fake_room(monkeypatch):
    """A room with GT steps g (frames 1..n) and a run whose increments cover frames idx; pair 3 has no GT."""
    g = np.array([0.010, 0.020, 0.030, 0.040, 0.050, 0.060])
    valid = np.array([True, True, False, True, True, True])

    def setup(pred, idx):
        inc = {'meta': {'scene': 'roomX'}, 'index': np.asarray(idx), 'rel_increment': _steps(np.asarray(pred)),
               'source': np.array(['tracked'] * len(idx))}
        monkeypatch.setattr(S, 'load_increments', lambda path: inc)
        monkeypatch.setattr(S, 'gt_steps', lambda scene, tau=0.0: (_steps(g), valid))
        return S.pair_table('roomX.inc.npz')
    monkeypatch.setattr(S, 'MIN_INDEX', [1])
    return g, valid, setup


def test_copy_prev_uses_own_previous_prediction_before_the_filter(fake_room):
    g, valid, setup = fake_room
    pred = np.array([0.011, 0.018, 0.033, 0.041, 0.047, 0.066])
    tab = setup(pred, np.arange(1, 7))
    # pair 3 (invalid GT) is dropped, but pair 4 still copies the prediction of pair 3 (deployable: no GT used)
    np.testing.assert_allclose(tab['gt'], g[valid])
    np.testing.assert_allclose(tab['pred'], pred[valid])
    np.testing.assert_allclose(tab['prev_pred'], np.array([0.011, 0.011, 0.033, 0.041, 0.047]))


def test_copy_prev_with_min_index_and_partial_range(fake_room, monkeypatch):
    _, _, setup = fake_room
    monkeypatch.setattr(S, 'MIN_INDEX', [5])
    tab = setup(np.array([0.030, 0.042, 0.048, 0.061]), np.arange(3, 7))    # increments of frames 3..6
    np.testing.assert_allclose(tab['pred'], [0.048, 0.061])                 # scored pairs 5, 6
    np.testing.assert_allclose(tab['prev_pred'], [0.042, 0.048])            # their own predictions of pairs 4, 5
    monkeypatch.setattr(S, 'MIN_INDEX', [1])
    tab = setup(np.array([0.030, 0.042, 0.048, 0.061]), np.arange(3, 7))
    np.testing.assert_allclose(tab['prev_pred'], [0.030, 0.042, 0.048])     # first increment keeps its own length


def test_summarize_mm_per_frame_and_copy_prev(fake_room):
    _, _, setup = fake_room
    pred = np.array([0.011, 0.018, 0.033, 0.041, 0.047, 0.066])
    tab = setup(pred, np.arange(1, 7))
    s = S.summarize([tab])
    gt, p, prev = tab['gt'], tab['pred'], tab['prev_pred']
    assert s['mean_err_mm'] == pytest.approx(np.mean(np.abs(p - gt)) * 1e3)            # (1+2+1+3+6)/5 = 2.6 mm
    assert s['mean_err_mm'] == pytest.approx(2.6)
    assert s['copy_prev_mean_err_mm'] == pytest.approx(np.mean(np.abs(prev - gt)) * 1e3)
    assert s['copy_prev_item03'] == pytest.approx(np.mean(np.abs(prev - gt) / gt) * 100)
    assert s['item03'] == pytest.approx(np.mean(np.abs(p - gt) / gt) * 100)            # existing key unchanged


def test_summarize_pools_rooms_by_pairs():
    rng = np.random.default_rng(0)
    tabs = []
    for n in (50, 200):
        gt = rng.uniform(0.002, 0.06, n)
        pred = gt * (1 + 0.05 * rng.standard_normal(n))
        tabs.append({'gt': gt, 'pred': pred, 'pct': np.abs(pred - gt) / gt * 100, 'tracked': np.ones(n, bool),
                     'prev_pred': np.r_[pred[:1], pred[:-1]]})
    s = S.summarize(tabs)
    gt, pred, prev = (np.concatenate([t[k] for t in tabs]) for k in ('gt', 'pred', 'prev_pred'))
    assert s['pairs'] == 250
    assert s['mean_err_mm'] == pytest.approx(np.abs(pred - gt).mean() * 1e3)
    assert s['copy_prev_mean_err_mm'] == pytest.approx(np.abs(prev - gt).mean() * 1e3)
    # a constant-speed baseline is worse than the run on independent steps
    assert s['copy_prev_item03'] > s['item03']


def test_constant_speed_motion_copy_prev_is_exact():
    gt = np.full(20, 0.02)
    pred = gt * 1.01
    s = S.summarize([{'gt': gt, 'pred': pred, 'pct': np.abs(pred - gt) / gt * 100, 'tracked': np.ones(20, bool),
                      'prev_pred': np.r_[gt[:1], gt[:-1]]}])
    assert s['copy_prev_item03'] == pytest.approx(0.0)
    assert s['copy_prev_mean_err_mm'] == pytest.approx(0.0)
    assert s['mean_err_mm'] == pytest.approx(0.2)
