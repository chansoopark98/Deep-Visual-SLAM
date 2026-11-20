"""EXP-016 audit: window -> pair mapping, central-window choice, depth normalisation and increment composition must
recover synthetic displacements exactly."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.exp016_audit import (
    N,
    analyse,
    angle_deg,
    central_choice,
    compose_increments,
    pair_ids,
    rot_deg,
    summarise,
    window_rel,
    window_starts,
)


def _trajectory(n=200, seed=0):
    """Random smooth-ish camera trajectory with known poses (steps 1-60 mm, small rotations)."""
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    for t in range(1, n):
        d = np.eye(4)
        d[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix()
        d[:3, 3] = rng.normal(0, 1, 3)
        d[:3, 3] *= rng.uniform(0.001, 0.06) / np.linalg.norm(d[:3, 3])
        T[t] = T[t - 1] @ d
    return T


def _model_windows(T, starts, gap, scale):
    """What a model returns: poses re-based to frame 0, translations (and depth) times a per-window scale."""
    out = np.stack([np.linalg.inv(T[t]) @ T[t + gap * np.arange(N)] for t in starts])
    out[:, :, :3, 3] *= scale[:, None, None]
    return out


@pytest.mark.parametrize('gap', [1, 2, 5, 10])
def test_window_pairs_recover_known_displacements(gap):
    T = _trajectory()
    starts = window_starts(len(T), gap)
    assert starts[-1] + (N - 1) * gap <= len(T) - 1 < starts[-1] + 2 * gap + (N - 1) * gap
    rel = window_rel(_model_windows(T, starts, gap, np.ones(len(starts))))
    pid = pair_ids(starts, gap)
    for (w, k), i in np.ndenumerate(pid):
        np.testing.assert_allclose(rel[w, k], np.linalg.inv(T[i]) @ T[i + gap], atol=1e-12)
    ids, counts = np.unique(pid, return_counts=True)
    assert np.all(ids % gap == 0)
    inner = (ids >= (N - 2) * gap) & (ids <= starts[-1])
    assert set(counts[inner & (ids // gap % 2 == 0)]) == {4} and set(counts[inner & (ids // gap % 2 == 1)]) == {3}
    cen = central_choice(pid)
    assert all(cen[int(i)][1] == (3 if i // gap % 2 else 2) for i in ids[inner])


@pytest.mark.parametrize('gap', [1, 5])
def test_analyse_exact_with_window_scales_and_depth_normalisation(gap):
    T = _trajectory(seed=1)
    valid = np.ones(len(T), bool)
    starts = window_starts(len(T), gap)
    scale = np.random.default_rng(2).uniform(0.5, 2.0, len(starts))          # model's per-window unit
    a = analyse(window_rel(_model_windows(T, starts, gap, scale)), starts, gap, T, valid,
                depth_med=np.outer(scale, np.full(N, 1.7)))                 # depth carries the same window unit
    assert a['ids'].min() >= 4 and np.all(a['ids'] % gap == 0)
    np.testing.assert_allclose(a['g'], [np.linalg.norm((np.linalg.inv(T[i]) @ T[i + gap])[:3, 3]) for i in a['ids']])
    np.testing.assert_allclose(a['p1'], a['g'] * scale[a['w1']], rtol=1e-12)
    np.testing.assert_allclose(a['p2'], a['p1'], rtol=1e-12)                 # all windows rescaled to w* agree
    np.testing.assert_allclose(a['s_w'], 1 / scale, rtol=1e-12)              # per-window oracle undoes the unit
    assert np.nanmax(a['rot']) < 1e-5 and np.nanmax(a['dir']) < 1e-5
    s = summarise(a)
    assert s['B']['perwin_item03'] < 1e-9 and s['C_dn']['sd'] < 1e-12
    if gap == 1:
        assert s['C_raw']['sd'] > 0.1                                       # raw window units disagree


def test_unit_scale_gives_zero_error_and_unit_oracle():
    T = _trajectory(seed=3)
    starts = window_starts(len(T), 2)
    a = analyse(window_rel(_model_windows(T, starts, 2, np.full(len(starts), 0.25))), starts, 2, T,
                np.ones(len(T), bool), depth_med=np.ones((len(starts), N)))
    s = summarise(a)
    assert s['s1'] == pytest.approx(4.0) and s['A1']['p90'] < 1e-9 and s['B']['jitter'] < 1e-12


def test_valid_filter_drops_pairs_without_gt_at_an_end():
    T = _trajectory(seed=4)
    valid = np.ones(len(T), bool)
    valid[[20, 21]] = False
    starts = window_starts(len(T), 1)
    a = analyse(window_rel(_model_windows(T, starts, 1, np.ones(len(starts)))), starts, 1, T, valid)
    assert not set(a['ids']) & {19, 20, 21} and {18, 22} <= set(a['ids']) and a['ids'][0] == 4


def test_angles_exact_and_robust_to_float32_non_orthonormality():
    R = Rotation.from_rotvec([0.0, 0.0, np.radians(30.0)]).as_matrix()
    assert rot_deg(R) == pytest.approx(30.0) and rot_deg(np.eye(3)) == 0.0
    assert rot_deg(np.eye(3) + 1e-5 * np.array([[1, 2, 0], [2, -1, 3], [0, 3, 1]])) < 1e-6
    assert angle_deg(np.array([1.0, 0, 0]), np.array([1.0, 1.0, 0])) == pytest.approx(45.0)
    assert angle_deg(np.array([1.0, 0, 0]), np.array([1.0, 1e-9, 0])) == pytest.approx(np.degrees(1e-9), rel=1e-6)


@pytest.mark.parametrize('gap', [1, 3, 10])
def test_compose_increments_equals_relative_pose(gap):
    T = _trajectory(seed=5)
    index = np.arange(1, len(T))
    inc = np.linalg.inv(T[:-1]) @ T[1:]                                       # increment t: (t - 1) -> t
    pairs = np.arange(0, len(T) - gap, gap)
    out = compose_increments(index, inc, pairs, gap)
    for n, i in enumerate(pairs):
        np.testing.assert_allclose(out[n], np.linalg.inv(T[i]) @ T[i + gap], atol=1e-10)
