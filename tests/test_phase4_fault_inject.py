"""FaultInjectOdometry (plan v3 item 4.3) and the true re-tracking scorer (4.1), on a fake odometry: no real data."""
import cv2
import numpy as np
import pytest

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from dvslam.odometry.fault_inject import Fault, FaultInjectOdometry
from dvslam.runtime.replay import apply_drop_policy
from dvslam.runtime.retrack import score_retracked


class FakeOdo:
    """Tracks while the image has texture; records what it was given and its re-initialisations."""

    def __init__(self):
        self.seen, self.resets, self.g, self.n_m, self.last_inc_ba = [], 0, 0.0, 0, np.eye(4)
        self.prev, self.fail = None, set()

    def reset(self, camera):
        self.resets += 1
        self.prev, self.g, self.n_m = None, 0.0, 0

    def _solve_pair(self):
        return True

    def step(self, frame):
        img = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        self.seen.append((frame.index, float(img.mean()), float(img.std())))
        if self.prev is None:
            self.prev, self.g, self.n_m = frame.index, 1.5, 7
            return None
        self._solve_pair()
        if frame.index in self.fail:                      # the base's own failure path: motion model, DEGRADED
            return PoseEstimate(frame.index, float(frame.index), self.last_inc_ba, 'DEGRADED', 'motion_model')
        inc = np.eye(4)
        inc[0, 3] = 0.01
        self.last_inc_ba = inc
        return PoseEstimate(frame.index, float(frame.index), inc, 'OK', 'tracked')


def make_frames(tmp_path, n=40):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        p = tmp_path / f'{i:06d}.png'
        cv2.imwrite(str(p), rng.integers(0, 255, (48, 64), dtype=np.uint8))
        out.append(Frame(i, i / 30, p, p))
    return out


def run(odo, frames):
    odo.reset(PinholeCamera(np.eye(3), 64, 48))
    return [e for f in frames if (e := odo.step(f)) is not None]


def test_no_faults_is_transparent(tmp_path):
    est = run(FaultInjectOdometry(FakeOdo(), []), make_frames(tmp_path, 10))
    assert [e.source for e in est] == ['tracked'] * 9 and {e.tracking_state for e in est} == {'OK'}


def test_black_freeze_starve_change_only_the_base_input(tmp_path):
    frames = make_frames(tmp_path)
    base = FakeOdo()
    odo = FaultInjectOdometry(base, [Fault('black', 5, 3), Fault('freeze', 12, 2), Fault('starve', 20, 2)])
    run(odo, frames)
    seen = {i: (m, s) for i, m, s in base.seen}
    assert all(seen[i] == (0.0, 0.0) for i in (5, 6, 7))                    # black
    assert seen[12] == seen[11] and seen[13] == seen[11]                    # duplicated last good frame
    assert seen[20][1] < 0.5 * seen[19][1]                                  # starved: texture almost gone
    assert seen[9][1] > 50 and seen[14][1] > 50                             # untouched frames stay textured
    assert [e['fault'] for e in odo.log if e['index'] in (5, 12, 20)] == ['black', 'freeze', 'starve']


def test_exception_is_swallowed_flagged_and_lost_then_recovers(tmp_path):
    base = FakeOdo()
    odo = FaultInjectOdometry(base, [Fault('exception', 10, 5)], lost_after=3)
    est = {e.index: e for e in run(odo, make_frames(tmp_path))}
    # exc(10) DEGRADED, re-init(11) BOOTSTRAP, exc(12) DEGRADED, re-init(13) BOOTSTRAP, exc(14) LOST, then recovery
    assert [est[i].tracking_state for i in range(10, 15)] == ['DEGRADED', 'BOOTSTRAP', 'DEGRADED', 'BOOTSTRAP', 'LOST']
    assert all(est[i].source == 'motion_model' for i in range(10, 15))
    assert est[15].tracking_state == 'BOOTSTRAP' and est[15].source == 'motion_model'    # re-init after LOST
    assert all(est[i].tracking_state == 'OK' and est[i].source == 'tracked' for i in range(16, 40))
    assert base.resets == 1 + 3                                             # initial + one re-init per exception frame
    assert (base.g, base.n_m) == (1.5, 7)                                   # g kept across the reset
    np.testing.assert_allclose(est[10].rel_increment[0, 3], 0.01)           # motion model = last emitted increment


def test_single_exception_costs_two_motion_model_frames(tmp_path):
    est = {e.index: e for e in run(FaultInjectOdometry(FakeOdo(), [Fault('exception', 20, 1)]), make_frames(tmp_path))}
    assert [est[i].source for i in (19, 20, 21, 22)] == ['tracked', 'motion_model', 'motion_model', 'tracked']
    assert est[20].tracking_state == 'DEGRADED'


def test_ba_exception_needs_a_ba_hook():
    with pytest.raises(ValueError):
        FaultInjectOdometry(FakeOdo(), [Fault('exception', 1, 1, where='ba')])
    with pytest.raises(ValueError):
        Fault('nonsense', 0, 1)


def test_score_retracked_matches_composed_replay_when_measurements_compose():
    n = 30
    inc = np.tile(np.eye(4), (n - 1, 1, 1))
    inc[:, 0, 3] = 0.01
    processed = np.zeros(n, bool)
    processed[[0, 3, 4, 9]] = True
    meas = {'index': np.array([3, 4, 9]), 'D': np.stack([np.linalg.matrix_power(inc[0], k) for k in (3, 1, 5)]),
            'tracked': np.array([True, True, True])}
    r = score_retracked(n, processed, meas)
    ref = apply_drop_policy(inc, processed)
    np.testing.assert_allclose(r['rel_increment'][:9], ref['rel_increment'][:9], atol=1e-12)
    assert (r['source'][:9] == ref['source'][:9]).all()
    assert r['drop_rate'] == pytest.approx(1 - processed[1:].mean())


def test_score_retracked_tracker_failure_is_motion_model():
    processed = np.array([True, False, True, True])
    D = np.eye(4)
    D[0, 3] = 0.02
    meas = {'index': np.array([2, 3]), 'D': np.stack([D, D]), 'tracked': np.array([True, False])}
    r = score_retracked(4, processed, meas)
    assert list(r['source']) == ['motion_model', 'tracked', 'motion_model'] and r['n_tracker_fail'] == 1
    np.testing.assert_allclose(r['rel_increment'][1][0, 3], 0.01)           # 0.02 over a 2-frame span
    np.testing.assert_allclose(r['rel_increment'][2][0, 3], 0.01)           # failure: previous scored increment


def test_base_motion_model_frames_are_promoted_to_lost_after_three(tmp_path):
    base = FakeOdo()
    base.fail = set(range(10, 16))
    est = {e.index: e for e in run(FaultInjectOdometry(base, [], lost_after=3), make_frames(tmp_path))}
    assert [est[i].tracking_state for i in range(10, 17)] == ['DEGRADED', 'DEGRADED', 'LOST', 'LOST', 'LOST', 'LOST', 'OK']
    assert est[16].source == 'tracked' and base.resets == 1                 # no supervisor reset: the base handled it
