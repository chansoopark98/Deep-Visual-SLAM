"""SE(3) log/exp, 30 Hz mailbox simulation and the drop scoring rule (plan v3 §4.2 F9, item 4.1)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.geometry.se3 import se3_exp, se3_log
from dvslam.runtime.replay import apply_drop_policy, simulate_mailbox


def rand_T(rng, rot=0.3, trans=0.01):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rng.normal(0, rot, 3)).as_matrix()
    T[:3, 3] = rng.normal(0, trans, 3)
    return T


def test_se3_log_exp_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = rand_T(rng, rot=1.0, trans=1.0)
        np.testing.assert_allclose(se3_exp(se3_log(T)), T, atol=1e-10)


def test_half_geodesic_squared_is_the_full_step():
    T = rand_T(np.random.default_rng(1))
    H = se3_exp(se3_log(T) / 2)
    np.testing.assert_allclose(H @ H, T, atol=1e-12)


def test_mailbox_fast_tracker_processes_everything():
    assert simulate_mailbox(np.full(100, 10.0)).all()


def test_mailbox_slow_tracker_drops_oldest():
    done = simulate_mailbox(np.full(100, 50.0))          # 50 ms per frame at 33.3 ms spacing
    assert done[0] and 0.3 < 1 - done.mean() < 0.6
    assert not (done[1:] & done[:-1]).all()


def test_drop_policy_no_drops_is_identity():
    rng = np.random.default_rng(2)
    inc = np.stack([rand_T(rng) for _ in range(20)])
    r = apply_drop_policy(inc, np.ones(21, bool))
    np.testing.assert_allclose(r['rel_increment'], inc)
    assert (r['source'] == 'tracked').all() and r['drop_rate'] == 0


def test_single_drop_splits_the_measured_motion_evenly():
    rng = np.random.default_rng(3)
    inc = np.stack([rand_T(rng) for _ in range(5)])
    processed = np.ones(6, bool)
    processed[2] = False                                 # frame 2 dropped
    r = apply_drop_policy(inc, processed)
    assert r['source'][1] == 'motion_model'
    np.testing.assert_allclose(r['rel_increment'][1], inc[0])         # constant velocity
    half = r['rel_increment'][2]
    np.testing.assert_allclose(half @ half, inc[1] @ inc[2], atol=1e-12)
    # the unscored output trajectory still lands on the measured pose at the next tracked frame
    np.testing.assert_allclose(r['T_wc'][3], inc[0] @ inc[1] @ inc[2], atol=1e-12)
    assert r['drop_rate'] == pytest.approx(0.2)
