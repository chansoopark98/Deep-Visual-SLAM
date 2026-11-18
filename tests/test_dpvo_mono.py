"""Plan v3 item 0.6: DPVO mono adapter. CPU tests cover the causal-hook indexing; the GPU smoke test runs DPVO."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvslam.odometry.dpvo_mono import DPVO_ROOT, causal_increment, pose7_to_matrix, resolve_pose_cw
from dvslam.geometry.se3 import relative_pose

KEYFRAME_INDEX = 4   # DPVO cfg.KEYFRAME_INDEX: keyframe() looks at slot n-4


def _random_T(rng, step=0.01):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix()
    T[:3, 3] = rng.normal(0, step, 3)
    return T


def _simulate_dpvo_graph(T_cw_true, cull):
    """Mimic DPVO.__call__/keyframe slot bookkeeping (dpvo.py:266-300) on exact poses.

    Yields (t, slot_of, poses_cw, deltas) after each frame, as the adapter sees it right after __call__.
    """
    slots, deltas = [], {}
    for t in range(len(T_cw_true)):
        slots.append(t)
        n = len(slots)
        if n > KEYFRAME_INDEX + 1 and cull(t):
            k = n - KEYFRAME_INDEX
            t0, t1 = slots[k - 1], slots[k]
            deltas[t1] = (t0, T_cw_true[t1] @ np.linalg.inv(T_cw_true[t0]))   # dP = P_k · P_{k-1}^-1
            del slots[k]
        slot_of = {ts: i for i, ts in enumerate(slots)}
        yield t, slot_of, np.stack([T_cw_true[ts] for ts in slots]), dict(deltas)


def test_pose7_matches_lietorch_convention():
    q = Rotation.from_euler('xyz', [10, -20, 30], degrees=True)
    T = pose7_to_matrix([1, 2, 3, *q.as_quat()])
    assert np.allclose(T[:3, :3], q.as_matrix()) and np.allclose(T[:3, 3], [1, 2, 3])


def test_causal_increment_is_inv_Twc_prev_times_Twc_cur():
    rng = np.random.default_rng(0)
    T_wc_a, T_wc_b = _random_T(rng, 1), _random_T(rng, 1)
    dT = causal_increment(np.linalg.inv(T_wc_a), np.linalg.inv(T_wc_b))
    assert np.allclose(dT, relative_pose(T_wc_a, T_wc_b))


@pytest.mark.parametrize('cull', [lambda t: True, lambda t: False, lambda t: t % 3 != 0])
def test_resolver_recovers_every_frame_and_prev_frame_stays_in_graph(cull):
    rng = np.random.default_rng(1)
    T_wc = [np.eye(4)]
    for _ in range(40):
        T_wc.append(T_wc[-1] @ _random_T(rng))
    T_cw = np.stack([np.linalg.inv(T) for T in T_wc])
    for t, slot_of, poses, deltas in _simulate_dpvo_graph(T_cw, cull):
        assert t in slot_of
        if t >= 1:
            assert t - 1 in slot_of, 'culling at slot n-4 must never remove t-1 at time t'
            dT = causal_increment(resolve_pose_cw(t - 1, slot_of, poses, deltas),
                                  resolve_pose_cw(t, slot_of, poses, deltas))
            assert np.allclose(dT, relative_pose(T_wc[t - 1], T_wc[t]), atol=1e-9)
        for s in range(t + 1):   # every past frame resolves, culled ones via the fixed deltas
            assert np.allclose(resolve_pose_cw(s, slot_of, poses, deltas), T_cw[s], atol=1e-9)


def test_resolver_follows_preinit_identity_deltas():
    # motion-probe skip (dpvo.py:443): delta[c-1] = (c-2, Id), the frame never gets a slot
    T_cw = pose7_to_matrix([0.1, 0, 0, 0, 0, 0, 1])
    pose = resolve_pose_cw(3, {0: 0, 1: 1}, np.stack([np.eye(4), T_cw]), {3: (2, np.eye(4)), 2: (1, np.eye(4))})
    assert np.allclose(pose, T_cw)


def test_scatter_shim_matches_reference():
    torch = pytest.importorskip('torch')
    sys.path.insert(0, str(DPVO_ROOT))
    from dpvo.scatter import scatter_softmax, scatter_sum
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 50, 8, generator=g)
    ix = torch.randint(0, 7, (50,), generator=g)
    s = scatter_sum(x, ix, dim=1)
    ref = torch.stack([x[:, ix == k].sum(1) for k in range(7)], 1)
    assert torch.allclose(s, ref, atol=1e-6)
    w = scatter_softmax(x, ix, dim=1)
    for k in range(7):
        assert torch.allclose(w[:, ix == k], torch.softmax(x[:, ix == k], dim=1), atol=1e-6)
    assert scatter_sum(x, ix, dim=1, dim_size=10).shape == (1, 10, 8)


@pytest.mark.gpu
def test_dpvo_smoke_on_validation_scene():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA')
    from dvslam.data import redwood
    from dvslam.odometry.base import Frame, PinholeCamera
    from dvslam.odometry.dpvo_mono import DPVOMonoOdometry
    try:
        seq = redwood.load_sequence('bedroom', 'validation')
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f'no Redwood data: {e}')
    odo = DPVOMonoOdometry(patches=48, seed=0)
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    n = 60
    ests = [odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i])) for i in range(n)]
    assert ests[0] is None and [e.index for e in ests[1:]] == list(range(1, n))
    assert any(e.tracking_state == 'OK' for e in ests[1:]), 'DPVO never initialised in 60 frames'
    for e in ests[1:]:
        assert e.rel_increment.shape == (4, 4) and np.isfinite(e.rel_increment).all()
        if e.tracking_state == 'BOOTSTRAP':
            assert e.source == 'motion_model' and np.allclose(e.rel_increment, np.eye(4))
    extra = odo.finalize()
    assert extra['final_T_wc'].shape == (n, 4, 4)
    assert np.isfinite(extra['rho'][extra['rho'] == extra['rho']]).all()
    assert (extra['edges'][1:][~extra['preinit_pair'][1:]] > 0).all()
