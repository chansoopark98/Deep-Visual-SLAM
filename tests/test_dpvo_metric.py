"""Plan v3 items 1.1 / 1.2: DPVO + metric prior. CPU tests (default) cover slot/prior alignment on culling (A-N5),
the fixed-shape graph ops against their dynamic references, the g filter and the depth-worker schedule.
GPU tests (-m gpu) check the BA prior unary (λ → ∞ reproduces the prior), padded vs compact update operator
and a short run."""
import sys

import numpy as np
import pytest
import torch

from dvslam.odometry.dpvo_metric import (DPVO_ROOT, DepthWorker, GFilter, huber_location, shift_slots_loop,
                                         shift_slots_vec)

if str(DPVO_ROOT) not in sys.path:
    sys.path.insert(0, str(DPVO_ROOT))
from dpvo.scatter import scatter_softmax  # noqa: E402
from dpvo.static_graph import group_ids, static_neighbors, static_scatter_softmax  # noqa: E402

KEYFRAME_INDEX = 4
M, PMEM, MEM = 3, 36, 36


def _buffers(N, frames, device='cpu'):
    """Slot buffers tagged with the frame id they belong to (patch depth, prior, weight, src, maps)."""
    tag = torch.zeros(N, dtype=torch.float32)
    tag[:len(frames)] = torch.tensor(frames, dtype=torch.float32)
    patches = tag.view(N, 1, 1).repeat(1, M, 3)            # (N, M, 3) stand-in for pg.patches_
    prior_d = tag.view(N, 1).repeat(1, M) + 0.5
    prior_w = tag.view(N, 1).repeat(1, M) + 0.25
    src = np.zeros(N, dtype=np.int64)
    src[:len(frames)] = frames
    tstamps = src.copy()
    imap = torch.zeros(PMEM, M)
    fmap = torch.zeros(1, MEM, 2)
    for s, f in enumerate(frames):
        imap[s % PMEM] = f
        fmap[0, s % MEM] = f
    return dict(patches=patches, prior_d=prior_d, prior_w=prior_w, src=src, tstamps=tstamps, imap=imap, fmap=fmap)


def _shift(b, k, n, fn):
    fn(k, n, [b['tstamps'], b['patches'], b['prior_d'], b['prior_w'], b['src']],
       [(b['imap'], PMEM), (b['fmap'][0], MEM)])


@pytest.mark.parametrize('fn', [shift_slots_loop, shift_slots_vec])
def test_prior_buffers_follow_patch_slots_on_culling(fn):
    """A-N5: after any sequence of culls (k = n - 4), every slot's prior / weight / src stays with its patches."""
    rng = np.random.default_rng(0)
    N = 64
    frames = []
    b = _buffers(N, [])
    for t in range(200):
        n = len(frames)
        if n >= N - 1:
            break
        frames.append(t)
        b['tstamps'][n] = b['src'][n] = t
        b['patches'][n] = t
        b['prior_d'][n] = t + 0.5
        b['prior_w'][n] = t + 0.25
        b['imap'][n % PMEM] = t
        b['fmap'][0, n % MEM] = t
        n += 1
        if n > KEYFRAME_INDEX + 1 and rng.random() < 0.9:
            k = n - KEYFRAME_INDEX
            _shift(b, k, n, fn)
            del frames[k]
            n -= 1
        f = torch.tensor(frames, dtype=torch.float32)
        assert np.array_equal(b['tstamps'][:n], frames)
        assert torch.equal(b['patches'][:n, :, 0], f.view(-1, 1).expand(n, M))
        assert torch.equal(b['prior_d'][:n], f.view(-1, 1).expand(n, M) + 0.5)
        assert torch.equal(b['prior_w'][:n], f.view(-1, 1).expand(n, M) + 0.25)
        assert np.array_equal(b['src'][:n], frames)
        for s in range(max(0, n - 20), n):   # ring buffers: the live window
            assert b['imap'][s % PMEM, 0] == frames[s]
            assert b['fmap'][0, s % MEM, 0] == frames[s]


def test_vectorised_shift_equals_upstream_loop():
    frames = list(range(10, 30))
    a, b = _buffers(40, frames), _buffers(40, frames)
    _shift(a, 16, 20, shift_slots_loop)
    _shift(b, 16, 20, shift_slots_vec)
    for key in a:
        x, y = a[key], b[key]
        assert (np.array_equal(x, y) if isinstance(x, np.ndarray) else torch.equal(x, y)), key


def _neighbors_ref(kk, jj):
    """Python port of cuda_ba.neighbors (ba.cpp): previous / next edge of the same patch ordered by jj."""
    E = len(kk)
    ix, jx = np.full(E, -1), np.full(E, -1)
    groups = {}
    for e in range(E):
        groups.setdefault(int(kk[e]), []).append(e)
    for idx in groups.values():
        idx = sorted(idx, key=lambda e: int(jj[e]))     # stable, like std::stable_sort
        for p, e in enumerate(idx):
            ix[e] = idx[p - 1] if p > 0 else -1
            jx[e] = idx[p + 1] if p < len(idx) - 1 else -1
    return ix, jx


def _random_edges(rng, E_real, E_pad, n=30, M_=8):
    pairs = set()
    while len(pairs) < E_real:
        kk = int(rng.integers(0, n * M_))
        jj = int(rng.integers(max(kk // M_ - 12, 0), n))
        pairs.add((kk, jj))
    kk = np.array([p[0] for p in pairs] + [0] * E_pad)
    jj = np.array([p[1] for p in pairs] + [0] * E_pad)
    ii = kk // M_
    valid = np.array([True] * E_real + [False] * E_pad)
    perm = rng.permutation(len(kk))
    return [torch.as_tensor(x[perm]) for x in (ii, jj, kk, valid)]


def test_static_neighbors_match_upstream_on_real_edges():
    rng = np.random.default_rng(1)
    ii, jj, kk, valid = _random_edges(rng, 400, 120)
    g_kk, _, _, _ = group_ids(ii, jj, kk, valid, patch_ring=36 * 8)
    ix, jx = static_neighbors(g_kk, jj, valid, 36 * 8, torch.tensor(30 - 64))
    real = torch.nonzero(valid).view(-1)
    rix, rjx = _neighbors_ref(kk[real].numpy(), jj[real].numpy())
    remap = lambda a: np.where(a >= 0, real.numpy()[np.maximum(a, 0)], -1)   # noqa: E731
    assert np.array_equal(ix[real].numpy(), remap(rix))
    assert np.array_equal(jx[real].numpy(), remap(rjx))
    pad = ~valid
    assert (ix[pad] == -1).all() and (jx[pad] == -1).all()   # padding never links to anything


def test_static_softmax_matches_dynamic_on_real_edges():
    rng = np.random.default_rng(2)
    ii, jj, kk, valid = _random_edges(rng, 300, 50)
    g_kk, n_kk, g_ij, n_ij = group_ids(ii, jj, kk, valid, patch_ring=36 * 8)
    x = torch.randn(1, len(kk), 5, dtype=torch.float64)
    real = torch.nonzero(valid).view(-1)
    for g, n in ((g_kk, n_kk), (g_ij, n_ij)):
        out = static_scatter_softmax(x, g, n)
        _, inv = torch.unique(g[real], return_inverse=True)
        ref = scatter_softmax(x[:, real], inv, dim=1)
        torch.testing.assert_close(out[:, real], ref)
    # (i, j) pair groups equal upstream's ii * 12345 + jj grouping on real edges
    _, a = torch.unique(g_ij[real], return_inverse=True)
    _, b = torch.unique(ii[real] * 12345 + jj[real], return_inverse=True)
    assert torch.equal(a, b)


def test_g_filter_fixed_until_valid_then_tracks_robustly():
    f = GFilter(tau_s=1.0, min_samples=5, clip=0.05)
    for _ in range(5):
        f.add(0.1)
        assert f.g == 0.0
    assert f.valid
    for _ in range(300):
        f.add(0.1)
    assert abs(f.g - 0.1) < 1e-3
    f.add(10.0)                                   # one outlier moves g by at most alpha * clip
    assert f.g - 0.1 <= f.alpha * 0.05 + 1e-9
    off = GFilter(enabled=False)
    for _ in range(50):
        off.add(0.3)
    assert off.g == 0.0


def test_huber_location_ignores_outliers():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0.02, 0.01, 90), rng.normal(1.0, 0.1, 10)])
    assert abs(huber_location(x) - 0.02) < 0.01


def test_depth_worker_latest_wins_schedule():
    w, got = DepthWorker(lag=3), []
    for t in range(20):
        f = w.due(t)
        if f is not None:
            got.append((t, f))
        w.submit(t, t)
    assert got == [(t, t - 3) for t in range(3, 20, 3)]   # frames 0, 3, 6, … each arrive 3 frames later


# ------------------------------------------------------------------------------------------- GPU
@pytest.mark.gpu
def test_padded_update_operator_equals_upstream():
    from dpvo.net import Update
    from dpvo.static_graph import static_update
    torch.manual_seed(0)
    upd = Update(3).cuda().eval()
    rng = np.random.default_rng(4)
    ii, jj, kk, valid = [t.cuda() for t in _random_edges(rng, 500, 200)]
    E = len(kk)
    net = torch.randn(1, E, 384, device='cuda')
    inp = torch.randn(1, E, 384, device='cuda')
    corr = torch.randn(1, E, 2 * 49 * 9, device='cuda')
    real = torch.nonzero(valid).view(-1)
    with torch.no_grad():
        ref_net, (ref_d, ref_w, _) = upd(net[:, real], inp[:, real], corr[:, real], None, ii[real], jj[real], kk[real])
        out_net, out_d, out_w = static_update(upd, net, inp, corr, ii, jj, kk, valid, 36 * 8,
                                                 torch.tensor(30 - 64, device='cuda'))
    torch.testing.assert_close(out_net[:, real], ref_net, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_d[:, real], ref_d, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_w[:, real], ref_w, atol=1e-4, rtol=1e-4)


def _run(n_frames, **opts):
    from dvslam.data import redwood
    from dvslam.odometry.base import Frame, PinholeCamera
    from dvslam.odometry.dpvo_metric import DPVOMetricOdometry
    seq = redwood.load_sequence('bedroom', 'validation')
    odo = DPVOMetricOdometry(**opts)
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    ests = [odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i], seq.depth_paths[i]))
            for i in range(n_frames)]
    return odo, ests


@pytest.mark.gpu
def test_oracle_prior_with_huge_lambda_is_reproduced_by_ba():
    odo, _ = _run(80, patches=48, prior='oracle', prior_lambda=1e6, prior_lag=0)
    e = odo.slam
    n = e.n
    z, dp, w = e.pg.patches_[:n, :, 2, 1, 1], e.prior_d_[:n], e.prior_w_[:n]
    ok = (w > 0) & (dp > 0)
    ok[:max(n - e.cfg.OPTIMIZATION_WINDOW, 1)] = False       # slots the last BA optimised
    assert ok.sum() > 100
    r = (torch.log(1 / z) - torch.log(dp * float(np.exp(e.log_g))))[ok].abs()
    assert float(r.median()) < 1e-4 and float(r.max()) < 1e-2


@pytest.mark.gpu
def test_short_run_with_async_perturbed_prior():
    odo, ests = _run(120, patches=48, prior='perturbed', prior_lag=3, ring=1, e_max=4000)
    assert ests[0] is None and [e.index for e in ests[1:]] == list(range(1, 120))
    states = {e.tracking_state for e in ests[1:]}
    assert states <= {'BOOTSTRAP', 'OK', 'DEGRADED'} and 'OK' in states
    ex = odo.finalize()
    assert ex['ba_failures'] == 0 and ex['attach_direct'] > 0 and ex['edges_evicted'] > 0
    assert np.nanmax(ex['edges']) <= 4000
    steps = np.array([np.linalg.norm(e.rel_increment[:3, 3]) for e in ests[1:] if e.source == 'tracked'])
    assert 1e-4 < np.median(steps) < 0.05                    # metric scale: mm-to-cm steps at 30 Hz


@pytest.mark.gpu
def test_cuda_graph_replay_equals_eager_step():
    """1.2: the captured reproject → corr → update graph reproduces the eager computation on the live state.
    Run in a fresh process: cuda_graph=1 needs the stream-aware extensions (setup_stream.py) loaded first."""
    import subprocess
    code = r'''
import sys, torch
sys.path[:0] = [".", "tests"]
from test_dpvo_metric import _run
odo, _ = _run(60, patches=48, prior="perturbed", ring=1, e_max=4000, cuda_graph=1)
e = odo.slam
assert e._graph is not None
net0 = e.pg.net.clone()
with torch.no_grad():
    t1, w1 = [x.clone() for x in e._net_step()]
    n1 = e.pg.net.clone()
    e.pg.net.copy_(net0)
    t2, w2 = [x.clone() for x in e._net_step_graphed()]
    n2 = e.pg.net.clone()
# fp16 (autocast) scatter sums are order-dependent: a few elements differ by ~1e-2 px between any two runs
torch.testing.assert_close(t2, t1, atol=5e-2, rtol=0)
torch.testing.assert_close(w2, w1, atol=5e-2, rtol=0)
assert (t2 - t1).abs().median() < 1e-4
torch.testing.assert_close(n2.float(), n1.float(), atol=5e-2, rtol=5e-2)
print("OK")
'''
    from dvslam.odometry.dpvo_metric import REPO_ROOT
    r = subprocess.run([sys.executable, '-c', code], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0 and 'OK' in r.stdout, r.stderr[-2000:]
