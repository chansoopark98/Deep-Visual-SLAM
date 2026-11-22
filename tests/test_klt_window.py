"""Plan v3 item 1.3: KLTWindowOdometry (window BA + prior factor + g) on a rendered synthetic point cloud."""
import cv2
import numpy as np
import pytest
from scipy.optimize import least_squares as _real_least_squares
from scipy.spatial.transform import Rotation

import dvslam.odometry.klt_window as kw
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import Frame, PinholeCamera
from dvslam.odometry.klt_window import KLTWindowOdometry

W, H, N_FRAMES = 320, 240, 40
K = np.array([[262.5, 0, 159.5], [0, 262.5, 119.5], [0, 0, 1]])


def _trajectory():
    T = np.tile(np.eye(4), (N_FRAMES, 1, 1))
    k = np.arange(N_FRAMES)
    T[:, :3, 3] = np.c_[0.006 * k, 0.001 * np.sin(k / 5), 0.002 * k]
    T[:, :3, :3] = Rotation.from_euler('xyz', np.c_[0.001 * k, 0.002 * k, 0.0005 * k]).as_matrix()
    return T


def _render(T_wc, pts):
    T_cw = np.linalg.inv(T_wc)
    Xc = pts @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv = Xc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    img, depth = np.full((H, W), 40.0), np.zeros((H, W), np.uint16)
    yy, xx = np.mgrid[-4:5, -4:5]
    for k in np.argsort(-Xc[:, 2]):
        (u, v), z = uv[k], Xc[k, 2]
        if z <= 0 or not (5 <= u < W - 5 and 5 <= v < H - 5):
            continue
        iu, iv = int(u), int(v)
        blob = 40 + 180 * np.exp(-((xx + iu - u) ** 2 + (yy + iv - v) ** 2) / (2 * 1.2 ** 2))
        np.copyto(img[iv - 4:iv + 5, iu - 4:iu + 5], blob, where=blob > 41)
        depth[iv - 3:iv + 4, iu - 3:iu + 4] = round(z * 1000)
    return np.clip(img, 0, 255).astype(np.uint8), depth


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1.5, -1.2, 1.0], [1.7, 1.2, 4.0], (260, 3))
    T = _trajectory()
    root = tmp_path_factory.mktemp('synthetic_window')
    frames = []
    for i, T_wc in enumerate(T):
        img, depth = _render(T_wc, pts)
        cv2.imwrite(str(root / f'{i:06d}.png'), img)
        cv2.imwrite(str(root / f'{i:06d}_d.png'), depth)
        frames.append(Frame(i, i / 30, root / f'{i:06d}.png', root / f'{i:06d}_d.png'))
    gt = np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, N_FRAMES)])
    return frames, gt


def _run(frames, **opts):
    odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150, **opts)
    odo.reset(PinholeCamera(K, W, H))
    out = [odo.step(f) for f in frames]
    assert out[0] is None and [e.index for e in out[1:]] == list(range(1, len(frames)))
    return out[1:], odo


def test_oracle_prior_recovers_metric_increments(scene):
    frames, gt = scene
    est, odo = _run(frames, prior='oracle')
    assert all(e.source == 'tracked' for e in est)
    assert sum(e.is_keyframe for e in est) >= 5 and len(odo.kfs) == 6       # the window slid
    inc = np.stack([e.rel_increment for e in est])
    err_mm = np.linalg.norm(inc[:, :3, 3] - gt[:, :3, 3], axis=1) * 1e3
    assert err_mm.max() < 0.3, err_mm.max()
    assert abs(odo.g) < 0.01


def test_prior_bias_sets_metric_scale_and_window_averages_jitter(scene):
    frames, gt = scene
    g = np.linalg.norm(gt[:, :3, 3], axis=1)

    def ratios(**o):
        est, _ = _run(frames, prior='perturbed', **o)
        p = np.linalg.norm(np.stack([e.rel_increment[:3, 3] for e in est]), axis=1)
        return p / g

    r = ratios(prior_bias=0.1, prior_jitter=0.0) / ratios(prior_bias=0.0, prior_jitter=0.0)
    assert abs(np.median(r) - 1.1) < 0.005, np.median(r)      # the prior defines the metric scale
    # per-frame prior jitter (5 %): the window BA beats the 0.7 path (prior back-projection per keyframe) on both
    # the per-pair scale spread and the mean scale error, averaged over 3 prior seeds
    def stats(**o):
        lr = [np.log(ratios(prior_bias=0.0, prior_jitter=0.05, prior_seed=s, **o)) for s in (1, 2, 3)]
        return np.mean([x.std() for x in lr]), np.mean([abs(x.mean()) for x in lr])
    ba, base = stats(), stats(ba=False, kf_align=False)
    assert ba[0] < base[0] and ba[1] < base[1], (ba, base)


def test_ba_jacobian_matches_finite_differences(scene, monkeypatch):
    frames, _ = scene
    calls = []

    def spy(fun, x0, jac, **kw_):
        calls.append((fun, jac, x0.copy()))
        return _real_least_squares(fun, x0, jac=jac, **kw_)

    monkeypatch.setattr(kw, 'least_squares', spy)
    _run(frames[:16], prior='perturbed', prior_jitter=0.05)
    fun, jac, x0 = calls[-1]
    x = x0 + np.random.default_rng(0).normal(0, 1e-3, len(x0))
    J = jac(x).toarray()
    eps = 1e-7
    Jn = np.stack([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(len(x))], 1)
    # rotation columns use the small-angle (right-Jacobian ≈ I) approximation
    np.testing.assert_allclose(J, Jn, atol=2e-2 * np.abs(Jn).max(), rtol=0)
    np.testing.assert_allclose(J[:, -1], Jn[:, -1], rtol=1e-5, atol=1e-6)


def test_g_scales_output_translation_only(scene):
    frames, _ = scene
    est, odo = _run(frames[:12], prior='oracle')
    hi = odo.head_inputs
    np.testing.assert_allclose(est[-1].rel_increment[:3, 3], hi['T_raw'][:3, 3] * np.exp(-hi['g']), rtol=1e-12)
    np.testing.assert_allclose(est[-1].rel_increment[:3, :3], hi['T_raw'][:3, :3], atol=1e-15)


def test_ba_guard_keeps_runaway_depth_finite(scene):
    """EXP-003: a landmark whose log inverse depth ran to x = -800 (zero parallax, no prior) makes exp(-x) overflow in
    the BA residual; with ba_guard the residual stays finite and the landmark ends inside the depth range or
    unmapped."""
    frames, _ = scene

    def runaway(**o):
        _, odo = _run(frames[:16], prior='perturbed', prior_jitter=0.05, **o)
        lms = odo._ba_problem()[1]
        odo.lm_x[lms[0]] = -800.0
        return odo, lms[0]

    odo, j = runaway()
    with np.errstate(over='raise'), pytest.raises(FloatingPointError):
        odo._bundle_adjust()
    odo, j = runaway(ba_guard=True)
    with np.errstate(over='raise'):
        odo._bundle_adjust()
    x = odo.lm_x[j]
    assert np.isnan(x) or -np.log(kw.BA_DEPTH_HI) <= x <= -np.log(kw.BA_DEPTH_LO), x
    assert odo.n_ba_reject == 0


def test_cv_threads_follow_omp_cap(monkeypatch):
    """EXP-003: OpenCV's pthreads pool ignores OMP_NUM_THREADS; reset() applies the cap (or cv_threads)."""
    before = cv2.getNumThreads()
    try:
        monkeypatch.setenv('OMP_NUM_THREADS', '3')
        KLTWindowOdometry(prior='oracle').reset(PinholeCamera(K, W, H))
        assert cv2.getNumThreads() == 3
        KLTWindowOdometry(prior='oracle', cv_threads=2).reset(PinholeCamera(K, W, H))
        assert cv2.getNumThreads() == 2
    finally:
        cv2.setNumThreads(before)


def test_fallback_reason_is_recorded(scene, tmp_path):
    """A frame on which every KLT track is lost falls back to the motion model with reason few_mapped."""
    frames, _ = scene
    blank = tmp_path / 'blank.png'
    cv2.imwrite(str(blank), np.full((H, W), 40, np.uint8))
    fr = list(frames[:14])
    fr[10] = Frame(10, fr[10].timestamp, blank, fr[10].depth_path)
    est, odo = _run(fr, prior='oracle')
    fail = odo.finalize()['fail']
    assert est[9].source == 'motion_model' and fail[9] == 'few_mapped', (est[9].source, fail[9])
    assert all(f == '' for f in fail[:9])


def test_jump_gate_does_not_lock_onto_motion_model(scene):
    """EXP-003: after a wrong increment (0.2 m) was repeated by the motion model, a correct solve differs from it by
    more than max_jump_m. With jump_ref_tracked the gate is skipped after an untracked frame, so tracking resumes."""
    frames, gt = scene

    def resume(**o):
        odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150,
                                prior='oracle', **o)
        odo.reset(PinholeCamera(K, W, H))
        for f in frames[:10]:
            odo.step(f)
        odo.last_inc_ba = np.eye(4)
        odo.last_inc_ba[2, 3] = 0.2                     # stale motion-model increment
        odo.fail_run = 1                                # the previous pair came from the motion model
        return odo.step(frames[10])

    assert resume().source == 'motion_model'
    e = resume(jump_ref_tracked=True)
    assert e.source == 'tracked'
    assert np.linalg.norm(e.rel_increment[:3, 3] - gt[9, :3, 3]) < 1e-3


def test_photo_refine_recovers_increment_on_textured_plane():
    """EXP-007 klt_photo.refine: from a zero-motion start, LM on a rendered textured plane recovers X1 = R X0 + t."""
    from dvslam.odometry.klt_photo import refine, select_points
    n, f, c = 256, 190.0, 127.5
    Kp = np.array([[f, 0, c], [0, f, c], [0, 0, 1]])
    uu, vv = np.meshgrid(np.arange(n), np.arange(n))
    ray = np.stack([(uu - c) / f, (vv - c) / f, np.ones((n, n))], -1)

    def render(R, t):                                   # camera X_c = R X_w + t, plane Z_w = 2 m
        rw, cw = ray @ R, -R.T @ t
        d = (2.0 - cw[2]) / rw[..., 2]
        P = cw + d[..., None] * rw
        img = 0.5 + 0.2 * np.sin(9 * P[..., 0]) * np.cos(7 * P[..., 1]) + 0.1 * np.sin(23 * P[..., 0] + 11 * P[..., 1])
        return img.astype(np.float32), d

    R1 = Rotation.from_euler('xyz', [0.3, -0.4, 0.2], degrees=True).as_matrix()
    t1 = np.array([0.02, -0.01, 0.03])
    I0, D0 = render(np.eye(3), np.zeros(3))
    I1, _ = render(R1, t1)
    M = np.ones((n, n), bool)
    pts = select_points(I0, M, 8, 0.002)
    z = D0[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    R, t, info = refine(1.1 * I0 + 0.05, I1, M, pts, z, Kp, np.eye(3), np.zeros(3), levels=3, iters=30)
    assert info['cost'] < info['cost0'] and info['n'] > 200
    assert np.linalg.norm(t - t1) < 5e-4, t - t1                       # < 0.5 mm of a 37 mm step
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ R1).as_rotvec())) < 0.02


def test_flow_tracks_follow_cached_flow(scene):
    """EXP-007 flow_cache: tracks move by the sampled forward flow; a backward-flow mismatch drops the track."""
    frames, _ = scene
    odo = KLTWindowOdometry(prior='oracle')
    odo.reset(PinholeCamera(K, W, H))
    odo.px = np.array([[50.0, 60.0], [100.5, 80.25], [200.0, 150.0]], np.float32)
    odo.anchor_idx, odo.anchor_px = np.zeros(3, np.int64), odo.px.astype(np.float64)
    odo.X, odo.par, odo.uid = np.full((3, 3), np.nan), np.zeros(3), np.arange(3)
    fwd = np.zeros((2, H, W), np.float32)
    fwd[0], fwd[1] = 1.5, -0.5
    bwd = -fwd.copy()
    bwd[:, 140:160, 195:210] = 0                        # third track lands here: backward flow disagrees by 1.6 px

    class Fake:
        def pair(self, path, prev_gray, gray):
            return fwd, bwd
    odo.flow, odo._cur_path = Fake(), frames[1].image_path
    p0, p1 = odo._track(None)
    np.testing.assert_allclose(p1, [[51.5, 59.5], [102.0, 79.75]], atol=1e-5)
    assert len(p0) == 2 and list(odo.uid) == [0, 1]


def test_hybrid_flow_reacquires_only_lost_tracks(scene):
    """EXP-008 flow_mode='hybrid': LK survivors keep their LK position; when LK keeps < flow_hybrid_frac of the
    tracks, the lost ones are re-acquired from the flow; with enough LK survivors the flow is not used at all."""
    frames, _ = scene
    odo = KLTWindowOdometry(prior='oracle', flow_mode='hybrid', flow_hybrid_frac=0.7)
    odo.reset(PinholeCamera(K, W, H))
    fwd = np.zeros((2, H, W), np.float32)
    fwd[0], fwd[1] = 1.5, -0.5

    class Fake:
        def pair(self, path, prev_gray, gray):
            return fwd, -fwd

    def setup(lk_ok):
        odo.px = np.array([[50.0, 60.0], [100.0, 80.0], [200.0, 150.0]], np.float32)
        odo.anchor_idx, odo.anchor_px = np.zeros(3, np.int64), odo.px.astype(np.float64)
        odo.X, odo.par, odo.uid = np.full((3, 3), np.nan), np.zeros(3), np.arange(3)
        odo.flow, odo._cur_path = Fake(), frames[1].image_path
        odo._lk = lambda gray: (odo.px + np.float32([3.0, 0.0]), np.array(lk_ok))

    setup([True, False, False])                         # LK keeps 1/3 < 0.7: re-acquire the other two from the flow
    _, p1 = odo._track(None)
    np.testing.assert_allclose(p1, [[53.0, 60.0], [101.5, 79.5], [201.5, 149.5]], atol=1e-5)
    assert odo._flow_used
    setup([True, True, True])
    _, p1 = odo._track(None)
    assert not odo._flow_used and len(p1) == 3


def test_wide_canvas_camera_cap_and_prior_embedding():
    """EXP-011 wide=768: K shifted to the canvas centre, field-of-view cap by angle, 512 prior embedded centrally."""
    from dvslam.data import tumvi
    odo = KLTWindowOdometry(prior='oracle', wide=768, wide_fov_deg=60.0)
    odo.reset(PinholeCamera(tumvi.K_OUT, 512, 512))
    assert odo.size == (768, 768) and odo.K[0, 2] == 383.5 and odo.K[0, 0] == 190.0
    m = odo.detect_mask
    assert m[383, 383] == 255 and m[0, 0] == 0                         # corner of the canvas is at 70.7 deg
    edge = round(383.5 + 190.0 * np.tan(np.radians(59.0)))
    assert m[383, edge] == 255 and m[383, 767] == 0                     # 59 deg inside, 63.7 deg outside

    class Ones:
        def predict(self, frame):
            return np.ones((512, 512), np.float32), None
    odo.prior = Ones()
    d = odo._prior_map(Frame(0, 0.0, 'x.png'))
    assert d.shape == (768, 768) and d[128:640, 128:640].min() == 1 and d[:128].max() == 0 and d[:, 640:].max() == 0
    with pytest.raises(ValueError):
        KLTWindowOdometry(prior='oracle', wide=768).reset(PinholeCamera(K, W, H))


def _run_lag(frames, **opts):
    odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150, **opts)
    odo.reset(PinholeCamera(K, W, H))
    returned = [odo.step(f) for f in frames]
    return returned, odo.flush(), odo


def test_out_lag_default_unchanged_and_alignment(scene):
    """EXP-013 out_lag (conditional, default 0): lag 0 is the old output; with lag L the first L+1 steps return None,
    flush() returns the last L pairs, and every pair 1..n-1 is emitted once, in order, with its own timestamp."""
    frames, _ = scene
    base, _ = _run(frames, prior='oracle')
    ret0, fl0, _ = _run_lag(frames, prior='oracle', out_lag=0)
    assert fl0 == []
    np.testing.assert_array_equal(np.stack([e.rel_increment for e in base]),
                                  np.stack([e.rel_increment for e in ret0[1:]]))
    for mode in ('smooth', 'window'):
        ret, fl, _ = _run_lag(frames, prior='oracle', out_lag=2, out_lag_mode=mode)
        assert all(r is None for r in ret[:3]) and all(r is not None for r in ret[3:]) and len(fl) == 2
        est = [r for r in ret if r is not None] + fl
        assert [e.index for e in est] == list(range(1, len(frames)))
        assert [e.timestamp for e in est] == [f.timestamp for f in frames[1:]]


def test_out_lag_window_mode_applies_only_later_window_changes(scene):
    """window mode = causal increment × (change of the window's relative pose since it was emitted): with no BA the
    keyframe poses never move, so the lagged output equals the causal one; with BA it differs (and stays sane)."""
    frames, gt = scene
    base, _ = _run(frames, prior='oracle', ba=False, kf_align=False)
    ret, fl, _ = _run_lag(frames, prior='oracle', ba=False, kf_align=False, out_lag=2, out_lag_mode='window')
    got = np.stack([e.rel_increment for e in [r for r in ret if r is not None] + fl])
    np.testing.assert_allclose(got, np.stack([e.rel_increment for e in base]), atol=1e-12)
    ret, fl, _ = _run_lag(frames, prior='oracle', out_lag=2, out_lag_mode='window')
    inc = np.stack([e.rel_increment for e in [r for r in ret if r is not None] + fl])
    assert np.linalg.norm(inc[:, :3, 3] - gt[:, :3, 3], axis=1).max() * 1e3 < 1.0


def test_wide_rejects_cached_flow_directory():
    """A flow cache holds flow of the standard 512 image; wide tracks live on the canvas, so the pair is refused.
    Online RAFT sees the canvas images and stays allowed."""
    with pytest.raises(ValueError, match='wide'):
        KLTWindowOdometry(prior='oracle', wide=768, flow_cache='/some/flow/dir')
    KLTWindowOdometry(prior='oracle', wide=768, flow_cache='raft_small')
    KLTWindowOdometry(prior='oracle', wide_fov_deg=60.0, flow_cache='/some/flow/dir')   # standard canvas: fine


def test_wide_prior_gets_the_standard_camera(monkeypatch):
    """wide: the prior predicts on the standard image, so it is built with K_OUT and 512 x 512 (live SSLDepth
    rejects any other K), as the cached prior whose maps are embedded in the canvas by _prior_map."""
    from dvslam.data import tumvi
    got = []
    monkeypatch.setattr(kw, 'make_prior', lambda name, K, *a, **k: got.append((K.copy(), k.get('hw'))) or None)
    odo = KLTWindowOdometry(prior='ssl', wide=768)
    odo.reset(PinholeCamera(tumvi.K_OUT, 512, 512))
    assert odo.K[0, 2] == 383.5
    np.testing.assert_array_equal(got[0][0], tumvi.K_OUT)
    assert got[0][1] == (512, 512)


def test_cv_threads_zero_leaves_opencv_pool(monkeypatch):
    """cv_threads=0 keeps OpenCV's current pool (it does not call cv2.setNumThreads(0), which disables threading)."""
    before = cv2.getNumThreads()
    try:
        cv2.setNumThreads(3)
        monkeypatch.setenv('OMP_NUM_THREADS', '2')
        KLTWindowOdometry(prior='oracle', cv_threads=0).reset(PinholeCamera(K, W, H))
        assert cv2.getNumThreads() == 3
    finally:
        cv2.setNumThreads(before)


def test_option_values_checked_and_lk_defaults_unchanged():
    """Unknown photo / flow_mode / out_lag_mode values are a ValueError; lk_iters / lk_eps default to the former
    hard-coded LK stop criteria (30 iterations, eps 0.001)."""
    for kw_bad in ({'photo': 'on'}, {'flow_mode': 'blend'}, {'out_lag_mode': 'rts'}):
        with pytest.raises(ValueError, match=next(iter(kw_bad))):
            KLTWindowOdometry(prior='oracle', **kw_bad)
    assert KLTWindowOdometry(prior='oracle').lk['criteria'][1:] == (30, 0.001)
    assert KLTWindowOdometry(prior='oracle', lk_iters=50, lk_eps=0.01).lk['criteria'][1:] == (50, 0.01)


class _UnitNorm:
    """Stand-in for klt_photo.PhotoNorm on the synthetic scene: image / 255, all pixels valid, unit exposure."""

    def __call__(self, image_path, gray, ratio):
        return gray.astype(np.float32) / 255 * ratio, np.ones(gray.shape, bool)

    def exposure(self, image_path):
        return 1.0


def test_photo_bidir_combines_both_directions(scene, monkeypatch):
    """EXP-018 photo_bidir: the t -> t-1 refinement runs on every refined pair, the output is the midpoint of the two
    accepted refinements (differs from forward-only; here closer to GT, 0.07 vs 0.11 mm mean), and photo_ref_max > 1
    is a ValueError."""
    frames, gt = scene

    def run(**o):
        odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150,
                                prior='oracle', photo='all', photo_cell=2, photo_grad=0.002, **o)
        odo.photo_norm = _UnitNorm()
        odo.reset(PinholeCamera(K, W, H))
        est = [odo.step(f) for f in frames][1:]
        return np.stack([e.rel_increment for e in est]), np.array(odo.diag['photo'])

    back = []
    real_back = KLTWindowOdometry._photo_back
    monkeypatch.setattr(KLTWindowOdometry, '_photo_back', lambda self, *a: back.append(real_back(self, *a)) or back[-1])
    fwd, ph_f = run()
    assert not back
    bid, ph_b = run(photo_bidir=True)
    assert len(back) == (ph_b > 0).sum() and (ph_f == 1).sum() >= 10 and (ph_b == 1).sum() >= 10
    both = [k for k, (T, info) in enumerate(back) if info is not None and info['cost'] < info['cost0']]
    assert len(both) >= 10
    err = lambda inc: np.linalg.norm(inc[:, :3, 3] - gt[:, :3, 3], axis=1) * 1e3
    assert np.abs(bid - fwd).max() > 1e-7                          # the backward result enters the output
    assert err(bid).max() < 0.5 and err(bid).mean() < err(fwd).mean(), (err(bid).mean(), err(fwd).mean())
    with pytest.raises(ValueError, match='photo_bidir'):
        KLTWindowOdometry(prior='oracle', photo_bidir=True, photo_ref_max=2)


def test_subset_mask_regions_and_random_control():
    """G0c subset_mask: left / right and blocks_a / blocks_b are complementary halves (blocks: block×block
    checkerboard); random_a / random_b keep half of the cells each, independently (overlap ≈ a quarter of the image),
    constant inside a cell, reproducible per seed; unknown modes are a ValueError (also in the tracker)."""
    m = {k: kw.subset_mask(k, 512, 512) for k in kw.PT_SUBSETS[1:]}
    for a, b in (('left', 'right'), ('blocks_a', 'blocks_b')):
        assert m[a].mean() == 0.5 and not (m[a] & m[b]).any() and (m[a] | m[b]).all()
    assert m['left'][:, :256].all() and not m['left'][:, 256:].any()
    assert m['blocks_a'][0, 0] and m['blocks_a'][127, 127] and not m['blocks_a'][0, 128] and m['blocks_a'][128, 128]
    ra, rb = m['random_a'], m['random_b']
    assert 0.45 < ra.mean() < 0.55 and 0.45 < rb.mean() < 0.55 and 0.2 < (ra & rb).mean() < 0.3
    assert (ra.reshape(32, 16, 32, 16) == ra[::16, ::16][:, None, :, None]).all()          # 16 px cells
    assert (kw.subset_mask('random_a', 512, 512) == ra).all()
    assert not (kw.subset_mask('random_a', 512, 512, seed=1) == ra).all()
    assert kw.subset_mask('random_b', 100, 70, cell=16).shape == (70, 100)
    with pytest.raises(ValueError, match='pt_subset'):
        kw.subset_mask('odd', 64, 64)
    with pytest.raises(ValueError, match='pt_subset'):
        KLTWindowOdometry(prior='oracle', pt_subset='even')


def test_pt_subset_restricts_tracks_and_photometric_points(scene, monkeypatch):
    """pt_subset: off leaves the masks untouched; left keeps every track and every photometric point (both
    directions) in the left half and halves n_features; random_a detects only in its cells but keeps a track that
    moves out of them."""
    import dvslam.odometry.klt_photo as kp
    frames, _ = scene
    seen = []
    real = kp.refine_prepared
    monkeypatch.setattr(kp, 'refine_prepared', lambda P0, P1, ratio, pts, *a: seen.append(pts) or
                        real(P0, P1, ratio, pts, *a))

    def run(**o):
        odo = KLTWindowOdometry(min_distance=10, lk_win=15, kf_max=5, window=6, g_min_m=2, n_features=150,
                                prior='oracle', photo='all', photo_cell=2, photo_grad=0.002, photo_bidir=True, **o)
        odo.photo_norm = _UnitNorm()
        odo.reset(PinholeCamera(K, W, H))
        px, anchors = [], []
        for f in frames:
            odo.step(f)
            px.append(odo.px.copy())
            anchors.append(odo.anchor_px.copy())
        return odo, np.concatenate(px), np.concatenate(anchors)

    odo, _, _ = run()
    assert odo._pt_mask is None and odo.track_mask is None and odo.detect_mask is None and odo.n_features == 150
    seen.clear()
    odo, px, _ = run(pt_subset='left')
    assert odo.n_features == 75 and len(px) > 0 and px[:, 0].max() < W / 2 + 1   # cornerSubPix may move 1 px
    assert len(seen) >= 20 and max(p[:, 0].max() for p in seen) < W / 2
    assert (np.array(odo.diag['photo']) == 1).sum() >= 10
    odo, px, anchors = run(pt_subset='random_a', pt_cell=32)
    mask = odo._pt_mask
    inside = lambda p: mask[np.round(p[:, 1]).astype(int).clip(0, H - 1), np.round(p[:, 0]).astype(int).clip(0, W - 1)]
    assert odo.track_mask is None and inside(anchors).all() and not inside(px).all()


def test_prior_obs_weight_bounds_and_direction():
    """S4: w = clip(ref / θ, w_min, w_max); low parallax → stronger prior, degenerate θ → w_max."""
    f = kw.prior_obs_weight
    assert f(1.0, 1.0, 0.5, 2.0) == 1.0
    assert f(0.8, 1.0, 0.5, 2.0) == pytest.approx(1.25) and f(1.6, 1.0, 0.5, 2.0) == pytest.approx(0.625)
    assert f(0.1, 1.0, 0.5, 2.0) == 2.0 and f(50.0, 1.0, 0.5, 2.0) == 0.5          # bounded
    assert f(0.0, 1.0, 0.5, 2.0) == 2.0 and f(float('nan'), 1.0, 0.5, 2.0) == 2.0
    th = np.geomspace(0.01, 100, 50)
    w = np.array([f(t, 1.3, 0.25, 4.0) for t in th])
    assert (np.diff(w) <= 0).all() and w.min() == 0.25 and w.max() == 4.0


def test_window_parallax_two_keyframes():
    """S4: a landmark 2 m in front of the host seen from a keyframe 0.2 m to the side has ray angle atan(0.1);
    a landmark seen only by its host has 0; the median runs over the prior landmarks only."""
    R = np.stack([np.eye(3), np.eye(3)])
    t = np.array([[0.0, 0, 0], [0.2, 0, 0]])
    bear = np.array([[0.0, 0, 1], [0.0, 0, 1], [0.0, 0, 1]])
    x = -np.log(np.array([2.0, 2.0, 4.0]))
    host = np.array([0, 0, 0])
    o_lm, o_kf = np.array([0, 0, 1, 2, 2]), np.array([0, 1, 0, 0, 1])
    a1, a2 = np.degrees(np.arctan(0.1)), np.degrees(np.arctan(0.05))
    assert kw.window_parallax(R, t, bear, x, host, o_lm, o_kf, np.array([0])) == pytest.approx(a1)
    assert kw.window_parallax(R, t, bear, x, host, o_lm, o_kf, np.array([0, 1, 2])) == pytest.approx(a2)
    assert kw.window_parallax(R, t * 7.0, bear, x - np.log(7.0), host, o_lm, o_kf, np.array([0, 2])) == \
        pytest.approx((a1 + a2) / 2)                                                   # scale-free
    assert np.isnan(kw.window_parallax(R, t, bear, x, host, o_lm, o_kf, np.array([], int)))


def test_window_parallax_extreme_depth_is_finite():
    """S4: log inverse depths far outside the ba_guard range (exp overflow, zero depth) are clipped to it, so the
    angle stays finite; in-range depths are unchanged."""
    R = np.stack([np.eye(3), np.eye(3)])
    t = np.array([[0.0, 0, 0], [0.2, 0, 0]])
    bear = np.array([[0.0, 0, 1], [0.0, 0, 1]])
    host, o_lm, o_kf = np.array([0, 0]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        theta = kw.window_parallax(R, t, bear, np.array([-1000.0, 800.0]), host, o_lm, o_kf, np.array([0, 1]))
    far, near = np.degrees(np.arctan(0.2 / kw.BA_DEPTH_HI)), np.degrees(np.arctan(0.2 / kw.BA_DEPTH_LO))
    assert np.isfinite(theta) and theta == pytest.approx((far + near) / 2)
    x = -np.log(np.array([2.0, 4.0]))
    assert kw.window_parallax(R, t, bear, x, host, o_lm, o_kf, np.array([0])) == \
        pytest.approx(np.degrees(np.arctan(0.1)))


def test_photo_feat_and_backend_option_checks():
    """photo_feat / photo_backend need photo on; photo_feat needs photo_levels=1 (the net's training level)."""
    for o in ({'photo_backend': 'torch'}, {'photo_feat': 'x.pt', 'photo_backend': 'torch', 'photo_levels': 1}):
        with pytest.raises(ValueError, match='photo=all'):
            KLTWindowOdometry(prior='oracle', **o)
    with pytest.raises(ValueError, match='photo_levels=1'):
        KLTWindowOdometry(prior='oracle', photo='all', photo_feat='x.pt', photo_backend='torch', photo_levels=2)
    odo = KLTWindowOdometry(prior='oracle', photo='all', photo_feat='x.pt', photo_backend='torch', photo_levels=1)
    assert odo.photo_feat == 'x.pt' and odo._photo_net is None          # the checkpoint is loaded lazily


@pytest.mark.parametrize('kind', ['huber', 'cauchy'])
def test_robust_log_residual_cost_and_derivative(kind):
    """S4: Σψ² = 2c²Σρ(r / c), ψ keeps the sign, ψ ≈ r near 0, dψ/dr matches central differences."""
    c = 0.05
    r = np.r_[-0.6, -0.08, -0.05, -0.02, -1e-9, 0.0, 1e-9, 0.03, 0.049, 0.051, 0.2, 1.5]
    psi, d = kw.robust_log_residual(r, c, kind)
    s = r / c
    rho = np.where(np.abs(s) <= 1, s * s / 2, np.abs(s) - 0.5) if kind == 'huber' else np.log1p(s * s) / 2
    np.testing.assert_allclose(psi ** 2, 2 * c * c * rho, rtol=1e-12, atol=1e-30)
    assert (np.sign(psi) == np.sign(r)).all() and (np.abs(psi) <= np.abs(r) + 1e-15).all()
    np.testing.assert_allclose(psi[np.abs(r) < 1e-6], r[np.abs(r) < 1e-6], rtol=1e-9, atol=1e-20)
    if kind == 'huber':
        np.testing.assert_allclose(psi[np.abs(s) <= 1], r[np.abs(s) <= 1], rtol=1e-12)
    eps = 1e-7
    dn = (kw.robust_log_residual(r + eps, c, kind)[0] - kw.robust_log_residual(r - eps, c, kind)[0]) / (2 * eps)
    smooth = np.abs(np.abs(s) - 1) > 1e-3                 # huber's ψ' jumps at |s| = 1 only in its 2nd derivative
    np.testing.assert_allclose(d[smooth], dn[smooth], rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match='kind'):
        kw.robust_log_residual(r, c, 'tukey')


def test_s4_options_default_identity_jacobian_and_log(scene, monkeypatch):
    """S4: prior_obs with w fixed to 1 reproduces the default run bit for bit (and logs θ_w per BA); the weight and
    ψ change the run; the BA Jacobian with both options matches finite differences; bad prior_loss is an error."""
    frames, _ = scene

    def incs(**o):
        est, odo = _run(frames, prior='perturbed', prior_jitter=0.05, prior_per_kf=False, **o)
        return np.stack([e.rel_increment for e in est]), odo

    base, odo0 = incs()
    assert odo0.prior_obs_log == []
    same, odo1 = incs(prior_obs=True, prior_obs_min=1.0, prior_obs_max=1.0)
    np.testing.assert_array_equal(same, base)
    log = np.array(odo1.prior_obs_log)
    assert len(log) >= 5 and (log[:, 2] == 1.0).all() and (log[:, 1] > 0).all()
    ref = float(np.median(log[:, 1]))
    obs, odo2 = incs(prior_obs=True, prior_obs_ref=ref, prior_obs_min=0.25, prior_obs_max=4.0)
    w = np.array(odo2.prior_obs_log)[:, 2]
    assert ((w >= 0.25) & (w <= 4.0)).all() and w.std() > 0 and not np.array_equal(obs, base)
    cau, _ = incs(prior_loss='cauchy', prior_loss_c=0.02)
    assert not np.array_equal(cau, base) and np.abs(cau[:, :3, 3] - base[:, :3, 3]).max() < 5e-3
    hub, _ = incs(prior_loss='huber', prior_loss_c=1e3)       # ψ = c·(r / c) inside the kernel: rounding only,
    np.testing.assert_allclose(hub, base, atol=1e-4)          # grown by later accept / keyframe decisions
    with pytest.raises(ValueError, match='prior_loss'):
        KLTWindowOdometry(prior='oracle', prior_loss='l1')

    calls = []

    def spy(fun, x0, jac, **kw_):
        calls.append((fun, jac, x0.copy()))
        return _real_least_squares(fun, x0, jac=jac, **kw_)

    monkeypatch.setattr(kw, 'least_squares', spy)
    _run(frames[:16], prior='perturbed', prior_jitter=0.05, prior_obs=True, prior_obs_ref=ref, prior_loss='cauchy',
         prior_loss_c=0.02)
    fun, jac, x0 = calls[-1]
    x = x0 + np.random.default_rng(0).normal(0, 1e-3, len(x0))
    J = jac(x).toarray()
    eps = 1e-7
    Jn = np.stack([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(len(x))], 1)
    np.testing.assert_allclose(J, Jn, atol=2e-2 * np.abs(Jn).max(), rtol=0)
    np.testing.assert_allclose(J[:, -1], Jn[:, -1], rtol=1e-5, atol=1e-6)
