"""Demo geometry: camera re-projection into the TUM-VI canvas (scripts/demo_track.py) and the renderer's 3D pieces
(scripts/render_demo.py). Synthetic inputs only (CPU, no data)."""
import numpy as np
from scipy.spatial.transform import Rotation

import scripts.demo_track as D
import scripts.render_demo as R

K_EUROC = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1.0]])


def _project(K, X):
    u = X @ K.T
    return u[:, :2] / u[:, 2:]


def test_canvas_affine_maps_projections_between_cameras():
    X = np.random.default_rng(0).normal(size=(20, 3)) + [0, 0, 4]
    A = D.canvas_affine(K_EUROC)
    u = np.c_[_project(K_EUROC, X), np.ones(len(X))] @ A.T
    np.testing.assert_allclose(u, _project(D.K_CANVAS, X), atol=1e-9)


def test_to_canvas_places_a_bright_square_where_the_rays_land():
    img = np.zeros((480, 752), np.uint8)
    img[200:300, 300:400] = 255                        # pixel centres 200..299 -> edges at 199.5 / 299.5
    c = D.to_canvas(img, K_EUROC)
    assert c.shape == (512, 512)
    A = D.canvas_affine(K_EUROC)
    (x0, y0), (x1, y1) = np.array([[299.5, 199.5, 1], [399.5, 299.5, 1]]) @ A.T
    ys, xs = np.nonzero(c > 127)
    assert abs(xs.min() - x0) < 1.0 and abs(xs.max() + 1 - x1) < 1.0
    assert abs(ys.min() - y0) < 1.0 and abs(ys.max() + 1 - y1) < 1.0


def test_valid_mask_is_inside_the_reprojected_image_and_eroded():
    m0 = D.valid_mask(np.ones((480, 752), np.float32), K_EUROC, 0)
    m8 = D.valid_mask(np.ones((480, 752), np.float32), K_EUROC, 8)
    A = D.canvas_affine(K_EUROC)
    (x0, y0), (x1, y1) = np.array([[-0.5, -0.5, 1], [751.5, 479.5, 1]]) @ A.T
    ys, xs = np.nonzero(m0)
    assert xs.min() >= np.floor(x0) and xs.max() <= np.ceil(x1) and ys.min() >= np.floor(y0) and ys.max() <= np.ceil(y1)
    assert (m8 > 0).sum() < (m0 > 0).sum() and not np.any((m8 > 0) & (m0 == 0))
    ys8, xs8 = np.nonzero(m8)
    assert xs8.min() - xs.min() == 8 and ys.max() - ys8.max() == 8


def test_flat_photo_norm_masks_outside_and_keeps_exposure_one():
    mask = np.zeros((4, 4), np.uint8)
    mask[1:3, 1:3] = 255
    nm = D.FlatPhotoNorm(mask)
    gray = np.full((4, 4), 102, np.uint8)
    img, ok = nm('x.png', gray, 1.0)
    assert nm.exposure('x.png') == 1.0
    np.testing.assert_array_equal(ok, mask > 0)
    np.testing.assert_allclose(img[1:3, 1:3], 0.4, atol=1e-6)
    assert not img[0].any()


def _poses(n=30, seed=1):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)).as_matrix()
    T[:, :3, 3] = np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)
    return T


def test_world_points_place_camera_points_with_their_frames_pose():
    T = _poses()
    Xw = np.random.default_rng(2).normal(size=(50, 3)) * 3
    rows = np.random.default_rng(3).integers(0, len(T), 50)
    xc = np.einsum('nji,nj->ni', T[rows, :3, :3], Xw - T[rows, :3, 3])      # R^T (X - t)
    np.testing.assert_allclose(R.world_points(T, rows, xc), Xw, atol=1e-9)


def test_ate_is_zero_for_a_rigidly_moved_copy_and_positive_for_a_scaled_one():
    p = np.random.default_rng(4).normal(size=(100, 3))
    Rm = Rotation.from_rotvec([0.3, -0.2, 1.0]).as_matrix()
    assert R.ate_rmse(p, p @ Rm.T + [1.0, 2.0, -3.0]) < 1e-9
    assert R.ate_rmse(p, 1.1 * p) > 0.01


def test_up_rotation_maps_the_cameras_minus_y_to_display_z():
    T = _poses()
    T[:, :3, :3] = Rotation.from_rotvec([0.4, 0.1, -0.3]).as_matrix() @ Rotation.from_euler(
        'y', np.linspace(0, 2, len(T))).as_matrix()   # cameras turning about their own y axis, tilted world
    Rd = R.up_rotation(T, z_up=False)
    np.testing.assert_allclose(Rd @ Rd.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(Rd) > 0
    np.testing.assert_allclose(Rd @ -T[0, :3, 1], [0, 0, 1], atol=1e-9)
    np.testing.assert_array_equal(R.up_rotation(T, z_up=True), np.eye(3))


def test_map_state_keeps_the_latest_position_and_counts_frames():
    frames = np.array([0, 0, 1, 2, 2, 3])
    gids = np.array([7, 9, 7, 7, 9, 5])
    X = np.arange(18, dtype=float).reshape(6, 3)
    ms = R.MapState(frames, gids, X)                   # landmark indices: 5 -> 0, 7 -> 1, 9 -> 2
    now = ms.advance(1)
    np.testing.assert_array_equal(now, [1])
    np.testing.assert_array_equal(ms.count, [0, 2, 1])
    np.testing.assert_array_equal(ms.pos[1], X[2])
    assert np.isnan(ms.pos[0]).all()
    now = ms.advance(3)
    np.testing.assert_array_equal(now, [0])
    np.testing.assert_array_equal(ms.count, [1, 3, 2])
    np.testing.assert_array_equal(ms.pos, X[[5, 3, 4]])


def test_view3d_is_orthographic_with_z_up_and_fits_the_box():
    box = np.array([[-2.0, -1.0, 0.0], [2.0, 1.0, 1.5]])
    corners = np.array([[box[i][0], box[j][1], box[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    v = R.View3D(corners, 200, 10, azim=-60, elev=30)
    px, _ = v(corners)
    assert px.min() >= 10 - 1e-9 and px.max() <= 190 + 1e-9
    assert np.isclose(np.ptp(px, 0).max(), 180)       # fitted: the larger extent fills size - 2 margin
    a, _ = v(np.array([[0, 0, 0.0], [0, 0, 1.0]]))
    assert a[1, 1] < a[0, 1] and abs(a[1, 0] - a[0, 0]) < 1e-9   # up in the world = up on screen
    b, _ = v(np.array([[0, 0, 0.0], [1, 0, 0.0], [3, 0, 0.0]]))
    np.testing.assert_allclose(b[2] - b[0], 3 * (b[1] - b[0]), atol=1e-9)   # parallel projection: linear
    _, dep = v(np.array([[0, 0, 0.0], [0, 0, 1.0]]))
    assert dep[1] > dep[0]                              # elevated view: higher points are nearer the viewer


def test_frustum_apex_at_camera_and_corners_at_depth():
    T = _poses()[7]
    K = D.K_CANVAS
    f = R.frustum(T, K, (0, 0, 511, 511), 0.5)
    np.testing.assert_allclose(f[0], T[:3, 3], atol=1e-12)
    pc = (f[1:] - T[:3, 3]) @ T[:3, :3]
    np.testing.assert_allclose(pc[:, 2], 0.5, atol=1e-12)
    np.testing.assert_allclose(_project(K, pc), [[0, 0], [511, 0], [511, 511], [0, 511]], atol=1e-9)


def test_thin_keyframes_keeps_the_first_and_spaces_the_rest():
    pos = np.array([[0, 0, 0], [0.05, 0, 0], [0.2, 0, 0], [0.25, 0, 0], [0.02, 0, 0], [0.5, 0, 0.0]])
    np.testing.assert_array_equal(R.thin_keyframes(pos, np.arange(6), 0.1), [0, 2, 5])
    np.testing.assert_array_equal(R.thin_keyframes(pos, np.array([1, 3, 4]), 0.1), [1, 3])
