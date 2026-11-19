"""dvslam.ssl.geometry: warp, SE(3) helpers and the pose convention T_{A<-B} = inv(T_wc_A) @ T_wc_B on a synthetic
textured plane rendered analytically (CPU, 64x64)."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dvslam.geometry.se3 import relative_pose
from dvslam.ssl import geometry
from dvslam.ssl.models import SSLModel

H = W = 64
K_NP = np.array([[60.0, 0, 31.5], [0, 60.0, 31.5], [0, 0, 1]])
PLANE_Z = 1.5


def texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (0.5 + 0.2 * np.sin(2 * np.pi * x / 0.45) + 0.15 * np.cos(2 * np.pi * y / 0.37)
            + 0.1 * np.sin(2 * np.pi * (x + y) / 0.6))


def render_plane(T_wc: np.ndarray, K: np.ndarray = K_NP) -> tuple[np.ndarray, np.ndarray]:
    """(image, z-depth) of the textured world plane Z = PLANE_Z seen by a camera with pose T_wc."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])   # camera rays with z = 1
    d = T_wc[:3, :3] @ rays
    lam = (PLANE_Z - T_wc[2, 3]) / d[2]                                              # = z-depth in the camera
    P = T_wc[:3, 3:] + d * lam
    return texture(P[0], P[1]).reshape(H, W).astype(np.float32), lam.reshape(H, W).astype(np.float32)


def pose(rotvec, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def tt(x: np.ndarray) -> torch.Tensor:
    """numpy image (H, W) / pose (4, 4) / K -> batched float32 tensor."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return x[None, None] if x.shape == (H, W) else x[None]


def test_identity_warp_reproduces_image():
    img = torch.rand(2, 1, H, W)
    depth = torch.rand(2, 1, H, W) + 0.5
    K = tt(K_NP).expand(2, 3, 3)
    warped, mask, z, pix = geometry.warp(img, depth, torch.eye(4).expand(2, 4, 4), K)
    assert (pix - geometry.pixel_grid(2, H, W, img.device)).abs().max() < 1e-3
    assert (warped - img).abs().max() < 1e-3
    assert mask.all() and torch.allclose(z, depth, atol=1e-5)


def test_translation_and_rotation_give_analytic_shift():
    depth = torch.full((1, 1, H, W), 2.0)
    K = tt(K_NP)
    grid = geometry.pixel_grid(1, H, W, depth.device)
    pix, z = geometry.reproject(depth, tt(pose([0, 0, 0], [0.1, 0, 0])), K)   # X_src = X_tgt + t
    assert torch.allclose(pix[:, 0], grid[:, 0] + 60.0 * 0.1 / 2.0, atol=1e-4)
    assert torch.allclose(pix[:, 1], grid[:, 1], atol=1e-4) and torch.allclose(z, depth)
    T = pose([0, np.radians(3), 0], [0, 0, 0])                                 # pure rotation: depth-free homography
    pix, _ = geometry.reproject(depth, tt(T), K)
    hom = K_NP @ T[:3, :3] @ np.linalg.inv(K_NP) @ np.stack([grid[0, 0].numpy().ravel(), grid[0, 1].numpy().ravel(),
                                                             np.ones(H * W)])
    np.testing.assert_allclose(pix[0].reshape(2, -1).numpy(), hom[:2] / hom[2], atol=1e-3)


def test_inverse_compose_and_rotation_log():
    torch.manual_seed(0)
    w = torch.randn(5, 3) * 0.8
    T = geometry.pose_vec_to_mat(w, torch.randn(5, 3))
    I = torch.eye(4).expand(5, 4, 4)
    assert torch.allclose(T @ geometry.se3_inverse(T), I, atol=1e-5)
    assert torch.allclose(geometry.se3_inverse(T) @ T, I, atol=1e-5)
    assert torch.allclose(geometry.rotation_log(T[:, :3, :3]), w, atol=1e-5)
    np.testing.assert_allclose(T[:, :3, :3].numpy(), Rotation.from_rotvec(w.numpy()).as_matrix(), atol=1e-5)
    np.testing.assert_allclose(geometry.rotation_angle(T[:, :3, :3]).numpy(), w.norm(dim=1).numpy(), atol=1e-5)
    # zero rotation: exact identity, finite gradients through exp and log
    w0 = torch.zeros(2, 3, requires_grad=True)
    T0 = geometry.pose_vec_to_mat(w0, torch.zeros(2, 3))
    assert torch.equal(T0, torch.eye(4).expand(2, 4, 4))
    (geometry.rotation_log(T0[:, :3, :3]).sum() + geometry.rotation_angle(T0[:, :3, :3]).sum()).backward()
    assert torch.isfinite(w0.grad).all()


def test_pose_convention_warps_A_into_B_on_rendered_plane():
    T_wc_A = pose([0.02, -0.03, 0.01], [0.0, 0.0, 0.0])
    T_wc_B = pose([-0.03, 0.05, 0.02], [0.12, -0.05, 0.08])
    img_A, _ = render_plane(T_wc_A)
    img_B, depth_B = render_plane(T_wc_B)
    T_AB = np.linalg.inv(T_wc_A) @ T_wc_B
    np.testing.assert_allclose(T_AB, relative_pose(T_wc_A, T_wc_B))
    warped, mask, z, _ = geometry.warp(tt(img_A), tt(depth_B), tt(T_AB), tt(K_NP))
    m = mask[0, 0].bool()
    assert m.float().mean() > 0.6
    err = (warped[0, 0] - tt(img_B)[0, 0]).abs()[m]
    assert err.max() < 0.02, err.max()
    # z in A's frame is the depth A renders at the reprojected pixel; the wrong direction (T_BA) does not warp
    wrong, mask_w, _, _ = geometry.warp(tt(img_A), tt(depth_B), tt(np.linalg.inv(T_AB)), tt(K_NP))
    assert (wrong[0, 0] - tt(img_B)[0, 0]).abs()[(mask_w[0, 0] > 0) & m].mean() > 5 * err.mean()


def test_residual_update_recovers_true_pose():
    T_wc_A = pose([0, 0, 0], [0, 0, 0])
    T_wc_B = pose([0.02, 0.04, -0.01], [0.1, 0.03, 0.05])
    img_A, _ = render_plane(T_wc_A)
    img_B, depth_B = render_plane(T_wc_B)
    T_true = relative_pose(T_wc_A, T_wc_B)
    T_est = T_true @ pose([0.01, -0.02, 0.0], [0.02, 0.0, -0.01])
    T_res = np.linalg.inv(T_est) @ T_true                                     # exact T_{w<-tgt}
    np.testing.assert_allclose(T_est @ T_res, T_true, atol=1e-12)
    a, d, K = tt(img_A), tt(depth_B), tt(K_NP)
    err = {}
    for name, T in (('est', T_est), ('updated', T_est @ T_res)):
        warped, mask, _, _ = geometry.warp(a, d, tt(T), K)
        err[name] = (warped - tt(img_B)).abs()[mask.bool()].mean()
    assert err['updated'] < 0.2 * err['est'], err

    # SSLModel.pose applies exactly T <- T @ residual_net(warped(A), B) with B as the target
    model = SSLModel(0.1, 20.0, False, 0.01, pose_iters=2, pretrained=False)
    seen = {}

    def residual(warped: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seen['warped'], seen['target'] = warped, target
        return tt(T_res)

    model.pose_net.forward = lambda x, y: tt(T_est)
    model.residual_net.forward = residual
    img3 = lambda x: tt(x).expand(1, 3, H, W)                                 # noqa: E731
    T_final, T_init = model.pose(img3(img_A), img3(img_B), K, d)
    np.testing.assert_allclose(T_final[0].double().numpy(), T_true, atol=1e-5)
    np.testing.assert_allclose(T_init[0].double().numpy(), T_est, atol=1e-6)
    assert torch.equal(seen['target'], img3(img_B)) and not seen['warped'].requires_grad
    assert torch.allclose(seen['warped'], geometry.warp(img3(img_A), d, tt(T_est), K)[0])
