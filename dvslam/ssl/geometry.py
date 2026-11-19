"""Batched SE(3) and view-synthesis geometry for the self-supervised losses (torch, always float32).

Conventions:
  poses     : (B, 4, 4); T_{A<-B} maps points from camera B into camera A: X_A = T_{A<-B} X_B (metres).
  pixels    : (u, v) with integer pixel centres, u to the right, v down; K is the pinhole matrix in these pixels.
              grid_sample runs with align_corners=True, so normalised -1 / +1 are the centres of the edge pixels.
  depth     : (B, 1, H, W) z-depth in metres. Warping source image S into the target view uses the target depth
              and T_{src<-tgt}: pixel p of the target samples S at proj(K, T_{src<-tgt} · backproj(p, depth(p))).
  rotations : axis-angle (rotation vector) in radians; R maps X_A = R X_B like the 3x3 block of T_{A<-B}.
Every public function disables autocast and casts its floating inputs to float32.
"""
import functools

import torch
import torch.nn.functional as F

Z_MIN = 1e-3          # projected depth (m) below which a point counts as behind the camera
EDGE_TOL = 1e-3       # px; float32 round-off at the image border still counts as in view


def _fp32(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        dev = next(a for a in args if torch.is_tensor(a)).device.type
        with torch.autocast(dev, enabled=False):
            return fn(*(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args))
    return wrapper


def _hat(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[:, 0])
    return torch.stack([torch.stack([z, -w[:, 2], w[:, 1]], -1),
                        torch.stack([w[:, 2], z, -w[:, 0]], -1),
                        torch.stack([-w[:, 1], w[:, 0], z], -1)], 1)


@_fp32
def pose_vec_to_mat(axisangle: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """(B, 3) rotation vector + (B, 3) translation -> (B, 4, 4) [R t; 0 1], R = exp(hat(w)) (exact at w = 0)."""
    T = torch.zeros(axisangle.shape[0], 4, 4, device=axisangle.device)
    T[:, :3, :3] = torch.linalg.matrix_exp(_hat(axisangle))
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1
    return T


@_fp32
def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rt = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ T[..., :3, 3:]
    out[..., 3, 3] = 1
    return out


@_fp32
def rotation_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> rotation vector (..., 3); valid for angles below pi, smooth (finite gradient) at R = I."""
    vee = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = torch.sqrt((vee ** 2).sum(-1) + 1e-12)            # sin(angle), kept away from 0
    c = 0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1)
    return vee * (torch.atan2(s, c) / s)[..., None]


@_fp32
def rotation_angle(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> rotation angle (...,) in radians, with a finite gradient at R = I."""
    vee = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    c = 0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1)
    return torch.atan2(torch.sqrt((vee ** 2).sum(-1) + 1e-12), c)


def pixel_grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """(b, 2, h, w) pixel coordinates (u, v)."""
    v, u = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    return torch.stack([u, v])[None].expand(b, 2, h, w)


@_fp32
def reproject(depth: torch.Tensor, T: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Target pixels with `depth` moved by T = T_{src<-tgt} into the source: (pix (B, 2, H, W), z_src (B, 1, H, W))."""
    b, _, h, w = depth.shape
    uv = pixel_grid(b, h, w, depth.device).reshape(b, 2, -1)
    rays = torch.linalg.inv(K) @ torch.cat([uv, torch.ones_like(uv[:, :1])], 1)
    X = T[:, :3, :3] @ (rays * depth.reshape(b, 1, -1)) + T[:, :3, 3:]
    p = K @ X
    z = p[:, 2:]
    pix = p[:, :2] / z.clamp(min=Z_MIN)
    return pix.reshape(b, 2, h, w), z.reshape(b, 1, h, w)


@_fp32
def sample(img: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of img (B, C, H, W) at pixel coordinates pix (B, 2, H', W'), border padding."""
    h, w = img.shape[-2:]
    grid = torch.stack([pix[:, 0] / (w - 1), pix[:, 1] / (h - 1)], -1) * 2 - 1
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


@_fp32
def in_view(pix: torch.Tensor, z: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, 1, H, W) float mask: projection inside the source image and in front of the source camera."""
    u, v = pix[:, :1], pix[:, 1:]
    lo, hi_u, hi_v = -EDGE_TOL, w - 1 + EDGE_TOL, h - 1 + EDGE_TOL
    return ((u >= lo) & (u <= hi_u) & (v >= lo) & (v <= hi_v) & (z > Z_MIN)).float()


@_fp32
def warp(img: torch.Tensor, depth: torch.Tensor, T: torch.Tensor,
         K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source img into the target view with the target depth and T = T_{src<-tgt}.

    Returns (warped (B, C, H, W), in-view mask (B, 1, H, W), z in the source frame (B, 1, H, W), pix (B, 2, H, W)).
    """
    pix, z = reproject(depth, T, K)
    return sample(img, pix), in_view(pix, z, *img.shape[-2:]), z, pix


@_fp32
def rigid_flow(pix: torch.Tensor) -> torch.Tensor:
    """Flow target -> source in pixels (B, 2, H, W) from the reprojected coordinates of `reproject`."""
    return pix - pixel_grid(pix.shape[0], *pix.shape[-2:], pix.device)
