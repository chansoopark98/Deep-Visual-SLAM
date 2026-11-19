"""Metric monocular depth + relative pose networks for self-supervised training.

Conventions:
  images : (B, 3, H, W) in [0, 1] (gray replicated to 3 channels); ImageNet normalisation is inside the encoders.
  depth  : sigmoid disparity d -> depth = 1 / (1/max_depth + (1/min_depth - 1/max_depth) d), metres. Every scale is
           bilinearly upsampled to the input resolution (Monodepth2), scale 0 is the full-resolution output.
  sigma  : with `uncertainty`, a second decoder channel gives sigma = sigma_min + (1 - sigma_min) sigmoid in
           (sigma_min, 1] per pixel (D3VO-style photometric uncertainty).
  pose   : pose(A, B) = T_{A<-B} (X_A = T X_B). B is the TARGET (its depth is used), A the SOURCE. For the scored pair
           pose(I_{t-1}, I_t) is the target-metric rel_increment inv(T_wc[t-1]) @ T_wc[t].
  refine : pose_iters - 1 residual steps (MonoIndoor-style). A is warped into B's view with depth(B) and the current
           T (warped image detached); the residual net on (warped, B) returns T_{w<-B} of the virtual camera w with
           X_w = T^-1 T_true X_B, so T <- T @ T_{w<-B}. The residual net is shared across steps and starts at identity.
  amp    : set_amp(dtype) runs the network forwards alone (depth encoder / decoder, pose encoder / head) under autocast
           in `dtype`; their outputs are cast back to float32 before anything else, so the pose vector -> matrix, the
           composition T @ residual, the warps and every loss stay float32. Off (None) by default and in checkpoints.
"""
import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from dvslam.ssl import geometry
from model.depthnet import DepthNet
from model.resnet_encoder import ResnetEncoder

NUM_SCALES = 4
POSE_SCALE = 0.1


def autocast(x: torch.Tensor, dtype: torch.dtype | None):
    """Autocast to `dtype` on x's device, or no context at all for None (the float32 default path is unchanged)."""
    return contextlib.nullcontext() if dtype is None else torch.autocast(x.device.type, dtype=dtype)


class PoseNet(nn.Module):
    """ResNet-18 on the 6-channel pair (A, B) + conv head -> T_{A<-B}; 6 outputs x 0.1 = (rotvec rad, t m)."""

    def __init__(self, pretrained: bool, zero_init: bool = False):
        super().__init__()
        self.amp_dtype = None
        self.encoder = ResnetEncoder(18, pretrained, num_input_images=2)
        self.head = nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(256, 6, 1))
        if zero_init:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with autocast(a, self.amp_dtype):
            x = self.head(self.encoder(torch.cat([a, b], 1))[-1])
        x = x.float().mean((2, 3)) * POSE_SCALE
        return geometry.pose_vec_to_mat(x[:, :3], x[:, 3:])


class SSLModel(nn.Module):
    def __init__(self, min_depth: float, max_depth: float, uncertainty: bool, sigma_min: float, pose_iters: int,
                 pretrained: bool):
        super().__init__()
        self.min_depth, self.max_depth = min_depth, max_depth
        self.uncertainty, self.sigma_min, self.pose_iters = uncertainty, sigma_min, pose_iters
        self.depth_net = DepthNet(18, pretrained, scales=range(NUM_SCALES), num_output_channels=1 + uncertainty)
        self.pose_net = PoseNet(pretrained)
        self.residual_net = PoseNet(pretrained, zero_init=True) if pose_iters > 1 else None
        self.amp_dtype = None

    def set_amp(self, dtype: torch.dtype | None) -> None:
        """Autocast dtype of the network forwards (None = float32 everywhere); see the module docstring."""
        self.amp_dtype = self.pose_net.amp_dtype = dtype
        if self.residual_net is not None:
            self.residual_net.amp_dtype = dtype

    def disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        return 1 / (1 / self.max_depth + (1 / self.min_depth - 1 / self.max_depth) * disp)

    def depth(self, img: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Per scale (0..3) disparity (B, 1, H, W) and sigma (or None), all at the input resolution."""
        with autocast(img, self.amp_dtype):
            out = self.depth_net(img)
        disps, sigmas = [], []
        for s in range(NUM_SCALES):
            x = out[('disp', s)].float()
            if x.shape[-2:] != img.shape[-2:]:
                x = F.interpolate(x, img.shape[-2:], mode='bilinear', align_corners=False)
            disps.append(x[:, :1])
            sigmas.append(self.sigma_min + (1 - self.sigma_min) * x[:, 1:])
        return disps, sigmas if self.uncertainty else None

    def depth_m(self, img: torch.Tensor) -> torch.Tensor:
        """Full-resolution metric depth (B, 1, H, W)."""
        return self.disp_to_depth(self.depth(img)[0][0])

    def pose(self, a: torch.Tensor, b: torch.Tensor, K: torch.Tensor,
             depth_b: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(final T_{A<-B}, initial T_{A<-B} of pose_net). depth_b (target depth) is computed if refinement needs it."""
        T0 = T = self.pose_net(a, b)
        if self.pose_iters > 1 and depth_b is None:
            depth_b = self.depth_m(b)
        for _ in range(self.pose_iters - 1):
            warped = geometry.warp(a, depth_b, T, K)[0].detach()
            T = T @ self.residual_net(warped, b)
        return T, T0
