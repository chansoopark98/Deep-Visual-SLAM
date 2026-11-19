"""Self-supervised loss terms for metric depth + pose on TUM-VI (stereo cam1 = metric anchor, gyro, flow teacher).

Batch (see dvslam.ssl.tumvi_data): 'net' (B, 3, 3, H, W) cam0 [t-1, t, t+1] network images; 'pho' (B, 4, 1, H, W)
loss images [cam0 t-1, t, t+1, cam1 t]; 'valid' (B, 4, 1, H, W); 'K' (B, 3, 3); 'T_stereo' (B, 4, 4) = T_{cam1<-cam0};
'R_gyro' (B, 2, 3, 3) = R_{src<-tgt} for sources [t-1, t+1]; 'gyro_ok' (B,). The target is cam0 at t.

Warps use T_{src<-tgt}: src t-1 -> pose(I_{t-1}, I_t); src t+1 -> inv(pose(I_t, I_{t+1})); cam1 -> T_stereo.
Terms (each switched off, with its extra forward passes, by weight 0):
  temporal  : per scale (0..3, depth upsampled to full res), per-pixel min over the two sources of
              p = 0.85 SSIM + 0.15 L1 on 'pho', Monodepth2 auto-mask (identity residual + 1e-5 noise). A source is
              unusable at a pixel when out of view or 'valid' of the target / warped source is 0 anywhere in the
              3x3 SSIM window (mask eroded), so invalid pixels never reach the loss. The geometric
              weight (1 - D_diff, scale 0, detached) of the selected source multiplies the residual at every scale.
  stereo    : the same p for warped cam1 vs target, same masking, no auto-mask.
  uncertainty (model with sigma): residual r -> r / sigma + log sigma in both photometric terms.
  smooth    : edge-aware smoothness of the mean-normalised disparity on the target loss image, / 2^scale.
  geometric : scale 0, D_diff = |z_proj - D_src(pix)| / (z_proj + D_src(pix)) over in-view valid pixels (SC-Depth).
  gyro      : angle (rad) of R_pred^T R_gyro of the final T_{src<-tgt}, both sources, where gyro_ok.
  pose_consistency : initial pose net on (A, B) and (B, A); M = T_AB T_BA; mean|t(M)| / 0.05 + mean|rotvec(M)| / 0.05.
  flow      : scale 0, Huber (delta 1 px) between the rigid flow target -> source and a frozen RAFT-large teacher flow
              on the 'net' images, over in-view valid pixels.
Photometric, smoothness and the masks are averaged over the 4 scales; every term returns a scalar.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dvslam.ssl import geometry
from dvslam.ssl.models import NUM_SCALES, SSLModel

TERMS = ('temporal', 'stereo', 'smooth', 'geometric', 'gyro', 'pose_consistency', 'flow')
SSIM_W = 0.85
UNUSABLE = 1e4          # residual of an unusable source, larger than any real photometric residual
POSE_CONSISTENCY_SCALE = 0.05


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-pixel (1 - SSIM) / 2 in [0, 1], 3x3 window, reflection padding (Monodepth2)."""
    x, y = F.pad(x, (1, 1, 1, 1), mode='reflect'), F.pad(y, (1, 1, 1, 1), mode='reflect')
    mu_x, mu_y = F.avg_pool2d(x, 3, 1), F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return torch.clamp((1 - n / d) / 2, 0, 1)


def photometric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) residual 0.85 SSIM + 0.15 L1 (channel mean)."""
    return SSIM_W * ssim(a, b).mean(1, True) + (1 - SSIM_W) * (a - b).abs().mean(1, True)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


def with_uncertainty(r: torch.Tensor, sigma: torch.Tensor | None) -> torch.Tensor:
    return r if sigma is None else r / sigma + torch.log(sigma)


def erode(mask: torch.Tensor) -> torch.Tensor:
    """3x3 erosion of a (N, 1, H, W) {0, 1} mask: a pixel stays usable only if its whole SSIM window is."""
    return -F.max_pool2d(-mask, 3, 1, 1)


def temporal_term(target: torch.Tensor, warped: list[torch.Tensor], masks: list[torch.Tensor],
                  sources: list[torch.Tensor], source_masks: list[torch.Tensor],
                  geo_weights: list[torch.Tensor] | None = None, sigma: torch.Tensor | None = None) -> torch.Tensor:
    """Min over sources with auto-mask. masks: usable warped pixels; sources / source_masks: the unwarped source
    loss images and their usable pixels (identity residual; an unusable source pixel cannot win the auto-mask)."""
    n = len(sources)
    res = torch.cat([torch.where(m > 0, photometric(w, target), UNUSABLE) for w, m in zip(warped, masks)], 1)
    ident = torch.cat([torch.where(m > 0, photometric(s, target), UNUSABLE) for s, m in zip(sources, source_masks)], 1)
    ident = ident + torch.randn_like(ident) * 1e-5
    best, idx = torch.cat([ident, res], 1).min(1, keepdim=True)
    keep = (idx >= n).float() * torch.cat(masks, 1).amax(1, True)
    if geo_weights is not None:
        best = best * torch.cat(geo_weights, 1).gather(1, (idx - n).clamp(min=0))
    return masked_mean(with_uncertainty(best, sigma), keep)


def stereo_term(target: torch.Tensor, warped: torch.Tensor, mask: torch.Tensor,
                sigma: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean(with_uncertainty(photometric(warped, target), sigma), mask)


def smooth_term(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    disp = disp / (disp.mean((2, 3), True).clamp(min=1e-3) + 1e-7)
    gx = (disp[..., :, :-1] - disp[..., :, 1:]).abs()
    gy = (disp[..., :-1, :] - disp[..., 1:, :]).abs()
    gx = gx * torch.exp(-(img[..., :, :-1] - img[..., :, 1:]).abs().mean(1, True))
    gy = gy * torch.exp(-(img[..., :-1, :] - img[..., 1:, :]).abs().mean(1, True))
    return gx.mean() + gy.mean()


def geometric_term(z_proj: torch.Tensor, d_src: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss, D_diff (B, 1, H, W)); d_src is the source depth sampled at the reprojected pixels."""
    d_diff = (z_proj - d_src).abs() / (z_proj + d_src).clamp(min=1e-6)
    return masked_mean(d_diff, mask), d_diff


def gyro_term(R_pred: torch.Tensor, R_gyro: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """R_pred, R_gyro (B, S, 3, 3) as R_{src<-tgt}; ok (B,) bool. Mean angle in rad over ok samples."""
    ang = geometry.rotation_angle(R_pred.transpose(-1, -2) @ R_gyro)
    return masked_mean(ang, ok[:, None].float().expand_as(ang))


def pose_consistency_term(T_ab: torch.Tensor, T_ba: torch.Tensor) -> torch.Tensor:
    M = T_ab @ T_ba
    return (M[:, :3, 3].abs().mean() + geometry.rotation_log(M[:, :3, :3]).abs().mean()) / POSE_CONSISTENCY_SCALE


def flow_term(rigid: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(F.huber_loss(rigid, teacher, reduction='none', delta=1.0).sum(1, True), mask)


class RaftTeacher(nn.Module):
    """Frozen torchvision RAFT-large (default pretrained weights); flow img1 -> img2 for [0, 1] inputs."""

    def __init__(self):
        super().__init__()
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.net = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.net(img1 * 2 - 1, img2 * 2 - 1)[-1]


class SSLLoss(nn.Module):
    """Weighted sum of the terms; weights from the config `loss` section. `teacher` replaces RAFT (tests)."""

    def __init__(self, weights: dict, teacher: nn.Module | None = None):
        super().__init__()
        self.w = {k: float(weights[k]) for k in TERMS}
        self.teacher = (teacher or RaftTeacher()) if self.w['flow'] > 0 else None

    def forward(self, model: SSLModel, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self.w
        net, pho, valid, K = batch['net'], batch['pho'], batch['valid'], batch['K']
        target, valid_t = pho[:, 1], valid[:, 1]
        disps, sigmas = model.depth(net[:, 1])
        depths = [model.disp_to_depth(d) for d in disps]
        terms = {}

        T_src = None
        if any(w[k] > 0 for k in ('temporal', 'geometric', 'gyro', 'flow', 'pose_consistency')):
            d_src = [None, None]
            if w['geometric'] > 0:
                d_src = [model.depth_m(net[:, 0]), model.depth_m(net[:, 2])]
            elif model.pose_iters > 1:
                with torch.no_grad():
                    d_src[1] = model.depth_m(net[:, 2])
            T_prev, T_prev0 = model.pose(net[:, 0], net[:, 1], K, depths[0])     # T_{t-1<-t}
            T_next, T_next0 = model.pose(net[:, 1], net[:, 2], K, d_src[1])      # T_{t<-t+1}
            T_src = [T_prev, geometry.se3_inverse(T_next)]
            if w['pose_consistency'] > 0:
                terms['pose_consistency'] = 0.5 * (
                    pose_consistency_term(T_prev0, model.pose_net(net[:, 1], net[:, 0]))
                    + pose_consistency_term(T_next0, model.pose_net(net[:, 2], net[:, 1])))
            if w['gyro'] > 0:
                R_pred = torch.stack([T[:, :3, :3] for T in T_src], 1)
                terms['gyro'] = gyro_term(R_pred, batch['R_gyro'], batch['gyro_ok'])

        def warp_masked(src: int, depth: torch.Tensor, T: torch.Tensor):
            """(warped loss image, usable mask, z_src, pix); usable = in view, target and warped source valid."""
            out, vis, z, pix = geometry.warp(torch.cat([pho[:, src], valid[:, src]], 1), depth, T, K)
            return out[:, :1], erode(vis * valid_t * (out[:, 1:] > 0.999).float()), z, pix

        geo_w = [] if w['geometric'] > 0 else None
        if w['geometric'] > 0 or w['flow'] > 0:
            geo, flow = [], []
            for i, src in enumerate((0, 2)):
                _, mask, z, pix = warp_masked(src, depths[0], T_src[i])
                if w['geometric'] > 0:
                    g, d_diff = geometric_term(z, geometry.sample(d_src[i], pix), mask)
                    geo.append(g)
                    geo_w.append((1 - d_diff).clamp(0, 1).detach())   # d_diff is unbounded where z_proj <= 0
                if w['flow'] > 0:
                    teacher = self.teacher(net[:, 1], net[:, src])
                    flow.append(flow_term(geometry.rigid_flow(pix), teacher, mask))
            if geo:
                terms['geometric'] = sum(geo) / 2
            if flow:
                terms['flow'] = sum(flow) / 2

        for name in ('temporal', 'stereo', 'smooth'):
            if w[name] > 0:
                terms[name] = 0.0
        for s in range(NUM_SCALES):
            sigma = sigmas[s] if sigmas is not None else None
            if w['temporal'] > 0:
                warped = [warp_masked(src, depths[s], T_src[i]) for i, src in enumerate((0, 2))]
                terms['temporal'] = terms['temporal'] + temporal_term(
                    target, [x[0] for x in warped], [x[1] for x in warped], [pho[:, 0], pho[:, 2]],
                    [erode(valid[:, 0]), erode(valid[:, 2])], geo_w, sigma)
            if w['stereo'] > 0:
                warped, mask, _, _ = warp_masked(3, depths[s], batch['T_stereo'])
                terms['stereo'] = terms['stereo'] + stereo_term(target, warped, mask, sigma)
            if w['smooth'] > 0:
                terms['smooth'] = terms['smooth'] + smooth_term(disps[s], target) / 2 ** s
        for name in ('temporal', 'stereo', 'smooth'):
            if w[name] > 0:
                terms[name] = terms[name] / NUM_SCALES

        total = sum(w[k] * v for k, v in terms.items())
        return total, terms
