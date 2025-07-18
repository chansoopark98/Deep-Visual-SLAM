import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

class DepthLearner:
    def __init__(self, model: nn.Module,
                 config: Dict[str, Any],
                 device: torch.device) -> None:
        """
        Args:
            model (nn.Module): PyTorch model that outputs a list of disparity tensors.
            config (Dict[str, Any]): Configuration dict with keys:
                - Train.mode: 'relative' or 'metric'
                - Train.min_depth: float
                - Train.max_depth: float
        """
        self.model = model
        self.train_mode = config['Train']['mode']
        self.min_depth = config['Train']['min_depth']
        self.max_depth = config['Train']['max_depth']
        self.num_scales = 4
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Weights for multi-scale losses
        # self.alphas = [1/2, 1/4, 1/8, 1/16]
        self.alphas = [0.5, 0.25, 0.125, 0.125]

    def disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        """Convert network's output disparity to depth."""
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / scaled_disp
        return depth.float()

    def scaled_depth_to_disp(self, depth: torch.Tensor) -> torch.Tensor:
        """Convert depth back to normalized disparity."""
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = 1.0 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return disp.float()

    @staticmethod
    def compute_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute absolute gradients in x and y directions.
        Expects x of shape [B, C, H, W].
        Returns dx: [B, C, H, W-1], dy: [B, C, H-1, W].
        """
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return dx, dy

    def get_smooth_loss(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Edge-aware smoothness loss on disparity.
        disp: [B,1,H,W], img: [B,3,H,W]
        """
        # normalize disparity
        disp_mean = disp.mean(dim=[2, 3], keepdim=True) + 1e-7
        norm_disp = disp / disp_mean

        disp_dx, disp_dy = self.compute_gradients(norm_disp)
        img_dx, img_dy = self.compute_gradients(img)

        # compute edge-aware weights
        weight_x = torch.exp(-img_dx.mean(dim=1, keepdim=True))
        weight_y = torch.exp(-img_dy.mean(dim=1, keepdim=True))

        smooth_x = disp_dx * weight_x
        smooth_y = disp_dy * weight_y

        return smooth_x.mean() + smooth_y.mean()

    def l1_loss(self, pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        L1 loss over valid pixels.
        pred, gt: [B,1,H,W], valid_mask: [B,1,H,W] bool
        """
        diff = torch.abs(pred - gt)
        # mask out invalid pixels
        valid = diff[valid_mask]
        return valid.mean()

    def silog_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        variance_focus: float = 0.5
    ) -> torch.Tensor:
        """
        Scale-invariant log loss.
        prediction, target: [B,1,H,W], valid_mask: [B,1,H,W] bool
        """
        eps = 1e-6
        pred = torch.clamp(prediction, min=eps)
        # select valid pixels
        pred_v = pred[valid_mask]
        tgt_v = target[valid_mask]
        d = torch.log(pred_v) - torch.log(tgt_v)
        d2_mean = (d ** 2).mean()
        d_mean = d.mean()
        silog = d2_mean - variance_focus * (d_mean ** 2)
        return torch.sqrt(silog)

    def multi_scale_loss(
        self,
        pred_depths: List[torch.Tensor],
        gt_depth: torch.Tensor,
        rgb: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combine smoothness, SILog, and (optionally) L1 losses across scales.
        pred_depths: list of 4 tensors, each [B,1,h_i,w_i]
        gt_depth, rgb: [B,1,H,W], [B,3,H,W]
        valid_mask: [B,1,H,W] bool
        """
        B, _, H, W = gt_depth.shape
        total_smooth = 0.0
        total_silog = 0.0
        total_loss = 0.0

        for i, alpha in enumerate(self.alphas):
            pd = pred_depths[i]
            # upsample to match ground truth resolution
            pd_up = F.interpolate(pd, size=(H, W), mode='bilinear', align_corners=False)
            # convert to disparity to compute smoothness
            disp_up = self.scaled_depth_to_disp(pd_up)

            total_smooth += self.get_smooth_loss(disp_up, rgb) * alpha
            total_silog += self.silog_loss(pd_up, gt_depth, valid_mask) * alpha

        total_loss = total_smooth + total_silog

        return total_loss, total_silog, total_smooth

    @torch.compile
    def forward_step(
        self,
        sample: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Execute a forward pass, compute losses.
        rgb: [B,3,H,W], depth: [B,1,H,W] or [B,H,W]
        """
        rgb = sample['image'].to(self.device)  # [B,3,H,W]
        depth = sample['depth'].to(self.device)  # [B,1,H,W] or [B,H,W]

        # ensure depth has channel dim
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # get list of predicted disparities
        pred_disps = self.model(rgb)  # list of [B,1,h_i,w_i]
        # build valid mask
        max_d = 1.0 if self.train_mode == 'relative' else self.max_depth
        valid = (depth > 0.0) & (depth < max_d)

        # convert to depth predictions
        pred_depths = []
        for scale in range(len(pred_disps)):
            pred_disp = pred_disps[("disp", scale)]
            pred_depth = self.disp_to_depth(pred_disp)
            pred_depths.append(pred_depth)
        # compute multi-scale losses
        total_loss, total_silog, total_smooth = self.multi_scale_loss(pred_depths, depth, rgb, valid)

        return total_loss, total_silog, total_smooth, pred_depths
