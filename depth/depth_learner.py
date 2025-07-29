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
        self.min_depth = config['Train']['min_depth']
        self.max_depth = config['Train']['max_depth']
        self.num_scales = 4
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Multi-scale weights 조정 (큰 스케일에 더 큰 가중치)
        self.alphas = [1.0, 0.5, 0.25, 0.125]
        
        # Loss weights
        self.smooth_weight = config['Train'].get('smooth_weight', 0.1)
        self.silog_weight = config['Train'].get('silog_weight', 1.0)
    

    def disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        """Convert network's output disparity to depth."""
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / scaled_disp
        return depth.float()
    
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
        Edge-aware smoothness loss on disparity (MonoDepth2 스타일).
        disp: [B,1,H,W], img: [B,3,H,W]
        """
        # Normalize disparity
        disp_mean = disp.mean(dim=[2, 3], keepdim=True).clamp(min=1e-7)
        norm_disp = disp / disp_mean

        # Compute gradients
        disp_dx, disp_dy = self.compute_gradients(norm_disp)
        img_dx, img_dy = self.compute_gradients(img)

        # Edge-aware weights
        weight_x = torch.exp(-img_dx.mean(dim=1, keepdim=True))
        weight_y = torch.exp(-img_dy.mean(dim=1, keepdim=True))

        # Smoothness loss
        smooth_x = disp_dx * weight_x
        smooth_y = disp_dy * weight_y

        return smooth_x.mean() + smooth_y.mean()

    def silog_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        variance_focus: float = 0.85
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
    ):
        B, _, H, W = gt_depth.shape
        total_smooth = total_silog = 0.0

        for i, alpha in enumerate(self.alphas):
            pred_depth = F.interpolate(pred_depths[i], (H, W), mode='bilinear', align_corners=False)

            smooth_loss = self.get_smooth_loss(pred_depth, rgb)

            silog = self.silog_loss(pred_depth, gt_depth, valid_mask)

            total_smooth += alpha * smooth_loss
            total_silog  += alpha * silog

        total_loss = self.silog_weight * total_silog + self.smooth_weight * total_smooth
        return total_loss, total_silog, total_smooth

    def forward_step(
        self,
        sample: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Execute a forward pass, compute losses.
        rgb: [B,3,H,W],
        depth: Relative depth [B,1,H,W] or [B,H,W]
        """
        rgb = sample['image'].to(self.device)  # [B,3,H,W]
        depth = sample['depth'].to(self.device)  # [B,1,H,W] or [B,H,W]
        valid_mask = sample['valid_mask'].to(self.device)

        # Get list of predicted disparities
        outputs = self.model(rgb)  # dict of {("disp", scale): [B,1,h_i,w_i]}
        
        # Convert to depth predictions
        pred_depths = []
        for scale in range(self.num_scales):
            pred_disp = outputs[("disp", scale)]
            pred_depth = self.disp_to_depth(pred_disp)
            pred_depths.append(pred_depth)

        # Compute multi-scale losses
        total_loss, total_silog, total_smooth = self.multi_scale_loss(
            pred_depths, depth, rgb, valid_mask
        )

        return total_loss, total_silog, total_smooth, pred_depths