"""Depth Anything V2 metric (Hypersim, indoor, max 20 m), official `metric_depth` code (third_party/depth_anything_v2).

Replaces the repo's `model/depth_anything_v2` path for priors: that code is the same metric DPT (Sigmoid x max_depth)
but `model/depth_anything_v2/test.py` loads `depth_anything_v2_vits.pth`, which is the *relative* checkpoint
(sha256 715fade1... = HF depth-anything/Depth-Anything-V2-Small), so its output is sigmoid(disparity logit) x 20
(research 13 §1.1). Here the metric checkpoint is loaded with strict=True.

Camera model: none. The network has no intrinsics input; its metric scale is implicitly tied to the Hypersim training
camera (f = 886.81 px at 1024 wide, f/W = 0.866, HFOV 60.0°). Redwood is f = 525 px at 640 wide (f/W = 0.820).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from dvslam.depth import WEIGHTS_ROOT, add_third_party, read_bgr
from dvslam.odometry.base import Frame

HYPERSIM_F_OVER_W = 886.81 / 1024.0
_CONFIGS = {'vits': dict(features=64, out_channels=[48, 96, 192, 384])}


class DAv2Metric:
    name = 'dav2s'

    def __init__(self, weights=None, encoder: str = 'vits', max_depth: float = 20.0, input_size: int = 518,
                 fp16: bool = False, device: str = 'cuda'):
        add_third_party('depth_anything_v2')
        from dav2_metric.dpt import DepthAnythingV2
        weights = weights or WEIGHTS_ROOT / 'dav2' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
        self.model = DepthAnythingV2(encoder=encoder, max_depth=max_depth, **_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.model = self.model.to(device).eval()
        self.input_size, self.fp16, self.device = input_size, fp16, device
        self.last_forward_ms = float('nan')

    @torch.inference_mode()
    def predict_image(self, bgr: np.ndarray) -> tuple[np.ndarray, None]:
        # Official preprocessing: BGR->RGB, /255, lower-bound resize to `input_size`, multiple of 14, INTER_CUBIC,
        # ImageNet normalisation (metric_depth/depth_anything_v2/dpt.py image2tensor).
        x, (h, w) = self.model.image2tensor(bgr, self.input_size)
        x = x.to(self.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            depth = self.model(x)
        depth = F.interpolate(depth[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0]
        torch.cuda.synchronize()
        self.last_forward_ms = (time.perf_counter() - t0) * 1e3
        return depth.cpu().numpy().astype(np.float32), None

    def predict(self, frame: Frame) -> tuple[np.ndarray, None]:
        return self.predict_image(read_bgr(frame.image_path))
