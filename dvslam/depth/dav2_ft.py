"""Fine-tuned DAv2-S metric depth prior (plan v3 Phase 3, §4.5). DepthPrior: predict(frame) -> (depth_m, None).

Fixed deployment pipeline (must match training, `dvslam.train.depth_data`): the image is resized to 476x630 (H x W)
with `prepare_image`, the network predicts depth for the canonical camera (Redwood K: f/W = 525/640), and the output
is converted to the input camera by d = d_canon / canon_factor(K), then bilinearly resized to the input resolution.
On Redwood (K = deployment K) the factor is 1. With `ckpt=None` the original Hypersim weights run through the same
fixed pipeline (zero-shot reference at the deployment resolution).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dvslam.depth import WEIGHTS_ROOT, add_third_party, read_bgr
from dvslam.odometry.base import Frame
from dvslam.train.depth_data import DEPLOY_K, canon_factor, prepare_image

_CONFIGS = {'vits': dict(features=64, out_channels=[48, 96, 192, 384])}


def load_dav2_metric(ckpt: str | Path | None = None, encoder: str = 'vits', max_depth: float = 20.0):
    """DepthAnythingV2 metric network with Hypersim weights (ckpt=None) or a Phase-3 checkpoint ({'model': sd})."""
    add_third_party('depth_anything_v2')
    from dav2_metric.dpt import DepthAnythingV2
    net = DepthAnythingV2(encoder=encoder, max_depth=max_depth, **_CONFIGS[encoder])
    path = ckpt or WEIGHTS_ROOT / 'dav2' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
    sd = torch.load(path, map_location='cpu', weights_only=True)
    net.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True)
    return net


class DAv2FT:
    name = 'dav2s_ft'

    def __init__(self, ckpt: str | Path | None = None, K: np.ndarray = DEPLOY_K, width: int = 640,
                 fp16: bool = False, device: str = 'cuda'):
        self.model = load_dav2_metric(ckpt).to(device).eval()
        K = np.asarray(K, np.float64)
        self.factor = canon_factor(K[0, 0], K[1, 1], width)
        self.fp16, self.device = fp16, device
        self.last_forward_ms = float('nan')

    @torch.inference_mode()
    def predict_image(self, bgr: np.ndarray) -> tuple[np.ndarray, None]:
        h, w = bgr.shape[:2]
        x = torch.from_numpy(prepare_image(bgr))[None].to(self.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            d = self.model(x)
        d = F.interpolate(d[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0] / self.factor
        torch.cuda.synchronize()
        self.last_forward_ms = (time.perf_counter() - t0) * 1e3
        return d.cpu().numpy().astype(np.float32), None

    def predict(self, frame: Frame) -> tuple[np.ndarray, None]:
        return self.predict_image(read_bgr(frame.image_path))
