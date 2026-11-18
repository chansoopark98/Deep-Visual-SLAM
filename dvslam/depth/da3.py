"""Depth Anything 3 DA3METRIC-LARGE, minimal copy of the official network code (third_party/depth_anything_3).

Camera model: canonical focal 300 px. metric depth = f * net_output / 300 with f = (fx + fy) / 2 at the *processed*
resolution (README FAQ; `utils/alignment.py apply_metric_scaling`). Redwood fx = fy = 525 at 640x480 is scaled by the
resize factor (504/640 at the default process_res) before use, so the true focal is applied.

Mirrors the official mono path (`api.py`, `model/da3.py` DepthAnything3Net.forward) for a single view:
upper-bound resize to `process_res` (cv2 INTER_AREA when shrinking), nearest multiple of 14, ImageNet normalisation;
backbone under autocast, DPT head with autocast disabled, sky pixels (sky < 0.3) set to the 99th percentile of
non-sky depth (exact quantile here; upstream subsamples 100k pixels at random). Upstream autocasts in bf16;
`fp16=True` autocasts the backbone in fp16, `fp16=False` runs everything in fp32.
"""
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dvslam.depth import WEIGHTS_ROOT, add_third_party, read_bgr
from dvslam.odometry.base import Frame

CANONICAL_FOCAL = 300.0
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _nearest_multiple(x: int, p: int = 14) -> int:
    down = (x // p) * p
    up = down + p
    return up if abs(up - x) <= abs(x - down) else down


class DA3Metric:
    name = 'da3m'

    def __init__(self, K: np.ndarray, weights=None, process_res: int = 504, fp16: bool = False, device: str = 'cuda'):
        add_third_party('depth_anything_3')
        from safetensors.torch import load_file
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dpt import DPT
        # Config da3metric-large.yaml (identical to the HF repo's config.json).
        net = nn.Module()
        net.backbone = DinoV2(name='vitl', out_layers=[4, 11, 17, 23], alt_start=-1, qknorm_start=-1, rope_start=-1,
                              cat_token=False)
        net.head = DPT(dim_in=1024, output_dim=1, features=256, out_channels=[256, 512, 1024, 1024])
        sd = load_file(str(weights or WEIGHTS_ROOT / 'da3' / 'model.safetensors'))
        net.load_state_dict({k.removeprefix('model.'): v for k, v in sd.items()}, strict=True)
        self.net = net.to(device).eval()
        self.K = np.asarray(K, np.float64)
        self.process_res, self.fp16, self.device = process_res, fp16, device
        self.last_forward_ms = float('nan')

    def _preprocess(self, bgr: np.ndarray) -> tuple[torch.Tensor, float]:
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = self.process_res / max(h0, w0)
        w1, h1 = max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))
        if (w1, h1) != (w0, h0):
            rgb = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        w2, h2 = _nearest_multiple(w1), _nearest_multiple(h1)
        if (w2, h2) != (w1, h1):
            up = w2 > w1 or h2 > h1
            rgb = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
        sx, sy = w2 / w0, h2 / h0
        focal = (self.K[0, 0] * sx + self.K[1, 1] * sy) / 2.0
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None, None].contiguous()  # (B=1, S=1, 3, H, W)
        return x.to(self.device), focal

    @torch.inference_mode()
    def predict_image(self, bgr: np.ndarray) -> tuple[np.ndarray, None]:
        h0, w0 = bgr.shape[:2]
        x, focal = self._preprocess(bgr)
        H, W = x.shape[-2:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            feats, _ = self.net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy='saddle_balanced')
        with torch.autocast('cuda', enabled=False):
            feats = [tuple(t.float() if torch.is_tensor(t) else t for t in f) for f in feats]
            out = self.net.head(feats, H, W, patch_start_idx=0)
        depth, sky = out['depth'][0, 0].float(), out['sky'][0, 0].float()
        non_sky = sky < 0.3
        if 10 < int(non_sky.sum()) < non_sky.numel() - 10:
            depth = torch.where(non_sky, depth, torch.quantile(depth[non_sky].flatten(), 0.99))
        depth = depth * (focal / CANONICAL_FOCAL)
        depth = F.interpolate(depth[None, None], (h0, w0), mode='bilinear', align_corners=False)[0, 0]
        torch.cuda.synchronize()
        self.last_forward_ms = (time.perf_counter() - t0) * 1e3
        return depth.cpu().numpy().astype(np.float32), None

    def predict(self, frame: Frame) -> tuple[np.ndarray, None]:
        return self.predict_image(read_bgr(frame.image_path))
