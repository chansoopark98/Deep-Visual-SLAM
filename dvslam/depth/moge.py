"""MoGe-2 ViT-S (moge-2-vits-normal), minimal copy of the official network code (third_party/moge).

Camera model: MoGe-2 predicts an affine-invariant point map plus one global `metric_scale` from the class token.
With `use_fov=True` (default) the true horizontal FoV from K (Redwood: 2·atan(320/525) = 62.7°) fixes the focal and
only the z-shift is solved; with `use_fov=False` focal and shift are both solved from the point map (upstream default
when fov_x is None). Mirrors upstream `MoGeModel.infer` (moge/model/v2.py) without `force_projection` (only depth is
needed): forward at `num_tokens` (default resolution_level 9 -> 3600 tokens), mask > 0.5, shift by scipy LM on a
64x64 downsample (CPU), depth = (z + shift) · metric_scale, masked pixels = inf. Upstream autocasts the forward in fp16
by default; `fp16=False` runs it in fp32. The post-processing is always fp32.
conf = predicted validity-mask probability.
"""
import math
import time

import numpy as np
import torch

from dvslam.depth import WEIGHTS_ROOT, add_third_party, read_bgr
from dvslam.odometry.base import Frame


class MoGe2:
    name = 'moge2s'

    def __init__(self, K: np.ndarray, weights=None, num_tokens: int | None = None, resolution_level: int = 9,
                 use_fov: bool = True, fp16: bool = False, device: str = 'cuda'):
        add_third_party('moge')
        from moge.model.v2 import MoGeModel
        ckpt = torch.load(weights or WEIGHTS_ROOT / 'moge' / 'model.pt', map_location='cpu', weights_only=True)
        model = MoGeModel(**ckpt['model_config'])
        model.load_state_dict(ckpt['model'], strict=True)
        self.model = model.to(device).eval()
        lo, hi = self.model.num_tokens_range
        self.num_tokens = num_tokens or int(lo + resolution_level / 9 * (hi - lo))
        self.K = np.asarray(K, np.float64)
        self.use_fov, self.fp16, self.device = use_fov, fp16, device
        self.last_forward_ms = float('nan')

    @torch.inference_mode()
    def predict_image(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from moge.utils.geometry import recover_focal_shift
        h, w = bgr.shape[:2]
        img = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(self.device)
        img = img.permute(2, 0, 1)[None].float() / 255.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            out = self.model(img, num_tokens=self.num_tokens)
        points, mask, scale = out['points'].float(), out['mask'].float(), out['metric_scale'].float()
        mask_bin = mask > 0.5
        if self.use_fov:
            aspect = w / h
            fov_x = 2.0 * math.atan(w / (2.0 * self.K[0, 0]))
            focal = torch.tensor([aspect / math.sqrt(1 + aspect ** 2) / math.tan(fov_x / 2)], device=self.device)
            _, shift = recover_focal_shift(points, mask_bin, focal=focal)
        else:
            _, shift = recover_focal_shift(points, mask_bin)
        depth = (points[..., 2] + shift[:, None, None]) * scale[:, None, None]
        mask_bin &= depth > 0
        depth = torch.where(mask_bin, depth, torch.inf)
        torch.cuda.synchronize()
        self.last_forward_ms = (time.perf_counter() - t0) * 1e3
        return depth[0].cpu().numpy().astype(np.float32), mask[0].cpu().numpy().astype(np.float32)

    def predict(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_image(read_bgr(frame.image_path))
