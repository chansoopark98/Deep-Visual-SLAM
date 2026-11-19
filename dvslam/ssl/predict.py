"""The one inference path of the self-supervised model (validation, odometry wrapper, depth prior).

Input: gray uint8 images (H, W) or (B, H, W), numpy or torch, converted exactly like the dataset's 'net' images
(gray / 255 replicated to 3 channels). increment(prev, cur) = T_{prev<-cur} = inv(T_wc[prev]) @ T_wc[cur] (float64,
metres); depth(img) = metric z-depth (float32, metres) at the input resolution.
"""
from pathlib import Path

import numpy as np
import torch

from dvslam.ssl.models import SSLModel


def build_model(model_cfg: dict, pretrained: bool) -> SSLModel:
    return SSLModel(model_cfg['min_depth'], model_cfg['max_depth'], model_cfg['uncertainty'], model_cfg['sigma_min'],
                    model_cfg['pose_iters'], pretrained)


class Predictor:
    def __init__(self, model: SSLModel, K: np.ndarray, size: tuple[int, int], device: str | torch.device):
        """size = (H, W) the model was trained at; K (3, 3) at that size."""
        self.model = model.to(device).eval()
        self.K = np.asarray(K, np.float64)
        self.size = tuple(size)
        self.device = torch.device(device)
        self._K = torch.as_tensor(self.K, dtype=torch.float32, device=self.device)[None]

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = 'cuda', ema: bool = True) -> 'Predictor':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = build_model(ckpt['config']['model'], pretrained=False)
        if ema:
            sd = {k[len('module.'):]: v for k, v in ckpt['ema'].items() if k.startswith('module.')}
        else:
            sd = ckpt['model']
        model.load_state_dict(sd)
        return cls(model, ckpt['K'], ckpt['size'], device)

    def _prep(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(img).to(self.device)
        if x.dtype != torch.uint8:
            raise TypeError(f'expected gray uint8 images, got {x.dtype}')
        x = x[None] if x.ndim == 2 else x
        if tuple(x.shape[-2:]) != self.size:
            raise ValueError(f'image size {tuple(x.shape[-2:])} != model size {self.size}')
        return (x.float() / 255)[:, None].expand(-1, 3, -1, -1)

    @torch.no_grad()
    def increment(self, img_prev: np.ndarray | torch.Tensor, img_cur: np.ndarray | torch.Tensor) -> np.ndarray:
        """(4, 4) for single images, (B, 4, 4) for batches."""
        a, b = self._prep(img_prev), self._prep(img_cur)
        T = self.model.pose(a, b, self._K.expand(len(a), 3, 3))[0].double().cpu().numpy()
        return T[0] if img_prev.ndim == 2 else T

    @torch.no_grad()
    def depth(self, img: np.ndarray | torch.Tensor) -> np.ndarray:
        """(H, W) for a single image, (B, H, W) for a batch."""
        d = self.model.depth_m(self._prep(img))[:, 0].float().cpu().numpy()
        return d[0] if img.ndim == 2 else d
