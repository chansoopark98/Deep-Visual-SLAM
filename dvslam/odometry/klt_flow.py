"""Dense optical flow cached on disk as a replacement for KLT frame-to-frame tracking (EXP-007 diagnostic).

`flow_cache=<dir>` in KLTWindowOdometry: every track is moved t-1 -> t by the bilinearly sampled forward flow and
checked by the backward flow (|p0 − (p1 + bwd(p1))| < fb_px), in place of pyramidal LK. Everything else (corner
detection, keyframes, host re-track `kf_refine`, PnP, window BA) is unchanged.
Layout of <dir> (written by scripts/exp007_cache_flow.py): <scene>.fwd.npy / <scene>.bwd.npy (n, 2, H, W) float32,
row t = flow (x, y) in pixels from frame t-1 to t / from t to t-1 (row 0 is zeros), and <scene>.paths.txt with the
image path of frame t on line t. Frames are looked up by image path, as in klt_depth_cache.py. <scene>.meta.json
(klt_depth_cache.write_meta) records the model and the flow H x W: FlowCache(hw=(H, W)) raises ValueError on a
mismatch; a scene without meta.json loads unchecked, as before; loaded metas are kept in `.meta[scene]`.
"""
import json
from pathlib import Path

import numpy as np


def bilinear(field: np.ndarray, px: np.ndarray) -> np.ndarray:
    """field (C, H, W) sampled at sub-pixel px (n, 2) -> (n, C); positions are clamped to the image."""
    _, h, w = field.shape
    x = np.clip(px[:, 0], 0, w - 1.001)
    y = np.clip(px[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f = lambda yy, xx: field[:, yy, xx].T
    return ((1 - fy) * ((1 - fx) * f(y0, x0) + fx * f(y0, x0 + 1))
            + fy * ((1 - fx) * f(y0 + 1, x0) + fx * f(y0 + 1, x0 + 1))).astype(np.float64)


class FlowCache:
    def __init__(self, root: str | Path, hw: tuple[int, int] | None = None):
        self.root, self.hw = Path(root), hw            # hw: (H, W) of the tracked images; None = no check
        self._index: dict[str, tuple[str, int]] | None = None
        self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.meta: dict[str, dict] = {}

    def pair(self, image_path, prev_gray=None, gray=None) -> tuple[np.ndarray, np.ndarray]:
        """(forward flow t-1 -> t, backward flow t -> t-1), each (2, H, W), for the frame t at image_path."""
        if self._index is None:
            self._index = {}
            for f in sorted(self.root.glob('*.paths.txt')):
                scene = f.name[:-len('.paths.txt')]
                self._index.update({p: (scene, i) for i, p in enumerate(f.read_text().splitlines())})
        key = str(image_path)
        if key not in self._index:
            raise KeyError(f'{key} is not in the flow cache {self.root}')
        scene, i = self._index[key]
        if scene not in self._maps:
            m = self.root / f'{scene}.meta.json'
            if m.exists():
                self.meta[scene] = json.loads(m.read_text())
                if self.hw is not None and tuple(self.meta[scene]['hw']) != tuple(self.hw):
                    raise ValueError(f'flow cache {m}: flow is {" x ".join(map(str, self.meta[scene]["hw"]))}, '
                                     f'the tracker needs {self.hw[0]} x {self.hw[1]}')
            self._maps[scene] = tuple(np.load(self.root / f'{scene}.{d}.npy', mmap_mode='r') for d in ('fwd', 'bwd'))
        fwd, bwd = self._maps[scene]
        return np.asarray(fwd[i]), np.asarray(bwd[i])


class OnlineRAFT:
    """EXP-008: torchvision RAFT (official weights) evaluated when the tracker asks for a pair: forward and backward
    flow as one batch of 2, gray replicated to 3 channels in [-1, 1] (as scripts/exp007_cache_flow.py). GPU time of
    every call (CUDA-synchronised wall time) is appended to self.ms."""

    def __init__(self, model: str = 'small', iters: int = 12, device: str = 'cuda'):
        import torch
        from torchvision.models import optical_flow as of
        self.torch, self.iters, self.device = torch, iters, device
        self.net = (of.raft_small(weights=of.Raft_Small_Weights.DEFAULT) if model == 'small'
                    else of.raft_large(weights=of.Raft_Large_Weights.DEFAULT)).eval().to(device)
        self.ms: list[float] = []

    def pair(self, image_path, prev_gray: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import time
        torch = self.torch
        t0 = time.perf_counter()
        with torch.no_grad():
            a, b = (torch.from_numpy(g).to(self.device).float().div(255).mul(2).sub(1)[None].expand(3, -1, -1)
                    for g in (prev_gray, gray))
            f = self.net(torch.stack([a, b]), torch.stack([b, a]), num_flow_updates=self.iters)[-1].float().cpu()
        self.ms.append((time.perf_counter() - t0) * 1e3)
        return f[0].numpy(), f[1].numpy()
