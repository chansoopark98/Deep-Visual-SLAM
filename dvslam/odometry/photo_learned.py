"""Plan v3.4 §14.4 S2: learned alignment features / per-pixel weights for the photometric pair refinement, trained
through the unrolled LM of klt_photo_torch.refine_torch (apps/train_photo_s2.py; no ground truth in any loss), and the
tracker hook (KLTWindowOdometry photo_feat=<ckpt>, needs photo_backend=torch|torch32).

PhotoNet: a small U-Net (4 scales, receptive field ~100 px) on ONE frame's normalised image I and valid mask M (the
level-0 channels 0 and 3 of a klt_photo.prepare stack, i.e. exactly what the tracker already holds) ->
  weights  (mode w / fw): per-pixel log-weight ℓ (centred per image, soft-clamped to ±LOGW_MAX by tanh); the weight
             of a point of one LM call is exp(ℓ) normalised to mean 1 over that call's points (point_weights:
             scale-free, cannot collapse to zero).
  features (mode f / fw): one alignment channel F = I + δ used instead of I (prepare_torch builds its gradients).
Both heads are zero-initialised: an untrained net reproduces the S1 refinement (weights mode bit for bit, since all
weights are exactly 1 and the numpy stacks are used unchanged; feature modes up to prepare_torch vs cv2, ~1e-5).
A frame's outputs are computed once and serve both pairs it belongs to (reference of t -> t+1, target of t-1 -> t).

Training losses (functions below; the plan's choice: sparse-track reprojection is the main loss, no NLL):
  reproj_loss    robust (Cauchy) reprojection error of the tracker's sparse tracks under a refined (or composed
                 multi-gap) pose, with the frozen depth of the source frame in map units.
  bidir term     |t_f − t_b| of the two refinement directions (train script; tracker-estimated step >= 5 mm only).
  photo_cost     raw-intensity photometric cost of a pose (unweighted, a / b fitted in closed form): the anchor that
                 keeps learned features / weights from moving the pose off the image evidence.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dvslam.odometry.klt_photo_torch import (
    _upload,
    prepare_torch,
    refine_torch,
    sample,
    so3_exp,
)

LOGW_MAX = 3.0
MODES = ('w', 'f', 'fw')


class PhotoNet(nn.Module):
    """(B, 1, H, W) image I and valid mask M -> (feature (B, 1, H, W) or None, log-weight (B, 1, H, W) or None)."""

    def __init__(self, mode: str = 'w', width: int = 16):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
        self.mode, self.width = mode, width
        w = width

        def block(cin, cout, stride=1):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(cout, cout, 3, 1, 1), nn.LeakyReLU(0.1))
        self.e0, self.e1, self.e2, self.e3 = block(2, w), block(w, 2 * w, 2), block(2 * w, 4 * w, 2), \
            block(4 * w, 4 * w, 2)
        self.d2, self.d1, self.d0 = block(8 * w, 4 * w), block(6 * w, 2 * w), block(3 * w, w)
        self.n_out = ('w' in mode) + ('f' in mode)
        self.head = nn.Conv2d(w, self.n_out, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, img: torch.Tensor, valid: torch.Tensor):
        m = valid.to(img.dtype)
        x0 = self.e0(torch.cat([(img - 0.5) * m, m], 1))
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)

        def up(a, b):
            return torch.cat([F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1)
        y = self.d0(up(self.d1(up(self.d2(up(x3, x2)), x1)), x0))
        out = self.head(y)
        k = 0
        feat = logw = None
        if 'f' in self.mode:
            feat, k = img + out[:, :1], 1
        if 'w' in self.mode:                           # centred on the valid pixels, soft-clamped (a hard clamp
            o = out[:, k:k + 1]                        # saturated in a pilot and stopped all gradients)
            o = o - (o * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp_min(1)
            logw = LOGW_MAX * torch.tanh(o / LOGW_MAX)
        return feat, logw


def point_weights(logw: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """(H, W) log-weight map at the integer pixel centres pts (n, 2) -> (n,) weights exp(ℓ) / mean exp(ℓ)."""
    u, v = pts[:, 0].round().long(), pts[:, 1].round().long()
    w = torch.exp(logw[v, u])
    n = torch.full((), len(w), dtype=w.dtype, device=w.device)   # a device tensor: true division (CUDA turns a
    return w / (w.sum() / n)                        # Python divisor into × 1/n), so a constant map gives exactly 1


def save_net(path, net: PhotoNet, extra: dict | None = None) -> None:
    torch.save({'mode': net.mode, 'width': net.width, 'state': net.state_dict(), 'extra': extra or {}}, path)


def load_net(path, device=None) -> PhotoNet:
    ck = torch.load(Path(path), map_location='cpu', weights_only=True)
    net = PhotoNet(ck['mode'], ck['width'])
    net.load_state_dict(ck['state'])
    return net.to(device or 'cpu').eval()


class LearnedPhoto:
    """Tracker drop-in for klt_photo_torch.refine_prepared_torch (same arguments) that runs PhotoNet on the two
    frames of the pair (outputs cached on the numpy stack's identity, like klt_photo_torch._upload)."""

    _CACHE_MAX = 4

    def __init__(self, ckpt, device=None):
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net = load_net(ckpt, self.dev)
        self._cache: dict = {}

    def _frame(self, P: list, levels: int, dtype):
        key = (id(P[0]), dtype, levels)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is P[0]:
            return hit[1]
        img = torch.from_numpy(np.ascontiguousarray(P[0][..., 0])).to(self.dev, torch.float32)
        valid = torch.from_numpy(np.ascontiguousarray(P[0][..., 3] > 0.5)).to(self.dev)
        feat, logw = self.net(img[None, None], valid[None, None])
        if feat is None:
            S = [_upload(p, self.dev, dtype) for p in P[:levels]]
        else:
            S = prepare_torch(feat[0].to(dtype), valid, levels)
        out = (S, None if logw is None else logw[0, 0])
        if len(self._cache) >= self._CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (P[0], out)
        return out

    def __call__(self, P0: list, P1: list, ratio: float, pts: np.ndarray, z: np.ndarray, K: np.ndarray, R: np.ndarray,
                 t: np.ndarray, levels: int = 2, iters: int = 10, huber: float = 0.03, tol: float = 0.0,
                 dtype: torch.dtype = torch.float64):
        with torch.no_grad():
            S0, lw0 = self._frame(P0, levels, dtype)
            S1, _ = self._frame(P1, levels, dtype)
            pt = torch.from_numpy(np.asarray(pts, np.float64)).to(self.dev)
            w = None if lw0 is None else point_weights(lw0, pt)
            Rt, tt, info = refine_torch(S0, S1, ratio, pt, torch.from_numpy(np.asarray(z, np.float64)).to(self.dev),
                                        K, R, t, levels, iters, huber, tol, w)
        return Rt.numpy(), tt.numpy(), {'cost0': info['cost0'], 'cost': info['cost'], 'n': info['n']}


# ------------------------------------------------------------------ poses (4 x 4 float64 tensors, autograd-friendly)
def to_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.cat([R, t[:, None]], 1), torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=R.dtype)], 0)


def inv_T(T: torch.Tensor) -> torch.Tensor:
    R, t = T[:3, :3], T[:3, 3]
    return to_T(R.T, -R.T @ t)


def so3_log_small(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector of a rotation by θ ≪ 1 rad: asin(|v|) v / |v|, v = vee(R − Rᵀ) / 2 (exact for θ < π / 2; used
    only for the midpoint of the two refinement directions, θ ~ 1e-4 rad)."""
    v = 0.5 * torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = torch.linalg.norm(v)
    if float(s.detach()) < 1e-12:
        return v
    return v * (torch.asin(s.clamp(max=1.0)) / s)


def pose_mid(Ta: torch.Tensor, Tb: torch.Tensor) -> torch.Tensor:
    """The tracker's photo_bidir midpoint: mean translation, geodesic rotation midpoint Ra exp(log(Raᵀ Rb) / 2)."""
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    R = Ra @ so3_exp(0.5 * so3_log_small(Ra.T @ Rb))
    return to_T(R, 0.5 * (Ta[:3, 3] + Tb[:3, 3]))


# ------------------------------------------------------------------ losses
def cauchy(e2: torch.Tensor, c: float) -> torch.Tensor:
    """Cauchy (Lorentzian) ρ of a squared residual: c² / 2 · log(1 + e² / c²) (≈ e² / 2 for e ≪ c)."""
    return 0.5 * c * c * torch.log1p(e2 / (c * c))


def reproj_loss(T: torch.Tensor, p_src: torch.Tensor, z_src: torch.Tensor, p_dst: torch.Tensor, K, c: float = 0.5):
    """Mean Cauchy reprojection error (px²) of tracks p_src (n, 2) with depths z_src (n,) in the source camera, mapped
    by X_dst = T X_src (T 4 x 4) and compared to p_dst (n, 2). Points behind the camera are dropped."""
    K = torch.as_tensor(K, dtype=T.dtype)
    X = torch.cat([p_src, torch.ones_like(p_src[:, :1])], 1) @ torch.linalg.inv(K).T * z_src[:, None]
    Y = X @ T[:3, :3].T + T[:3, 3]
    ok = Y[:, 2] > 1e-3
    if not bool(ok.any()):
        return T.sum() * 0
    Y = Y[ok]
    uv = Y[:, :2] / Y[:, 2:3] * torch.stack([K[0, 0], K[1, 1]]) + K[:2, 2]
    return cauchy(((uv - p_dst[ok]) ** 2).sum(1), c).mean()


def photo_cost(P0: list, P1: list, ratio: float, pts: torch.Tensor, z: torch.Tensor, K, R: torch.Tensor,
               t: torch.Tensor, huber: float = 0.03) -> torch.Tensor:
    """Unweighted Huber photometric cost of X1 = R X0 + t on raw level-0 stacks (h, w, 4) [I, ∂x, ∂y, valid], with the
    affine brightness a, b fitted by least squares (closed form, differentiable): the anchor of the learned modes."""
    dev, dt = P1[0].device, P1[0].dtype
    K = torch.as_tensor(K, dtype=torch.float64)
    ray = torch.cat([pts, torch.ones_like(pts[:, :1])], 1).to(torch.float64) @ torch.linalg.inv(K).T.to(dev)
    X0 = ray * z.to(torch.float64)[:, None]
    X1 = X0 @ R.to(dev).T + t.to(dev)
    Z = X1[:, 2]
    Kd = K.to(dev)
    uv = X1[:, :2] / Z.clamp_min(1e-6)[:, None] * torch.stack([Kd[0, 0], Kd[1, 1]]) + Kd[:2, 2]
    h, w = P1[0].shape[:2]
    S = sample(P1[0], uv.to(dt))
    ok = (Z > 1e-3) & (uv[:, 0] > 1) & (uv[:, 1] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] < h - 2) & (S[:, -1] > 0.99)
    y = (ratio * S[ok, 0]).to(torch.float64)
    x = P0[0][pts[ok, 1].round().long(), pts[ok, 0].round().long(), 0].to(torch.float64)
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp_min(1e-12)
    e = (y - (a * (x - xm) + ym)).abs()
    return torch.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber)).mean()
