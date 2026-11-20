"""PyTorch port of the photometric pair refinement in klt_photo.py (plan v3.4 §14.4 S1), differentiable end to end.

Same model and the same Levenberg-Marquardt schedule as klt_photo.refine_prepared: X1 = R X0 + t, residual
r = ratio · F1(π(X1)) − (a · F0(p) + b), left perturbation (v, ω) of [R | t] plus the affine brightness a, b, Huber IRLS
weights, λ ×10 / ÷10, early stops. The per-point work (warp, bilinear sampling, Jacobian, normal equations, cost) runs
batched on the device; the 8 × 8 solve, the SO(3) exponential and the pose state stay float64 on the CPU (as numpy /
LAPACK do), so the float64 path follows the numpy one up to reduction order.

Generalisations for learned alignment (S2), both defaulting to the numpy behaviour:
  features : a pyramid level is an (h, w, 3C + 1) stack [F (C), ∂F/∂x (C), ∂F/∂y (C), valid]; C = 1 with the image
             is klt_photo.prepare's [I, ∂x, ∂y, valid] layout. prepare_torch builds it from a (C, h, w) tensor with
             differentiable Sobel / 8 and pyrDown (cv2 reflect-101 borders). a, b are shared by all channels.
  weight   : (n,) per-point weights multiplying the Huber IRLS weight in the normal equations and the robust cost.
Gradients: every accepted step d = −(H + λ diag H)⁻¹ g is a differentiable function of the residuals, Jacobians and
weights, so a loss on the returned (R, t) back-propagates through all accepted LM steps into the feature maps, their
gradient channels and the weights (accept / reject decisions and λ are piecewise constant). Points, depths, K and
the initial pose may also carry gradients.

The tracker path (refine_prepared_torch) is a drop-in for klt_photo.refine_prepared on its numpy stacks: dtype
float64 (default) or float32 (per-point work in float32; normal equations, costs and the solve in float64).
"""
import numpy as np
import torch
import torch.nn.functional as F

_DEV_CACHE: dict = {}          # id(numpy stack) -> (stack, device tensor): stacks are reused by the next pair
_DEV_CACHE_MAX = 8
_NB = 256                      # row blocks of the normal-equation reduction


def _upload(a: np.ndarray, device, dtype) -> torch.Tensor:
    """numpy stack -> device tensor, cached on the array's identity (the array is kept alive while cached)."""
    key = (id(a), str(device), dtype)
    hit = _DEV_CACHE.get(key)
    if hit is not None and hit[0] is a:
        return hit[1]
    x = torch.from_numpy(np.ascontiguousarray(a)).to(device).to(dtype)
    if len(_DEV_CACHE) >= _DEV_CACHE_MAX:
        _DEV_CACHE.pop(next(iter(_DEV_CACHE)))
    _DEV_CACHE[key] = (a, x)
    return x


def _conv_reflect(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """(C, h, w) correlated with a 2-D kernel per channel, cv2 BORDER_REFLECT_101 (torch 'reflect')."""
    p = k.shape[-1] // 2
    y = F.pad(x[:, None], (p, p, p, p), mode='reflect')
    return F.conv2d(y, k[None, None].to(x))[:, 0]


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_PYR = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
_PYR = torch.outer(_PYR, _PYR) / 256


def prepare_torch(feat: torch.Tensor, valid: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """klt_photo.prepare for a (C, h, w) (or (h, w)) feature tensor: per level an (h, w, 3C + 1) stack
    [F, ∂F/∂x, ∂F/∂y, valid] with Sobel / 8 gradients and cv2.pyrDown between levels (valid pyrDown-ed as a float).
    Differentiable in feat."""
    f = feat[None] if feat.ndim == 2 else feat
    m = valid.to(f.dtype)[None]
    out = []
    for lv in range(levels):
        if lv:
            f, m = _conv_reflect(f, _PYR)[:, ::2, ::2], _conv_reflect(m, _PYR)[:, ::2, ::2]
        gx = _conv_reflect(f, _SOBEL_X) / 8
        gy = _conv_reflect(f, _SOBEL_X.T) / 8
        out.append(torch.cat([f, gx, gy, m]).permute(1, 2, 0))
    return out


def sample(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """klt_photo._sample: bilinear sample of (h, w) or (h, w, K) at uv (n, 2), positions clamped to the image."""
    h, w = img.shape[:2]
    flat = img.reshape(h * w, -1)
    x, y = uv[:, 0].clamp(0, w - 1.001), uv[:, 1].clamp(0, h - 1.001)
    x0, y0 = x.detach().floor().long(), y.detach().floor().long()
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    i = y0 * w + x0
    out = ((1 - fy) * ((1 - fx) * flat[i] + fx * flat[i + 1])
           + fy * ((1 - fx) * flat[i + w] + fx * flat[i + w + 1]))
    return out[:, 0] if img.ndim == 2 else out


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrix of a rotation vector (Rodrigues; second-order series below 1e-6 rad)."""
    th = torch.linalg.norm(w)
    Kx = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -w[2], w[1], w[2], -w[0], -w[1], w[0]
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    if float(th.detach()) < 1e-6:
        return eye + Kx + 0.5 * Kx @ Kx
    return eye + torch.sin(th) / th * Kx + (1 - torch.cos(th)) / th ** 2 * (Kx @ Kx)


def refine_torch(P0: list, P1: list, ratio: float, pts: torch.Tensor, z: torch.Tensor, K, R, t, levels: int = 2,
                 iters: int = 10, huber: float = 0.03, tol: float = 0.0, weight: torch.Tensor | None = None):
    """Differentiable klt_photo.refine_prepared. P0 / P1: per-level (h, w, 3C + 1) device tensors (reference frame
    t-1 / frame t, the latter scaled by ratio); pts (n, 2) full-resolution pixels of the reference frame, z (n,)
    depths; K (3, 3); R (3, 3), t (3,) the initial X1 = R X0 + t; weight (n,) per-point weights (None = 1).
    Returns (R, t, info): float64 CPU tensors carrying the autograd graph, info = {'cost0', 'cost', 'n', 'a', 'b'}
    (robust mean cost at full resolution, start / end, as in numpy)."""
    dev, dt = P1[0].device, P1[0].dtype
    C = (P1[0].shape[-1] - 1) // 3
    f64 = {'dtype': torch.float64}
    K = torch.as_tensor(K, **f64)
    R, t = torch.as_tensor(R, **f64).cpu(), torch.as_tensor(t, **f64).cpu()
    a, b = torch.ones((), **f64), torch.zeros((), **f64)
    pts, z = torch.as_tensor(pts, device=dev), torch.as_tensor(z, device=dev)
    Kd = K.to(dev)
    ray = torch.cat([pts.to(torch.float64), torch.ones_like(pts[:, :1], dtype=torch.float64)], 1) @ \
        torch.linalg.inv(Kd).T
    X0 = (ray * z.to(torch.float64)[:, None]).to(dt)
    wpt = None if weight is None else torch.as_tensor(weight, device=dev).to(dt)
    scale = torch.tensor([ratio] * (3 * C) + [1.0], dtype=dt, device=dev)

    def setup(lv):
        s = 0.5 ** lv
        Kl = K.clone()
        Kl[:2] *= s
        Kl[0, 2], Kl[1, 2] = (K[0, 2] + 0.5) * s - 0.5, (K[1, 2] + 0.5) * s - 0.5
        ref = sample(P0[lv][..., :C], ((pts.to(torch.float64) + 0.5) * s - 0.5).to(dt))
        return Kl.to(dev).to(dt), ref

    def residual(Kl, ref, lv, R, t, a, b):
        Rd, td = R.to(dev).to(dt), t.to(dev).to(dt)
        X1 = (X0[:, None, :] * Rd[None]).sum(-1) + td
        Z = X1[:, 2]
        uv = X1[:, :2] / Z.clamp_min(1e-6)[:, None] * torch.stack([Kl[0, 0], Kl[1, 1]]) + Kl[:2, 2]
        h, w = P1[lv].shape[:2]
        S = sample(P1[lv], uv) * scale
        ok = (Z > 1e-3) & (uv[:, 0] > 1) & (uv[:, 1] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] < h - 2) & (S[:, -1] > 0.99)
        r = S[:, :C] - (a.to(dev).to(dt) * ref + b.to(dev).to(dt))
        e = r.abs()
        rho = torch.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber))
        if wpt is not None:
            rho = rho * wpt[:, None]
        tot = torch.stack([torch.where(ok[:, None], rho, 0).to(torch.float64).sum(), ok.sum().to(torch.float64)])
        s_c, n = tot.detach().cpu().tolist()
        c = s_c / (n * C) if n > 0 else np.inf
        return r, ok, X1, S[:, C:3 * C], c, int(n)

    def normal_eq(Kl, ref, r, ok, X1, g):
        Zs = torch.where(ok, X1[:, 2], 1)[:, None]
        X, Y = X1[:, 0:1], X1[:, 1:2]
        gx, gy = g[:, :C], g[:, C:]
        fx, fy = Kl[0, 0], Kl[1, 1]
        gu = torch.stack([gx * fx / Zs, gy * fy / Zs, -(gx * fx * X + gy * fy * Y) / Zs ** 2], -1)   # (n, C, 3)
        Jw = torch.cat([gu, torch.linalg.cross(X1[:, None, :].expand_as(gu), gu, dim=-1)], -1)
        J = torch.cat([Jw, -ref[..., None], -torch.ones_like(ref)[..., None]], -1).reshape(-1, 8).to(torch.float64)
        wgt = torch.minimum(torch.ones_like(r), huber / r.abs().clamp_min(1e-12))
        if wpt is not None:
            wgt = wgt * wpt[:, None]
        wgt = torch.where(ok[:, None], wgt, 0).reshape(-1).to(torch.float64)
        # J^T W [J | r] as a sum of _NB block products: one (8 x n) @ (n x 9) GEMM gets no parallelism from cuBLAS
        # (measured 16 ms per call for n = 250k in float64 on an RTX 5090)
        A = torch.cat([J, r.reshape(-1, 1).to(torch.float64)], 1)
        A = F.pad(A, (0, 0, 0, -len(A) % _NB)).view(_NB, -1, 9)
        Wt = F.pad(J * wgt[:, None], (0, 0, 0, -len(J) % _NB)).view(_NB, -1, 8).transpose(1, 2)
        HG = torch.bmm(Wt, A).sum(0).cpu()
        return HG[:, :8], HG[:, 8]

    Kl, ref = setup(0)
    info = {'cost0': residual(Kl, ref, 0, R, t, a, b)[4]}
    c, n = info['cost0'], 0
    for lv in range(levels - 1, -1, -1):
        Kl, ref = setup(lv)
        lam = 1e-3
        r, ok, X1, g, c, n = residual(Kl, ref, lv, R, t, a, b)
        for _ in range(iters):
            if n < 30:
                break
            H, gr = normal_eq(Kl, ref, r, ok, X1, g)
            c_old, step = c, None
            while lam < 1e6:
                d = -torch.linalg.solve(H + lam * torch.diag(torch.diag(H) + 1e-12), gr)
                dR = so3_exp(d[3:6])
                Rn, tn, an, bn = dR @ R, dR @ t + d[:3], a + d[6], b + d[7]
                rn, okn, X1n, gn, cn, nn = residual(Kl, ref, lv, Rn, tn, an, bn)
                if cn < c:
                    R, t, a, b, r, ok, X1, g, c, n = Rn, tn, an, bn, rn, okn, X1n, gn, cn, nn
                    lam, step = max(lam / 10, 1e-7), d
                    break
                lam *= 10
            if step is None:
                break
            if float(torch.linalg.norm(step[:6].detach())) < 1e-7 or (tol > 0 and c_old - c < tol * c_old):
                break
    info.update(cost=c, n=n, a=a, b=b)
    return R, t, info


def refine_prepared_torch(P0: list, P1: list, ratio: float, pts: np.ndarray, z: np.ndarray, K: np.ndarray,
                          R: np.ndarray, t: np.ndarray, levels: int = 2, iters: int = 10, huber: float = 0.03,
                          tol: float = 0.0, dtype: torch.dtype = torch.float64, device=None):
    """Drop-in for klt_photo.refine_prepared on its numpy prepare() stacks (no autograd). Returns numpy (R, t, info)
    with info = {'cost0', 'cost', 'n'}. device None = CUDA when available."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        S0 = [_upload(p, dev, dtype) for p in P0[:levels]]
        S1 = [_upload(p, dev, dtype) for p in P1[:levels]]
        Rt, tt, info = refine_torch(S0, S1, ratio, torch.from_numpy(np.asarray(pts, np.float64)).to(dev),
                                    torch.from_numpy(np.asarray(z, np.float64)).to(dev), K, R, t, levels, iters,
                                    huber, tol)
    return Rt.numpy(), tt.numpy(), {'cost0': info['cost0'], 'cost': info['cost'], 'n': info['n']}
