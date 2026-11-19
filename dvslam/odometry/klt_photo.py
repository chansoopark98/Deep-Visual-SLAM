"""Direct photometric refinement of one frame-to-frame increment (EXP-007, `photo=` option of KLTWindowOdometry).

Images: exposure- and vignette-normalised with dvslam/ssl/tumvi_data.py (I = gray / 255 / V, frame t scaled by
e_{t-1} / e_t so both frames are on frame t-1's exposure; pixels with V < v_min or saturated are invalid).
Points: semi-dense, the strongest-gradient valid pixel of every `cell`×`cell` block of frame t-1 (gradient > grad_min),
with depth from the SSL map of frame t-1 rescaled to the tracker's map units (median ratio over the tracked map points)
so the refinement keeps the tracker's metric scale and does not inherit the network's per-frame scale jitter.
Model: X1 = R X0 + t (frame t-1 -> t), residual r = I1(π(X1)) − (a · I0(p) + b), unknowns δ = (v, ω) (left
perturbation of [R | t]) and the affine brightness a, b. Levenberg-Marquardt with Huber weights, coarse to fine over
`levels` pyramid levels (K scaled with the pixel-centre convention). Single pixels, no patch pattern.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class PhotoNorm:
    """Normalised image of a TUM-VI frame from its rectified cache path (scene = the cache path's grandparent).
    wide > 0 (EXP-011): the vignette is remapped to the wide canvas tumvi.wide_K(wide) instead of K_OUT."""

    def __init__(self, v_min: float = 0.25, sat_level: int = 250, wide: int = 0):
        self.v_min, self.sat_level, self.wide = v_min, sat_level, wide
        self._scenes: dict[str, tuple[np.ndarray, dict[str, float]]] = {}

    def exposure(self, image_path) -> float:
        return self._scene(image_path)[1][str(image_path)]

    def _scene(self, image_path):
        scene = Path(image_path).parent.parent.name
        if scene not in self._scenes:
            from dvslam.ssl.tumvi_data import scene_frames
            f = scene_frames(scene)
            V = f['V']['cam0']
            if self.wide:
                from dvslam.data import tumvi
                d = tumvi.sequence_dir(scene)
                v = cv2.imread(str(d / 'dso/cam0/vignette.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
                _, intrinsics, dist = tumvi.load_cam_calib(d, 'cam0')
                V = cv2.remap(v / v.max(), *tumvi.rectify_maps_wide(intrinsics, dist, self.wide), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
            self._scenes[scene] = (V, dict(zip(f['cam0'], f['exposure_ms'].tolist())))
        return self._scenes[scene]

    def __call__(self, image_path, gray: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        from dvslam.ssl.tumvi_data import photometric
        V = self._scene(image_path)[0]
        img, ok = photometric(gray, V, ratio, self.v_min, self.sat_level)
        return img, ok > 0


def select_points(I0: np.ndarray, M0: np.ndarray, cell: int, grad_min: float) -> np.ndarray:
    """(n, 2) pixel centres: per cell×cell block the valid pixel with the largest gradient magnitude > grad_min."""
    gx = cv2.Sobel(I0, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(I0, cv2.CV_32F, 0, 1, ksize=3) / 8
    return select_points_g(gx, gy, M0, cell, grad_min)[0]


def select_points_g(gx: np.ndarray, gy: np.ndarray, M0: np.ndarray, cell: int, grad_min: float,
                    max_pts: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """select_points from precomputed gradients; max_pts > 0 keeps the strongest max_pts. Returns (pts, |∇I|)."""
    G = np.hypot(gx, gy) * cv2.erode(M0.astype(np.uint8), np.ones((5, 5), np.uint8))
    h, w = G.shape
    hc, wc = h // cell, w // cell
    B = G[:hc * cell, :wc * cell].reshape(hc, cell, wc, cell).transpose(0, 2, 1, 3).reshape(hc, wc, -1)
    k = B.argmax(2)
    best = np.take_along_axis(B, k[..., None], 2)[..., 0]
    yy, xx = np.mgrid[:hc, :wc]
    v, u = yy * cell + k // cell, xx * cell + k % cell
    m = (best > grad_min) & (u >= 4) & (v >= 4) & (u < w - 4) & (v < h - 4)
    pts, gm = np.c_[u[m], v[m]].astype(np.float64), best[m]
    if max_pts and len(pts) > max_pts:
        keep = np.argsort(-gm)[:max_pts]
        pts, gm = pts[keep], gm[keep]
    return pts, gm


def prepare(I: np.ndarray, M: np.ndarray, levels: int) -> list[np.ndarray]:
    """Per pyramid level an (h, w, 4) float32 stack [image, ∂x, ∂y, valid] (Sobel / 8, valid pyrDown-ed as a float).
    Linear in I, so the stack of r·I is r times the stack (the valid channel aside): frames are prepared once at
    exposure ratio 1 and reused as the next pair's reference."""
    out, img, m = [], I.astype(np.float32), M.astype(np.float32)
    for lv in range(levels):
        if lv:
            img, m = cv2.pyrDown(img), cv2.pyrDown(m)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8
        out.append(np.dstack([img, gx, gy, m]))
    return out


def _sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of (h, w) or (h, w, C) at uv (n, 2) -> (n,) or (n, C); positions clamped to the image."""
    h, w = img.shape[:2]
    x, y = np.clip(uv[:, 0], 0, w - 1.001), np.clip(uv[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x0 + 1])
            + fy * ((1 - fx) * img[y0 + 1, x0] + fx * img[y0 + 1, x0 + 1]))


def refine(I0: np.ndarray, I1: np.ndarray, M1: np.ndarray, pts: np.ndarray, z: np.ndarray, K: np.ndarray,
           R: np.ndarray, t: np.ndarray, levels: int = 2, iters: int = 10, huber: float = 0.03, tol: float = 0.0):
    """Refine X1 = R X0 + t. pts (n, 2) full-resolution pixels of frame t-1 with depths z (n,).
    Returns (R, t, info) with info = {'cost0', 'cost', 'n'} (robust mean cost at full resolution, start / end)."""
    return refine_prepared(prepare(I0, np.ones_like(M1), levels), prepare(I1, M1, levels), 1.0, pts, z, K, R, t,
                           levels, iters, huber, tol)


def refine_prepared(P0: list, P1: list, ratio: float, pts: np.ndarray, z: np.ndarray, K: np.ndarray, R: np.ndarray,
                    t: np.ndarray, levels: int = 2, iters: int = 10, huber: float = 0.03, tol: float = 0.0):
    """refine() on prepare()-d stacks; frame t's image is ratio · P1. tol > 0 ends a level once an accepted step
    lowers the cost by less than the fraction tol."""
    X0 = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(K).T * z[:, None]
    a, b = 1.0, 0.0
    scale = np.array([ratio, ratio, ratio, 1.0], np.float32)

    def setup(lv):
        s = 0.5 ** lv
        Kl = K.copy()
        Kl[:2] *= s
        Kl[0, 2], Kl[1, 2] = (K[0, 2] + 0.5) * s - 0.5, (K[1, 2] + 0.5) * s - 0.5
        return Kl, _sample(P0[lv][..., 0], (pts + 0.5) * s - 0.5)

    def residual(Kl, ref, lv, R, t, a, b):
        X1 = X0 @ R.T + t
        Z = X1[:, 2]
        uv = X1[:, :2] / np.maximum(Z, 1e-6)[:, None] * Kl[[0, 1], [0, 1]] + Kl[:2, 2]
        h, w = P1[lv].shape[:2]
        S = _sample(P1[lv], uv) * scale
        ok = (Z > 1e-3) & (uv[:, 0] > 1) & (uv[:, 1] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] < h - 2) & (S[:, 3] > 0.99)
        return S[:, 0] - (a * ref + b), ok, X1, S[:, 1:3]

    def cost(r, ok):
        e = np.abs(r[ok])
        return float(np.mean(np.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber)))) if ok.any() else np.inf

    Kl, ref = setup(0)
    info = {'cost0': cost(*residual(Kl, ref, 0, R, t, a, b)[:2])}
    for lv in range(levels - 1, -1, -1):
        Kl, ref = setup(lv)
        lam = 1e-3
        r, ok, X1, g = residual(Kl, ref, lv, R, t, a, b)
        c = cost(r, ok)
        for _ in range(iters):
            if ok.sum() < 30:
                break
            Xo, ro, go = X1[ok], r[ok], g[ok]
            Z = Xo[:, 2]
            fx, fy = Kl[0, 0], Kl[1, 1]
            gu = np.c_[go[:, 0] * fx / Z, go[:, 1] * fy / Z,
                       -(go[:, 0] * fx * Xo[:, 0] + go[:, 1] * fy * Xo[:, 1]) / Z ** 2]
            Jw = np.c_[gu, np.cross(Xo, gu)]                   # d r / d(v, ω): gu · [I | −[X1]×]
            J = np.c_[Jw, -ref[ok], -np.ones(len(ro))]
            wgt = np.minimum(1.0, huber / np.maximum(np.abs(ro), 1e-12))
            H = (J * wgt[:, None]).T @ J
            gr = (J * wgt[:, None]).T @ ro
            c_old = c
            while lam < 1e6:
                d = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), gr)
                dR = Rotation.from_rotvec(d[3:6]).as_matrix()
                Rn, tn, an, bn = dR @ R, dR @ t + d[:3], a + d[6], b + d[7]
                rn, okn, X1n, gn = residual(Kl, ref, lv, Rn, tn, an, bn)
                cn = cost(rn, okn)
                if cn < c:
                    R, t, a, b, r, ok, X1, g, c = Rn, tn, an, bn, rn, okn, X1n, gn, cn
                    lam = max(lam / 10, 1e-7)
                    break
                lam *= 10
            else:
                break
            if np.linalg.norm(d[:6]) < 1e-7 or (tol > 0 and c_old - c < tol * c_old):
                break
    info.update(cost=c, n=int(ok.sum()))
    return R, t, info
