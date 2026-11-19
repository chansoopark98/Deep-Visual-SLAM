"""EXP-009 diagnostic (NOT deployable: uses cam1): stereo-matched metric depth for cam0 of a TUM-VI room, cached in
the klt_depth_cache layout so the tracker can take it as its depth prior (`prior=cache prior_ckpt=<out>`).

  python scripts/exp009_stereo_depth.py --scene room5 --out runs/eval/exp009/depth_stereo [--every 1]

cam0 and cam1 caches are each rectified to the pinhole K_OUT with R = identity (dvslam/data/tumvi.py), so they are not
a rectified stereo pair. Here they are stereo-rectified with cv2.stereoRectify (R, T = cam0 -> cam1 from the camchain
T_cn_cnm1, no distortion), matched with SGBM (left-right check, uniqueness, speckle filter) and the disparity is mapped
back to cam0's pinhole image: for every cam0 pixel the ray is rotated into the rectified frame, the disparity sampled
(nearest; invalid where the 3x3 disparity patch varies by more than 1 px), depth along the ray and its z in cam0 taken.
Invalid pixels are 0 (the tracker's sample_depth treats depth <= depth_min as no prior).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry, tumvi
from dvslam.odometry.klt_depth_cache import write_cache
from dvslam.ssl.tumvi_data import scene_frames


class StereoDepth:
    def __init__(self, T_c1_c0: np.ndarray, size: int, num_disp: int = 64, block: int = 5):
        K = tumvi.K_OUT
        R, T = T_c1_c0[:3, :3], T_c1_c0[:3, 3]
        self.R1, R2, self.P1, P2, _, _, _ = cv2.stereoRectify(K, None, K, None, (size, size), R, T, alpha=0)
        self.m0 = cv2.initUndistortRectifyMap(K, None, self.R1, self.P1, (size, size), cv2.CV_32FC1)
        self.m1 = cv2.initUndistortRectifyMap(K, None, R2, P2, (size, size), cv2.CV_32FC1)
        self.fB = abs(P2[0, 3])                         # f_rect · baseline
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                          P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=1,
                                          uniquenessRatio=10, speckleWindowSize=100, speckleRange=1,
                                          mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        # every cam0 pixel: its ray in the rectified frame and its pixel there
        uu, vv = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
        ray = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(K).T
        self.ray_r = ray @ self.R1.T
        pr = self.ray_r @ self.P1[:, :3].T
        self.ur = pr[..., 0] / pr[..., 2]
        self.vr = pr[..., 1] / pr[..., 2]
        self.size = size

    def __call__(self, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
        r0 = cv2.remap(g0, *self.m0, cv2.INTER_LINEAR)
        r1 = cv2.remap(g1, *self.m1, cv2.INTER_LINEAR)
        d = self.sgbm.compute(r0, r1).astype(np.float32) / 16
        d[d <= 0.5] = np.nan
        dmin = -cv2.dilate(-np.nan_to_num(d, nan=1e6), np.ones((3, 3)))
        dmax = cv2.dilate(np.nan_to_num(d, nan=1e6), np.ones((3, 3)))
        d[(dmax - dmin) > 1.0] = np.nan                  # depth edges / holes
        n = self.size
        ui, vi = np.round(self.ur).astype(int), np.round(self.vr).astype(int)
        inside = (ui >= 0) & (ui < n) & (vi >= 0) & (vi < n)
        disp = np.full((n, n), np.nan, np.float32)
        disp[inside] = d[vi[inside], ui[inside]]
        Zr = self.fB / disp                              # depth along the rectified optical axis
        X = self.ray_r * (Zr / self.ray_r[..., 2])[..., None]
        z0 = (X @ self.R1)[..., 2]                       # back to cam0: X0 = R1^T Xr
        return np.where(np.isfinite(z0) & (z0 > 0), z0, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    seq = registry.load_sequence('tumvi', args.scene, None, registry.default_root('tumvi'), 0.0)
    f = scene_frames(args.scene)
    cam1 = dict(zip(f['cam0'], f['cam1']))
    sd = StereoDepth(f['T_stereo'], seq.width)
    print(f'{args.scene}: f·B = {sd.fB:.2f} px·m', flush=True)
    out = np.zeros((len(seq), seq.height, seq.width), np.float32)
    for i, p in enumerate(seq.image_paths):
        g0 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(cam1[str(p)], cv2.IMREAD_GRAYSCALE)
        out[i] = sd(g0, g1)
    write_cache(args.out, args.scene, [str(p) for p in seq.image_paths], out,
                model='exp009_stereo_depth.py StereoDepth (cam0+cam1 SGBM, no checkpoint)')
    v = out[::50] > 0
    print(f'{args.scene}: valid {100 * v.mean():.1f} %, median depth {np.median(out[::50][v]):.3f} m')


if __name__ == '__main__':
    main()
