"""Stereo depth hints for SSL training (training-only signal: uses cam1), one cache per TUM-VI scene.

  python scripts/cache_stereo_hints.py --out $DVSLAM_CACHE_ROOT/tumvi_stereo_hints room1 room2 ...
-> <out>/<scene>.npy (N, 512, 512) float16 metres, 0 = no hint, indexed like dvslam.ssl.tumvi_data.scene_frames
   (cam0 frame i). The depth is exactly scripts/exp009_stereo_depth.py's StereoDepth (same SGBM settings, the depth
   that gave the EXP-009 1.0 pp bound), stored as float16 (relative rounding <= 5e-4).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from exp009_stereo_depth import StereoDepth

from dvslam.ssl.tumvi_data import scene_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('scenes', nargs='+')
    args = ap.parse_args()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        f = scene_frames(scene)
        g = cv2.imread(f['cam0'][0], cv2.IMREAD_GRAYSCALE)
        sd = StereoDepth(f['T_stereo'], g.shape[1])
        tmp = args.out / f'{scene}.tmp.npy'
        out = np.lib.format.open_memmap(tmp, 'w+', np.float16, (len(f['cam0']), *g.shape))
        for i, (p0, p1) in enumerate(zip(f['cam0'], f['cam1'])):
            out[i] = sd(cv2.imread(p0, cv2.IMREAD_GRAYSCALE), cv2.imread(p1, cv2.IMREAD_GRAYSCALE))
        out.flush()
        del out
        tmp.rename(args.out / f'{scene}.npy')
        d = np.load(args.out / f'{scene}.npy', mmap_mode='r')[::50].astype(np.float32)
        print(f'{scene}: {len(f["cam0"])} frames, f·B {sd.fB:.2f}, valid {100 * (d > 0).mean():.1f} %, '
              f'median {np.median(d[d > 0]):.3f} m', flush=True)


if __name__ == '__main__':
    main()
