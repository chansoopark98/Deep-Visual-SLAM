"""EXP-011: build the wide cam0 pinhole caches (tumvi.rectified_paths_wide) and check that their central 512 crop
equals the standard rectified image.

  python scripts/exp011_wide_cache.py --scenes room5 --sizes 640 768 1024
Prints per size: frames, the share of crop pixels that differ from the standard image (every 50th frame), the largest
difference in gray levels, and the share of canvas pixels that are black (outside the fisheye source).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry, tumvi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[768])
    args = ap.parse_args()
    root = registry.default_root('tumvi')
    for scene in args.scenes:
        seq = registry.load_sequence('tumvi', scene, None, root, 0.0)
        d = tumvi.sequence_dir(scene, root)
        _, images = tumvi.camera_frames(d, 'cam0')
        _, intrinsics, dist = tumvi.load_cam0_calib(d)
        for size in args.sizes:
            tumvi.rectified_paths_wide(scene, images, intrinsics, dist, size, build=True)
            o = tumvi.wide_offset(size)
            diff, mx, black = [], 0, []
            for p in seq.image_paths[::50]:
                a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(int)
                b = cv2.imread(str(tumvi.wide_path(p, size)), cv2.IMREAD_GRAYSCALE).astype(int)
                c = b[o:o + 512, o:o + 512]
                diff.append((c != a).mean())
                mx = max(mx, int(np.abs(c - a).max()))
                black.append((b == 0).mean())
            print(f'{scene} {size}: {len(images)} frames; crop pixels differing {100 * np.mean(diff):.4f} %, '
                  f'max |diff| {mx} gray levels; black canvas {100 * np.mean(black):.1f} %', flush=True)


if __name__ == '__main__':
    main()
