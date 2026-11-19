"""EXP-003: cache the SSL depth prior of every TUM-VI room frame once, so tracker sweeps need no GPU.

  CUDA_VISIBLE_DEVICES=3 python scripts/exp003_cache_depth.py --ckpt runs/ssl/core/best.pt --out runs/eval/exp003/depth
-> <out>/<scene>.npy (n, 512, 512) float32 metres + <scene>.paths.txt + <scene>.meta.json (checkpoint path and
sha256; read by dvslam/odometry/klt_depth_cache.py).
One image per forward pass (batch 1), exactly as the live tracker calls SSLDepth.predict.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry
from dvslam.depth.ssl import SSLDepth
from dvslam.odometry.base import Frame
from dvslam.odometry.klt_depth_cache import write_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scenes', nargs='+', default=['room1', 'room2', 'room3', 'room4', 'room5', 'room6'])
    args = ap.parse_args()
    root = registry.default_root('tumvi')
    model = None
    for scene in args.scenes:
        seq = registry.load_sequence('tumvi', scene, None, root, 0.0)
        model = model or SSLDepth(args.ckpt, seq.K)
        d = np.empty((len(seq), seq.height, seq.width), np.float32)
        for i, p in enumerate(seq.image_paths):
            d[i] = model.predict(Frame(i, float(seq.timestamps[i]), p))[0]
        write_cache(args.out, scene, [str(p) for p in seq.image_paths], d, model='SSLDepth', ckpt=args.ckpt)
        print(f'{scene}: {len(seq)} maps, median depth {float(np.median(d[::50])):.3f} m', flush=True)


if __name__ == '__main__':
    main()
