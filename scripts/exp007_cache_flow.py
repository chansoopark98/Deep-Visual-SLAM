"""EXP-007: RAFT-large flow between consecutive frames of a TUM-VI room, both directions, cached for klt_flow.py.

  CUDA_VISIBLE_DEVICES=<g> python scripts/exp007_cache_flow.py --scene room5 --out runs/eval/exp007/flow [--iters 20]
      [--model small] [--scale 0.5]       # EXP-008: torchvision raft_small, reduced input resolution
-> <out>/<scene>.fwd.npy, <scene>.bwd.npy (n, 2, 512, 512) float32 px, <scene>.paths.txt, <scene>.meta.json
(model, iters, scale, H x W; klt_depth_cache.write_meta). Input: the rectified gray
cam0 image replicated to 3 channels in [-1, 1] (as dvslam/ssl/losses.py RaftTeacher), torchvision default weights.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry
from dvslam.odometry.klt_depth_cache import write_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--model', choices=['large', 'small'], default='large')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='input resize factor (EXP-008); the flow is resized back bilinearly and multiplied by 1/scale')
    args = ap.parse_args()
    import torch.nn.functional as Fn
    from torchvision.models import optical_flow as of
    net = (of.raft_large(weights=of.Raft_Large_Weights.DEFAULT) if args.model == 'large'
           else of.raft_small(weights=of.Raft_Small_Weights.DEFAULT)).eval().cuda()
    seq = registry.load_sequence('tumvi', args.scene, None, registry.default_root('tumvi'), 0.0)
    n, h, w = len(seq), seq.height, seq.width
    args.out.mkdir(parents=True, exist_ok=True)
    fwd = np.lib.format.open_memmap(args.out / f'{args.scene}.fwd.npy', 'w+', np.float32, (n, 2, h, w))
    bwd = np.lib.format.open_memmap(args.out / f'{args.scene}.bwd.npy', 'w+', np.float32, (n, 2, h, w))
    fwd[0], bwd[0] = 0, 0

    def img(i):
        g = torch.from_numpy(cv2.imread(str(seq.image_paths[i]), cv2.IMREAD_GRAYSCALE)).float() / 255
        return (g * 2 - 1)[None].expand(3, -1, -1)

    with torch.no_grad():
        for s in range(1, n, args.batch):
            idx = list(range(s, min(n, s + args.batch)))
            a = torch.stack([img(i - 1) for i in idx]).cuda()
            b = torch.stack([img(i) for i in idx]).cuda()
            x, y = torch.cat([a, b]), torch.cat([b, a])
            if args.scale != 1.0:
                # int(): with a NumPy h, round() returns a NumPy float on NumPy < 2
                hs = int(round(h * args.scale / 8)) * 8  # noqa: RUF046
                x, y = (Fn.interpolate(v, size=(hs, hs), mode='bilinear', align_corners=False) for v in (x, y))
            f = net(x, y, num_flow_updates=args.iters)[-1].float()
            if args.scale != 1.0:
                f = Fn.interpolate(f, size=(h, w), mode='bilinear', align_corners=False) * (h / f.shape[-2])
            f = f.cpu().numpy()
            fwd[idx], bwd[idx] = f[:len(idx)], f[len(idx):]
            if s % 400 == 1:
                print(f'{args.scene}: {s}/{n}, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)
    fwd.flush(), bwd.flush()
    (args.out / f'{args.scene}.paths.txt').write_text(''.join(f'{p}\n' for p in seq.image_paths))
    write_meta(args.out, args.scene, (h, w), f'torchvision raft_{args.model} DEFAULT weights, iters={args.iters}, '
               f'scale={args.scale}')
    print(args.scene, 'done')


if __name__ == '__main__':
    main()
