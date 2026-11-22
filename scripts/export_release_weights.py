"""Strip a training checkpoint to an inference-only release file.

  python scripts/export_release_weights.py SRC DST [--verify-images IMG.png ...] [--device cuda]

Two source formats:
  SSL depth + pose (dvslam/ssl/trainer.py: config, model, ema, optimizer, scheduler, scaler, epoch, step, best,
      git_commit, K, size) -> {config, ema, K, size, epoch, step, best, git_commit, source_sha256, format}. The EMA
      weights are what dvslam.ssl.predict.Predictor.from_checkpoint loads by default (ema=True), so they are kept
      unchanged ('module.' prefix, n_averaged); the raw training weights, optimizer, scheduler and AMP scaler are
      dropped. K (numpy) becomes a nested list of floats and size a tuple of ints, so the file loads with
      torch.load(weights_only=True). Predictor.from_checkpoint and the tracker's live prior (prior=ssl, SSLDepth)
      read it unchanged.
  PhotoNet (dvslam/odometry/photo_learned.py save_net: mode, width, state, extra) is already inference-only and loads
      with weights_only=True: it is checked with load_net and copied byte for byte.
--verify-images (SSL only): the depth of every given gray image through SSLDepth (the tracker's live prior path) must
be bit-identical for SRC and DST; otherwise exit 1.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FORMAT = 'dvslam-ssl-inference-v1'
KEEP = ('config', 'ema', 'epoch', 'step', 'best', 'git_commit')


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def strip_ssl(ckpt: dict, source_sha256: str = '') -> dict:
    """Inference-only dict of an SSL training checkpoint (see the module docstring)."""
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out['K'] = np.asarray(ckpt['K'], np.float64).tolist()
    out['size'] = tuple(int(s) for s in ckpt['size'])
    out['source_sha256'] = source_sha256
    out['format'] = FORMAT
    return out


def is_photonet(ckpt: dict) -> bool:
    return {'mode', 'width', 'state'} <= set(ckpt)


def verify_depth(src: Path, dst: Path, images: list[Path], device: str) -> float:
    """Max |depth(src) − depth(dst)| over `images` through SSLDepth (0.0 = bit-identical)."""
    import cv2

    from dvslam.depth.ssl import SSLDepth
    from dvslam.odometry.base import Frame
    K = np.asarray(torch.load(dst, map_location='cpu', weights_only=True)['K'])
    a, b = SSLDepth(str(src), K, device), SSLDepth(str(dst), K, device)
    worst = 0.0
    for i, p in enumerate(images):
        if cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) is None:
            raise FileNotFoundError(p)
        da, db = a.predict(Frame(i, 0.0, p))[0], b.predict(Frame(i, 0.0, p))[0]
        same = np.array_equal(da, db)
        worst = max(worst, float(np.abs(da - db).max()))
        print(f'{p.name}: bit-identical={same} max|diff|={np.abs(da - db).max():.3g} m, median depth '
              f'{float(np.median(da)):.3f} m')
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--verify-images', nargs='+', type=Path, default=[])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    if is_photonet(ckpt):
        from dvslam.odometry.photo_learned import load_net
        load_net(args.src)
        shutil.copyfile(args.src, args.dst)
        print(f'PhotoNet ({ckpt["mode"]}, width {ckpt["width"]}): copied unchanged -> {args.dst}')
    else:
        torch.save(strip_ssl(ckpt, sha256_of(args.src)), args.dst)
        dropped = sorted(set(ckpt) - set(KEEP) - {'K', 'size'})
        print(f'SSL: kept {list(KEEP)} + K, size; dropped {dropped} -> {args.dst}')
    out = torch.load(args.dst, map_location='cpu', weights_only=True)      # must load without pickled objects
    print(f'{args.dst}: {args.dst.stat().st_size / 1e6:.1f} MB, sha256 {sha256_of(args.dst)}, keys {sorted(out)}')
    if args.verify_images:
        if is_photonet(ckpt):
            sys.exit('--verify-images applies to SSL depth checkpoints only')
        worst = verify_depth(args.src, args.dst, args.verify_images, args.device)
        print(f'verify: max |depth diff| over {len(args.verify_images)} images = {worst:.3g} m')
        if worst != 0.0:
            sys.exit(1)


if __name__ == '__main__':
    main()
