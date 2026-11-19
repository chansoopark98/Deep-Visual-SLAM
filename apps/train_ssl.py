"""Self-supervised metric depth + pose training on TUM-VI (dvslam/ssl)

  python apps/train_ssl.py --config configs/ssl_tumvi.yaml --name base [--set loss.flow=0 --set train.epochs=30]
  python apps/train_ssl.py --name base --resume            # continues from runs/ssl/base/last.pt (its config.yaml)
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 apps/train_ssl.py --name base   # one process per GPU
  -> runs/ssl/<name>/config.yaml, train_log.jsonl, val_log.jsonl, last.pt, best.pt
With torchrun --name is required (the default name is a per-process timestamp).
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '4')    # before numpy / torch / cv2 are imported (shared workstation)

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.ssl.trainer import Trainer, apply_overrides, init_distributed, make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/ssl_tumvi.yaml')
    ap.add_argument('--name', default=None, help='run directory runs/ssl/<name> (default: timestamp)')
    ap.add_argument('--resume', action='store_true', help='continue runs/ssl/<name>/last.pt with its config.yaml')
    ap.add_argument('--set', action='append', default=[], metavar='KEY.SUB=VALUE', help='config override (repeatable)')
    args = ap.parse_args()
    rank, world = init_distributed()
    if (args.resume or world > 1) and not args.name:
        sys.exit('--resume and torchrun need --name')
    out = REPO_ROOT / 'runs/ssl' / (args.name or f'{datetime.now():%Y%m%d-%H%M%S}')
    with open(out / 'config.yaml' if args.resume else args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    if world > 1:
        torch.distributed.barrier()     # every process has read the config before rank 0 rewrites it (--resume)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=args.resume)
        with open(out / 'config.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    Trainer(cfg, out, make_loader(cfg, rank, world), rank=rank, world=world).fit(resume=args.resume)


if __name__ == '__main__':
    main()
