"""Self-supervised metric depth + pose training on TUM-VI (dvslam/ssl)

  python apps/train_ssl.py --config configs/ssl_tumvi.yaml --name base [--set loss.flow=0 --set train.epochs=30]
  python apps/train_ssl.py --name base --resume            # continues from runs/ssl/base/last.pt (its config.yaml)
  -> runs/ssl/<name>/config.yaml, train_log.jsonl, val_log.jsonl, last.pt, best.pt
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '4')    # before numpy / torch / cv2 are imported (shared workstation)

import argparse  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.ssl.trainer import Trainer, apply_overrides, make_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/ssl_tumvi.yaml')
    ap.add_argument('--name', default=None, help='run directory runs/ssl/<name> (default: timestamp)')
    ap.add_argument('--resume', action='store_true', help='continue runs/ssl/<name>/last.pt with its config.yaml')
    ap.add_argument('--set', action='append', default=[], metavar='KEY.SUB=VALUE', help='config override (repeatable)')
    args = ap.parse_args()
    if args.resume and not args.name:
        sys.exit('--resume needs --name')
    out = REPO_ROOT / 'runs/ssl' / (args.name or f'{datetime.now():%Y%m%d-%H%M%S}')
    with open(out / 'config.yaml' if args.resume else args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    out.mkdir(parents=True, exist_ok=args.resume)
    with open(out / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    Trainer(cfg, out, make_loader(cfg)).fit(resume=args.resume)


if __name__ == '__main__':
    main()
