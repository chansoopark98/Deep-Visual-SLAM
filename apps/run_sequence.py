"""Run an odometry over Redwood sequences and write causal increments (plan v3 §4.8, item 0.4).

  python apps/run_sequence.py --odometry legacy_posenet --split test
  -> runs/<id>/<scene>.inc.npz, <scene>.causal.tum, timings.csv
"""
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera, causal_trajectory, save_increments, write_tum  # noqa: E402


def make_odometry(name: str):
    if name == 'legacy_posenet':
        from dvslam.odometry.legacy_posenet import LegacyPoseNetOdometry
        return LegacyPoseNetOdometry()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odometry', choices=['legacy_posenet'], required=True)
    ap.add_argument('--split', choices=['validation', 'test'], default='test')
    ap.add_argument('--scenes', nargs='+', default=list(redwood.SCENES))
    ap.add_argument('--data-root', type=Path, default=redwood.DEFAULT_ROOT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None, help='default: runs/<odometry>-<split>-<YYYYmmdd-HHMMSS>')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out or REPO_ROOT / 'runs' / f'{args.odometry}-{args.split}-{datetime.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    odo = make_odometry(args.odometry)

    with open(out / 'timings.csv', 'w', newline='') as tf:
        tw = csv.writer(tf)
        tw.writerow(['scene', 'index', 'total_ms'])
        for scene in args.scenes:
            seq = redwood.load_sequence(scene, args.split, args.data_root)
            odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
            estimates = []
            for i in range(len(seq)):
                est = odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i], seq.depth_paths[i]))
                if est is not None:
                    estimates.append(est)
                    tw.writerow([scene, est.index, f'{est.timings_ms.get("total", float("nan")):.3f}'])
            if [e.index for e in estimates] != list(range(1, len(seq))):
                raise RuntimeError(f'{scene}: odometry must emit one increment per frame t >= 1')
            meta = {'scene': scene, 'split': args.split, 'odometry': args.odometry, 'seed': args.seed,
                    'data_root': str(args.data_root), 'n_frames': len(seq), 'git_commit': commit,
                    'frame_rate_hz': redwood.FRAME_RATE_HZ, 'frame_rate_verified': redwood.FRAME_RATE_VERIFIED}
            save_increments(out / f'{scene}.inc.npz', estimates, meta)
            T_out = causal_trajectory(np.stack([e.rel_increment for e in estimates]))
            write_tum(out / f'{scene}.causal.tum', seq.timestamps, T_out)
            print(f'{scene}: {len(estimates)} increments -> {out / f"{scene}.inc.npz"}')
    print(out)


if __name__ == '__main__':
    main()
