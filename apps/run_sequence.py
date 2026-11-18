"""Run an odometry over Redwood sequences and write causal increments (plan v3 §4.8, item 0.4).

  python apps/run_sequence.py --odometry legacy_posenet --split test
  -> runs/<id>/<scene>.inc.npz, <scene>.causal.tum, timings.csv
"""
import argparse
import csv
import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:  # torch-free envs (e.g. the py3.10 cuVSLAM env) can still run non-torch trackers
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera, causal_trajectory, save_increments, write_tum  # noqa: E402


# name -> (module, class). Modules are imported lazily so one tracker's heavy deps never block another.
ODOMETRY = {
    'legacy_posenet': ('dvslam.odometry.legacy_posenet', 'LegacyPoseNetOdometry'),
    'klt_pnp': ('dvslam.odometry.klt_pnp', 'KLTPnPOdometry'),           # plan 0.7 (C)
    'dpvo_mono': ('dvslam.odometry.dpvo_mono', 'DPVOMonoOdometry'),     # plan 0.6 (B, depth-free)
    'rgbd_icp': ('dvslam.odometry.rgbd_icp', 'RGBDICPOdometry'),        # plan 0.8 (E0, sensor depth)
    'cuvslam_rgbd': ('dvslam.odometry.cuvslam_rgbd', 'CuVSLAMRGBDOdometry'),  # plan 0.8/1.4 (A); needs py3.10 env
    'dpvo_metric': ('dvslam.odometry.dpvo_metric', 'DPVOMetricOdometry'),  # plan 1.1/1.2 (B + depth prior)
    'klt_window': ('dvslam.odometry.klt_window', 'KLTWindowOdometry'),    # plan 1.3 (C + window BA + prior)
    'klt_head': ('dvslam.odometry.klt_head', 'KLTHeadOdometry'),          # plan 1.5 (precision head over any base)
}


def parse_opts(items: list[str]) -> dict:
    """--opt k=v pairs -> kwargs; values are parsed as JSON when possible (numbers, bools), else kept as str."""
    out = {}
    for it in items:
        k, v = it.split('=', 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def make_odometry(name: str, opts: dict):
    module, cls = ODOMETRY[name]
    return getattr(importlib.import_module(module), cls)(**opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odometry', choices=sorted(ODOMETRY), required=True)
    ap.add_argument('--opt', action='append', default=[], help='tracker option k=v (repeatable)')
    ap.add_argument('--split', choices=['validation', 'test'], default='test')
    ap.add_argument('--scenes', nargs='+', default=list(redwood.SCENES))
    ap.add_argument('--data-root', type=Path, default=redwood.DEFAULT_ROOT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None, help='default: runs/<odometry>-<split>-<YYYYmmdd-HHMMSS>')
    args = ap.parse_args()

    if torch is not None:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out or REPO_ROOT / 'runs' / f'{args.odometry}-{args.split}-{datetime.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    opts = parse_opts(args.opt)
    odo = make_odometry(args.odometry, opts)

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
            meta = {'scene': scene, 'split': args.split, 'odometry': args.odometry, 'opts': opts, 'seed': args.seed,
                    'data_root': str(args.data_root), 'n_frames': len(seq), 'git_commit': commit,
                    'frame_rate_hz': redwood.FRAME_RATE_HZ, 'frame_rate_verified': redwood.FRAME_RATE_VERIFIED}
            save_increments(out / f'{scene}.inc.npz', estimates, meta)
            T_out = causal_trajectory(np.stack([e.rel_increment for e in estimates]))
            write_tum(out / f'{scene}.causal.tum', seq.timestamps, T_out)
            if hasattr(odo, 'finalize'):   # optional per-scene diagnostics (dict of arrays)
                np.savez(out / f'{scene}.extra.npz', **odo.finalize())
            print(f'{scene}: {len(estimates)} increments -> {out / f"{scene}.inc.npz"}')
    print(out)


if __name__ == '__main__':
    main()
