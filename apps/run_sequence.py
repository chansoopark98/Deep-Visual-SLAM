"""Run an odometry over a dataset (Redwood by default; --dataset tumvi | euroc) and write causal increments

  python apps/run_sequence.py --odometry klt_window --dataset tumvi --split train test   # official set (room1-4, 6)
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

try:  # torch-free envs can still run the numpy tracker path
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera, causal_trajectory, save_increments, write_tum  # noqa: E402


# name -> (module, class). Modules are imported lazily so one tracker's heavy deps never block another.
ODOMETRY = {
    'klt_window': ('dvslam.odometry.klt_window', 'KLTWindowOdometry'),    # plan 1.3 (C + window BA + prior)
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
    ap.add_argument('--dataset', choices=registry.DATASETS, default='redwood')
    ap.add_argument('--split', nargs='+', choices=['train', 'validation', 'test'], default=['test'],
                    help='one or more splits; several only where each scene belongs to one split (TUM-VI, EuRoC)')
    ap.add_argument('--scenes', nargs='+', default=None, help='default: every scene of the split(s)')
    ap.add_argument('--data-root', type=Path, default=None, help="default: the dataset's own root")
    ap.add_argument('--time-offset', type=float, default=0.0,
                    help='GT lookup offset in seconds (TUM-VI / EuRoC); it also decides which end frames have GT')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None, help='default: runs/<odometry>-<split>-<YYYYmmdd-HHMMSS>')
    args = ap.parse_args()
    args.data_root = args.data_root or registry.default_root(args.dataset)
    splits = list(dict.fromkeys(args.split))
    if len(splits) > 1 and args.dataset == 'redwood':
        sys.exit('Redwood has the same five scenes in every split: run one split per output directory')
    # scene -> split; with several splits a scene must belong to exactly one of them
    scene_split = {}
    for split in splits:
        for scene in registry.expected_scenes(args.dataset, split):
            scene_split.setdefault(scene, split)
    if not args.scenes:
        args.scenes = [s for split in splits for s in registry.scenes(args.dataset, split, args.data_root)]
        missing = sorted(set(scene_split) - set(args.scenes))
        if missing:
            print(f'INCOMPLETE split: {args.dataset} {"+".join(splits)} is missing {missing} on disk')
    elif len(splits) > 1:
        unknown = sorted(set(args.scenes) - set(scene_split))
        if unknown:
            sys.exit(f'{unknown} not in split(s) {splits}')
    frame_rate_hz, frame_rate_verified = registry.frame_rate(args.dataset)

    if torch is not None:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prefix = args.odometry if args.dataset == 'redwood' else f'{args.odometry}-{args.dataset}'
    out = args.out or REPO_ROOT / 'runs' / f'{prefix}-{"+".join(splits)}-{datetime.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    opts = parse_opts(args.opt)
    odo = make_odometry(args.odometry, opts)

    with open(out / 'timings.csv', 'w', newline='') as tf:
        tw = csv.writer(tf)
        tw.writerow(['scene', 'index', 'total_ms'])
        for scene in args.scenes:
            split = scene_split.get(scene, splits[0])
            seq = registry.load_sequence(args.dataset, scene, split, args.data_root, args.time_offset)
            odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
            estimates = []
            for i in range(len(seq)):
                depth_path = seq.depth_paths[i] if seq.depth_paths else None   # TUM-VI / EuRoC have no sensor depth
                est = odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i], depth_path))
                if est is not None:
                    estimates.append(est)
                    tw.writerow([scene, est.index, f'{est.timings_ms.get("total", float("nan")):.3f}'])
            if [e.index for e in estimates] != list(range(1, len(seq))):
                raise RuntimeError(f'{scene}: odometry must emit one increment per frame t >= 1')
            meta = {'dataset': args.dataset, 'scene': scene, 'split': split, 'odometry': args.odometry,
                    'opts': opts, 'seed': args.seed,
                    'data_root': str(args.data_root), 'n_frames': len(seq), 'git_commit': commit,
                    'frame_rate_hz': frame_rate_hz, 'frame_rate_verified': frame_rate_verified,
                    'time_offset_s': args.time_offset}
            save_increments(out / f'{scene}.inc.npz', estimates, meta)
            T_out = causal_trajectory(np.stack([e.rel_increment for e in estimates]))
            write_tum(out / f'{scene}.causal.tum', seq.timestamps, T_out)
            if hasattr(odo, 'finalize'):   # optional per-scene diagnostics (dict of arrays)
                np.savez(out / f'{scene}.extra.npz', **odo.finalize())
            print(f'{scene}: {len(estimates)} increments -> {out / f"{scene}.inc.npz"}')
    print(out)


if __name__ == '__main__':
    main()
