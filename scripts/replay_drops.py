"""Plan v3 item 4.1: replay a finished run at 30 Hz with drop-oldest and an injected per-frame latency.

  python scripts/replay_drops.py --runs runs/<id> --out runs/<id>-replay-x8 --scale 8
  python scripts/replay_drops.py --runs runs/<id> --out runs/<id>-replay-40ms --fixed-ms 40
  python apps/eval_pairs.py --runs runs/<id>-replay-x8      # all pairs (official) and tracked-only

Latency per frame = recorded 5090 time × scale (e.g. the 3.6–23× NX range of plan G1), or a fixed value.
"""
import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.odometry.base import PoseEstimate, load_increments, save_increments  # noqa: E402
from dvslam.runtime.replay import apply_drop_policy, simulate_mailbox  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--scale', type=float, help='multiply recorded per-frame times')
    g.add_argument('--fixed-ms', type=float, help='constant per-frame time')
    args = ap.parse_args()

    times = defaultdict(dict)
    with open(args.runs / 'timings.csv') as f:
        for r in csv.DictReader(f):
            times[r['scene']][int(r['index'])] = float(r['total_ms'])
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.runs / 'timings.csv', args.out / 'timings.csv')
    for p in sorted(args.runs.glob('*.inc.npz')):
        inc = load_increments(p)
        meta = inc['meta']
        n = meta['n_frames']
        if args.fixed_ms is not None:
            proc = np.full(n, args.fixed_ms)
        else:
            t = times[meta['scene']]
            med = float(np.median(list(t.values())))
            proc = np.array([t.get(i, med) for i in range(n)]) * args.scale
        processed = simulate_mailbox(proc)
        r = apply_drop_policy(inc['rel_increment'], processed)
        est = [PoseEstimate(int(i), float(ts), dT, str(st) if src == 'tracked' else 'DEGRADED', str(src))
               for i, ts, dT, st, src in zip(inc['index'], inc['timestamp'], r['rel_increment'],
                                             inc['tracking_state'], r['source'])]
        meta['replay'] = {'scale': args.scale, 'fixed_ms': args.fixed_ms, 'drop_rate': r['drop_rate'],
                          'source_run': str(args.runs)}
        save_increments(args.out / p.name, est, meta)
        print(f'{meta["scene"]}: drop rate {r["drop_rate"]*100:.2f} %')


if __name__ == '__main__':
    main()
