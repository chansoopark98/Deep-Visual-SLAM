"""Plan v3 item 4.1: 30 Hz drop-oldest replay of tracker C with injected latency, composed vs true re-tracking.

  python scripts/phase4_replay.py --out runs/p4-replay --quiet-timings runs/p4-memory-as6g/timings.csv

Per (seed, scene, latency model) the mailbox mask is computed (`simulate_mailbox`); then
  composed : offline raw increments composed across gaps (`apply_drop_policy`; optimistic)
  retrack  : tracker C stepped only on the processed frames (`retrack`) and scored with the same rule
Both are followed by the causal CV prior (alpha 0.6, as in G1'), then item 03 = mean per-pair error, mean of scene means,
mean over seeds. `offline` is the same CV'd per-frame run without drops. Latency models: recorded per-frame times
x {3.6, 8, 23} (the recorded times are inflated by host load), re-measured quiet times x the same factors, and fixed
20 / 33 / 50 ms.
"""
import argparse
import sys
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase4_common import REPO_ROOT, SCENES, gt_pairs_t, item03, load_val, make_c, pooled  # noqa: E402
from dvslam.odometry.base import load_increments  # noqa: E402
from dvslam.runtime.replay import apply_drop_policy, simulate_mailbox  # noqa: E402
from dvslam.runtime.retrack import retrack, score_retracked  # noqa: E402

SCALES = (3.6, 8.0, 23.0)
FIXED = (20.0, 33.0, 50.0)


def read_times(path: Path) -> dict[str, dict[int, float]]:
    t = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            t[r['scene']][int(r['index'])] = float(r['total_ms'])
    return t


def proc_ms(times: dict[int, float], n: int, scale: float) -> np.ndarray:
    med = float(np.median(list(times.values())))
    return np.array([times.get(i, med) for i in range(n)]) * scale


def _retrack_task(args):
    cfg, seed, scene, mask, cache = args
    if cache.exists():
        return args[:3], dict(np.load(cache))
    cv_threads = 1
    import cv2
    cv2.setNumThreads(cv_threads)
    m = retrack(make_c(seed), load_val(scene), mask)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **m)
    return args[:3], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--quiet-timings', type=Path, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scales', type=float, nargs='*', default=list(SCALES))
    ap.add_argument('--fixed', type=float, nargs='*', default=list(FIXED))
    ap.add_argument('--scenes', nargs='+', default=list(SCENES))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seqs = {s: load_val(s) for s in args.scenes}
    gt = {s: gt_pairs_t(seqs[s]) for s in args.scenes}
    raw = {(sd, s): load_increments(REPO_ROOT / f'runs/p1c-base07-s{sd}/{s}.inc.npz')['rel_increment']
           for sd in args.seeds for s in args.scenes}
    models = {f'rec x{x:g}': ('rec', x) for x in args.scales}
    if args.quiet_timings:
        models.update({f'quiet x{x:g}': ('quiet', x) for x in args.scales})
    models.update({f'fixed {f:g} ms': ('fix', f) for f in args.fixed})
    rec = {sd: read_times(REPO_ROOT / f'runs/p1c-base07-s{sd}/timings.csv') for sd in args.seeds}
    quiet = read_times(args.quiet_timings) if args.quiet_timings else None

    masks, jobs = {}, []
    for name, (kind, x) in models.items():
        for sd in args.seeds:
            for s in args.scenes:
                n = len(seqs[s])
                p = np.full(n, x) if kind == 'fix' else proc_ms((rec[sd] if kind == 'rec' else quiet)[s], n, x)
                masks[name, sd, s] = simulate_mailbox(p)
                if not masks[name, sd, s].all():
                    jobs.append((name, sd, s, masks[name, sd, s],
                                 args.out / 'retrack' / f'{name.replace(" ", "_")}-s{sd}-{s}.npz'))
    print(f'{len(jobs)} re-tracking jobs, {sum(int(j[3].sum()) for j in jobs)} frames', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        meas = {k: m for k, m in ex.map(_retrack_task, sorted(jobs, key=lambda j: -j[3].sum()))}

    def agg(per_scene_seed: dict[str, list[np.ndarray]]):
        per_scene = [np.mean(per_scene_seed[s], axis=0) for s in args.scenes]
        return per_scene, pooled(per_scene)

    off = {s: [item03(raw[sd, s], gt[s]) for sd in args.seeds] for s in args.scenes}
    off_scene, (off_pt, off_lo, off_hi) = agg(off)
    rows = [{'model': 'offline (no drops)', 'drop_pct': 0.0, 'composed': off_pt, 'composed_ci': [off_lo, off_hi],
             'retrack': off_pt, 'retrack_ci': [off_lo, off_hi]}]
    for name in models:
        row = {'model': name}
        drops, fails, tr_only = [], [], []
        res = {'composed': defaultdict(list), 'retrack': defaultdict(list)}
        for sd in args.seeds:
            for s in args.scenes:
                mask, n = masks[name, sd, s], len(seqs[s])
                c = apply_drop_policy(raw[sd, s], mask)
                res['composed'][s].append(item03(c['rel_increment'], gt[s]))
                if mask.all():
                    r = {'rel_increment': raw[sd, s], 'source': np.full(n - 1, 'tracked'), 'drop_rate': 0.0, 'n_tracker_fail': 0}
                else:
                    r = score_retracked(n, mask, meas[name, sd, s])
                res['retrack'][s].append(item03(r['rel_increment'], gt[s]))
                drops.append(r['drop_rate'])
                fails.append(r['n_tracker_fail'])
                ok = r['source'] == 'tracked'
                tr_only.append(res['retrack'][s][-1][ok].mean() if ok.any() else np.nan)
        row['drop_pct'] = 100 * float(np.mean(drops))
        row['processed_fail'] = int(np.sum(fails))
        row['retrack_tracked_only'] = float(np.nanmean(tr_only))
        for k in ('composed', 'retrack'):
            per_scene, (pt, lo, hi) = agg(res[k])
            d, dlo, dhi = pooled([a - b for a, b in zip(per_scene, off_scene)])
            row[k], row[k + '_ci'], row[k + '_delta'], row[k + '_delta_ci'] = pt, [lo, hi], d, [dlo, dhi]
        rows.append(row)
        print(f'{name:>14}: drop {row["drop_pct"]:5.1f} %  composed {row["composed"]:.2f} (Δ {row["composed_delta"]:+.2f})  '
              f'retrack {row["retrack"]:.2f} (Δ {row["retrack_delta"]:+.2f})  fails {row["processed_fail"]}', flush=True)
    (args.out / 'replay_summary.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
