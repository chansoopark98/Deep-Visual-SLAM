"""Plan v3 item 4.3: fault injection into tracker C (blackout / duplicated frames / corner starvation / forced exceptions).

  python scripts/phase4_faults.py --out runs/p4-faults            # run all configs (seed 0, validation) and analyse
  python scripts/phase4_faults.py --out runs/p4-faults --analyse-only

Events start at frame 150 + 300 k in every scene (each followed by >= 240 clean frames). Per event we report whether
the fault frames were flagged motion_model, whether LOST was reached, how many frames until C tracks again, and the
item-03 cost (extra per-pair error vs the fault-free run, inside the event window and over the whole scene).
The fault-free reference is `runs/p1c-base07-s0` (a fresh C run reproduces it bit-exactly, see phase4_memory.py).
"""
import argparse
import sys
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase4_common import REPO_ROOT, SCENES, gt_pairs_t, item03, load_val, make_c, pooled  # noqa: E402
from dvslam.data import redwood  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera, load_increments, save_increments  # noqa: E402
from dvslam.odometry.fault_inject import Fault, FaultInjectOdometry  # noqa: E402

CONFIGS = {   # name -> (fault kind, length, where)
    'black5': ('black', 5, 'pnp'), 'black30': ('black', 30, 'pnp'), 'freeze10': ('freeze', 10, 'pnp'),
    'starve20': ('starve', 20, 'pnp'), 'starve60': ('starve', 60, 'pnp'),
    'exc1': ('exception', 1, 'pnp'), 'exc3': ('exception', 3, 'pnp'), 'exc10': ('exception', 10, 'pnp'),
}
FIRST, SPACING, AFTER = 150, 300, 60       # event starts, and the clean margin kept after each event
WIN = 30                                   # recovery window (frames) after the fault ends, for the event-local cost


def events(cfg: str, n: int) -> list[Fault]:
    kind, length, where = CONFIGS[cfg]
    return [Fault(kind, s, length, where) for s in range(FIRST, n - length - AFTER, SPACING)]


def _run(task):
    cfg, scene, out = task
    import cv2
    cv2.setNumThreads(1)
    seq = load_val(scene)
    odo = FaultInjectOdometry(make_c(0), events(cfg, len(seq)))
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    est = []
    for i in range(len(seq)):
        e = odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i], seq.depth_paths[i]))
        if e is not None:
            est.append(e)
    meta = {'scene': scene, 'split': 'validation', 'odometry': 'klt_window+fault_inject', 'opts': {'ba': False, 'kf_align': False},
            'fault_config': cfg, 'seed': 0, 'data_root': str(redwood.DEFAULT_ROOT), 'n_frames': len(seq)}
    out.mkdir(parents=True, exist_ok=True)
    save_increments(out / f'{scene}.inc.npz', est, meta)
    np.save(out / f'{scene}.fault.npy', np.array([l['fault'] or '' for l in odo.log]))
    return cfg, scene


def analyse(out: Path) -> dict:
    gt = {s: gt_pairs_t(load_val(s)) for s in SCENES}
    clean = {s: load_increments(REPO_ROOT / f'runs/p1c-base07-s0/{s}.inc.npz') for s in SCENES}
    clean_err = {s: item03(clean[s]['rel_increment'], gt[s]) for s in SCENES}
    clean_n_mm = sum(int((clean[s]['source'] == 'motion_model').sum()) for s in SCENES)
    res = {'clean': {'item03': pooled([clean_err[s] for s in SCENES]), 'motion_model_pairs': clean_n_mm}}
    for cfg in CONFIGS:
        per_scene, ev_rows = [], []
        for s in SCENES:
            inc = load_increments(out / cfg / f'{s}.inc.npz')
            err = item03(inc['rel_increment'], gt[s])
            per_scene.append(err)
            idx, src, st = inc['index'], inc['source'], inc['tracking_state']
            for f in events(cfg, len(gt[s]) + 1):
                a, b = f.start, f.start + f.length                  # fault frames [a, b); pair k <-> frame k + 1
                fr = np.arange(a, b)
                w = np.arange(a, b + WIN)
                rec = next((t for t in range(b, b + 200) if t <= idx[-1] and src[t - 1] == 'tracked' and st[t - 1] != 'LOST'), None)
                ev_rows.append({
                    'scene': s, 'start': a, 'len': f.length,
                    'flagged_frac': float((src[fr - 1] == 'motion_model').mean()),
                    'lost': bool((st[w - 1] == 'LOST').any()),
                    'degraded_only': bool((st[w - 1] == 'DEGRADED').any() and not (st[w - 1] == 'LOST').any()),
                    'recover_lag': None if rec is None else int(rec - b),
                    'mm_pairs_window': int((src[w - 1] == 'motion_model').sum()),
                    'excess_pct_sum': float((err[w - 1] - clean_err[s][w - 1]).sum()),
                    'excess_pct_mean_window': float((err[w - 1] - clean_err[s][w - 1]).mean()),
                    'post_excess_mean': float((err[b:b + 60] - clean_err[s][b:b + 60]).mean()),
                })
        pt, lo, hi = pooled(per_scene)
        d, dlo, dhi = pooled([e - clean_err[s] for e, s in zip(per_scene, SCENES)])
        ev = ev_rows
        res[cfg] = {
            'item03': [pt, lo, hi], 'delta_pp': [d, dlo, dhi], 'events': len(ev),
            'fault_frames': int(sum(e['len'] for e in ev)),
            'flagged_frac': float(np.average([e['flagged_frac'] for e in ev], weights=[e['len'] for e in ev])),
            'lost_frac': float(np.mean([e['lost'] for e in ev])),
            'degraded_only_frac': float(np.mean([e['degraded_only'] for e in ev])),
            'recovered_frac': float(np.mean([e['recover_lag'] is not None for e in ev])),
            'recover_lag_median': float(np.median([e['recover_lag'] for e in ev if e['recover_lag'] is not None])),
            'recover_lag_max': max((e['recover_lag'] for e in ev if e['recover_lag'] is not None), default=None),
            'mm_pairs_window_mean': float(np.mean([e['mm_pairs_window'] for e in ev])),
            'excess_pct_mean_window': float(np.mean([e['excess_pct_mean_window'] for e in ev])),
            'post_excess_mean': float(np.mean([e['post_excess_mean'] for e in ev])),
            'motion_model_pairs_total': int(sum((load_increments(out / cfg / f'{s}.inc.npz')['source'] == 'motion_model').sum()
                                                for s in SCENES)) - clean_n_mm,
        }
        (out / cfg / 'events.json').write_text(json.dumps(ev_rows, indent=1))
    (out / 'fault_summary.json').write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--analyse-only', action='store_true')
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    args = ap.parse_args()
    if not args.analyse_only:
        tasks = [(c, s, args.out / c) for c in args.configs for s in SCENES if not (args.out / c / f'{s}.inc.npz').exists()]
        with ProcessPoolExecutor(args.workers) as ex:
            for c, s in ex.map(_run, tasks):
                print('done', c, s, flush=True)
    res = analyse(args.out)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
