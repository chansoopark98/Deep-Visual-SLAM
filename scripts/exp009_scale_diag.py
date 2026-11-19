"""EXP-009: how much of a run's item 03 is scale? Per room: median step ratio (pred / GT), item 03, item 03 after the
best single per-room rescale, and after a windowed oracle rescale (1 s = 20 pairs, 2 s). All use GT: diagnostics only.

  python scripts/exp009_scale_diag.py runs/eval/exp008/final_hybrid [more runs ...]
Pooled rows (room1-4, official = room1-4 + room6) weight each room's values by its number of scored pairs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.eval import diagnostics as D
from scripts.exp003_score import pair_table

FPS = 20


def room_stats(tab: dict) -> dict:
    g, p = tab['gt'][:, None], tab['pred'][:, None]
    return {'pairs': len(g), 'item03': float(tab['pct'].mean()), 'ratio': D.median_step_ratio(g, p),
            'rescale_s': D.best_rescale(g, p)[0], 'rescaled': D.best_rescale(g, p)[1],
            'win1s': D.windowed_rescale_mape(g, p, FPS), 'win2s': D.windowed_rescale_mape(g, p, 2 * FPS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    args = ap.parse_args()
    keys = ('item03', 'rescaled', 'win2s', 'win1s')
    for run in args.runs:
        files = {f.name.split('.')[0]: f for f in sorted(run.rglob('room*.inc.npz'))}
        st = {s: room_stats(pair_table(f)) for s, f in files.items()}
        print(f'== {run}')
        print(f'{"set":<10} {"pairs":>6} {"ratio":>6} {"best s":>7} {"03 %":>7} {"rescaled":>9} {"win 2 s":>8} {"win 1 s":>8}')
        for s, v in st.items():
            print(f'{s:<10} {v["pairs"]:>6} {v["ratio"]:6.3f} {v["rescale_s"]:7.3f} '
                  + ' '.join(f'{v[k]:>8.3f}' for k in keys))
        for name, rooms in (('room1-4', ('room1', 'room2', 'room3', 'room4')),
                            ('official', ('room1', 'room2', 'room3', 'room4', 'room6'))):
            if all(r in st for r in rooms):
                w = np.array([st[r]['pairs'] for r in rooms], float)
                print(f'{name:<10} {int(w.sum()):>6} {"":>6} {"":>7} '
                      + ' '.join(f'{np.dot(w, [st[r][k] for r in rooms]) / w.sum():>8.3f}' for k in keys))


if __name__ == '__main__':
    main()
