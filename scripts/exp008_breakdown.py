"""EXP-008: where does the item-03 error of a run come from? Contribution (pp of the pooled mean) by pair source,
GT-step bin, worst 5 % of pairs and room, for the official set and room5.

  python scripts/exp008_breakdown.py runs/eval/exp007/final_photo
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.exp003_score import OFFICIAL, pair_table


def breakdown(name: str, tabs: dict[str, dict]) -> None:
    t = {k: np.concatenate([x[k] for x in tabs.values()]) for k in next(iter(tabs.values()))}
    room = np.concatenate([[s] * len(x['pct']) for s, x in tabs.items()])
    n, pct = len(t['pct']), t['pct']
    err = np.abs(t['pred'] - t['gt']) * 1e3
    print(f'== {name}: {n} pairs, item 03 {pct.mean():.3f} %')

    def row(label, m):
        print(f'  {label:<26} share {100 * m.mean():5.1f} %  item03 {pct[m].mean():7.2f} %  median |err| '
              f'{np.median(err[m]):5.2f} mm  contributes {pct[m].sum() / n:5.2f} pp')
    row('tracked', t['tracked'])
    row('motion_model', ~t['tracked'])
    g = t['gt'] * 1e3
    for lo, hi in ((0, 5), (5, 20), (20, 40), (40, np.inf)):
        row(f'GT step {lo}-{hi} mm', (g >= lo) & (g < hi))
        row('  of which motion_model', (g >= lo) & (g < hi) & ~t['tracked'])
    worst = pct >= np.quantile(pct, 0.95)
    row('worst 5 % of pairs', worst)
    print(f'    worst 5 %: motion_model {100 * (~t["tracked"][worst]).mean():.1f} %, GT step < 5 mm '
          f'{100 * (g[worst] < 5).mean():.1f} %, median GT step {np.median(g[worst]):.1f} mm')
    print(f'  without the worst 5 %: {pct[~worst].mean():.3f} %;  without < 5 mm pairs: {pct[g >= 5].mean():.3f} %;  '
          f'tracked only and >= 5 mm: {pct[(g >= 5) & t["tracked"]].mean():.3f} %')
    for s in tabs:
        row(f'room {s}', room == s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    files = {f.name.split('.')[0]: f for f in sorted(args.run.rglob('room*.inc.npz'))}
    tabs = {s: pair_table(f) for s, f in files.items()}
    if all(s in tabs for s in OFFICIAL):
        breakdown('official', {s: tabs[s] for s in OFFICIAL})
    if 'room5' in tabs:
        breakdown('room5', {'room5': tabs['room5']})


if __name__ == '__main__':
    main()
