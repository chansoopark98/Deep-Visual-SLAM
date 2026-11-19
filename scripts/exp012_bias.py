"""EXP-012 step 1 (measurement only): signed step-length error pred − gt by GT-step bin.

  python scripts/exp012_bias.py runs/eval/exp009b/a_F_scale --rooms room5 / room1 room2 room3 room4

Per bin (< 2, 2-5, 5-10, 10-20, > 20 mm): pairs, mean and median of ‖t_pred‖ − ‖t_gt‖ (mm), share of pairs with a
positive error, item 03 and its contribution to the pooled mean (pp), and the contribution after an ORACLE per-bin
constant shift (pred − median signed error of that bin, uses GT: an upper bound on what removing the bias can give).
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.exp003_score import pair_table

EDGES = (0, 2, 5, 10, 20, np.inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--rooms', nargs='+', required=True)
    args = ap.parse_args()
    tabs = [pair_table(args.run / f'{r}.inc.npz') for r in args.rooms]
    g = np.concatenate([t['gt'] for t in tabs]) * 1e3
    p = np.concatenate([t['pred'] for t in tabs]) * 1e3
    pct = np.concatenate([t['pct'] for t in tabs])
    n = len(g)
    e = p - g
    print(f'{args.run} {"+".join(args.rooms)}: {n} pairs, item 03 {pct.mean():.3f} %')
    print(f'{"GT step mm":<11} {"pairs":>6} {"mean e":>7} {"med e":>7} {"e>0":>6} {"03 %":>7} {"pp":>6} '
          f'{"pp oracle-shift":>15}')
    tot_gain = 0.0
    for lo, hi in itertools.pairwise(EDGES):
        m = (g >= lo) & (g < hi)
        sh = np.where(g[m] > 1e-3, np.abs(np.maximum(p[m] - np.median(e[m]), 0) - g[m])   # same guard as pct
                      / np.maximum(g[m], 1e-9) * 100, 0.0)                                # (1e-6 m, here in mm)
        gain = (pct[m].sum() - sh.sum()) / n
        tot_gain += gain
        pos = 100 * (e[m] > 0).mean()
        print(f'{f"{lo}-{hi}":<11} {m.sum():>6} {e[m].mean():7.3f} {np.median(e[m]):7.3f} {pos:5.1f}% '
              f'{pct[m].mean():7.2f} {pct[m].sum() / n:6.3f} {sh.sum() / n:15.3f}')
    print(f'oracle per-bin shift would lower item 03 by {tot_gain:.3f} pp')


if __name__ == '__main__':
    main()
