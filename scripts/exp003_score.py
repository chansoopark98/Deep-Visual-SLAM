"""EXP-003 scorer: item 03 per pair for TUM-VI room runs, pooled the way docs/EXPERIMENTS.md reports it.

  python scripts/exp003_score.py runs/eval/core-klt [more run dirs ...] [--json out.json]

A run dir holds room<N>.inc.npz (searched recursively, one file per room). Reported per run:
  official = room1-4 + room6 pooled by pairs, room5, room6, per room; mean item 03 (%), median item 03, median
  |‖t_pred‖ − ‖t_gt‖| (mm), tracked-only item 03 and share of motion_model pairs, item 03 / median |norm error| by
  GT-step bin (< 5, 5-20, 20-40, > 40 mm). Only pairs with measured GT at both ends are scored (as eval_pairs.py).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import load_increments

OFFICIAL = ('room1', 'room2', 'room3', 'room4', 'room6')
BINS_MM = (0, 5, 20, 40, np.inf)
MIN_INDEX = [1]             # score only pairs (t-1, t) with t >= this (--min-index; EXP-007 start-frame perturbation)
_GT = {}


def gt_steps(scene: str, time_offset_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(GT relative poses (n-1, 4, 4), pair_valid (n-1,)) of a room, cached per process."""
    key = (scene, time_offset_s)
    if key not in _GT:
        seq = registry.load_sequence('tumvi', scene, None, registry.default_root('tumvi'), time_offset_s,
                                     load_images=False)
        rel = np.stack([relative_pose(seq.T_wc[t - 1], seq.T_wc[t]) for t in range(1, len(seq))])
        _GT[key] = (rel, seq.pair_valid)
    return _GT[key]


def pair_table(inc_path: Path, time_offset_s: float | None = None) -> dict:
    """Per scored pair: GT and predicted step length (m), item-03 value (%), tracked flag. time_offset_s overrides
    the run's GT offset (diagnostic re-scoring); ValueError when it changes the frame range with GT."""
    inc = load_increments(inc_path)
    meta = inc['meta']
    tau = meta.get('time_offset_s', 0.0) if time_offset_s is None else time_offset_s
    rel, valid = gt_steps(meta['scene'], tau)
    idx = inc['index']
    if tau != meta.get('time_offset_s', 0.0) and not np.array_equal(idx, np.arange(1, len(rel) + 1)):
        raise ValueError(f'{inc_path}: increments must cover frames 1..{len(rel)}')
    if idx.min() < 1 or idx.max() > len(rel) or np.any(np.diff(idx) != 1):
        raise ValueError(f'{inc_path}: increments must be a contiguous range inside frames 1..{len(rel)}')
    keep = valid[idx - 1] & (idx >= MIN_INDEX[0])
    rel = rel[idx - 1]
    g = np.linalg.norm(rel[:, :3, 3], axis=1)
    p = np.linalg.norm(inc['rel_increment'][:, :3, 3], axis=1)
    pct = np.where(g > 1e-6, np.abs(p - g) / np.maximum(g, 1e-12) * 100, 0.0)
    tracked = inc['source'] == 'tracked'
    return {k: v[keep] for k, v in (('gt', g), ('pred', p), ('pct', pct), ('tracked', tracked))}


def summarize(tabs: list[dict]) -> dict:
    t = {k: np.concatenate([x[k] for x in tabs]) for k in tabs[0]}
    err_mm = np.abs(t['pred'] - t['gt']) * 1e3
    out = {'pairs': len(t['pct']), 'item03': float(t['pct'].mean()), 'median_pct': float(np.median(t['pct'])),
           'median_err_mm': float(np.median(err_mm)),
           'tracked_item03': float(t['pct'][t['tracked']].mean()) if t['tracked'].any() else float('nan'),
           'fallback_share': float(1 - t['tracked'].mean())}
    gmm = t['gt'] * 1e3
    for lo, hi in itertools.pairwise(BINS_MM):
        m = (gmm >= lo) & (gmm < hi)
        out[f'bin_{lo}_{hi}'] = {'share': float(m.mean()), 'item03': float(t['pct'][m].mean()),
                                 'median_err_mm': float(np.median(err_mm[m])),
                                 'contrib_pp': float(t['pct'][m].sum() / len(t['pct']))}
    return out


def score_run(run: Path) -> dict:
    files = {f.name.split('.')[0]: f for f in sorted(run.rglob('room*.inc.npz'))}
    tabs = {s: pair_table(f) for s, f in files.items()}
    res = {s: summarize([t]) for s, t in tabs.items()}
    if all(s in tabs for s in OFFICIAL):
        res['official'] = summarize([tabs[s] for s in OFFICIAL])
    return res


def fmt(name: str, s: dict) -> str:
    b = ' '.join(f'{s[k]["item03"]:6.2f}/{s[k]["median_err_mm"]:4.2f}' for k in s if k.startswith('bin_'))
    return (f'{name:<9} {s["pairs"]:>6} {s["item03"]:7.3f} {s["median_pct"]:6.2f} {s["median_err_mm"]:6.3f} '
            f'{s["tracked_item03"]:7.3f} {100 * s["fallback_share"]:6.2f}%  {b}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--min-index', type=int, default=1)
    args = ap.parse_args()
    MIN_INDEX[0] = args.min_index
    allres = {}
    for run in args.runs:
        res = score_run(run)
        allres[str(run)] = res
        print(f'== {run}')
        print(f'{"set":<9} {"pairs":>6} {"03 %":>7} {"med%":>6} {"med mm":>6} {"trk03":>7} {"fallbk":>7}  '
              f'bins <5 / 5-20 / 20-40 / >40 mm: 03 % / median |err| mm')
        for k in ['official', 'room5', 'room6'] + [s for s in sorted(res) if s not in ('official', 'room5', 'room6')]:
            if k in res:
                print(fmt(k, res[k]))
    if args.json:
        args.json.write_text(json.dumps(allres, indent=1))


if __name__ == '__main__':
    main()
