"""EXP-015 (E0): split the tracker's item-03 error into a scale-state part and a per-pair pose-noise part (diagnostic).

  python scripts/exp015_scale_oracle.py RUN [RUN ...] [--compare RUN_A RUN_B ...] [--json out.json]

Only already saved outputs are read (RUN/room<N>.inc.npz and, when present, room<N>.extra.npz). Every row except
'deployed' and the templates uses mocap GT to choose a scale: ORACLE diagnostics, never a deployable result.

Conventions
  pairs        : (t−1, t) with the filter of scripts/exp003_score.py (pair_table: GT valid at both ends, t ≥ 1);
                 checked against pair_table at load. p / g = predicted / GT step length (m). item 03 = mean over pairs
                 of |p − g| / g · 100, pairs with g ≤ 1e-6 m count as 0 (as pair_table). official = room1-4 + room6
                 pooled by pairs.
  oracle scale : one s > 0 multiplies every p of a group: a sequence ('seq') or a non-overlapping window of W frames
                 ('winW', window id (t − 1) // W). (i) 'metric': argmin_s Σ |s·p − g| / g, which is the weighted median
                 of g / p with weights p / g; (ii) 'median': median of g / p. Pairs with p = 0 or g ≤ 1e-6 m do not
                 move s; a group without such pairs keeps s = 1. 'winW shuffled' (control): the pairs of a room are
                 permuted in time (seed 0) before the same 'metric' window oracle, so its gain over 'deployed' is what
                 a W-frame oracle gets from fitting per-pair noise alone; 'winW metric' minus that is time-correlated.
  bins         : GT step < 5, 5-20, 20-40, > 40 mm: item 03 inside the bin and its contribution Σ_bin / N_all (pp)
                 for deployed, seq metric / median, win200 metric, win20 metric / shuffled.
  scale series : log s ('metric') per 200-frame window (windows with ≥ 100 scored pairs): mean, std, lag-1
                 autocorrelation. The tracker's own state g (extra.npz 'g' at the output frame; output = raw · e^−g):
                 per window mean of −g = deployed log scale; −g + log s = the log scale an ideal state would have had.
  templates    : T1 = p_{t−1} scored against g_t (one-frame persistence; the first pair keeps p_1). T2 = centred
                 3-frame mean (p_{t−1} + p_t + p_{t+1}) / 3, ends use the two available; needs one future frame (50 ms
                 lag). Both are computed from the run's own output; apply them to a causal run.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.odometry.base import load_increments
from scripts.exp003_score import BINS_MM, MIN_INDEX, OFFICIAL, gt_steps, pair_table

WINDOWS = (2000, 600, 200, 60, 20)
SERIES_W = 200
G_EPS = 1e-6
SETS = ('official', 'room5', 'room6', 'room1', 'room2', 'room3', 'room4')


def item03(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-pair item 03 (%), same guard as pair_table."""
    return np.where(g > G_EPS, np.abs(p - g) / np.maximum(g, 1e-12) * 100, 0.0)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Lower weighted median: a minimiser of Σ w_i |x_i − m|."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def oracle_scale(p: np.ndarray, g: np.ndarray, method: str) -> float:
    """ORACLE (uses GT) single scale for a group of pairs; 'metric' minimises item 03, 'median' = median(g / p)."""
    ok = (p > 0) & (g > G_EPS)
    if not ok.any():
        return 1.0
    r = g[ok] / p[ok]
    return weighted_median(r, p[ok] / g[ok]) if method == 'metric' else float(np.median(r))


def scale_groups(p: np.ndarray, g: np.ndarray, group: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """p with one oracle scale per group id; returns (scaled p, {group id: s})."""
    out, s = p.copy(), {}
    for k in np.unique(group):
        m = group == k
        s[int(k)] = oracle_scale(p[m], g[m], method)
        out[m] = p[m] * s[int(k)]
    return out, s


def templates(p_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T1 (previous step length) and T2 (centred 3-frame mean) over the full, unfiltered pair sequence of a room."""
    t1 = np.concatenate([p_all[:1], p_all[:-1]])
    s = np.convolve(p_all, np.ones(3), 'same')
    n = np.convolve(np.ones_like(p_all), np.ones(3), 'same')
    return t1, s / n


def load_room(inc_path: Path) -> dict:
    """Scored pairs of one room (filter identical to pair_table, verified) plus templates and the tracker's g."""
    inc = load_increments(inc_path)
    meta = inc['meta']
    rel, valid = gt_steps(meta['scene'], meta.get('time_offset_s', 0.0))
    idx = inc['index']
    g = np.linalg.norm(rel[idx - 1, :3, 3], axis=1)
    p = np.linalg.norm(inc['rel_increment'][:, :3, 3], axis=1)
    keep = valid[idx - 1] & (idx >= MIN_INDEX[0])
    ref = pair_table(inc_path)
    if not (np.array_equal(ref['gt'], g[keep]) and np.array_equal(ref['pred'], p[keep])):
        raise ValueError(f'{inc_path}: pair filter differs from exp003_score.pair_table')
    t1, t2 = templates(p)
    room = {'index': idx[keep], 'gt': g[keep], 'pred': p[keep], 'T1': t1[keep], 'T2': t2[keep], 'g_state': None}
    extra = inc_path.with_name(inc_path.name.replace('.inc.npz', '.extra.npz'))
    if extra.exists():
        with np.load(extra) as z:
            gmap = dict(zip(z['index'].tolist(), z['g'].tolist()))
        room['g_state'] = np.array([gmap.get(int(i), np.nan) for i in room['index']])
    return room


def variants(room: dict) -> tuple[dict, dict]:
    """Predicted step lengths per variant for one room, and the oracle scales used."""
    p, g, idx = room['pred'], room['gt'], room['index']
    pred = {'deployed': p, 'T1 persistence': room['T1'], 'T2 3-frame mean': room['T2']}
    scales = {}
    for method in ('metric', 'median'):
        pred[f'seq {method}'], s = scale_groups(p, g, np.zeros(len(p), int), method)
        scales[f'seq {method}'] = s[0]
    perm = np.random.default_rng(0).permutation(len(p))
    for w in WINDOWS:
        for method in ('metric', 'median'):
            pred[f'win{w} {method}'], _ = scale_groups(p, g, (idx - 1) // w, method)
        shuf, _ = scale_groups(p[perm], g[perm], (idx - 1) // w, 'metric')
        pred[f'win{w} shuffled'] = np.empty_like(p)
        pred[f'win{w} shuffled'][perm] = shuf
    return pred, scales


def scale_series(room: dict) -> dict:
    """log s ('metric') per SERIES_W-frame window and its relation to the tracker's own scale state g."""
    p, g, idx = room['pred'], room['gt'], room['index']
    wid = (idx - 1) // SERIES_W
    ids = [k for k in np.unique(wid) if (wid == k).sum() >= SERIES_W // 2]
    logs = np.array([np.log(oracle_scale(p[wid == k], g[wid == k], 'metric')) for k in ids])
    x = logs - logs.mean()
    out = {'windows': len(ids), 'mean_log_s': float(logs.mean()), 'std_log_s': float(logs.std()),
           'ac1_log_s': float((x[1:] * x[:-1]).sum() / (x * x).sum()), 'log_s': logs.tolist(),
           'window_ids': [int(k) for k in ids]}
    if room['g_state'] is not None:
        dep = np.array([-np.nanmean(room['g_state'][wid == k]) for k in ids])
        ideal = dep + logs
        out.update(std_deployed=float(dep.std()), std_ideal=float(ideal.std()),
                   corr_deployed_ideal=float(np.corrcoef(dep, ideal)[0, 1]) if dep.std() > 0 else float('nan'))
    return out


def pooled(per_room: dict, rooms: tuple, key: str) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate([per_room[r]['pred'][key] for r in rooms]),
            np.concatenate([per_room[r]['gt'] for r in rooms]))


def set_rooms(name: str, per_room: dict) -> tuple | None:
    rooms = OFFICIAL if name == 'official' else (name,)
    return rooms if all(r in per_room for r in rooms) else None


def analyse(run: Path) -> dict:
    files = {f.name.split('.')[0]: f for f in sorted(run.rglob('room*.inc.npz'))}
    per_room = {}
    for r, f in files.items():
        room = load_room(f)
        pred, scales = variants(room)
        per_room[r] = {'gt': room['gt'], 'pred': pred, 'scales': scales, 'series': scale_series(room)}
    res = {'item03': {}, 'bins': {}, 'scales': {r: v['scales'] for r, v in per_room.items()},
           'series': {r: v['series'] for r, v in per_room.items()}}
    keys = list(next(iter(per_room.values()))['pred'])
    for name in SETS:
        rooms = set_rooms(name, per_room)
        if rooms is None:
            continue
        res['item03'][name] = {k: float(item03(*pooled(per_room, rooms, k)).mean()) for k in keys}
        res['bins'][name] = {}
        for k in ('deployed', 'seq metric', 'seq median', 'win200 metric', 'win20 metric', 'win20 shuffled'):
            p, g = pooled(per_room, rooms, k)
            e, gmm = item03(p, g), g * 1e3
            res['bins'][name][k] = {f'{lo}-{hi}': {'item03': float(e[(gmm >= lo) & (gmm < hi)].mean()),
                                                   'contrib_pp': float(e[(gmm >= lo) & (gmm < hi)].sum() / len(e))}
                                    for lo, hi in itertools.pairwise(BINS_MM)}
    return res


def compare(a: dict, b: dict) -> dict:
    """Per room: log s series of run A minus run B on their common windows."""
    out = {}
    for r in sorted(set(a['series']) & set(b['series'])):
        sa, sb = a['series'][r], b['series'][r]
        common = sorted(set(sa['window_ids']) & set(sb['window_ids']))
        la = np.array([sa['log_s'][sa['window_ids'].index(k)] for k in common])
        lb = np.array([sb['log_s'][sb['window_ids'].index(k)] for k in common])
        d = la - lb
        out[r] = {'windows': len(common), 'mean_diff': float(d.mean()), 'std_diff': float(d.std()),
                  'corr': float(np.corrcoef(la, lb)[0, 1])}
    return out


def report(name: str, res: dict) -> None:
    print(f'== {name}   (all rows except deployed / T1 / T2 are ORACLE: scale chosen with mocap GT)')
    sets = [s for s in SETS if s in res['item03']]
    print(f'{"item 03 %":<18}' + ''.join(f'{s:>9}' for s in sets))
    for k in res['item03'][sets[0]]:
        print(f'{k:<18}' + ''.join(f'{res["item03"][s][k]:9.3f}' for s in sets))
    print('oracle scale per sequence (metric / median):  ' + '  '.join(
        f'{r} {v["seq metric"]:.4f}/{v["seq median"]:.4f}' for r, v in sorted(res['scales'].items())))
    for s in ('official', 'room5', 'room6'):
        if s not in res['bins']:
            continue
        print(f'bins {s:<9} GT mm: item 03 % / contribution pp')
        for k, b in res['bins'][s].items():
            print(f'  {k:<15}' + '  '.join(f'{lh} {v["item03"]:6.2f}/{v["contrib_pp"]:.3f}' for lh, v in b.items()))
    print(f'scale series (log s metric, {SERIES_W}-frame windows): room n mean std ac1 | std(-g) std(ideal) '
          f'corr(-g, ideal)')
    for r, v in sorted(res['series'].items()):
        tail = (f' | {v["std_deployed"]:.4f} {v["std_ideal"]:.4f} {v["corr_deployed_ideal"]:+.3f}'
                if 'std_deployed' in v else '')
        print(f'  {r} {v["windows"]:3d} {v["mean_log_s"]:+.4f} {v["std_log_s"]:.4f} {v["ac1_log_s"]:+.3f}{tail}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--compare', nargs=2, action='append', default=[], type=Path, metavar=('RUN_A', 'RUN_B'),
                    help='log-scale series of A minus B per room (both must also be listed in runs)')
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()
    allres = {}
    for run in args.runs:
        allres[str(run)] = analyse(run)
        report(str(run), allres[str(run)])
    for a, b in args.compare:
        c = compare(allres[str(a)], allres[str(b)])
        allres[f'compare {a} - {b}'] = c
        print(f'== log s ({SERIES_W}-frame windows) {a} minus {b}: room n mean std corr')
        for r, v in c.items():
            print(f'  {r} {v["windows"]:3d} {v["mean_diff"]:+.4f} {v["std_diff"]:.4f} {v["corr"]:+.3f}')
    if args.json:
        args.json.write_text(json.dumps(allres, indent=1))


if __name__ == '__main__':
    main()
