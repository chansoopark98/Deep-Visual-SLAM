"""Phase 1 C scoring: G1' metrics with 95 % moving-block CIs (30-pair blocks, 1000 reps) for run directories.

  python scripts/g1p_score.py --runs runs/p1c-win-s0 runs/p1c-win-s1 runs/p1c-win-s2 --label win --out x.json
Seeds (several --runs) are combined by averaging the per-pair errors across seeds before the bootstrap.
Variants: raw, cv06 (scripts/cv_prior.py α = 0.6), and for head runs headoff / headoff_cv06 (base increments from
<scene>.extra.npz 'head_raw_inc').
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dvslam.data import redwood  # noqa: E402
from dvslam.eval import diagnostics as D  # noqa: E402
from dvslam.eval.metrics import pair_metrics  # noqa: E402
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import causal_trajectory, load_increments  # noqa: E402
from scripts.cv_prior import cv_filter, windowed_pct  # noqa: E402

_SEQ = {}


def seq_of(meta):
    key = (meta['scene'], meta['split'])
    if key not in _SEQ:
        _SEQ[key] = redwood.load_sequence(meta['scene'], meta['split'], meta['data_root'])
    return _SEQ[key]


def per_pair(T_gt, inc):
    """Per-pair arrays for one scene."""
    gt_t = np.stack([relative_pose(T_gt[t - 1], T_gt[t])[:3, 3] for t in range(1, len(T_gt))])
    p = inc[:, :3, 3]
    e03 = D._pct_errors(gt_t, p)
    rot = np.array([pair_metrics(relative_pose(T_gt[t - 1], T_gt[t]), inc[t - 1])['rpy_abs_err_deg'].mean()
                    for t in range(1, len(T_gt))])
    k = 3                                                         # stride-3, same rule as D.stride_mape
    P = np.tile(np.eye(4), (len(inc) + 1, 1, 1))
    for i, dT in enumerate(inc):
        P[i + 1] = P[i] @ dT
    g3 = np.stack([relative_pose(T_gt[t - k], T_gt[t])[:3, 3] for t in range(k, len(T_gt))])
    p3 = np.stack([(np.linalg.inv(P[t - k]) @ P[t])[:3, 3] for t in range(k, len(T_gt))])
    e3 = D._pct_errors(g3, p3)
    m2 = np.linalg.norm(gt_t, axis=1) >= 2e-3
    T_est = causal_trajectory(inc)
    te = []
    for i in range(len(T_gt) - 30):
        E = np.linalg.inv(relative_pose(T_gt[i], T_gt[i + 30])) @ relative_pose(T_est[i], T_est[i + 30])
        te.append(np.linalg.norm(E[:3, 3]) ** 2)
    sp = D.sigma_parallel(gt_t, p)
    u = gt_t / np.maximum(np.linalg.norm(gt_t, axis=1, keepdims=True), 1e-12)
    epar = ((p - gt_t) * u).sum(1) * 1e3
    return {'e03': e03, 'e3': e3, 'e2mm': e03[m2], 'rpe2': np.array(te), 'rot': rot, 'epar': epar,
            'win1s': windowed_pct(gt_t, p, 30),
            'ate': D.ate_rmse(T_gt, T_est, sim3=False)[0], 'sig_rob': sp['sigma_par_robust_mm'],
            'sig_std': sp['sigma_par_mm']}


def boot(per_scene, stat=np.mean, post=lambda x: x, block=30, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    point = float(np.mean([post(stat(v)) for v in per_scene]))
    out = np.empty(reps)
    for r in range(reps):
        vals = []
        for v in per_scene:
            n = len(v)
            b = min(block, n)
            st = rng.integers(0, n - b + 1, size=-(-n // b))
            vals.append(post(stat(v[(st[:, None] + np.arange(b)).ravel()[:n]])))
        out[r] = np.mean(vals)
    lo, hi = np.quantile(out, [0.025, 0.975])
    return point, float(lo), float(hi)


def mad_sigma(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def load_variant(run: Path, variant: str):
    out = {}
    for f in sorted(run.glob('*.inc.npz')):
        z = load_increments(f)
        inc = z['rel_increment']
        if variant.startswith('headoff'):
            inc = np.load(f.with_name(f.name.replace('.inc.npz', '.extra.npz')))['head_raw_inc']
        if variant.endswith('cv06'):
            inc = cv_filter(inc, 0.6)
        out[z['meta']['scene']] = (z['meta'], inc)
    return out


def score(runs: list[Path], variant: str) -> dict:
    per_seed = [load_variant(r, variant) for r in runs]
    scenes = sorted(per_seed[0])
    A = {}
    for sc in scenes:
        meta = per_seed[0][sc][0]
        T = seq_of(meta).T_wc
        pp = [per_pair(T, s[sc][1]) for s in per_seed]
        A[sc] = {k: (np.mean([p[k] for p in pp], 0) if isinstance(pp[0][k], np.ndarray) else float(np.mean([p[k] for p in pp])))
                 for k in pp[0]}
    L = lambda k: [A[s][k] for s in scenes]  # noqa: E731
    res = {'scenes': scenes, 'n_seeds': len(runs)}
    for k, name in (('e03', 'item03'), ('e3', 'stride3'), ('e2mm', 'min2mm'), ('win1s', 'oracle1s'), ('rot', 'item04')):
        res[name] = boot(L(k))
    res['rpe1s'] = boot(L('rpe2'), post=np.sqrt)
    res['sig_rob_mm'] = boot(L('epar'), stat=mad_sigma)
    res['sig_std_mm'] = float(np.mean(L('sig_std')))
    res['ate_se3'] = float(np.mean(L('ate')))
    meta0 = per_seed[0][scenes[0]][0]
    res['opts'] = meta0.get('opts')
    ms = []
    for r in runs:
        tf = r / 'timings.csv'
        if tf.exists():
            with open(tf) as f:
                ms += [float(row['total_ms']) for row in csv.DictReader(f)]
    res['ms_median'] = float(np.median(ms)) if ms else float('nan')
    if variant in ('raw', 'cv06') and (runs[0] / f'{scenes[0]}.extra.npz').exists():
        acc, n, hm = 0, 0, []
        for r in runs:
            for sc in scenes:
                e = np.load(r / f'{sc}.extra.npz')
                if 'head_status' in e:
                    acc += int((e['head_status'] == 'accepted').sum())
                    n += len(e['head_status'])
                    hm += list(e['head_head_ms'])
        if n:
            res['head_accept_rate'] = acc / n
            res['head_ms_median'] = float(np.median(hm))
    return res


def paired(runs_a, var_a, runs_b, var_b, key='e3'):
    """Δ (a − b) of the seed-averaged per-pair metric, with block CI."""
    A, B = [load_variant(r, var_a) for r in runs_a], [load_variant(r, var_b) for r in runs_b]
    d = []
    for sc in sorted(A[0]):
        T = seq_of(A[0][sc][0]).T_wc
        a = np.mean([per_pair(T, s[sc][1])[key] for s in A], 0)
        b = np.mean([per_pair(T, s[sc][1])[key] for s in B], 0)
        d.append(a - b)
    return boot(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--variants', nargs='+', default=['raw', 'cv06'])
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    allres = json.loads(args.out.read_text()) if args.out.exists() else {}
    for v in args.variants:
        r = score(args.runs, v)
        r['runs'] = [str(x) for x in args.runs]
        allres[f'{args.label}/{v}'] = r
        print(f'{args.label}/{v}: 03 {r["item03"][0]:.2f} [{r["item03"][1]:.2f},{r["item03"][2]:.2f}]  '
              f'stride3 {r["stride3"][0]:.2f} [{r["stride3"][1]:.2f},{r["stride3"][2]:.2f}]  '
              f'min2mm {r["min2mm"][0]:.2f} [{r["min2mm"][1]:.2f},{r["min2mm"][2]:.2f}]  '
              f'RPE1s {r["rpe1s"][0]*1e3:.1f} mm  ATE {r["ate_se3"]*1e3:.0f} mm  04 {r["item04"][0]:.4f}°  '
              f'σ∥rob {r["sig_rob_mm"][0]:.3f} [{r["sig_rob_mm"][1]:.3f},{r["sig_rob_mm"][2]:.3f}]  '
              f'1s-oracle {r["oracle1s"][0]:.2f}  ms {r["ms_median"]:.0f}'
              + (f'  head acc {r["head_accept_rate"]:.3f}' if 'head_accept_rate' in r else ''), flush=True)
        args.out.write_text(json.dumps(allres, indent=1))


if __name__ == '__main__':
    main()
