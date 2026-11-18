"""Phase 2 analysis (items 2.1–2.4, E3 summaries) over run directories written by `scripts/phase2_run.py`.

  python scripts/phase2_analyze.py transfer  --out reports/phase2/transfer.json
  python scripts/phase2_analyze.py latency   --out reports/phase2/latency.json
  python scripts/phase2_analyze.py gfilter   --runs runs/p2-tr-b0.02-j0.03-s{0,1,2} --out reports/phase2/gfilter.json
  python scripts/phase2_analyze.py summary   --runs runs/p2-e3-val-da3 [--variant cv06|g60_cv06|...]

Metrics reuse the G1' scoring of `scripts/g1p_score.py` (per-pair item 03, stride-3, min-2mm, robust σ∥, 30-pair
moving-block bootstrap, seeds averaged per pair before the bootstrap). Variants:
  raw        tracker output
  cv06       causal CV prior α = 0.6 (`scripts/cv_prior.py`), the G1' C configuration
  g<τ>_cv06  robust g filter (`dvslam/scale/g_filter.py`, τ in frames) on m_prior, then CV
  a<τ>_cv06  same on m_sensor: diagnostic sensor-depth anchor (item 2.4, never submitted)
κ (item 2.1): per pair s = log(‖t_pred‖/‖t_gt‖) on pairs with ‖t_gt‖ ≥ 2 mm. The prior-induced part is isolated by
pairing each run with the oracle-prior run (b = 0, jitter 0; C is deterministic given the prior):
Δs = s − s_oracle, minus its per-scene median (removes the b offset), robust sd = 1.4826·MAD, pooled over scenes and
seeds. The non-jitter floor (same with jitter 0, b > 0) is removed in quadrature. κ = sd / jitter.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dvslam.odometry.base import load_increments  # noqa: E402
from dvslam.scale.g_filter import apply_g, g_series  # noqa: E402
from scripts.cv_prior import cv_filter  # noqa: E402

_spec = importlib.util.spec_from_file_location('p1c_score', REPO / 'scripts/g1p_score.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

B_SET, J_SET = (0.0, 0.01, 0.02, 0.05), (0.0, 0.01, 0.03, 0.10)


def tr_runs(b, j):
    seeds = (0,) if j == 0 else (0, 1, 2)
    return [REPO / f'runs/p2-tr-b{b:g}-j{j:g}-s{s}' for s in seeds]


def transform(inc, extra, variant):
    m = re.fullmatch(r'([ga])(\d+)_cv06', variant)
    if m:
        key = 'm_prior' if m[1] == 'g' else 'm_sensor'
        g = g_series(np.r_[np.nan, extra[key]], tau=float(m[2]))
        return cv_filter(apply_g(inc, g), 0.6), g
    if variant == 'cv06':
        return cv_filter(inc, 0.6), None
    if variant == 'raw':
        return inc, None
    raise ValueError(variant)


def load(run: Path, variant: str):
    out = {}
    for f in sorted(run.glob('*.inc.npz')):
        z = load_increments(f)
        ex = dict(np.load(f.with_name(f.name.replace('.inc.npz', '.extra.npz'))))
        inc, g = transform(z['rel_increment'], ex, variant)
        out[z['meta']['scene']] = (z['meta'], inc, g, ex)
    return out


def per_scene(runs, variant):
    """scene -> seed-averaged per-pair arrays (and per-seed raw extras)."""
    P = [load(r, variant) for r in runs]
    scenes = sorted(set.intersection(*(set(p) for p in P)))
    A = {}
    for sc in scenes:
        T = S.seq_of(P[0][sc][0]).T_wc
        pp = [S.per_pair(T, p[sc][1]) for p in P]
        A[sc] = {k: (np.mean([q[k] for q in pp], 0) if isinstance(pp[0][k], np.ndarray) else float(np.mean([q[k] for q in pp])))
                 for k in pp[0]}
        A[sc]['_P'] = [p[sc] for p in P]
    return A


def summarize(A):
    sc = sorted(A)
    L = lambda k: [A[s][k] for s in sc]  # noqa: E731
    r = {'scenes': sc}
    for k, name in (('e03', 'item03'), ('e3', 'stride3'), ('e2mm', 'min2mm'), ('rot', 'item04')):
        r[name] = S.boot(L(k))
    r['rpe1s_mm'] = [1e3 * v for v in S.boot(L('rpe2'), post=np.sqrt)]
    r['sig_rob_mm'] = S.boot(L('epar'), stat=S.mad_sigma)
    r['ate_mm'] = 1e3 * float(np.mean(L('ate')))
    return r


def paired(A, B, key='e03'):
    sc = sorted(set(A) & set(B))
    return S.boot([A[s][key] - B[s][key] for s in sc])


def paired_sigma(A, B):
    """Δ of robust σ∥ (mm), block bootstrap with common resampling indices."""
    sc = sorted(set(A) & set(B))
    rng = np.random.default_rng(0)
    point = float(np.mean([S.mad_sigma(A[s]['epar']) - S.mad_sigma(B[s]['epar']) for s in sc]))
    out = []
    for _ in range(1000):
        v = []
        for s in sc:
            n = len(A[s]['epar'])
            b = min(30, n)
            st = rng.integers(0, n - b + 1, size=-(-n // b))
            i = (st[:, None] + np.arange(b)).ravel()[:n]
            v.append(S.mad_sigma(A[s]['epar'][i]) - S.mad_sigma(B[s]['epar'][i]))
        out.append(np.mean(v))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return point, float(lo), float(hi)


def log_scale(T_gt, inc):
    from dvslam.geometry.se3 import relative_pose
    gt = np.stack([relative_pose(T_gt[t - 1], T_gt[t])[:3, 3] for t in range(1, len(T_gt))])
    ng, npd = np.linalg.norm(gt, axis=1), np.linalg.norm(inc[:, :3, 3], axis=1)
    s = np.full(len(gt), np.nan)
    ok = (ng >= 2e-3) & (npd > 0)
    s[ok] = np.log(npd[ok] / ng[ok])
    return s


def kappa_sd(runs, oracle_run, variant):
    """Pooled robust sd of Δs (prior-induced per-pair log-scale), and the pooled median of Δs."""
    O = load(oracle_run, variant)
    dev, meds = [], []
    for r in runs:
        P = load(r, variant)
        for sc in sorted(P):
            T = S.seq_of(P[sc][0]).T_wc
            d = log_scale(T, P[sc][1]) - log_scale(T, O[sc][1])
            d = d[np.isfinite(d)]
            meds.append(np.median(d))
            dev.append(d - np.median(d))
    dev = np.concatenate(dev)
    return 1.4826 * float(np.median(np.abs(dev))), float(np.std(dev)), float(np.median(meds))


def cmd_transfer(args):
    oracle = tr_runs(0.0, 0.0)
    res = {'rows': []}
    base = {v: per_scene(oracle, v) for v in ('raw', 'cv06')}
    floor = {}
    for b in B_SET:
        for j in J_SET:
            runs = tr_runs(b, j)
            if not all((r / 'loft.inc.npz').exists() for r in runs):
                continue
            row = {'b': b, 'jitter': j, 'seeds': len(runs)}
            for v in ('raw', 'cv06'):
                A = per_scene(runs, v)
                row[v] = summarize(A)
                row[v]['d_item03_vs_oracle'] = paired(A, base[v], 'e03')
                row[v]['d_stride3_vs_oracle'] = paired(A, base[v], 'e3')
                row[v]['d_sig_vs_oracle'] = paired_sigma(A, base[v])
                if b or j:
                    row[v]['ds_mad'], row[v]['ds_std'], row[v]['ds_median'] = kappa_sd(runs, oracle[0], v)
                    if j == 0:
                        floor.setdefault(v, []).append(row[v]['ds_mad'])
            res['rows'].append(row)
            print(b, j, {v: (round(row[v]['item03'][0], 3), [round(x, 3) for x in row[v]['d_item03_vs_oracle']],
                             round(row[v].get('ds_mad', 0), 5)) for v in ('raw', 'cv06')}, flush=True)
    for v in ('raw', 'cv06'):
        fl = float(np.median(floor.get(v, [0.0])))
        ks = []
        for row in res['rows']:
            if row['jitter'] > 0:
                k = np.sqrt(max(row[v]['ds_mad'] ** 2 - fl ** 2, 0)) / row['jitter']
                row[v]['kappa'] = float(k)
                ks.append((row['jitter'], k))
        res[f'floor_{v}'] = fl
        res[f'kappa_{v}'] = {str(jj): float(np.median([k for j2, k in ks if j2 == jj])) for jj in sorted({j for j, _ in ks})}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


def cmd_latency(args):
    res = []
    for d in sorted(REPO.glob('runs/p2-lat-*-s0')):
        tag = d.name[len('p2-lat-'):-3]
        runs = [REPO / f'runs/{d.name[:-3]}-s{s}' for s in (0, 1, 2)]
        runs = [r for r in runs if (r / 'loft.inc.npz').exists()]
        if not runs:
            continue
        seeds = [int(r.name[-1]) for r in runs]
        A = per_scene(runs, 'cv06')
        Bref = per_scene([REPO / f'runs/p2-tr-b0.02-j0.03-s{s}' for s in seeds], 'cv06')
        ages = np.concatenate([np.load(f)['prior_age'] for r in runs for f in r.glob('*.extra.npz')])
        row = {'tag': tag, 'seeds': seeds, 'item03': summarize(A)['item03'], 'd_item03': paired(A, Bref, 'e03'),
               'd_stride3': paired(A, Bref, 'e3'), 'd_sig': paired_sigma(A, Bref),
               'age_mean': float(ages.mean()) if len(ages) else 0.0}
        res.append(row)
        print(tag, [round(x, 3) for x in row['d_item03']], [round(x, 3) for x in row['d_stride3']], flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def cmd_gfilter(args):
    runs = [Path(r) for r in args.runs]
    base = per_scene(runs, 'cv06')
    res = {'runs': [str(r) for r in runs], 'base': summarize(base), 'rows': []}
    taus = [int(t) for t in args.taus.split(',')]
    for kind in ('g', 'a'):
        for tau in taus:
            v = f'{kind}{tau}_cv06'
            A = per_scene(runs, v)
            gm = []
            for sc in sorted(A):
                for meta, _, g, ex in A[sc]['_P']:
                    gm.append(float(np.nanmean(g)))
            row = {'variant': v, **summarize(A), 'd_item03': paired(A, base, 'e03'), 'd_stride3': paired(A, base, 'e3'),
                   'd_sig': paired_sigma(A, base), 'g_mean': float(np.mean(gm))}
            row['sig_change_pct'] = 100 * row['d_sig'][0] / res['base']['sig_rob_mm'][0]
            res['rows'].append(row)
            print(v, round(row['item03'][0], 3), [round(x, 3) for x in row['d_item03']],
                  [round(x, 4) for x in row['d_sig']], round(row['sig_change_pct'], 2), round(row['g_mean'], 4), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def anchor_table(runs, tau=300):
    """Per scene: mean over frames of g_anchor − g_prior (filtered, valid part) and of m_sensor − m_prior."""
    out = {}
    for r in runs:
        for f in sorted(Path(r).glob('*.extra.npz')):
            ex = np.load(f)
            ga, gp = g_series(ex['m_sensor'], tau=tau), g_series(ex['m_prior'], tau=tau)
            v = slice(len(ga) // 10, None)                          # skip the start-up transient
            dm = ex['m_sensor'] - ex['m_prior']
            out.setdefault(f.name.split('.')[0], []).append(
                {'g_diff': float(np.mean(ga[v] - gp[v])), 'm_diff_median': float(np.nanmedian(dm)),
                 'm_valid': float(np.isfinite(dm).mean())})
    return {k: {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]} for k, v in out.items()}


def cmd_anchor(args):
    res = {}
    for r in args.runs:
        res[r] = anchor_table([r], args.tau)
        print(r, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res[r].items()}), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def cmd_summary(args):
    for r in args.runs:
        A = per_scene([Path(r)], args.variant)
        s = summarize(A)
        print(f'{r} [{args.variant}]: 03 {s["item03"][0]:.2f} [{s["item03"][1]:.2f},{s["item03"][2]:.2f}]  '
              f'stride3 {s["stride3"][0]:.2f}  min2mm {s["min2mm"][0]:.2f}  04 {s["item04"][0]:.4f}  '
              f'σ∥ {s["sig_rob_mm"][0]:.3f}  RPE1s {s["rpe1s_mm"][0]:.1f} mm  ATE {s["ate_mm"]:.0f} mm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['transfer', 'latency', 'gfilter', 'anchor', 'summary'])
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--out', default=None)
    ap.add_argument('--variant', default='cv06')
    ap.add_argument('--taus', default='30,100,300,1000')
    ap.add_argument('--tau', type=int, default=300)
    args = ap.parse_args()
    {'transfer': cmd_transfer, 'latency': cmd_latency, 'gfilter': cmd_gfilter, 'anchor': cmd_anchor,
     'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
