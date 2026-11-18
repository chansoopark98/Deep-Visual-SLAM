"""Causal constant-velocity prior on scored increments (plan v3 item 0.11, §4.4).

Steady-state Kalman filter on the per-frame translation (and optionally the rotation vector) with a
constant-velocity model: the prediction for frame t is the filtered increment of t-1, and
  x_t = x_{t-1} + K (z_t - x_{t-1}),   K = 1 - alpha.
alpha = 0 leaves the run unchanged; one gain per run (covariance ratio q/r collapsed to the steady-state gain).
Only past increments are used. Works on any run directory of <scene>.inc.npz (KLT, DPVO, ...).

  python scripts/cv_prior.py --runs runs/klt-df-val --tune                 # grid on a validation run
  python scripts/cv_prior.py --runs runs/klt-df-test --out runs/klt-df-test-cv --alpha 0.3
  python scripts/cv_prior.py --runs runs/klt-df-test --out runs/klt-df-test-cv --alpha 0.3 --compare
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.eval import diagnostics as D  # noqa: E402
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import load_increments  # noqa: E402


def cv_filter(inc: np.ndarray, alpha: float, alpha_rot: float = 0.0) -> np.ndarray:
    """Causal blend of each increment with the previous filtered one. inc: (n, 4, 4) -> (n, 4, 4)."""
    out = inc.copy()
    t_prev = inc[0, :3, 3].copy()
    r_prev = Rotation.from_matrix(inc[0, :3, :3]).as_rotvec()
    for k in range(1, len(inc)):
        t_prev = alpha * t_prev + (1 - alpha) * inc[k, :3, 3]
        r_prev = alpha_rot * r_prev + (1 - alpha_rot) * Rotation.from_matrix(inc[k, :3, :3]).as_rotvec()
        out[k, :3, 3] = t_prev
        out[k, :3, :3] = Rotation.from_rotvec(r_prev).as_matrix()
    return out


def windowed_pct(gt_t: np.ndarray, pred_t: np.ndarray, window: int) -> np.ndarray:
    """Per-pair item-03 % after one oracle scale per window (same rule as D.windowed_rescale_mape)."""
    pct = np.empty(len(gt_t))
    for s in range(0, len(gt_t), window):
        g, p = gt_t[s:s + window], pred_t[s:s + window]
        ok = (np.linalg.norm(p, axis=1) > 0).any() and (np.linalg.norm(g, axis=1) > 1e-6).any()
        pct[s:s + window] = D._pct_errors(g, (D.best_rescale(g, p)[0] if ok else 1.0) * p)
    return pct


def load_run(runs: Path) -> list[tuple[Path, dict, np.ndarray]]:
    out = []
    for f in sorted(runs.glob('*.inc.npz')):
        inc = load_increments(f)
        m = inc['meta']
        seq = redwood.load_sequence(m['scene'], m['split'], m['data_root'])
        gt_t = np.stack([relative_pose(seq.T_wc[t - 1], seq.T_wc[t])[:3, 3] for t in inc['index']])
        out.append((f, inc, gt_t))
    return out


def score(run, alpha: float, alpha_rot: float) -> dict[str, list[np.ndarray]]:
    res = {'raw': [], 'win1s': [], 'win2s': []}
    for _, inc, gt_t in run:
        p = cv_filter(inc['rel_increment'], alpha, alpha_rot)[:, :3, 3]
        res['raw'].append(D._pct_errors(gt_t, p))
        res['win1s'].append(windowed_pct(gt_t, p, 30))
        res['win2s'].append(windowed_pct(gt_t, p, 60))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--alpha', type=float, default=0.0, help='weight of the previous filtered increment (translation)')
    ap.add_argument('--alpha-rot', type=float, default=0.0, help='same for the rotation vector (default: off)')
    ap.add_argument('--tune', action='store_true', help='grid over alpha on a validation run (refuses test)')
    ap.add_argument('--compare', action='store_true', help='print MAPE change vs alpha=0 with 95 %% block CI')
    args = ap.parse_args()
    run = load_run(args.runs)
    if not run:
        sys.exit(f'no *.inc.npz in {args.runs}')

    if args.tune:
        if any(inc['meta']['split'] != 'validation' for _, inc, _ in run):
            sys.exit('--tune only on validation runs (plan v3 F8)')
        print(f'{"alpha":>6} {"raw %":>9} {"win1s %":>8} {"win2s %":>8}')
        for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            r = score(run, a, args.alpha_rot)
            print(f'{a:>6.2f} ' + ' '.join(f'{np.mean([v.mean() for v in r[k]]):>8.3f}' for k in ('raw', 'win1s', 'win2s')))
        return

    if args.compare:
        base, filt = score(run, 0.0, 0.0), score(run, args.alpha, args.alpha_rot)
        for k in ('raw', 'win1s', 'win2s'):
            d, lo, hi = D.block_bootstrap_ci([f - b for f, b in zip(filt[k], base[k])])
            print(f'{k:>6}: {np.mean([v.mean() for v in base[k]]):.3f} -> {np.mean([v.mean() for v in filt[k]]):.3f} %  '
                  f'Δ {d:+.3f} %p  95 % block CI [{lo:+.3f}, {hi:+.3f}]')

    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        for f, inc, _ in run:
            meta = dict(inc['meta'], cv_prior={'alpha': args.alpha, 'alpha_rot': args.alpha_rot,
                                               'source_run': str(args.runs)})
            np.savez(args.out / f.name, index=inc['index'], timestamp=inc['timestamp'],
                     rel_increment=cv_filter(inc['rel_increment'], args.alpha, args.alpha_rot),
                     tracking_state=inc['tracking_state'], source=inc['source'], meta=np.array(json.dumps(meta)))
        print(args.out)


if __name__ == '__main__':
    main()
