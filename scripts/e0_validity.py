"""Plan v3 item 0.8 / G0-E0 validity rule: per-axis residual between E0 increments and GT increments.

  python scripts/e0_validity.py --runs runs/e0-rgbd-icp-test

Residual = std of (pred_t - gt_t) per axis, in mm, per scene and pooled. If every pooled axis is below
0.05 mm, E0 shares error with the GT (the GT is an Open3D reconstruction from this depth) -> INVALID for the gate.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import load_increments  # noqa: E402

THRESHOLD_MM = 0.05


def residuals(inc_path: Path, data_root: Path | None) -> tuple[str, str, np.ndarray]:
    inc = load_increments(inc_path)
    meta = inc['meta']
    seq = redwood.load_sequence(meta['scene'], meta['split'], data_root or meta['data_root'])
    gt = np.array([relative_pose(seq.T_wc[t - 1], seq.T_wc[t])[:3, 3] for t in inc['index']])
    return meta['scene'], meta['split'], inc['rel_increment'][:, :3, 3] - gt


def verdict(pooled_std_mm: np.ndarray) -> str:
    return 'INVALID' if bool((pooled_std_mm < THRESHOLD_MM).all()) else 'VALID'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=None)
    args = ap.parse_args()
    files = sorted(args.runs.glob('*.inc.npz'))
    if not files:
        sys.exit(f'no *.inc.npz in {args.runs}')
    lines = [f'=== e0_validity: {args.runs} ===', f'{"scene":<10} {"split":<11} {"std x mm":>9} {"std y mm":>9} '
             f'{"std z mm":>9} {"rms mm":>8} {"max|res| mm":>11}']
    pooled = []
    for f in files:
        scene, split, r = residuals(f, args.data_root)
        pooled.append(r)
        s = r.std(0) * 1e3
        lines.append(f'{scene:<10} {split:<11} {s[0]:>9.4f} {s[1]:>9.4f} {s[2]:>9.4f} '
                     f'{np.sqrt((r ** 2).sum(1).mean()) * 1e3:>8.4f} {np.abs(r).max() * 1e3:>11.3f}')
    P = np.concatenate(pooled)
    s = P.std(0) * 1e3
    lines.append(f'{"POOLED":<10} {"":<11} {s[0]:>9.4f} {s[1]:>9.4f} {s[2]:>9.4f} '
                 f'{np.sqrt((P ** 2).sum(1).mean()) * 1e3:>8.4f} {np.abs(P).max() * 1e3:>11.3f}')
    v = verdict(s)
    lines.append(f'G0-E0: pooled per-axis residual std {s.round(4).tolist()} mm vs {THRESHOLD_MM} mm -> {v}'
                 + (' (shares error with GT; excluded from the gate)' if v == 'INVALID'
                    else ' (a valid E0 <= 2.0 % is supporting evidence only)'))
    out = '\n'.join(lines)
    print(out)
    (args.runs / 'e0_validity.txt').write_text(out + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
