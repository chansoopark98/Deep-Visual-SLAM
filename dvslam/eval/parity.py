"""Stage parity between two runs of the same tracker, e.g. FP32 PyTorch vs ORT / TRT / C++ (plan v3 item 5.1).

Plan tolerances (5.3): item-03 difference <= 0.1 %p, and no drift of the chained trajectory over long runs.
"""
from pathlib import Path

import numpy as np

from dvslam.eval.diagnostics import _pct_errors
from dvslam.geometry.se3 import chain_increments
from dvslam.odometry.base import load_increments


def _angle_deg(R: np.ndarray) -> float:
    """Rotation angle via atan2 (well-conditioned near 0, unlike arccos of the trace)."""
    s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return float(np.degrees(np.arctan2(s, (np.trace(R) - 1) / 2)))


def compare_increments(inc_a: np.ndarray, inc_b: np.ndarray, gt_t: np.ndarray | None = None) -> dict:
    """inc_*: (N, 4, 4). gt_t: optional (N, 3) GT translations to compare item-03 values."""
    dt = np.linalg.norm(inc_a[:, :3, 3] - inc_b[:, :3, 3], axis=1)
    drot = np.array([_angle_deg(a[:3, :3] @ b[:3, :3].T) for a, b in zip(inc_a, inc_b)])
    Ta, Tb = chain_increments(np.eye(4), inc_a), chain_increments(np.eye(4), inc_b)
    drift = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=1)
    out = {'pairs': len(inc_a), 'max_dt_mm': float(dt.max() * 1e3), 'p99_dt_mm': float(np.quantile(dt, 0.99) * 1e3),
           'max_drot_deg': float(drot.max()), 'end_drift_mm': float(drift[-1] * 1e3),
           'max_drift_mm': float(drift.max() * 1e3)}
    if gt_t is not None:
        out['item03_diff_pp'] = float(_pct_errors(gt_t, inc_a[:, :3, 3]).mean()
                                      - _pct_errors(gt_t, inc_b[:, :3, 3]).mean())
    return out


def compare_runs(run_a: Path, run_b: Path) -> dict[str, dict]:
    res = {}
    for pa in sorted(Path(run_a).glob('*.inc.npz')):
        pb = Path(run_b) / pa.name
        if not pb.exists():
            continue
        a, b = load_increments(pa), load_increments(pb)
        if not np.array_equal(a['index'], b['index']):
            raise ValueError(f'{pa.name}: frame indices differ between runs')
        res[a['meta']['scene']] = compare_increments(a['rel_increment'], b['rel_increment'])
    return res


if __name__ == '__main__':
    import sys
    for scene, r in compare_runs(Path(sys.argv[1]), Path(sys.argv[2])).items():
        print(scene, ' '.join(f'{k}={v:.4g}' for k, v in r.items()))
