"""Convert the C++ frame loop's increments.csv (cpp/src/frame_loop.cpp) to the scored `.inc.npz` schema.

    python -m dvslam.export.inc_csv <run_dir>/increments.csv <out>.inc.npz --scene apartment --split validation
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dvslam.odometry.base import PoseEstimate, save_increments


def read_increments_csv(path: str | Path) -> list[PoseEstimate]:
    rows = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    rows = np.atleast_1d(rows)
    mats = np.stack([rows[f'm{i}{j}'].astype(np.float64) for i in range(4) for j in range(4)], 1).reshape(-1, 4, 4)
    return [PoseEstimate(int(r['index']), float(r['timestamp']), m, tracking_state=str(r['tracking_state']),
                         source=str(r['source'])) for r, m in zip(rows, mats)]


def read_processed_mask(frames_csv: str | Path) -> np.ndarray:
    f = np.genfromtxt(frames_csv, delimiter=',', names=True)
    return np.atleast_1d(f['processed']).astype(bool)


def csv_to_inc(csv_path: str | Path, out_path: str | Path, meta: dict) -> None:
    est = read_increments_csv(csv_path)
    save_increments(Path(out_path), est, {**meta, 'n_frames': len(est) + 1})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv')
    p.add_argument('out')
    p.add_argument('--scene', required=True)
    p.add_argument('--split', default='validation')
    p.add_argument('--odometry', default='cpp_loop')
    p.add_argument('--meta', default='{}', help='extra JSON merged into meta')
    a = p.parse_args()
    meta = {'scene': a.scene, 'split': a.split, 'odometry': a.odometry, 'frame_rate_hz': 30.0, **json.loads(a.meta)}
    csv_to_inc(a.csv, a.out, meta)


if __name__ == '__main__':
    main()
