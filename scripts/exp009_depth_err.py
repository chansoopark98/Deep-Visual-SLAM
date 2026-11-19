"""EXP-009: error of cached depth maps against the stereo reference (scripts/exp009_stereo_depth.py), one room.

  python scripts/exp009_depth_err.py --scene room5 --ref runs/eval/exp009/depth_stereo --dirs <cache dir> ...

Every 10th frame, pixels where the reference is valid and both depths lie in [0.3, 6] m; e = log(d / d_ref).
Reported: median e (global scale bias), median |e − median e| (shape error after the bias), per-frame scale
s_k = median_k e: its sd over frames (scale jitter) and the sd of s_k − s_{k−1} for consecutive sampled frames
(10 frames apart), which is what moves the tracker's map scale between keyframes.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--dirs', type=Path, nargs='+', required=True)
    ap.add_argument('--every', type=int, default=10)
    args = ap.parse_args()
    ref = np.load(args.ref / f'{args.scene}.npy', mmap_mode='r')
    idx = np.arange(0, ref.shape[0], args.every)
    print(f'{"depth":<34} {"bias %":>7} {"shape %":>8} {"frame sd %":>10} {"Δframe sd %":>11} {"valid %":>8}')
    for d in args.dirs:
        dep = np.load(d / f'{args.scene}.npy', mmap_mode='r')
        allv, s, n_ok, n_all = [], [], 0, 0
        for i in idx:
            r, x = np.asarray(ref[i]), np.asarray(dep[i])
            m = (r > 0.3) & (r < 6) & (x > 0.3) & (x < 6)
            n_ok, n_all = n_ok + m.sum(), n_all + m.size
            if m.sum() < 1000:
                s.append(np.nan)
                continue
            e = np.log(x[m] / r[m])
            allv.append(e[::20])
            s.append(np.median(e))
        e = np.concatenate(allv)
        s = np.array(s)
        b = np.median(e)
        ds = np.diff(s)
        print(f'{d!s:<34} {100 * b:7.2f} {100 * np.median(np.abs(e - b)):8.2f} {100 * np.nanstd(s):10.2f} '
              f'{100 * np.nanstd(ds):11.2f} {100 * n_ok / n_all:8.1f}')


if __name__ == '__main__':
    main()
