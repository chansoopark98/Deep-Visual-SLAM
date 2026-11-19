"""EXP-003: how much of item 03 is the ground truth itself? (1) white mocap position noise, (2) GT time offset.

  python scripts/exp003_gt_floor.py noise                       # σ of the 120 Hz mocap camera-centre noise, floor
  python scripts/exp003_gt_floor.py offset runs/eval/<run> ...  # item 03 of runs re-scored with GT offsets

(1) Camera centre c = p_imu + R_imu · t_imu_cam at every 120 Hz mocap sample. For runs of evenly spaced samples
    (dt 7-10 ms) the k-th difference of white noise of σ has variance C(2k, k) σ², while real motion contributes
    about |d^k c / dt^k| dt^k (0.07 mm for 2nd differences at 2 m/s²). σ is estimated per axis with the MAD of the
    2nd and of the 3rd differences; agreement of the two says the estimate is noise, not motion. The item-03 floor
    is then simulated: GT camera steps of the scored pairs + noise of the linearly interpolated GT at both ends
    (variance (w² + (1−w)²) σ² per end, w = the interpolation weight), mean over 20 draws. This is the item-03 value
    a perfect estimator would get from white mocap noise alone; slow (non-white) mocap errors are not included.
(2) Item 03 of a run re-scored with GT looked up at camera time + τ. Diagnostic only: τ is not chosen on GT here.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry, tumvi
from scripts.exp003_score import OFFICIAL, pair_table, summarize

ROOMS = ('room1', 'room2', 'room3', 'room4', 'room5', 'room6')


def mocap_centre(scene: str) -> tuple[np.ndarray, np.ndarray]:
    d = tumvi.sequence_dir(scene, registry.default_root('tumvi'))
    m = np.loadtxt(d / 'mav0/mocap0/data.csv', delimiter=',', comments='#')
    T_cam_imu = tumvi.load_cam0_calib(d)[0]
    t_imu_cam = np.linalg.inv(T_cam_imu)[:3, 3]
    R = Rotation.from_quat(m[:, [5, 6, 7, 4]]).as_matrix()
    return m[:, 0].astype(np.int64), m[:, 1:4] + R @ t_imu_cam


def noise_sigma(t_ns: np.ndarray, c: np.ndarray) -> dict:
    dt = np.diff(t_ns) * 1e-9
    good = (dt > 0.007) & (dt < 0.010)
    out = {}
    for k, var_k in ((2, 6.0), (3, 20.0)):
        ok = np.ones(len(c) - k, bool)
        for j in range(k):
            ok &= good[j:len(good) - k + 1 + j]
        dk = np.diff(c, n=k, axis=0)[ok]
        mad = 1.4826 * np.median(np.abs(dk - np.median(dk, 0)), 0)
        out[f'sigma_d{k}_mm'] = (mad / np.sqrt(var_k) * 1e3).tolist()
        out[f'n_d{k}'] = int(ok.sum())
    return out


def floor(scene: str, sigma_m: float, rng: np.random.Generator, draws: int = 20) -> np.ndarray:
    """Per scored pair: item 03 of perfect steps against GT with white noise σ (mean over draws)."""
    seq = registry.load_sequence('tumvi', scene, None, registry.default_root('tumvi'), 0.0, load_images=False)
    t_gt = np.loadtxt(tumvi.sequence_dir(scene, registry.default_root('tumvi')) / 'mav0/mocap0/data.csv',
                      delimiter=',', comments='#', usecols=0, dtype=np.int64)
    j = np.clip(np.searchsorted(t_gt, seq.timestamps_ns), 1, len(t_gt) - 1)
    w = (seq.timestamps_ns - t_gt[j - 1]) / np.maximum(t_gt[j] - t_gt[j - 1], 1)
    s_end = sigma_m * np.sqrt(w ** 2 + (1 - w) ** 2)
    step = np.diff(seq.T_wc[:, :3, 3], axis=0)[seq.pair_valid]
    s_pair = np.sqrt(s_end[:-1] ** 2 + s_end[1:] ** 2)[seq.pair_valid]
    g = np.linalg.norm(step, axis=1)
    pct = np.zeros(len(g))
    for _ in range(draws):
        noisy = step + rng.normal(size=step.shape) * s_pair[:, None]
        pct += np.abs(np.linalg.norm(noisy, axis=1) - g) / np.maximum(g, 1e-9) * 100
    return pct / draws, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['noise', 'offset'])
    ap.add_argument('runs', nargs='*', type=Path)
    ap.add_argument('--offsets-ms', type=float, nargs='+', default=[-20, -10, -5, -2, 0, 2, 5, 10, 20])
    args = ap.parse_args()
    if args.what == 'noise':
        rng = np.random.default_rng(0)
        sig, pct, gts = {}, {}, {}
        for r in ROOMS:
            s = noise_sigma(*mocap_centre(r))
            sig[r] = float(np.mean(s['sigma_d3_mm'])) * 1e-3
            print(r, {k: np.round(v, 4).tolist() if isinstance(v, list) else v for k, v in s.items()})
        for r in ROOMS:
            pct[r], gts[r] = floor(r, sig[r], rng)
        for name, rooms in (('official', OFFICIAL), ('room5', ('room5',)), ('room6', ('room6',))):
            p, g = np.concatenate([pct[r] for r in rooms]), np.concatenate([gts[r] for r in rooms]) * 1e3
            edges = ((0, 5), (5, 20), (20, 40), (40, 1e9))
            bins = ' '.join(f'{p[(g >= lo) & (g < hi)].mean():.2f}' for lo, hi in edges)
            print(f'{name}: white-noise GT floor item 03 = {p.mean():.3f} %  (bins <5/5-20/20-40/>40 mm: {bins})')
        return
    for run in args.runs:
        files = {f.name.split('.')[0]: f for f in sorted(run.rglob('room*.inc.npz'))}
        row = []
        for tau in args.offsets_ms:
            tabs = {}
            for s, f in files.items():
                try:
                    tabs[s] = pair_table(f, tau * 1e-3)
                except ValueError:
                    tabs[s] = None
            off = [tabs[s] for s in OFFICIAL if tabs.get(s) is not None]
            o = summarize(off)['item03'] if len(off) == len(OFFICIAL) else float('nan')
            r5 = summarize([tabs['room5']])['item03'] if tabs.get('room5') is not None else float('nan')
            row.append(f'τ {tau:+5.0f} ms: official {o:6.3f} room5 {r5:6.3f}')
        print(run, '\n  ' + '\n  '.join(row))


if __name__ == '__main__':
    main()
