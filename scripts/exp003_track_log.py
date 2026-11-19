"""EXP-003: run tracker C (KLTWindowOdometry) on one TUM-VI room and log its KLT tracks for offline BA.

  python scripts/exp003_track_log.py --scene room5 --out runs/eval/exp003/causal [--opt k=v ...] [--cv-threads 4]
-> <out>/<scene>.inc.npz (same format as apps/run_sequence.py, scorable by eval_pairs.py / exp003_score.py),
   <out>/<scene>.extra.npz (tracker diagnostics), <out>/<scene>.tracks.npz (obs_frame, obs_gid, obs_px: every KLT
   observation the tracker held at the end of each step, plus the tracks that survived the pair solve of a re-init
   frame, so the offline graph is connected across re-inits), <out>/<scene>.cpu.json (wall, CPU seconds, cores used).
Track ids restart at every tracker re-init; gid = uid + offset keeps them unique per sequence.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from apps.run_sequence import parse_opts
from dvslam.data import registry
from dvslam.odometry.base import Frame, PinholeCamera, save_increments
from dvslam.odometry.klt_window import KLTWindowOdometry


class LoggedKLTWindow(KLTWindowOdometry):
    def _clear_window(self) -> None:
        self._gid_base = getattr(self, '_gid_base', 0) + getattr(self, 'next_uid', 0)
        super()._clear_window()

    def _start_window(self, gray, frame, T_wc) -> None:
        if len(getattr(self, 'uid', ())):            # survivors of the pair solve at a re-init frame
            self._snap = (self.uid + self._gid_base, self.px.astype(np.float64).copy())
        super()._start_window(gray, frame, T_wc)

    def step(self, frame):
        self._snap = None
        est = super().step(frame)
        rec = [(self.uid + self._gid_base, self.px.astype(np.float64))]
        if self._snap is not None and frame.index > 0:
            rec.append(self._snap)
        for g, p in rec:
            self.log.append((np.full(len(g), frame.index, np.int32), g.astype(np.int64), p))
        return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--opt', action='append', default=[])
    ap.add_argument('--start', type=int, default=0,
                    help='first frame (EXP-007 run-to-run perturbation); increments then cover start+1..n-1')
    ap.add_argument('--cv-threads', type=int, default=None,
                    help='cv2.setNumThreads before the run (default: untouched)')
    args = ap.parse_args()
    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    args.out.mkdir(parents=True, exist_ok=True)
    opts = parse_opts(args.opt)
    seq = registry.load_sequence('tumvi', args.scene, None, registry.default_root('tumvi'), 0.0)
    odo = LoggedKLTWindow(**opts)
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    odo.log = []
    est = []
    r0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    for i in range(args.start, len(seq)):
        e = odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i]))
        if e is not None:
            est.append(e)
    if hasattr(odo, 'flush'):                          # EXP-013 out_lag: the pairs still held back at the end
        est += odo.flush()
    r1, wall = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter() - w0
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    meta = {'dataset': 'tumvi', 'scene': args.scene, 'split': seq.split, 'odometry': 'klt_window', 'opts': opts,
            'data_root': str(registry.default_root('tumvi')), 'n_frames': len(seq), 'time_offset_s': 0.0,
            'frame_rate_hz': 20.0}
    save_increments(args.out / f'{args.scene}.inc.npz', est, meta)
    np.savez(args.out / f'{args.scene}.extra.npz', **odo.finalize())
    f, g, p = (np.concatenate([x[k] for x in odo.log]) for k in range(3))
    np.savez(args.out / f'{args.scene}.tracks.npz', obs_frame=f, obs_gid=g, obs_px=p, K=seq.K)
    info = {'wall_s': wall, 'cpu_s': cpu, 'cores': cpu / wall, 'ms_per_frame': 1e3 * wall / len(seq),
            'cv_threads': cv2.getNumThreads(), 'omp': os.environ.get('OMP_NUM_THREADS'), 'n_frames': len(seq)}
    fl = getattr(odo, 'flow', None)
    if hasattr(fl, 'ms'):                              # EXP-008 online RAFT: GPU calls and their time
        info.update(flow_calls=len(fl.ms), flow_ms_mean=float(np.mean(fl.ms)) if fl.ms else 0.0,
                    flow_ms_per_frame=float(np.sum(fl.ms)) / len(seq))
    (args.out / f'{args.scene}.cpu.json').write_text(json.dumps(info))
    print(args.scene, json.dumps(info))


if __name__ == '__main__':
    main()
