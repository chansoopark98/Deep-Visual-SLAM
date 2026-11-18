"""Phase 2 runner for tracker C (`klt_window ba=false kf_align=false`, G1' selection) on one Redwood scene.

  python scripts/phase2_run.py --split validation --scene lobby --out runs/p2-x --prior perturbed --bias 0.02 \
      --jitter 0.03 --seed 0 [--latency 5 --warp] [--ckpt runs/phase3/single/model.pth]

Same tracker and options as `apps/run_sequence.py --odometry klt_window --opt ba=false --opt kf_align=false`
(prior_seed/bias/jitter as given); with latency 0 the increments are identical. Additions (none of them changes the
emitted increments at latency 0):
  * prior   : perturbed | oracle | da3 | dav2_ft (with --ckpt: a Phase-3 checkpoint; the tracker's own make_prior has
              no ckpt option, so the prior object is injected here).
  * latency : item 2.2. The prior of frame j arrives at j + L ("pipelined": one depth result per frame, age L) or,
              with --schedule latest, a latest-wins single depth thread (results for j = 0, L, 2L, ...; age L..2L−1).
              C only reads the prior at keyframes. Before any result has arrived (k < L) the prior of k is used
              synchronously (bootstrap, §4.3 item 1). With --warp the newest depth map is forward-warped into the
              keyframe with the tracker's own relative pose (z-buffer, 2x2 splat; the tracker's 3x3 edge check then
              rejects inconsistent pixels, §4.3); without it the stale map is used as is. With --delayed (pipelined
              only) a keyframe's new corners stay unmapped until the prior of that keyframe arrives at k + L; it is
              then attached exactly at their host pixels, before the PnP of frame k + L (the §4.3 "k still in
              the window" branch; nothing is warped).
  * m log   : items 2.3/2.4. After each step (latency 0 only), m_prior = Huber location of log z_map − log d_prior
              and m_sensor = the same against the sensor depth (diagnostic anchor) over mapped tracked points not
              created in this frame. Stored in <scene>.extra.npz; the g filter is applied afterwards
              (`dvslam/scale/g_filter.py`), which is exactly causal because g does not feed back into C.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.data.redwood import load_depth  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera, causal_trajectory, save_increments, write_tum  # noqa: E402
from dvslam.odometry.klt_window import KLTWindowOdometry, huber_location, sample_depth  # noqa: E402


def warp_depth(depth: np.ndarray, K: np.ndarray, T_src: np.ndarray, T_dst: np.ndarray) -> np.ndarray:
    """Forward-warp a depth map from camera src into camera dst (T_* = T_wc). 2x2 splat, nearest depth wins; 0 = hole."""
    h, w = depth.shape
    v, u = np.nonzero(depth > 0)
    d = depth[v, u].astype(np.float64)
    P = (np.linalg.inv(K) @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)) * d
    T = np.linalg.inv(T_dst) @ T_src
    Q = T[:3, :3] @ P + T[:3, 3:4]
    z = Q[2]
    ok = z > 1e-3
    uv = (K @ Q[:, ok])[:2] / z[ok]
    z = z[ok]
    out = np.full(h * w, np.inf)
    for du in (0, 1):
        for dv in (0, 1):
            x, y = np.floor(uv[0]).astype(int) + du, np.floor(uv[1]).astype(int) + dv
            m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            np.minimum.at(out, y[m] * w + x[m], z[m])
    out[~np.isfinite(out)] = 0.0
    return out.reshape(h, w).astype(np.float32)


class Phase2C(KLTWindowOdometry):
    """C with an injected prior, simulated prior latency and per-frame log-scale measurements."""

    def __init__(self, prior_obj=None, latency: int = 0, warp: bool = True, schedule: str = 'pipelined',
                 log_m: bool = True, delayed: bool = False, **kw):
        super().__init__(ba=False, kf_align=False, **kw)
        self.prior = prior_obj                       # reset() keeps a non-None prior
        self.latency, self.warp, self.schedule, self.log_m = int(latency), warp, schedule, log_m
        self.delayed = delayed

    def reset(self, camera: PinholeCamera) -> None:
        super().reset(camera)
        self.frames, self.poses, self.dcache = {}, {0: np.eye(4)}, {}
        self.m_log = {'m_prior': [], 'm_sensor': [], 'm_n': [], 'prior_age': []}
        self._T_kf, self._age, self.pending = None, [], []

    def _raw_prior(self, j: int) -> np.ndarray:
        if j not in self.dcache:
            if len(self.dcache) > 8:
                self.dcache.pop(min(self.dcache))
            self.dcache[j] = np.asarray(self.prior.predict(self.frames[j])[0], np.float32)
        return self.dcache[j]

    def _source_index(self, k: int) -> int:
        L = self.latency
        if L == 0 or k < L:
            return k
        if self.schedule == 'latest':
            return ((k - L) // L) * L
        return k - L

    def _add_keyframe(self, gray, frame: Frame, T_wc: np.ndarray) -> None:
        self._T_kf = T_wc
        super()._add_keyframe(gray, frame, T_wc)

    def _predict_prior(self, frame: Frame) -> np.ndarray:
        if self.latency == 0:
            return super()._predict_prior(frame)
        k = frame.index
        if self.last_prior is not None and self.last_prior[0] == k:
            return self.last_prior[1]
        j = self._source_index(k)
        if self.delayed and j != k and self.kfs:     # a window (re)start reads its prior synchronously (§4.3)
            self.pending.append(k)                   # corners stay unmapped until the prior of k arrives
            self.last_prior = (k, np.zeros(self.size[::-1], np.float32))
            return self.last_prior[1]
        d = self._raw_prior(j)
        if j != k and self.warp:
            d = warp_depth(d, self.K, self.poses[j], self._T_kf)
        self._age.append(k - j)
        self.last_prior = (k, d)
        return d

    def _attach(self, k: int) -> None:
        if k not in self.kf_prior:                   # keyframe already dropped (or window restarted)
            return
        depth = self._raw_prior(k)
        self.kf_prior[k] = depth
        u = np.flatnonzero(self.lm_alive[:self.next_uid] & (self.lm_host[:self.next_uid] == k))
        u = u[np.isnan(self.lm_x[u])]
        lp = np.log(sample_depth(depth, self.lm_hpx[u], self.depth_min, self.depth_max, self.depth_edge))
        self.lm_lp[u], self.lm_x[u] = lp, -lp
        self._age.append(self.latency)
        self._refresh_map()

    def step(self, frame: Frame):
        self.frames[frame.index] = frame
        while self.pending and self.pending[0] + self.latency <= frame.index:
            self._attach(self.pending.pop(0))
        est = super().step(frame)
        self.poses[frame.index] = self.T_prev.copy()
        if est is not None and self.log_m and self.latency == 0:
            self._log_m(frame)
        return est

    def _log_m(self, frame: Frame) -> None:
        lg = self.m_log
        ok = np.isfinite(self.X[:, 0]) & (self.lm_host[self.uid] != frame.index)
        mp = ms = np.nan
        if ok.sum() >= self.min_inliers:
            z = self._project(self.T_prev, self.X[ok])[1]
            px = self.px[ok].astype(np.float64)
            for name, dmap in (('p', self._predict_prior(frame)), ('s', load_depth(frame.depth_path))):
                d = sample_depth(dmap, px, self.depth_min, self.depth_max, self.depth_edge)
                v = np.isfinite(d) & (z > 1e-3)
                if v.sum() >= self.min_inliers:
                    val = huber_location(np.log(z[v]) - np.log(d[v]))
                    mp, ms = (val, ms) if name == 'p' else (mp, val)
        lg['m_prior'].append(mp), lg['m_sensor'].append(ms), lg['m_n'].append(int(ok.sum()))

    def finalize(self) -> dict[str, np.ndarray]:
        out = super().finalize()
        out.update({k: np.asarray(v) for k, v in self.m_log.items() if k != 'prior_age'})
        out['prior_age'] = np.asarray(self._age)
        return out


def make_prior_obj(args, K):
    if args.prior in ('perturbed', 'oracle'):
        from dvslam.depth.perturbed import PerturbedGTDepth
        b, j = (args.bias, args.jitter) if args.prior == 'perturbed' else (0.0, 0.0)
        return PerturbedGTDepth(bias=b, jitter_logsd=j, seed=args.seed)
    if args.prior == 'da3':
        from dvslam.depth import build
        return build('da3m', K)
    if args.prior == 'dav2_ft':
        from dvslam.depth.dav2_ft import DAv2FT
        return DAv2FT(ckpt=args.ckpt, K=K)
    raise ValueError(args.prior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['validation', 'test'], required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prior', choices=['perturbed', 'oracle', 'da3', 'dav2_ft'], default='perturbed')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--bias', type=float, default=0.02)
    ap.add_argument('--jitter', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--latency', type=int, default=0)
    ap.add_argument('--warp', action='store_true')
    ap.add_argument('--delayed', action='store_true', help='delayed exact attach (pipelined schedule only)')
    ap.add_argument('--schedule', choices=['pipelined', 'latest'], default='pipelined')
    ap.add_argument('--cv-threads', type=int, default=2)
    ap.add_argument('--no-m', action='store_true', help='skip the per-frame m log (saves prior calls)')
    args = ap.parse_args()

    cv2.setNumThreads(args.cv_threads)          # host is shared; results do not depend on the thread count
    np.random.seed(0)
    seq = redwood.load_sequence(args.scene, args.split, redwood.DEFAULT_ROOT)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / f'{args.scene}.inc.npz').exists():
        sys.exit(f'{args.out}/{args.scene}.inc.npz exists')
    odo = Phase2C(prior_obj=make_prior_obj(args, seq.K), latency=args.latency, warp=args.warp,
                  schedule=args.schedule, log_m=not args.no_m, prior_seed=args.seed,
                  delayed=args.delayed)
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    est, t0 = [], time.time()
    for i in range(len(seq)):
        e = odo.step(Frame(i, float(seq.timestamps[i]), seq.image_paths[i], seq.depth_paths[i]))
        if e is not None:
            est.append(e)
    commit = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    opts = {'ba': False, 'kf_align': False, 'prior': args.prior, 'prior_seed': args.seed}
    if args.prior == 'perturbed':
        opts.update(prior_bias=args.bias, prior_jitter=args.jitter)
    if args.ckpt:
        opts['ckpt'] = args.ckpt
    meta = {'scene': args.scene, 'split': args.split, 'odometry': 'klt_window', 'opts': opts, 'seed': 0,
            'data_root': str(redwood.DEFAULT_ROOT), 'n_frames': len(seq), 'git_commit': commit,
            'frame_rate_hz': redwood.FRAME_RATE_HZ, 'frame_rate_verified': redwood.FRAME_RATE_VERIFIED,
            'phase2': {'latency': args.latency, 'warp': args.warp, 'delayed': args.delayed, 'schedule': args.schedule,
                       'runner': 'scripts/phase2_run.py', 'argv': sys.argv[1:]}}
    save_increments(args.out / f'{args.scene}.inc.npz', est, meta)
    write_tum(args.out / f'{args.scene}.causal.tum', seq.timestamps, causal_trajectory(np.stack([e.rel_increment for e in est])))
    np.savez(args.out / f'{args.scene}.extra.npz', **odo.finalize())
    print(f'{args.scene}: {len(est)} increments, {time.time() - t0:.0f} s -> {args.out}')


if __name__ == '__main__':
    main()
