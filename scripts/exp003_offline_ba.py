"""EXP-003 ceiling: non-causal / fixed-lag bundle adjustment over ALL frames, on tracker C's own KLT tracks and the
same cached depth prior, increments read out after optimisation and scored per pair like the causal tracker.

  python scripts/exp003_offline_ba.py --scene room5 --tracks runs/eval/exp003/causal --depth runs/eval/exp003/depth \
      --mode smooth --win 120 --out runs/eval/exp003/smooth_w120
  python scripts/exp003_offline_ba.py ... --mode lag --lag 5 --win 40 --out runs/eval/exp003/lag5

Model (per window of consecutive frames a..b, every frame a pose, first pose fixed = gauge):
  landmark j  = one KLT track (gid) observed >= 2 times in the window, anchored at its first frame h in the window
                as log inverse depth x_j on the observed ray (host observation exact).
  reprojection: every non-host observation, (uv − obs) / px_sigma, Huber (f_scale = huber) like the tracker.
  prior       : r = (log d_p(i, px) − log z_ij) / σ_i at every observation with a valid prior sample (--prior all) or
                at the host only (--prior host, as the tracker); d_p from the same cached maps and sample_depth()
                (edge check) as the tracker. σ_i = prior_sigma · sqrt(n_i): one frame's prior is one common-mode
                jitter, so frame i contributes a total weight of 1 / prior_sigma² (tracker's prior_per_kf rule).
  init        : poses from the causal increments (<tracks>/<scene>.inc.npz), x from the raw prior at the host.
Readout:
  smooth   : windows of `win` frames every win/2 frames; each pair is read from the window where it is most central
             (>= win/4 frames of context on both sides except at the sequence ends).
  lag k    : one solve per frame t over [t+k−win+1, t+k] (warm-started, oldest pose fixed at its current estimate),
             pair (t−1, t) read out right after that solve: a fixed-lag smoother with output lag k frames.
Optional constant-velocity prior (--cv-sigma, --cv-rot-deg): second differences of position and relative rotation,
a weak motion model for stretches with few tracks (measured: without it, windows with 20-40 tracks per frame diverge).
The Huber loss applies to all residuals, as in the tracker.
Pairs whose two frames share fewer than `min_shared` tracks cannot be solved from tracks: they keep the causal
tracker's value and are tagged motion_model (counted in the log). Rotation uses the right-perturbation small-angle
Jacobian, as in klt_window.py.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.odometry.base import (
    PoseEstimate,
    causal_trajectory,
    load_increments,
    save_increments,
)
from dvslam.odometry.klt_window import _skew, sample_depth


class Problem:
    def __init__(self, tracks: dict, depth: np.ndarray, K: np.ndarray, args):
        self.K, self.Kinv, self.depth, self.a = K, np.linalg.inv(K), depth, args
        o = np.lexsort((tracks['obs_gid'], tracks['obs_frame']))
        self.f, self.g, self.px = tracks['obs_frame'][o], tracks['obs_gid'][o], tracks['obs_px'][o]
        self.start = np.searchsorted(self.f, np.arange(depth.shape[0] + 1))   # obs of frame i: start[i]:start[i+1]
        self._lp = {}

    def obs(self, a: int, b: int):
        s, e = self.start[a], self.start[b + 1]
        return self.f[s:e], self.g[s:e], self.px[s:e]

    def log_prior(self, i: int, px: np.ndarray, check: bool = True) -> np.ndarray:
        if check:
            return np.log(sample_depth(self.depth[i], px))
        h, w = self.depth.shape[1:]
        x = np.clip(np.round(px).astype(int), 0, [w - 1, h - 1])
        return np.log(np.clip(self.depth[i][x[:, 1], x[:, 0]], 0.2, 6.0).astype(np.float64))

    def shared(self, t: int) -> int:
        return len(np.intersect1d(self.obs(t - 1, t - 1)[1], self.obs(t, t)[1], assume_unique=True))

    def solve(self, a: int, b: int, T: dict, Xw: dict, max_nfev: int):
        """BA over frames a..b. T: frame -> T_wc (initial, updated in place); Xw: gid -> world point (warm start)."""
        f, g, px = self.obs(a, b)
        _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 2
        f, g, px = f[keep], g[keep], px[keep]
        lms, o_lm = np.unique(g, return_inverse=True)
        n_lm, n_f = len(lms), b - a + 1
        if n_lm < 6:
            return None
        first = np.full(n_lm, np.iinfo(np.int64).max)
        np.minimum.at(first, o_lm, f)
        is_host = f == first[o_lm]
        hpx = np.empty((n_lm, 2))
        hpx[o_lm[is_host]] = px[is_host]
        host = first - a
        bear = np.c_[hpx, np.ones(n_lm)] @ self.Kinv.T
        R0 = np.stack([T[i][:3, :3] for i in range(a, b + 1)])
        t0 = np.stack([T[i][:3, 3] for i in range(a, b + 1)])
        # landmark init: warm-start world point if it projects in front of the host, else the raw prior at the host
        x0 = np.empty(n_lm)
        for hh in np.unique(host):
            m = np.flatnonzero(host == hh)
            x0[m] = -self.log_prior(a + hh, hpx[m], check=False)
            have = np.array([u in Xw for u in lms[m]])
            if have.any():
                Xm = np.stack([Xw[u] for u in lms[m][have]])
                z = (Xm - t0[hh]) @ R0[hh][:, 2]
                ok = z > 0.05
                x0[m[have][ok]] = -np.log(z[ok])
        # observations: reprojection (non-host), prior (all or host only)
        fi = f - a
        rp = ~is_host
        r_lm, r_f, r_px = o_lm[rp], fi[rp], px[rp]
        lp = np.full(len(f), np.nan)
        for i in np.unique(f):
            m = f == i
            lp[m] = self.log_prior(i, px[m])
        use_p = np.isfinite(lp) & (is_host if self.a.prior == 'host' else True)
        p_lm, p_f, p_lp, p_host = o_lm[use_p], fi[use_p], lp[use_p], is_host[use_p]
        n_i = np.bincount(p_f, minlength=n_f)
        p_sig = self.a.prior_sigma * np.sqrt(n_i[p_f])
        oh_r, oh_p = host[r_lm], host[p_lm]
        n_p = 6 * (n_f - 1)
        K = self.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pxs = self.a.px_sigma

        def unpack(p):
            om, v = np.zeros((n_f, 3)), np.zeros((n_f, 3))
            om[1:], v[1:] = p[:n_p].reshape(-1, 6)[:, :3], p[:n_p].reshape(-1, 6)[:, 3:]
            return R0 @ Rotation.from_rotvec(om).as_matrix(), t0 + v, p[n_p:]

        def geom(R, t, x, lm, fr, oh):
            ph = bear[lm] * np.exp(-np.clip(x[lm], -6, 6))[:, None]
            pw = np.einsum('nij,nj->ni', R[oh], ph) + t[oh]
            pk = np.einsum('nji,nj->ni', R[fr], pw - t[fr])
            return ph, pk

        cvt, cvr = self.a.cv_sigma, self.a.cv_rot_deg * np.pi / 180
        n_cv = (n_f - 2) if cvt > 0 and n_f >= 3 else 0

        def motion(R, t):
            """Constant-velocity prior: second differences of position (world) and of the relative rotation."""
            if not n_cv:
                return np.empty(0)
            rt = (t[2:] - 2 * t[1:-1] + t[:-2]) / cvt
            w = Rotation.from_matrix(np.transpose(R[:-1], (0, 2, 1)) @ R[1:]).as_rotvec()
            return np.r_[rt.ravel(), ((w[1:] - w[:-1]) / cvr).ravel()]

        def fun(p):
            R, t, x = unpack(p)
            _, pk = geom(R, t, x, r_lm, r_f, oh_r)
            z = np.maximum(pk[:, 2], 1e-3)
            r1 = np.c_[(fx * pk[:, 0] / z + cx - r_px[:, 0]), (fy * pk[:, 1] / z + cy - r_px[:, 1])].ravel() / pxs
            _, qk = geom(R, t, x, p_lm, p_f, oh_p)
            lz = np.where(p_host, -x[p_lm], np.log(np.maximum(qk[:, 2], 1e-3)))
            return np.r_[r1, (p_lp - lz) / p_sig, motion(R, t)]

        def blocks(R, x, ph, pk, lm, fr, oh):
            """d pk / d(x, ω_h, v_h, ω_k, v_k) per observation."""
            RkT = np.transpose(R[fr], (0, 2, 1))
            RkTRh = RkT @ R[oh]
            dx = np.einsum('nij,nj->ni', RkTRh, -ph)
            dh = np.concatenate([-RkTRh @ _skew(ph), RkT], axis=2)
            dk = np.concatenate([_skew(pk), -RkT], axis=2)
            return dx, dh, dk

        def jac(p):
            R, t, x = unpack(p)
            rows, cols, vals = [], [], []

            def put(r0, J, lm, fr, oh, dx, dh, dk, mask):
                # J (n, m, 3): d residual / d pk for m residual rows per observation
                n, m_ = J.shape[:2]
                rr = r0 + m_ * np.arange(n)
                for d, c, ok in ((np.einsum('nmj,nj->nm', J, dx)[:, :, None], n_p + lm[:, None], mask),
                                 (J @ dh, 6 * (oh - 1)[:, None] + np.arange(6), mask & (oh > 0)),
                                 (J @ dk, 6 * (fr - 1)[:, None] + np.arange(6), mask & (fr > 0))):
                    for a_ in range(m_):
                        rows.append(np.repeat(rr[ok] + a_, d.shape[2])), cols.append(c[ok].ravel())
                        vals.append(d[ok, a_].ravel())

            ph, pk = geom(R, t, x, r_lm, r_f, oh_r)
            z = np.maximum(pk[:, 2], 1e-3)
            J = np.zeros((len(r_lm), 2, 3))
            J[:, 0, 0], J[:, 0, 2] = fx / z, -fx * pk[:, 0] / z ** 2
            J[:, 1, 1], J[:, 1, 2] = fy / z, -fy * pk[:, 1] / z ** 2
            put(0, J / pxs, r_lm, r_f, oh_r, *blocks(R, x, ph, pk, r_lm, r_f, oh_r), np.ones(len(r_lm), bool))
            n1 = 2 * len(r_lm)
            qh, qk = geom(R, t, x, p_lm, p_f, oh_p)
            Jq = np.zeros((len(p_lm), 1, 3))
            Jq[:, 0, 2] = -1.0 / np.maximum(qk[:, 2], 1e-3) / p_sig
            put(n1, Jq, p_lm, p_f, oh_p, *blocks(R, x, qh, qk, p_lm, p_f, oh_p), ~p_host)
            hp = np.flatnonzero(p_host)                     # host prior: r = (lp + x) / σ
            rows.append(n1 + hp), cols.append(n_p + p_lm[hp]), vals.append(1.0 / p_sig[hp])
            n2 = n1 + len(p_lm)
            if n_cv:   # second differences; rotation part uses the small-angle Jacobian (d rotvec ≈ dω_{k+1} − dω_k)
                for k in range(n_cv):
                    for c_, frame in ((1.0, k), (-2.0, k + 1), (1.0, k + 2)):
                        if frame == 0:
                            continue
                        for ax in range(3):
                            rows += [np.array([n2 + 3 * k + ax]), np.array([n2 + 3 * (n_cv + k) + ax])]
                            cols += [np.array([6 * (frame - 1) + 3 + ax]), np.array([6 * (frame - 1) + ax])]
                            vals += [np.array([c_ / cvt]), np.array([c_ / cvr])]
            return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n2 + 6 * n_cv, n_p + n_lm))

        p0 = np.r_[np.zeros(n_p), x0]
        res = least_squares(fun, p0, jac=jac, method='trf', loss=self.a.loss, f_scale=self.a.huber,
                            max_nfev=max_nfev, x_scale='jac', tr_solver='lsmr')
        R, t, x = unpack(res.x)
        for k in range(n_f):
            Tk = np.eye(4)
            Tk[:3, :3], Tk[:3, 3] = R[k], t[k]
            T[a + k] = Tk
        Xp = bear * np.exp(-np.clip(x, -6, 6))[:, None]
        Xw.update(zip(lms.tolist(), np.einsum('nij,nj->ni', R[host], Xp) + t[host]))
        return {'nfev': res.nfev, 'cost0': float(0.5 * np.sum(fun(p0) ** 2)), 'cost': float(res.cost),
                'n_lm': n_lm, 'n_reproj': len(r_lm), 'n_prior': len(p_lm)}


def segments(prob: Problem, n: int, min_shared: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    cut = np.array([prob.shared(t) < min_shared for t in range(1, n)])     # cut[t-1]: pair (t-1, t) unsolvable
    segs, s = [], 0
    for t in range(1, n):
        if cut[t - 1]:
            segs.append((s, t - 1))
            s = t
    segs.append((s, n - 1))
    return segs, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--tracks', type=Path, required=True, help='dir with <scene>.tracks.npz and <scene>.inc.npz')
    ap.add_argument('--depth', type=Path, required=True, help='depth cache dir (<scene>.npy)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['smooth', 'lag'], default='smooth')
    ap.add_argument('--win', type=int, default=120)
    ap.add_argument('--lag', type=int, default=0)
    ap.add_argument('--prior', choices=['all', 'host'], default='all')
    ap.add_argument('--prior-sigma', dest='prior_sigma', type=float, default=0.05)
    ap.add_argument('--px-sigma', dest='px_sigma', type=float, default=0.5)
    ap.add_argument('--huber', type=float, default=2.0)
    ap.add_argument('--loss', default='huber')
    ap.add_argument('--cv-sigma', dest='cv_sigma', type=float, default=0.0,
                    help='constant-velocity prior: σ of the position second difference per frame (m); 0 = off')
    ap.add_argument('--cv-rot-deg', dest='cv_rot_deg', type=float, default=1.0,
                    help='σ of the relative-rotation change per frame (deg), used when --cv-sigma > 0')
    ap.add_argument('--nfev', type=int, default=50)
    ap.add_argument('--min-shared', dest='min_shared', type=int, default=8)
    ap.add_argument('--every', type=int, default=1, help='lag mode: read out only every n-th pair (others: causal)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tracks = dict(np.load(args.tracks / f'{args.scene}.tracks.npz'))
    causal = load_increments(args.tracks / f'{args.scene}.inc.npz')
    depth = np.load(args.depth / f'{args.scene}.npy', mmap_mode='r')
    n = len(causal['index']) + 1
    prob = Problem(tracks, depth, tracks['K'], args)
    T_c = causal_trajectory(causal['rel_increment'])
    segs, cut = segments(prob, n, args.min_shared)
    inc = causal['rel_increment'].copy()
    src = np.array(['motion_model'] * (n - 1), dtype=object)
    stats, r0, w0 = [], resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    for s, e in segs:
        if e - s < 1:
            continue
        T = {i: T_c[i].copy() for i in range(s, e + 1)}
        Xw = {}
        if args.mode == 'smooth':
            W = min(args.win, e - s + 1)
            starts = list(range(s, e - W + 2, max(1, W // 2)))
            if starts[-1] != e - W + 1:
                starts.append(e - W + 1)
            best = {}
            for a in starts:
                b = a + W - 1
                Tw = {i: T[i].copy() for i in range(a, b + 1)}
                st = prob.solve(a, b, Tw, Xw, args.nfev)
                if st is None:
                    continue
                stats.append(st)
                T.update(Tw)                            # next window starts from these poses, causal beyond b
                for i in range(b + 1, e + 1):
                    T[i] = T[i - 1] @ causal['rel_increment'][i - 1]
                for t in range(a + 1, b + 1):
                    c = min(t - 1 - a, b - t)
                    if t not in best or c > best[t][0]:
                        best[t] = (c, np.linalg.inv(Tw[t - 1]) @ Tw[t])
            for t, (_, d) in best.items():
                inc[t - 1], src[t - 1] = d, 'tracked'
        else:
            for t in range(s + 1, e + 1):
                b = min(t + args.lag, e)
                a = max(s, b - args.win + 1, 0)
                a = min(a, t - 1)
                for i in range(a, b + 1):               # frames entering the window: chain the causal increments
                    if i not in T:
                        T[i] = T[i - 1] @ inc[i - 1]
                if (t - s) % args.every:
                    continue
                st = prob.solve(a, b, T, Xw, args.nfev)
                if st is None:
                    continue
                stats.append(st)
                inc[t - 1], src[t - 1] = np.linalg.inv(T[t - 1]) @ T[t], 'tracked'
    r1, wall = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter() - w0
    est = [PoseEstimate(int(i), float(ts), inc[k], 'OK', str(src[k])) for k, (i, ts) in
           enumerate(zip(causal['index'], causal['timestamp']))]
    meta = dict(causal['meta'])
    meta.update({'odometry': f'exp003_offline_ba_{args.mode}', 'ba_args': vars(args) | {
        'tracks': str(args.tracks), 'depth': str(args.depth), 'out': str(args.out)}})
    save_increments(args.out / f'{args.scene}.inc.npz', est, meta)
    info = {'scene': args.scene, 'segments': len(segs), 'cut_pairs': int(cut.sum()), 'solves': len(stats),
            'readout_pairs': int((src == 'tracked').sum()), 'pairs': n - 1, 'wall_s': wall,
            'cpu_s': (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime),
            'nfev_median': float(np.median([s['nfev'] for s in stats])) if stats else None,
            'hit_nfev_share': float(np.mean([s['nfev'] >= args.nfev for s in stats])) if stats else None,
            'n_lm_median': float(np.median([s['n_lm'] for s in stats])) if stats else None}
    (args.out / f'{args.scene}.ba.json').write_text(json.dumps(info))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
