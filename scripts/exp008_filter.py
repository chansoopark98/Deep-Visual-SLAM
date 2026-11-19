"""EXP-008: causal filtering of a run's increments (post-processing; no image data).

  python scripts/exp008_filter.py runs/eval/<run> --scene room5 --out runs/eval/<out> --mode cv --q 1e-3 --r 1e-3

The step of pair t is put in a world frame with the run's own causal orientation, v_t = R_wc(t−1) · t_inc,t (so it is
causal), filtered, and rotated back; rotations are untouched.
  cv  : Kalman, state v (3), v_t = v_{t−1} + w, w ~ N(0, q²) per axis (constant velocity of the camera centre)
  ca  : Kalman, state (v, a), v_t = v_{t−1} + a_{t−1}, a_t = a_{t−1} + w, w ~ N(0, q²)
  len : the cv filter on the step length only (scalar); the direction of the measurement is kept
Measurement v_t with noise r² per axis on tracked pairs; motion_model pairs are not measurements (prediction only).
--lag L (diagnostic, NOT causal): output of pair t uses measurements up to t+L (fixed-lag RTS smoother).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.odometry.base import (
    PoseEstimate,
    causal_trajectory,
    load_increments,
    save_increments,
)


def kalman(z: np.ndarray, meas: np.ndarray, mode: str, q: float, r: float, lag: int) -> np.ndarray:
    """z (n, d) measurements (d = 3 or 1), meas (n,) bool. Returns the filtered (lag 0) or smoothed velocity (n, d)."""
    n, d = z.shape
    if mode == 'ca':
        F = np.kron(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(d))
        Q = np.kron(np.diag([0.0, q ** 2]), np.eye(d))
        H = np.c_[np.eye(d), np.zeros((d, d))]
    else:
        F, Q, H = np.eye(d), q ** 2 * np.eye(d), np.eye(d)
    m = F.shape[0]
    x, P = np.zeros(m), np.eye(m) * 1.0
    xp, Pp, xf, Pf = (np.zeros((n, m)), np.zeros((n, m, m)), np.zeros((n, m)), np.zeros((n, m, m)))
    first = True
    for t in range(n):
        if first and meas[t]:                           # initialise on the first measurement
            x = np.zeros(m)
            x[:d] = z[t]
            P = np.eye(m) * r ** 2
            first = False
            xp[t], Pp[t] = x, P
            xf[t], Pf[t] = x, P
            continue
        x, P = F @ x, F @ P @ F.T + Q
        xp[t], Pp[t] = x, P
        if meas[t]:
            S = H @ P @ H.T + r ** 2 * np.eye(d)
            Kg = P @ H.T @ np.linalg.inv(S)
            x = x + Kg @ (z[t] - H @ x)
            P = (np.eye(m) - Kg @ H) @ P
        xf[t], Pf[t] = x, P
    if lag == 0:
        return xf[:, :d]
    out = np.zeros((n, d))
    for t in range(n):                                  # fixed-lag: RTS from min(t+lag, n-1) back to t
        e = min(t + lag, n - 1)
        xs = xf[e]
        Ps = Pf[e]
        for k in range(e - 1, t - 1, -1):
            C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
            xs = xf[k] + C @ (xs - xp[k + 1])
            Ps = Pf[k] + C @ (Ps - Pp[k + 1]) @ C.T
        out[t] = xs[:d]
    return out


def filter_run(inc: dict, mode: str, q: float, r: float, lag: int) -> np.ndarray:
    T = inc['rel_increment'].copy()
    Rw = causal_trajectory(T)[:-1, :3, :3]              # R_wc of frame t−1 for pair t (causal)
    v = np.einsum('nij,nj->ni', Rw, T[:, :3, 3])
    meas = inc['source'] == 'tracked'
    if mode == 'len':
        L = np.linalg.norm(v, axis=1)
        Lf = kalman(L[:, None], meas, 'cv', q, r, lag)[:, 0]
        T[:, :3, 3] *= (np.maximum(Lf, 0) / np.maximum(L, 1e-12))[:, None]
        return T
    vf = kalman(v, meas, mode, q, r, lag)
    T[:, :3, 3] = np.einsum('nji,nj->ni', Rw, vf)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--scene', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['cv', 'ca', 'len'], default='cv')
    ap.add_argument('--q', type=float, required=True, help='process noise per frame (m, or m/frame for ca)')
    ap.add_argument('--r', type=float, required=True, help='measurement noise per axis (m)')
    ap.add_argument('--lag', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for scene in args.scene:
        inc = load_increments(args.run / f'{scene}.inc.npz')
        T = filter_run(inc, args.mode, args.q, args.r, args.lag)
        est = [PoseEstimate(int(i), float(ts), T[k], 'OK', str(s)) for k, (i, ts, s) in
               enumerate(zip(inc['index'], inc['timestamp'], inc['source']))]
        meta = dict(inc['meta'])
        meta['filter'] = {'mode': args.mode, 'q': args.q, 'r': args.r, 'lag': args.lag, 'run': str(args.run)}
        save_increments(args.out / f'{scene}.inc.npz', est, meta)


if __name__ == '__main__':
    main()
