"""Fixed-lag output for KLTWindowOdometry (EXP-013, `out_lag=L`, `out_lag_mode=smooth|window`; default L = 0 = off).

CONDITIONAL: the increment of pair (k-1, k) is emitted only after frame k+L was processed (L frames = 50 L ms at 20 Hz).
Whether the evaluation allows that output delay is the user's decision; nothing here is a default.

smooth : fixed-lag RTS smoother of a constant-acceleration Kalman filter on the camera-centre step in the world
         frame, v_t = R_wc(t-1) t_inc,t (R_wc chained from the emitted rotations, so it uses no future data for the
         frame itself). State (v, a), v_t = v_{t-1} + a_{t-1}, a_t = a_{t-1} + w, w ~ N(0, q²) per axis; measurement v_t
         with noise r² on tracked pairs, none on motion-model pairs. The output of pair k is the RTS estimate using
         measurements up to k+L.
window : the causal increment is corrected by the change of the window's relative pose of (k-1, k) between the time
         pair k was emitted causally and the time frame k+L was processed (implemented in klt_window.py).
End of sequence: flush() emits the last L pairs with the data available (lag L-1, ..., 0), so every pair is emitted
once, in order.
"""
import numpy as np


class FixedLagCA:
    """Incremental constant-acceleration Kalman filter + fixed-lag RTS on 3-vectors."""

    def __init__(self, q: float, r: float, lag: int):
        self.q, self.r, self.lag = q, r, lag
        self.F = np.kron(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(3))
        self.Q = np.kron(np.diag([0.0, q ** 2]), np.eye(3))
        self.H = np.c_[np.eye(3), np.zeros((3, 3))]
        self.x, self.P, self.first = np.zeros(6), np.eye(6), True
        self.xp, self.Pp, self.xf, self.Pf = [], [], [], []
        self.n_out = 0

    def push(self, z: np.ndarray, measured: bool) -> list[np.ndarray]:
        """Add the step of the next pair; returns the smoothed steps that became final (0 or 1 of them)."""
        if self.first and measured:
            self.x = np.r_[z, np.zeros(3)]
            self.P = np.eye(6) * self.r ** 2
            self.first = False
            self.xp.append(self.x), self.Pp.append(self.P)
        else:
            self.x, self.P = self.F @ self.x, self.F @ self.P @ self.F.T + self.Q
            self.xp.append(self.x), self.Pp.append(self.P)
            if measured:
                S = self.H @ self.P @ self.H.T + self.r ** 2 * np.eye(3)
                Kg = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + Kg @ (z - self.H @ self.x)
                self.P = (np.eye(6) - Kg @ self.H) @ self.P
        self.xf.append(self.x), self.Pf.append(self.P)
        out = []
        while self.n_out + self.lag < len(self.xf):
            out.append(self._smooth(self.n_out, len(self.xf) - 1))
            self.n_out += 1
        return out

    def flush(self) -> list[np.ndarray]:
        out = []
        while self.n_out < len(self.xf):
            out.append(self._smooth(self.n_out, len(self.xf) - 1))
            self.n_out += 1
        return out

    def _smooth(self, t: int, last: int) -> np.ndarray:
        e = min(t + self.lag, last)
        xs = self.xf[e]
        for k in range(e - 1, t - 1, -1):
            C = self.Pf[k] @ self.F.T @ np.linalg.inv(self.Pp[k + 1])
            xs = self.xf[k] + C @ (xs - self.xp[k + 1])
        return xs[:3]
