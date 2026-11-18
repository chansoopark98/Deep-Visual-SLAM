"""Robust 1-D filter for the global log-scale g (plan v3 §4.5, item 2.3), applied as a causal output transform.

Convention (as in `klt_window`): map units = metric · e^g, and each measurement is
    m_t = robust location of (log z_map − log d_ref) over the mapped points seen in frame t,
where z_map is the point depth in camera t from the tracker's map and d_ref the reference depth at the tracked pixel:
the depth prior of frame t (the normal g) or the sensor depth (the diagnostic anchor, item 2.4; never submitted).
The filter is a clipped exponential smoother:
    g ← g + clip(m − g, ±c·s) / tau,
with s a running robust spread of the innovation. Before `min_m` valid measurements g stays 0 (BOOTSTRAP, §4.3 item 4);
at the `min_m`-th it is initialised to their median. The increment emitted at t uses only measurements up to t − 1
(g is updated after ΔT_t is emitted, §4.2): ΔT_t.t ← ΔT_t.t · e^{−g_{t−1}}.
"""
import numpy as np


class RobustGFilter:
    def __init__(self, tau: float = 60.0, c: float = 2.0, min_m: int = 5, s0: float = 0.05, s_tau: float = 30.0):
        self.tau, self.c, self.min_m, self.s, self.s_tau = float(tau), float(c), int(min_m), float(s0), float(s_tau)
        self.g, self.n, self._first = 0.0, 0, []

    @property
    def valid(self) -> bool:
        return self.n >= self.min_m

    def update(self, m: float) -> float:
        if not np.isfinite(m):
            return self.g
        self.n += 1
        if self.n < self.min_m:
            self._first.append(m)
            return self.g
        if self.n == self.min_m:
            self._first.append(m)
            self.g = float(np.median(self._first))
            return self.g
        r = m - self.g
        self.g += float(np.clip(r, -self.c * self.s, self.c * self.s)) / self.tau
        self.s += (1.4826 * min(abs(r), 3 * self.s) - self.s) / self.s_tau   # bounded update: robust spread
        return self.g


def g_series(m: np.ndarray, **kw) -> np.ndarray:
    """g after each measurement (same length as m; NaN measurements leave g unchanged)."""
    f = RobustGFilter(**kw)
    return np.array([f.update(float(v)) for v in m])


def apply_g(inc: np.ndarray, g_after: np.ndarray) -> np.ndarray:
    """Causal output transform. inc[k] is ΔT of frame k+1; g_after[i] is g after the measurement of frame i
    (i = 0..n). Frame t = k+1 uses g_after[t − 1] = g_after[k]."""
    out = inc.copy()
    out[:, :3, 3] *= np.exp(-np.asarray(g_after[:len(inc)]))[:, None]
    return out
