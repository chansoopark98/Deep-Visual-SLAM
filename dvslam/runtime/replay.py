"""30 Hz live replay with a 1-slot drop-oldest mailbox, and the drop scoring rule (plan v3 §4.2 F9, item 4.1).

Offline limitation: when the tracker skips frames, the real tracker would measure t-1 -> t+k directly. Here that
increment is approximated by composing the offline per-frame increments, so tracking accuracy on long gaps is
optimistic; the drop *scoring* rule is exact.
"""
import numpy as np

from dvslam.geometry.se3 import se3_exp, se3_log


def simulate_mailbox(proc_ms: np.ndarray, rate_hz: float = 30.0) -> np.ndarray:
    """Which frames a single tracker thread processes when frames arrive every 1/rate s into a 1-slot mailbox
    that keeps only the newest frame. proc_ms[i] = processing time of frame i. Frame 0 is always processed.
    Returns a bool mask (True = processed)."""
    n = len(proc_ms)
    dt = 1000.0 / rate_hz
    done = np.zeros(n, bool)
    free_at, i = 0.0, 0
    while i < n:
        start = max(free_at, i * dt)
        # newest frame that has arrived by the time the tracker is free
        j = max(i, min(n - 1, int(np.floor(start / dt + 1e-9))))
        done[j] = True
        free_at = start + proc_ms[j]
        i = j + 1
    return done


def apply_drop_policy(increments: np.ndarray, processed: np.ndarray) -> dict:
    """increments: (N, 4, 4) offline ΔT for frames 1..N (index k <-> frame k+1); processed: (N+1,) mask over frames.

    Dropped frame t: emit the previous scored increment (constant velocity), source = motion_model.
    Next tracked frame after a gap of g dropped frames: the measured increment D spans g+1 frames; its scored
    increment is exp(log(D) / (g+1)), so one drop does not dump its error onto a single pair. The chain correction
    (D versus the emitted motion-model steps) goes only into the unscored T_wc output.
    """
    n = len(increments)
    scored = np.empty_like(increments)
    source = np.empty(n, dtype='<U12')
    T_wc = [np.eye(4)]
    last_tracked_T = np.eye(4)
    prev = np.eye(4)
    gap = 0
    acc = np.eye(4)                      # composed measured motion since the last tracked frame
    for k in range(n):
        acc = acc @ increments[k]
        if processed[k + 1]:
            dT = se3_exp(se3_log(acc) / (gap + 1)) if gap else acc
            scored[k], source[k] = dT, 'tracked'
            last_tracked_T = last_tracked_T @ acc
            T_wc.append(last_tracked_T)
            acc, gap = np.eye(4), 0
        else:
            dT = prev
            scored[k], source[k] = dT, 'motion_model'
            T_wc.append(T_wc[-1] @ dT)
            gap += 1
        prev = dT
    return {'rel_increment': scored, 'source': source, 'T_wc': np.stack(T_wc),
            'drop_rate': float(1 - processed[1:].mean())}
