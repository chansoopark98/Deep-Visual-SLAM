"""True re-tracking replay (plan v3 item 4.1): run the tracker only on the frames a 1-slot drop-oldest mailbox would
process, then score with the F9 drop rule.

`dvslam.runtime.replay.apply_drop_policy` composes the offline per-frame increments across a gap, which is optimistic:
a real tracker measures frame i -> j directly (LK over a longer baseline, fewer surviving tracks). `retrack` does that
measurement; `score_retracked` applies the same scoring rule to the measured increments.
"""
import numpy as np

from dvslam.geometry.se3 import se3_exp, se3_log
from dvslam.odometry.base import Frame, PinholeCamera


def retrack(odo, seq, processed: np.ndarray) -> dict:
    """Step `odo` only over frames with processed[f]. Frame 0 must be processed. Frame indices stay the true ones.

    Returns arrays over the processed frames f >= 1: index, D (increment from the previously processed frame, metric),
    tracked (bool), state (str), ms (the tracker's own step time).
    """
    if not processed[0]:
        raise ValueError('frame 0 must be processed')
    odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
    ts = seq.timestamps
    idx, D, tracked, state, ms = [], [], [], [], []
    for f in np.flatnonzero(processed):
        est = odo.step(Frame(int(f), float(ts[f]), seq.image_paths[f], seq.depth_paths[f]))
        if est is None:
            continue
        idx.append(int(f)), D.append(est.rel_increment), tracked.append(est.source == 'tracked')
        state.append(est.tracking_state), ms.append(est.timings_ms.get('total', np.nan))
    return {'index': np.array(idx, np.int64), 'D': np.stack(D) if D else np.empty((0, 4, 4)),
            'tracked': np.array(tracked, bool), 'state': np.array(state), 'ms': np.array(ms)}


def score_retracked(n_frames: int, processed: np.ndarray, meas: dict) -> dict:
    """Scored increments for frames 1..n_frames-1 from re-tracked measurements.

    Dropped frame: previous scored increment (constant velocity), source motion_model.
    Processed + tracked frame f, last processed frame p: measured D spans f - p frames; its scored increment is
    exp(log(D) / (f - p)) (half-geodesic for one drop, §4.2 F9). Processed + tracker failure: motion_model too.
    Returns rel_increment (n_frames-1, 4, 4), source (n_frames-1,), drop_rate, n_tracker_fail.
    """
    by_idx = {int(i): (d, bool(t)) for i, d, t in zip(meas['index'], meas['D'], meas['tracked'])}
    out = np.empty((n_frames - 1, 4, 4))
    source = np.empty(n_frames - 1, dtype='<U12')
    prev, last_proc, n_fail = np.eye(4), 0, 0
    for f in range(1, n_frames):
        if processed[f]:
            D, ok = by_idx[f]
            span = f - last_proc
            last_proc = f
            if ok:
                dT = se3_exp(se3_log(D) / span) if span > 1 else D
                out[f - 1], source[f - 1] = dT, 'tracked'
                prev = dT
                continue
            n_fail += 1
        out[f - 1], source[f - 1] = prev, 'motion_model'
    return {'rel_increment': out, 'source': source, 'drop_rate': float(1 - processed[1:].mean()),
            'n_tracker_fail': n_fail}
