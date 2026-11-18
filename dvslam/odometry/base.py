"""Tracker interface and the scored output (plan v3 §4.2, §4.8).

Item 03/04 are scored from `rel_increment` = ΔT_t = inv(T_wc_{t-1}) · T_wc_t, one per frame t >= 1.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from dvslam.geometry.se3 import chain_increments

TrackingState = Literal['BOOTSTRAP', 'OK', 'DEGRADED', 'LOST']
Source = Literal['tracked', 'motion_model']


@dataclass
class PinholeCamera:
    K: np.ndarray
    width: int
    height: int


@dataclass
class Frame:
    index: int
    timestamp: float
    image_path: Path
    depth_path: Path | None = None


@dataclass
class PoseEstimate:
    index: int
    timestamp: float
    rel_increment: np.ndarray            # 4x4 float64, pose of camera t in camera t-1
    tracking_state: TrackingState = 'OK'
    source: Source = 'tracked'
    is_keyframe: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


class Odometry(Protocol):
    def reset(self, camera: PinholeCamera) -> None: ...
    def step(self, frame: Frame) -> PoseEstimate | None: ...   # None only for the first frame (no pair yet)


def save_increments(path: Path, estimates: list[PoseEstimate], meta: dict) -> None:
    np.savez(path,
             index=np.array([e.index for e in estimates], dtype=np.int64),
             timestamp=np.array([e.timestamp for e in estimates], dtype=np.float64),
             rel_increment=np.stack([e.rel_increment for e in estimates]).astype(np.float64),
             tracking_state=np.array([e.tracking_state for e in estimates]),
             source=np.array([e.source for e in estimates]),
             meta=np.array(json.dumps(meta)))


def load_increments(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out['meta'] = json.loads(str(out['meta']))
    return out


def write_tum(path: Path, timestamps: np.ndarray, T_wc: np.ndarray) -> None:
    q = Rotation.from_matrix(T_wc[:, :3, :3]).as_quat()  # x, y, z, w
    with open(path, 'w') as f:
        for t, T, qq in zip(timestamps, T_wc, q):
            f.write(f'{t:.6f} ' + ' '.join(f'{v:.9f}' for v in (*T[:3, 3], *qq)) + '\n')


def causal_trajectory(increments: np.ndarray) -> np.ndarray:
    """Chain increments from identity: T_out_0 = I, T_out_t = T_out_{t-1} · ΔT_t."""
    return chain_increments(np.eye(4), increments)
