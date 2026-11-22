"""Shared pieces for datasets without sensor depth (TUM-VI, EuRoC): validation stage V0.

`Sequence` uses the same attribute names as `redwood.RedwoodSequence` (image_paths, depth_paths, T_wc, K, width,
height, timestamps) so trackers and evaluation code can take either. Images are rectified to a pinhole model once
and cached on local disk; `image_paths` point at the cached files, so trackers that read by path need no change.
All translations are in metres.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DATA_ROOT = Path(__file__).resolve().parents[2] / 'data'   # <repo>/data; every default below can be overridden by env var
CACHE_ROOT = Path(os.environ.get('DVSLAM_CACHE_ROOT', DATA_ROOT / 'cache'))


@dataclass
class Sequence:
    dataset: str
    scene: str
    split: str
    frame_ids: np.ndarray        # (N,) int, 0..N-1 after trimming to the GT-covered range
    image_paths: list[Path]      # rectified pinhole images (cache)
    T_wc: np.ndarray             # (N, 4, 4) float64 camera-to-world, metres
    K: np.ndarray                # (3, 3) float64, intrinsics of the rectified images
    width: int
    height: int
    timestamps: np.ndarray       # (N,) float64 seconds since the first kept frame
    timestamps_ns: np.ndarray    # (N,) int64 sensor clock
    gt_valid: np.ndarray         # (N,) bool; False where the GT was interpolated across a dropout
    frame_rate_hz: float
    depth_paths: list | None = None

    def __len__(self) -> int:
        return len(self.frame_ids)

    @property
    def pair_valid(self) -> np.ndarray:
        """(N-1,) bool: pair (t-1, t) has measured GT at both ends."""
        return self.gt_valid[:-1] & self.gt_valid[1:]


def interpolate_poses(t_query_ns: np.ndarray, t_gt_ns: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray,
                      max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """GT poses at the query times: linear in position, slerp in rotation.

    Returns (T, valid): T is (Q, 4, 4); valid is False where the query lies outside the GT span or the two GT
    samples around it are more than `max_gap_s` apart (a dropout). T is only meaningful where valid is True,
    except inside a dropout, where it is still the interpolated value.
    """
    t_query_ns = np.asarray(t_query_ns, dtype=np.int64)
    t_gt_ns = np.asarray(t_gt_ns, dtype=np.int64)
    if np.any(np.diff(t_gt_ns) <= 0):
        raise ValueError('GT timestamps must be strictly increasing')
    inside = (t_query_ns >= t_gt_ns[0]) & (t_query_ns <= t_gt_ns[-1])
    tq = np.clip(t_query_ns, t_gt_ns[0], t_gt_ns[-1])
    j = np.clip(np.searchsorted(t_gt_ns, tq, side='left'), 1, len(t_gt_ns) - 1)
    gap_s = (t_gt_ns[j] - t_gt_ns[j - 1]) * 1e-9
    valid = inside & (gap_s <= max_gap_s)
    # float64 seconds relative to the first GT sample keep sub-microsecond resolution
    tg = (t_gt_ns - t_gt_ns[0]) * 1e-9
    tqs = (tq - t_gt_ns[0]) * 1e-9
    T = np.tile(np.eye(4), (len(tq), 1, 1))
    for c in range(3):
        T[:, c, 3] = np.interp(tqs, tg, pos[:, c])
    T[:, :3, :3] = Slerp(tg, Rotation.from_quat(quat_xyzw))(tqs).as_matrix()
    return T, valid


def cache_dir(dataset: str, scene: str, tag: str) -> Path:
    return CACHE_ROOT / dataset / scene / tag


def rectify_to_cache(src_paths: list[Path], dst_dir: Path, map_x: np.ndarray, map_y: np.ndarray,
                     meta: dict) -> list[Path]:
    """Remap every source image with (map_x, map_y) into dst_dir (PNG, same stem); skip files already present.

    `meta` (K of the rectified images, size, source model) is written to dst_dir/meta.json BEFORE the images, so
    an interrupted build stays tied to its parameters; an existing cache with a different meta is an error rather
    than being silently reused. 16-bit sources are stored as 8-bit (value >> 8): every tracker reads 8-bit images,
    and some readers saturate 16-bit PNGs instead of scaling them.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dst_dir / 'meta.json'
    meta = json.loads(json.dumps({**meta, 'n_frames': len(src_paths)}))
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise ValueError(f'{dst_dir}: cached rectification was built with different parameters; delete it')
    else:
        _write_atomic(meta_path, json.dumps(meta, indent=1).encode())
    threads = cv2.getNumThreads()
    cv2.setNumThreads(2)   # this machine froze once from thread oversubscription
    try:
        out = []
        for p in src_paths:
            q = dst_dir / f'{p.stem}.png'
            if not q.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f'cannot read {p}')
                if img.dtype == np.uint16:
                    img = (img >> 8).astype(np.uint8)
                rect = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                ok, buf = cv2.imencode('.png', rect)
                if not ok:
                    raise OSError(f'cannot encode {q}')
                _write_atomic(q, buf.tobytes())
            out.append(q)
    finally:
        cv2.setNumThreads(threads)   # do not leave the cap on the caller: tracker timings would be affected
    return out


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a per-process temp file so concurrent builders and interrupted runs never leave a partial file."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)
