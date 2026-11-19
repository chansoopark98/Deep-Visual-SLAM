"""EuRoC MAV machine-hall sequences, left camera (cam0) only: validation stage V0.

All five machine-hall sequences are recorded in the same place (one machine hall); the validation/test split
separates sequences, not places. Vicon-room sequences are not used.

Layout per scene: <root>/machine_hall/<scene>/mav0/ with cam0/data.csv (`#timestamp [ns],filename`),
cam0/data/*.png (752x480 grayscale, 20 Hz), cam0/sensor.yaml and state_groundtruth_estimate0/data.csv.

Camera: cam0/sensor.yaml gives `T_BS` = pose of the camera in the body (IMU) frame, p_B = T_BS p_S, and a
radial-tangential model (k1, k2, p1, p2). Images are undistorted once to a pinhole model with zero distortion,
same size and the sensor's own K, and cached (base.rectify_to_cache).

GT: state_groundtruth_estimate0 is the dataset's own state estimate, not a direct measurement. Its sensor.yaml
describes it as the nonlinear least-squares batch solution over the Leica MS50 laser-tracker positions and the
IMU, with time-offset estimation; the orientation is observed only by the IMU. There is no 6-DoF motion capture
in the machine hall. Rows are body (IMU) poses in the reference frame at 200 Hz, metres, quaternion w-first.
T_wc = T_w_body @ T_BS, interpolated to the image timestamps. The GT stream starts after and ends before the
image stream: frames outside it are trimmed; bridged GT dropouts inside it are kept with gt_valid=False.
`time_offset_s` (default 0) is added to the camera clock before the GT lookup; any camera-GT offset has to be
measured, not assumed.
The GT is a batch-smoothed estimate, so smoothing-based GT-floor proxies are low by construction on this dataset.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from dvslam.data import base

DEFAULT_ROOT = Path(os.environ.get('DVSLAM_EUROC_ROOT', '/home/park-ubuntu/park/datasets/euroc'))
SPLITS = {'validation': ('MH_01_easy', 'MH_02_easy'),
          'test': ('MH_03_medium', 'MH_04_difficult', 'MH_05_difficult')}
FRAME_RATE_HZ = 20.0
MAX_GT_GAP_S = 0.02
CACHE_TAG = 'cam0_pinhole'


def _scene_dir(scene: str, root: Path | str | None) -> Path:
    return (Path(root) if root else DEFAULT_ROOT) / 'machine_hall' / scene / 'mav0'


def _read_cam_csv(d: Path) -> tuple[np.ndarray, list[Path]]:
    rows = [line.strip().split(',') for line in (d / 'cam0' / 'data.csv').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    return np.array([int(r[0]) for r in rows], dtype=np.int64), [d / 'cam0' / 'data' / r[1].strip() for r in rows]


def available_scenes(split: str, root: Path | str | None = None) -> list[str]:
    """Scenes of `split` whose CSVs and every listed cam0 image are on disk."""
    out = []
    for s in SPLITS[split]:
        d = _scene_dir(s, root)
        if (d / 'cam0' / 'data.csv').exists() and (d / 'state_groundtruth_estimate0' / 'data.csv').exists():
            if all(p.exists() for p in _read_cam_csv(d)[1]):
                out.append(s)
    return out


def read_camera(sensor_yaml: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """(K, dist (k1, k2, p1, p2), T_BS, (width, height)) from a cam sensor.yaml."""
    y = yaml.safe_load(sensor_yaml.read_text())
    if y['distortion_model'] != 'radial-tangential':
        raise ValueError(f'{sensor_yaml}: unsupported distortion model {y["distortion_model"]}')
    fu, fv, cu, cv = y['intrinsics']
    K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float64)
    T_BS = np.array(y['T_BS']['data'], dtype=np.float64).reshape(4, 4)
    return K, np.array(y['distortion_coefficients'], dtype=np.float64), T_BS, tuple(y['resolution'])


def rectify_maps(K: np.ndarray, dist: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Maps from the output pinhole image (intrinsics K, no distortion) into the distorted source image."""
    return cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)


def load_sequence(scene: str, split: str | None = None, root: Path | str | None = None,
                  time_offset_s: float = 0.0, load_images: bool = True) -> base.Sequence:
    """`split=None` infers the split from SPLITS. `load_images=False` leaves the image cache untouched and returns
    the paths the cache would have (for tools that only need the GT)."""
    if split is None:
        split = next((k for k, v in SPLITS.items() if scene in v), None)
        if split is None:
            raise ValueError(f'unknown EuRoC scene {scene}; known: {SPLITS}')
    elif scene not in SPLITS.get(split, ()):
        raise ValueError(f'EuRoC scene {scene} is not in split {split!r}; splits: {SPLITS}')
    d = _scene_dir(scene, root)
    gt_csv = d / 'state_groundtruth_estimate0' / 'data.csv'
    if not (d / 'cam0' / 'data.csv').exists() or not gt_csv.exists():
        raise FileNotFoundError(f'EuRoC {scene} is not on disk under {d}')
    t_ns, src = _read_cam_csv(d)
    missing = [p for p in src if not p.exists()]
    if missing:
        raise FileNotFoundError(f'{scene}: {len(missing)} cam0 images missing, e.g. {missing[0]}')
    K, dist, T_BS, (w, h) = read_camera(d / 'cam0' / 'sensor.yaml')

    rows = [line.split(',') for line in gt_csv.read_text().splitlines() if line.strip() and not line.startswith('#')]
    t_gt = np.array([int(r[0]) for r in rows], dtype=np.int64)    # exact ns, not via float
    gt = np.array([[float(x) for x in r[1:8]] for r in rows])     # p_x, p_y, p_z, q_w, q_x, q_y, q_z
    T_wb, valid = base.interpolate_poses(t_ns + int(round(time_offset_s * 1e9)), t_gt, gt[:, 0:3],
                                         gt[:, [4, 5, 6, 3]], MAX_GT_GAP_S)
    if not valid.any():
        raise ValueError(f'{scene}: no image timestamp is covered by GT')
    i0, i1 = np.flatnonzero(valid)[[0, -1]]
    keep = slice(i0, i1 + 1)

    dst = base.cache_dir('euroc', scene, CACHE_TAG)
    if load_images:
        map_x, map_y = rectify_maps(K, dist, (w, h))
        meta = {'source': 'cam0 radial-tangential', 'dist': dist.tolist(), 'K': K.tolist(), 'size': [w, h]}
        # all frames are rectified, so the cache does not depend on time_offset_s
        images = base.rectify_to_cache(src, dst, map_x, map_y, meta)[keep]
    else:
        images = [dst / f'{q.stem}.png' for q in src[keep]]
    ts = t_ns[keep]
    return base.Sequence(dataset='euroc', scene=scene, split=split, frame_ids=np.arange(len(ts)),
                         image_paths=images, T_wc=T_wb[keep] @ T_BS, K=K, width=w, height=h,
                         timestamps=(ts - ts[0]) * 1e-9, timestamps_ns=ts, gt_valid=valid[keep],
                         frame_rate_hz=FRAME_RATE_HZ)
