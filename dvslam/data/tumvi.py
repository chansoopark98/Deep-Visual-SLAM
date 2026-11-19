"""TUM-VI room sequences, cam0, 512x512 version (plan v3.2 item T.1).

Splits (user decision 2026-11-19, target-metric track; replaces plan §12.3's room1-2 validation / room3-6 test): whole
sequences, train room1-4, validation room5, test room6. All six rooms are the same motion-capture room, so the split
separates trajectories, not scenes. The official target-metric evaluation set is train + test (room1-4, room6).
A Redwood-style 80/10/10 split in time within each room is a separate evaluation: apps/eval_pairs_temporal.py.

Layout per sequence <root>/dataset-room{N}_512_16/: mav0/cam0/data.csv + data/<ns>.png (grayscale fisheye),
mav0/mocap0/data.csv (120 Hz), dso/camchain.yaml (cam0: T_cam_imu, equidistant intrinsics and k1..k4).

Rectification: the equidistant (Kannala-Brandt) fisheye is remapped to ONE fixed pinhole K_OUT for every sequence:
512x512, fx = fy = 190.0, cx = cy = 255.5. f = 190 keeps the calibrated fisheye focal length (about 190.97 px), so
the image centre keeps its native resolution and the plan's "f about 190 px" holds. Field of view: 106.8 deg
horizontal and vertical, 124.6 deg diagonal (2 atan(256 / 190) and 2 atan(256 sqrt 2 / 190)); every output pixel
samples inside the 512x512 source. The remainder of the fisheye view (up to 195 deg) is dropped. K_OUT is written to
the cache meta.json next to the rectified PNGs. cam1 (dvslam.ssl.tumvi_data) is rectified the same way with
`rectified_paths`, also with R = identity: each camera keeps its own orientation.

GT: the mocap rows (p_RS_R, q_RS) are the pose of the IMU frame S in the mocap world R (the gyro matches the mocap
angular rate in the IMU frame, not in the camera frame). Camera pose: T_wc = T_w_imu @ inv(T_cam_imu).
The image taken at camera time t gets the mocap pose at t + time_offset_s. The mocap-to-camera offset tau is NOT known
to be 0 (plan §11.2, §12.6): it must be measured from a tracker run and passed in; the default 0 means "no correction
applied", not "tau is 0". Translations are metres (float64).

room1 check (tests/test_tumvi.py): the plan §11.2 figures (median step 53 mm, step roughness 3.7 %) are IMU-origin
values (measured 53.43 mm, 3.74 %); the camera centre returned here gives 56.85 mm and 4.02 %. The GT-vs-smoothed-GT
MAPE depends strongly on the smoothing window (SG(15,3) at 20 Hz: 4.39 % camera, 3.66 % IMU; plan 1.3 % not
reproduced).
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from dvslam.data import base

DEFAULT_ROOT = Path(os.environ.get('DVSLAM_TUMVI_ROOT', '/home/park-ubuntu/park/datasets/tum_vi'))
SCENES = ('room1', 'room2', 'room3', 'room4', 'room5', 'room6')
SPLITS = {'train': ('room1', 'room2', 'room3', 'room4'), 'validation': ('room5',), 'test': ('room6',)}
FRAME_RATE_HZ = 20.0
MAX_GAP_S = 0.02            # mocap is 120 Hz (8.3 ms); a larger gap between the two samples around a frame is a dropout
OUT_SIZE = 512
K_OUT = np.array([[190.0, 0.0, 255.5], [0.0, 190.0, 255.5], [0.0, 0.0, 1.0]])
CACHE_TAG = 'cam0_pinhole_512_f190'      # cam1 (stereo partner, SSL training only): 'cam1_pinhole_512_f190'


def sequence_dir(scene: str, root: Path | str | None = None) -> Path:
    return (Path(root) if root else DEFAULT_ROOT) / f'dataset-{scene}_{OUT_SIZE}_16'


def available_scenes(split: str, root: Path | str | None = None) -> list[str]:
    """Scenes of `split` whose sequence directory exists on disk."""
    return [s for s in SPLITS[split] if sequence_dir(s, root).is_dir()]


def load_cam_calib(seq_dir: Path, cam: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T_cam_imu (4,4), intrinsics [fx, fy, cx, cy], distortion k1..k4) of camera `cam` ('cam0' or 'cam1')."""
    with open(seq_dir / 'dso/camchain.yaml') as f:
        c = yaml.safe_load(f)[cam]
    if c['distortion_model'] != 'equidistant':
        raise ValueError(f'{seq_dir}: {cam} distortion model is {c["distortion_model"]}, expected equidistant')
    return (np.array(c['T_cam_imu'], dtype=np.float64), np.array(c['intrinsics'], dtype=np.float64),
            np.array(c['distortion_coeffs'], dtype=np.float64))


def load_cam0_calib(seq_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_cam_calib(seq_dir, 'cam0')


def camera_frames(seq_dir: Path, cam: str) -> tuple[np.ndarray, list[Path]]:
    """(timestamps ns (N,) int64, fisheye source image paths) of every frame of camera `cam`."""
    with open(seq_dir / f'mav0/{cam}/data.csv') as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
    t_ns = np.array([int(r[0]) for r in rows], dtype=np.int64)
    return t_ns, [seq_dir / f'mav0/{cam}/data' / r[1].strip() for r in rows]


def rectify_maps(intrinsics: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 (map_x, map_y) for cv2.remap: output pixel of K_OUT -> source pixel of the equidistant fisheye."""
    fx, fy, cx, cy = intrinsics
    K_src = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(K_src, dist, np.eye(3), K_OUT, (OUT_SIZE, OUT_SIZE), cv2.CV_32FC1)


def rectified_paths(scene: str, cam: str, images: list[Path], intrinsics: np.ndarray, dist: np.ndarray,
                    build: bool) -> list[Path]:
    """Cache paths of `images` rectified to K_OUT with R = identity; build=True first creates the missing files."""
    dst = base.cache_dir('tumvi', scene, CACHE_TAG.replace('cam0', cam))
    if not build:
        return [dst / f'{p.stem}.png' for p in images]
    map_x, map_y = rectify_maps(intrinsics, dist)
    meta = {'K': K_OUT.tolist(), 'size': [OUT_SIZE, OUT_SIZE], 'source_model': 'equidistant',
            'source_intrinsics': intrinsics.tolist(), 'source_distortion': dist.tolist(), 'bit_depth': 8}
    return base.rectify_to_cache(images, dst, map_x, map_y, meta)


def load_sequence(scene: str, split: str | None = None, root: Path | str | None = None,
                  time_offset_s: float = 0.0, load_images: bool = True) -> base.Sequence:
    """split=None infers the split from SPLITS; a split that does not contain `scene` is an error.

    load_images=False leaves the image cache untouched and returns the paths the cache would have (GT-only use).
    """
    if split is None:
        split = next(k for k, v in SPLITS.items() if scene in v)
    elif scene not in SPLITS.get(split, ()):
        raise ValueError(f'{scene} is not in split {split!r} (SPLITS: {SPLITS})')
    d = sequence_dir(scene, root)
    if not d.is_dir():
        raise FileNotFoundError(f'{scene}: {d} not found')
    t_cam, images = camera_frames(d, 'cam0')
    mocap = d / 'mav0/mocap0/data.csv'
    t_gt = np.loadtxt(mocap, delimiter=',', comments='#', usecols=0, dtype=np.int64)
    gt = np.loadtxt(mocap, delimiter=',', comments='#', usecols=range(1, 8), dtype=np.float64)
    T_w_imu, valid = base.interpolate_poses(t_cam + round(time_offset_s * 1e9), t_gt, gt[:, :3],
                                            gt[:, [4, 5, 6, 3]], MAX_GAP_S)
    ok = np.flatnonzero(valid)
    if len(ok) < 2:
        raise ValueError(f'{scene}: fewer than 2 frames with mocap GT')
    keep = slice(ok[0], ok[-1] + 1)
    T_cam_imu, intrinsics, dist = load_cam0_calib(d)
    T_wc = T_w_imu[keep] @ np.linalg.inv(T_cam_imu)
    t_cam = t_cam[keep]
    # all frames, independent of the GT range, so the cache does not depend on time_offset_s
    paths = rectified_paths(scene, 'cam0', images, intrinsics, dist, load_images)[keep]
    return base.Sequence(
        dataset='tumvi', scene=scene, split=split, frame_ids=np.arange(len(t_cam)), image_paths=paths, T_wc=T_wc,
        K=K_OUT.copy(), width=OUT_SIZE, height=OUT_SIZE, timestamps=(t_cam - t_cam[0]) * 1e-9, timestamps_ns=t_cam,
        gt_valid=valid[keep], frame_rate_hz=FRAME_RATE_HZ)
