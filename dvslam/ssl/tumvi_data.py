"""TUM-VI training samples for self-supervised depth + pose: cam0 triplets, the cam1 stereo partner, gyro rotations.

Images: cam0 and cam1 fisheyes are rectified to the same pinhole tumvi.K_OUT (512x512, f = 190) with R = identity, so
each rectified camera keeps its own orientation; cam0 reuses the cache of tumvi.load_sequence (identical meta), cam1
goes to 'cam1_pinhole_512_f190'. cam0 and cam1 share timestamps (checked on load) and exposure times (dso/cam*/times.txt
identical for all nine 512 sequences on disk, 2026-11-19), so only cam0's exposures are read.
Stereo: T_stereo = T_cam1_cam0 = camchain cam1 T_cn_cnm1, X_cam1 = T_stereo @ X_cam0, metres (baseline about 101 mm).
Photometry: the DSO vignette (fisheye domain) is normalised to max 1 and remapped with the image maps. Loss image of
frame k, normalised to the exposure of the target frame t: (gray / 255) / V * (e_t / e_k); 0 where V < v_min.
Gyro: the IMU rate (rad/s, IMU frame) is integrated on SO(3) between camera timestamps, linear interpolation of the
rate at the partial end intervals and the trapezoidal rate on each segment; no bias estimation. Rotations follow
X_a = R_{a<-b} X_b and are expressed in the camera frame with cam0's R_ci (X_cam = R_ci X_imu):
R_{cam_a<-cam_b} = R_ci R_{imu_a<-imu_b} R_ci^T. Mocap GT is not used.
Stereo hints (optional, `hints_dir`): <hints_dir>/<scene>.npy (N, H, W) float16, the stereo-matched metric z-depth of
cam0 frame i in the rectified cam0 pinhole (cam0 + cam1 SGBM, runs/eval/exp010_cache_hints.py), 0 = no hint. The
sample gets 'hint' (1, H, W) float32 of the target frame t. Training-only (cam1); never an inference input.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from dvslam.data import tumvi

FRAME_DT_NS = 50_000_000
FRAME_DT_TOL_NS = 5_000_000      # a neighbour gap outside 50 +- 5 ms is a frame drop
MAX_IMU_GAP_NS = 15_000_000      # IMU is 200 Hz (5 ms)
AUG_RANGE = (0.8, 1.25)          # gamma and brightness gain


def integrate_gyro(t_imu_ns: np.ndarray, omega: np.ndarray, t0_ns: int, t1_ns: int) -> tuple[np.ndarray, bool]:
    """R_{imu(t0)<-imu(t1)} (3,3) from the body rate `omega` (N, 3) sampled at `t_imu_ns`, t0 < t1.

    ok is False (and R the identity) if the IMU does not cover [t0, t1] or has a gap > MAX_IMU_GAP_NS around it.
    """
    i0 = np.searchsorted(t_imu_ns, t0_ns, side='right') - 1      # last sample <= t0
    i1 = np.searchsorted(t_imu_ns, t1_ns, side='left')           # first sample >= t1
    if i0 < 0 or i1 >= len(t_imu_ns) or np.diff(t_imu_ns[i0:i1 + 1]).max() > MAX_IMU_GAP_NS:
        return np.eye(3), False
    ts = (t_imu_ns[i0:i1 + 1] - t0_ns) * 1e-9
    knots = np.r_[0.0, ts[1:-1], (t1_ns - t0_ns) * 1e-9]
    w = np.stack([np.interp(knots, ts, omega[i0:i1 + 1, c]) for c in range(3)], 1)
    R = Rotation.identity()
    for step in Rotation.from_rotvec(0.5 * (w[:-1] + w[1:]) * np.diff(knots)[:, None]):
        R = R * step
    return R.as_matrix(), True


def photometric(gray: np.ndarray, V: np.ndarray, exposure_ratio: float, v_min: float,
                sat_level: int) -> tuple[np.ndarray, np.ndarray]:
    """(loss image, valid) float32 (H, W) of an 8-bit image: (gray / 255) / V * exposure_ratio, 0 where V < v_min."""
    ok_v = V >= v_min
    pho = np.where(ok_v, gray.astype(np.float32) / 255 / np.where(ok_v, V, 1) * exposure_ratio, 0)
    return pho.astype(np.float32), (ok_v & (gray < sat_level)).astype(np.float32)


def rectified_vignette(seq_dir: Path, cam: str, intrinsics: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Vignette V (H, W) float32 of `cam`, max 1, remapped (bilinear) like the rectified images."""
    v = cv2.imread(str(seq_dir / f'dso/{cam}/vignette.png'), cv2.IMREAD_UNCHANGED)
    if v is None:
        raise ValueError(f'cannot read {seq_dir}/dso/{cam}/vignette.png')
    v = v.astype(np.float32) / v.max()
    map_x, map_y = tumvi.rectify_maps(intrinsics, dist)
    return cv2.remap(v, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def gyro_rotations(seq_dir: Path, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R (N-1, 3, 3) = R_{cam0_i<-cam0_i+1} between consecutive timestamps `t_ns`, ok (N-1,) bool)."""
    imu_csv = seq_dir / 'mav0/imu0/data.csv'
    t_imu = np.loadtxt(imu_csv, delimiter=',', comments='#', usecols=0, dtype=np.int64)
    omega = np.loadtxt(imu_csv, delimiter=',', comments='#', usecols=(1, 2, 3), dtype=np.float64)
    R_ci = tumvi.load_cam_calib(seq_dir, 'cam0')[0][:3, :3]
    rel = [integrate_gyro(t_imu, omega, t_ns[i], t_ns[i + 1]) for i in range(len(t_ns) - 1)]
    return R_ci @ np.array([r for r, _ in rel]) @ R_ci.T, np.array([ok for _, ok in rel])


def scene_frames(scene: str, root: Path | str | None = None) -> dict:
    """Everything a scene contributes, rectified caches built: per cam0 frame i (N frames of the sequence):
    't_ns' (N,), 'cam0'/'cam1' cache paths, 'exposure_ms' (N,), 'R_next' (N-1, 3, 3) = R_{cam_i<-cam_i+1},
    'gyro_ok' (N-1,), plus 'V' {cam: (H, W)}, 'T_stereo' (4, 4).
    """
    d = tumvi.sequence_dir(scene, root)
    if not d.is_dir():
        raise FileNotFoundError(f'{scene}: {d} not found')
    out = {'V': {}}
    for cam in ('cam0', 'cam1'):
        t_ns, images = tumvi.camera_frames(d, cam)
        if cam == 'cam0':
            out['t_ns'] = t_ns
        elif not np.array_equal(t_ns, out['t_ns']):
            raise ValueError(f'{scene}: cam0 and cam1 timestamps differ')
        _, intrinsics, dist = tumvi.load_cam_calib(d, cam)
        out[cam] = [str(p) for p in tumvi.rectified_paths(scene, cam, images, intrinsics, dist, build=True)]
        out['V'][cam] = rectified_vignette(d, cam, intrinsics, dist)
    t_exp = np.loadtxt(d / 'dso/cam0/times.txt', comments='#', usecols=0, dtype=np.int64)
    exposure = np.loadtxt(d / 'dso/cam0/times.txt', comments='#', usecols=2, dtype=np.float64)
    j = np.clip(np.searchsorted(t_exp, out['t_ns']), 0, len(t_exp) - 1)
    if not np.array_equal(t_exp[j], out['t_ns']):
        raise ValueError(f'{scene}: dso/cam0/times.txt lacks exposure times for some frames')
    out['exposure_ms'] = exposure[j]
    with open(d / 'dso/camchain.yaml') as f:
        out['T_stereo'] = np.array(yaml.safe_load(f)['cam1']['T_cn_cnm1'], dtype=np.float64)
    out['R_next'], out['gyro_ok'] = gyro_rotations(d, out['t_ns'])
    return out


class TumviSSLDataset(torch.utils.data.Dataset):
    """Samples: every cam0 frame t with neighbours t-1 and t+1 both 50 +- 5 ms away (see module docstring for items)."""

    def __init__(self, scenes: list[str], train: bool, root: Path | str | None = None, v_min: float = 0.25,
                 sat_level: int = 250, hints_dir: Path | str | None = None):
        self.scenes = list(scenes)
        self.hints_dir = hints_dir
        self._hints = None               # memory maps, opened lazily in each worker (see __getstate__)
        self.v_min = v_min
        self.sat_level = sat_level
        self.aug = train                 # gamma / brightness augmentation of the network images
        self.data = [scene_frames(s, root) for s in self.scenes]
        self.samples = []
        for k, f in enumerate(self.data):
            gap_ok = np.abs(np.diff(f['t_ns']) - FRAME_DT_NS) <= FRAME_DT_TOL_NS
            self.samples += [(k, int(i) + 1) for i in np.flatnonzero(gap_ok[:-1] & gap_ok[1:])]

        if hints_dir is not None:
            for s, f in zip(self.scenes, self.data):
                n = np.load(Path(hints_dir) / f'{s}.npy', mmap_mode='r').shape[0]
                if n != len(f['t_ns']):
                    raise ValueError(f'{s}: {n} stereo hints for {len(f["t_ns"])} frames')

    def __getstate__(self) -> dict:
        return {**self.__dict__, '_hints': None}         # a pickled memmap would copy the whole array

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        cv2.setNumThreads(1)
        k, t = self.samples[idx]
        f = self.data[k]
        frames = [(f['cam0'][t - 1], 'cam0', t - 1), (f['cam0'][t], 'cam0', t), (f['cam0'][t + 1], 'cam0', t + 1),
                  (f['cam1'][t], 'cam1', t)]
        grays, pho, valid = [], [], []
        for path, cam, i in frames:
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise ValueError(f'cannot read {path}')
            p, v = photometric(g, f['V'][cam], f['exposure_ms'][t] / f['exposure_ms'][i], self.v_min, self.sat_level)
            grays.append(g)
            pho.append(p)
            valid.append(v)
        net = torch.from_numpy(np.stack(grays[:3]).astype(np.float32) / 255)
        if self.aug:
            gamma, gain = torch.empty(2).uniform_(*AUG_RANGE)
            net = (gain * net ** gamma).clamp(0, 1)
        extra = {}
        if self.hints_dir is not None:
            if self._hints is None:
                self._hints = [np.load(Path(self.hints_dir) / f'{s}.npy', mmap_mode='r') for s in self.scenes]
            extra['hint'] = torch.from_numpy(self._hints[k][t].astype(np.float32))[None]
        return {
            **extra,
            'net': net[:, None].expand(3, 3, *net.shape[1:]).contiguous(),
            'pho': torch.from_numpy(np.stack(pho)[:, None]),
            'valid': torch.from_numpy(np.stack(valid)[:, None]),
            'K': torch.from_numpy(tumvi.K_OUT.astype(np.float32)),
            'T_stereo': torch.from_numpy(f['T_stereo'].astype(np.float32)),
            'R_gyro': torch.from_numpy(np.stack([f['R_next'][t - 1], f['R_next'][t].T]).astype(np.float32)),
            'gyro_ok': torch.tensor(bool(f['gyro_ok'][t - 1] and f['gyro_ok'][t])),
            'scene': self.scenes[k],
            'index': t,
        }
