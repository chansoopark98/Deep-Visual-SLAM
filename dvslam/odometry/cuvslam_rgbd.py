"""E0 variant: NVIDIA cuVSLAM RGB-D odometry fed with the Redwood sensor depth (plan v3 item 0.8, optional).

Needs the official `cuvslam` wheel (NVIDIA Community License, research use) in the running interpreter.
Depth `prior`: sensor (default, as 0.8) | perturbed (PerturbedGTDepth, bias 2 %, jitter 3 %, `prior_seed`) |
oracle (PerturbedGTDepth with bias = jitter = 0) | cache:<dir> (float32 metre maps <dir>/<scene>/<frame stem>.npy).
Only `sensor` uses depth that the mono-only submission may not have; the others stand in for a depth network.
Increment = inv(world_from_rig_{t-1}) · world_from_rig_t. A lost frame repeats the previous increment.
"""
import time

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from dvslam.data.redwood import DEPTH_SCALE
from dvslam.depth.perturbed import PerturbedGTDepth
from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate


class CuVSLAMRGBDOdometry:
    def __init__(self, enable_depth_stereo_tracking: bool = False, async_sba: bool = False,
                 prior: str = 'sensor', prior_seed: int = 0):
        import cuvslam   # lazy
        self._prior_fn = None
        self._cache_dir = None
        if prior == 'perturbed':
            self._prior_fn = PerturbedGTDepth(seed=prior_seed)
        elif prior == 'oracle':
            self._prior_fn = PerturbedGTDepth(bias=0.0, jitter_logsd=0.0, seed=prior_seed)
        elif prior.startswith('cache:'):
            self._cache_dir = Path(prior[len('cache:'):])
        elif prior != 'sensor':
            raise ValueError(f'prior must be sensor|perturbed|oracle|cache:<dir>, got {prior!r}')
        self._cu = cuvslam
        self._stereo = enable_depth_stereo_tracking
        self._async_sba = async_sba
        self._tracker = None

    def reset(self, camera: PinholeCamera) -> None:
        cu = self._cu
        cam = cu.Camera()
        cam.size = (camera.width, camera.height)
        cam.focal = (float(camera.K[0, 0]), float(camera.K[1, 1]))
        cam.principal = (float(camera.K[0, 2]), float(camera.K[1, 2]))
        rgbd = cu.Tracker.OdometryRGBDSettings()
        rgbd.depth_scale_factor = DEPTH_SCALE
        rgbd.depth_camera_id = 0
        rgbd.enable_depth_stereo_tracking = self._stereo
        cfg = cu.Tracker.OdometryConfig(async_sba=self._async_sba, odometry_mode=cu.Tracker.OdometryMode.RGBD,
                                        rgbd_settings=rgbd)
        self._tracker = cu.Tracker(cu.Rig([cam]), cfg)
        self._first = True
        self._prev_T = None
        self._prev_inc = np.eye(4)

    def step(self, frame: Frame) -> PoseEstimate | None:
        t0 = time.perf_counter()
        color = np.array(Image.open(frame.image_path).convert('RGB'))
        depth = self._depth(frame)
        est, _ = self._tracker.track(int(round(frame.timestamp * 1e9)), images=[color], depths=[depth])
        pose = None if est.world_from_rig is None else est.world_from_rig.pose
        T = None
        if pose is not None:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(np.asarray(pose.rotation, dtype=np.float64)).as_matrix()   # x, y, z, w
            T[:3, 3] = np.asarray(pose.translation, dtype=np.float64)
        first, self._first = self._first, False
        prev_T = self._prev_T
        if T is not None:
            self._prev_T = T
        if first:
            return None
        if T is None or prev_T is None:
            return self._lost(frame, t0)
        inc = np.linalg.inv(prev_T) @ T
        self._prev_inc = inc
        return PoseEstimate(frame.index, frame.timestamp, inc,
                            timings_ms={'total': (time.perf_counter() - t0) * 1e3})

    def _depth(self, frame: Frame) -> np.ndarray:
        """uint16 depth in 1/DEPTH_SCALE m for cuVSLAM; 0 = invalid."""
        if self._prior_fn is None and self._cache_dir is None:
            return np.array(Image.open(frame.depth_path)).astype(np.uint16)
        if self._prior_fn is not None:
            d, _ = self._prior_fn.predict(frame)
        else:
            d = np.load(self._cache_dir / frame.depth_path.parent.parent.name / f'{frame.depth_path.stem}.npy')
        d = np.nan_to_num(np.asarray(d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(np.rint(d * DEPTH_SCALE), 0, 65535).astype(np.uint16)

    def _lost(self, frame: Frame, t0: float) -> PoseEstimate:
        return PoseEstimate(frame.index, frame.timestamp, self._prev_inc.copy(), tracking_state='LOST',
                            source='motion_model', timings_ms={'total': (time.perf_counter() - t0) * 1e3})
