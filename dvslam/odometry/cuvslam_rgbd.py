"""E0 variant: NVIDIA cuVSLAM RGB-D odometry fed with the Redwood sensor depth (plan v3 item 0.8, optional).

Needs the official `cuvslam` wheel (NVIDIA Community License, research use) in the running interpreter.
Increment = inv(world_from_rig_{t-1}) · world_from_rig_t. A lost frame repeats the previous increment.
"""
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from dvslam.data.redwood import DEPTH_SCALE
from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate


class CuVSLAMRGBDOdometry:
    def __init__(self, enable_depth_stereo_tracking: bool = False, async_sba: bool = False):
        import cuvslam   # lazy
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
        depth = np.array(Image.open(frame.depth_path)).astype(np.uint16)
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

    def _lost(self, frame: Frame, t0: float) -> PoseEstimate:
        return PoseEstimate(frame.index, frame.timestamp, self._prev_inc.copy(), tracking_state='LOST',
                            source='motion_model', timings_ms={'total': (time.perf_counter() - t0) * 1e3})
