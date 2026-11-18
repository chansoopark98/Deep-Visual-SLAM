"""E0: frame-to-frame RGB-D odometry with sensor depth (plan v3 item 0.8, G0-E0).

Open3D `compute_rgbd_odometry` with the hybrid (photometric + point-to-plane) Jacobian, pair (t-1, t).
The Redwood GT is itself an Open3D reconstruction from this depth, so this tracker can share error with the GT;
`scripts/e0_validity.py` decides whether the result counts for the gate.
"""
import time

import numpy as np

from dvslam.data.redwood import DEPTH_SCALE
from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate


class RGBDICPOdometry:
    def __init__(self, depth_trunc: float = 4.0, max_depth_diff: float = 0.07, min_depth: float = 0.0,
                 max_depth: float = 4.0, jacobian: str = 'hybrid'):
        import open3d as o3d   # lazy: the module stays importable without open3d
        self._o3d = o3d
        self._depth_trunc = depth_trunc
        self._option = o3d.pipelines.odometry.OdometryOption()
        self._option.depth_diff_max = max_depth_diff
        self._option.depth_min = min_depth
        self._option.depth_max = max_depth
        self._jac = {'hybrid': o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm,
                     'color': o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm}[jacobian]()
        self._intr = None
        self._prev = None
        self._prev_inc = np.eye(4)

    def reset(self, camera: PinholeCamera) -> None:
        K = camera.K
        self._intr = self._o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self._prev = None
        self._prev_inc = np.eye(4)

    def _load(self, frame: Frame):
        o3d = self._o3d
        color = o3d.io.read_image(str(frame.image_path))
        depth = o3d.io.read_image(str(frame.depth_path))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=DEPTH_SCALE, depth_trunc=self._depth_trunc, convert_rgb_to_intensity=True)

    def step(self, frame: Frame) -> PoseEstimate | None:
        t0 = time.perf_counter()
        cur = self._load(frame)
        prev, self._prev = self._prev, cur
        if prev is None:
            return None
        # source = t, target = t-1  ->  T maps points of camera t into camera t-1 = pose of camera t in camera t-1.
        ok, T, _ = self._o3d.pipelines.odometry.compute_rgbd_odometry(
            cur, prev, self._intr, np.eye(4), self._jac, self._option)
        if ok:
            inc, state, src = np.asarray(T, dtype=np.float64), 'OK', 'tracked'
        else:
            inc, state, src = self._prev_inc.copy(), 'DEGRADED', 'motion_model'
        self._prev_inc = inc
        return PoseEstimate(frame.index, frame.timestamp, inc, tracking_state=state, source=src,
                            timings_ms={'total': (time.perf_counter() - t0) * 1e3})
