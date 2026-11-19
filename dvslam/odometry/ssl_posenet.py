"""Self-supervised metric pose network (dvslam/ssl) behind the Odometry interface.

Increment = Predictor.increment(I_{t-1}, I_t) = T_{t-1<-t}, the scored rel_increment, from gray cam0 images.
The camera must have the checkpoint's training K and image size (no resizing or re-intrinsics here).
"""
import time

import cv2
import numpy as np

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from dvslam.ssl.predict import Predictor


class SSLPoseNetOdometry:
    def __init__(self, checkpoint: str, device: str = 'cuda'):
        self._pred = Predictor.from_checkpoint(checkpoint, device)
        self._prev: np.ndarray | None = None

    def reset(self, camera: PinholeCamera) -> None:
        if (camera.height, camera.width) != self._pred.size or not np.allclose(camera.K, self._pred.K, atol=1e-6):
            raise ValueError(f'camera K / size {camera.K.tolist()} {(camera.height, camera.width)} differs from the '
                             f'checkpoint {self._pred.K.tolist()} {self._pred.size}')
        self._prev = None

    def step(self, frame: Frame) -> PoseEstimate | None:
        img = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(frame.image_path)
        prev, self._prev = self._prev, img
        if prev is None:
            return None
        t0 = time.perf_counter()
        dT = self._pred.increment(prev, img)
        return PoseEstimate(frame.index, frame.timestamp, dT, timings_ms={'total': (time.perf_counter() - t0) * 1e3})
