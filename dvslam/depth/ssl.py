"""Depth of the self-supervised metric model (dvslam/ssl) as a DepthPrior: predict(frame) -> (depth_m, None).

Gray input read with cv2.IMREAD_GRAYSCALE; K and size must equal the checkpoint's training K and size.
"""
import cv2
import numpy as np

from dvslam.odometry.base import Frame
from dvslam.ssl.predict import Predictor


class SSLDepth:
    name = 'ssl'

    def __init__(self, ckpt: str, K: np.ndarray, device: str = 'cuda'):
        self._pred = Predictor.from_checkpoint(ckpt, device)
        if not np.allclose(K, self._pred.K, atol=1e-6):
            raise ValueError(f'K {np.asarray(K).tolist()} differs from the checkpoint K {self._pred.K.tolist()}')

    def predict(self, frame: Frame) -> tuple[np.ndarray, None]:
        img = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(frame.image_path)
        return self._pred.depth(img), None
