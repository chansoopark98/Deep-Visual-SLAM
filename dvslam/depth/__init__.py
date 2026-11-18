"""Zero-shot metric depth priors (plan v3 §3.2, §4.8 DepthPrior protocol, item 0.10 / E1).

Each model class exposes `predict(frame) -> (depth_m HxW float32, conf HxW float32 | None)` at the input image
resolution, plus `predict_image(bgr)` for an already loaded uint8 BGR image. Model code is a minimal copy of the
official inference code under `third_party/<name>/` (see each NOTICE.md); weights are read from
`$DVSLAM_DEPTH_WEIGHTS` (default `runs/e1_weights/`, git-ignored).
"""
import os
import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from dvslam.odometry.base import Frame

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / 'third_party'
WEIGHTS_ROOT = Path(os.environ.get('DVSLAM_DEPTH_WEIGHTS', REPO_ROOT / 'runs' / 'e1_weights'))
MODELS = ('dav2s', 'da3m', 'moge2s')


class DepthPrior(Protocol):
    def predict(self, frame: Frame) -> tuple[np.ndarray, np.ndarray | None]: ...


def add_third_party(name: str) -> None:
    path = str(THIRD_PARTY / name)
    if path not in sys.path:
        sys.path.insert(0, path)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def build(name: str, K: np.ndarray, **kwargs) -> DepthPrior:
    """K (3x3, at the input image resolution) is used by the models that take a focal (DA3METRIC, MoGe-2)."""
    if name == 'dav2s':
        from dvslam.depth.dav2 import DAv2Metric
        return DAv2Metric(**kwargs)
    if name == 'da3m':
        from dvslam.depth.da3 import DA3Metric
        return DA3Metric(K, **kwargs)
    if name == 'moge2s':
        from dvslam.depth.moge import MoGe2
        return MoGe2(K, **kwargs)
    raise ValueError(f'unknown depth model {name!r}, expected one of {MODELS}')
