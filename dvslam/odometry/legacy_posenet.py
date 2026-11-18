"""The 2026 Monodepth2-style PoseNet behind the Odometry interface (plan v3 item 0.4).

Increment = PoseNet(I_{t-1}, I_t) with invert=True, exactly as vo/eval_redwood.py scores it.
"""
import time
from pathlib import Path

import yaml

from dvslam.odometry.base import Frame, PinholeCamera, PoseEstimate
from vo.eval_redwood import RedwoodEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (REPO_ROOT / 'weights/vo/mode=axisAngle_res=(480, 640)_ep=31_bs=16_initLR=0.0001_endLR=1e-05'
                   '_prefix=Pose-MobileNetV4HybridLarge/pose_net_epoch_31.pth')


class LegacyPoseNetOdometry:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, config: Path = REPO_ROOT / 'vo/config.yaml'):
        with open(config) as f:
            cfg = yaml.safe_load(f)
        # data_root is unused by predict_pose; RedwoodEvaluator only needs it for relative paths.
        self._net = RedwoodEvaluator(cfg, str(weights), data_root='/')
        self._prev: Frame | None = None

    def reset(self, camera: PinholeCamera) -> None:
        self._prev = None

    def step(self, frame: Frame) -> PoseEstimate | None:
        prev, self._prev = self._prev, frame
        if prev is None:
            return None
        t0 = time.perf_counter()
        dT = self._net.predict_pose(str(prev.image_path), str(frame.image_path))
        return PoseEstimate(frame.index, frame.timestamp, dT,
                            timings_ms={'total': (time.perf_counter() - t0) * 1e3})
