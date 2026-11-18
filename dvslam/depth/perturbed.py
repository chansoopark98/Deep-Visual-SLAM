"""Perturbed GT depth prior for tracker ranking (plan v3 item 1.0, critic C1).

depth_t = GT_t · (1 + b) · exp(n_t),  n_t ~ N(0, jitter²) per frame. Ranking default: b = 2 %, jitter = 3 %.
n_t is a deterministic function of (seed, frame index), so re-running a frame gives the same prior.
"""
import numpy as np

from dvslam.data.redwood import load_depth


class PerturbedGTDepth:
    def __init__(self, bias: float = 0.02, jitter_logsd: float = 0.03, seed: int = 0):
        self.bias, self.jitter, self.seed = bias, jitter_logsd, seed

    def scale(self, index: int) -> float:
        n = np.random.default_rng([self.seed, index]).standard_normal()
        return (1 + self.bias) * float(np.exp(self.jitter * n))

    def predict(self, frame) -> tuple[np.ndarray, None]:
        d = load_depth(frame.depth_path)
        return d * np.float32(self.scale(frame.index)), None
