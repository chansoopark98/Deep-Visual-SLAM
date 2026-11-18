"""Shared helpers for the Phase 4 scripts: tracker C construction, GT pairs, item-03 per-pair errors."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.eval import diagnostics as D  # noqa: E402
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from scripts.cv_prior import cv_filter  # noqa: E402

CV_ALPHA = 0.6
SCENES = redwood.SCENES


def make_c(seed: int, **kw):
    """Tracker C exactly as the G1' validation runs: klt_window ba=false kf_align=false, perturbed prior."""
    from dvslam.odometry.klt_window import KLTWindowOdometry
    return KLTWindowOdometry(ba=False, kf_align=False, prior_seed=seed, **kw)


def load_val(scene: str):
    return redwood.load_sequence(scene, 'validation')


def gt_pairs_t(seq) -> np.ndarray:
    return np.stack([relative_pose(seq.T_wc[t - 1], seq.T_wc[t])[:3, 3] for t in range(1, len(seq))])


def item03(inc: np.ndarray, gt_t: np.ndarray, cv: bool = True) -> np.ndarray:
    """Per-pair item-03 % for raw increments (n,4,4); the causal CV prior (alpha 0.6) is applied first."""
    p = (cv_filter(inc, CV_ALPHA) if cv else inc)[:, :3, 3]
    return D._pct_errors(gt_t, p)


def pooled(per_scene: list[np.ndarray]) -> tuple[float, float, float]:
    """Mean of scene means with 95 % block-bootstrap CI."""
    return D.block_bootstrap_ci(per_scene)
