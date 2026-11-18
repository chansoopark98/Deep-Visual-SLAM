"""target-metric items 03/04 on per-pair (t-1, t) relative poses.

03: per pair |‖t_pred‖ − ‖t_gt‖| / ‖t_gt‖ · 100 (meters, magnitude only; 0 % when ‖t_gt‖ <= 1e-6 m),
    mean over pairs per scene, then mean over scenes. Target <= 2.96 %.
04: per pair |Δ| of ZYX roll/pitch/yaw of the relative rotation (deg, wrapped), mean over pairs and axes
    per scene, then mean over scenes. Target <= 3.4 deg.
"""
import numpy as np

from dvslam.geometry.se3 import euler_zyx, rotation_angle_deg

POS_MAPE_TARGET_PCT = 2.96
ROT_ABS_TARGET_DEG = 3.4
ZERO_MOTION_M = 1e-6


def position_error_pct(t_gt: np.ndarray, t_pred: np.ndarray) -> float:
    gt_norm = np.linalg.norm(t_gt)
    pred_norm = np.linalg.norm(t_pred)
    return abs(pred_norm - gt_norm) / gt_norm * 100 if gt_norm > ZERO_MOTION_M else 0


def rpy_abs_err_deg(euler_gt: np.ndarray, euler_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.degrees((euler_gt - euler_pred + np.pi) % (2 * np.pi) - np.pi))


def pair_metrics(T_gt_rel: np.ndarray, T_pred_rel: np.ndarray) -> dict:
    """Every per-pair quantity the target-metric CSV reports."""
    gt_t, pred_t = T_gt_rel[:3, 3], T_pred_rel[:3, 3]
    gt_euler, pred_euler = euler_zyx(T_gt_rel[:3, :3]), euler_zyx(T_pred_rel[:3, :3])
    gt_rot_norm, pred_rot_norm = np.linalg.norm(gt_euler), np.linalg.norm(pred_euler)
    return {
        'gt_t': gt_t, 'pred_t': pred_t, 'gt_euler': gt_euler, 'pred_euler': pred_euler,
        'pos_diff': gt_t - pred_t,
        'rot_diff': gt_euler - pred_euler,
        'pos_error': np.linalg.norm(gt_t - pred_t),
        'rot_error_deg': rotation_angle_deg(T_gt_rel[:3, :3] @ T_pred_rel[:3, :3].T),
        'pos_error_percent': position_error_pct(gt_t, pred_t),
        'rot_error_percent': abs(pred_rot_norm - gt_rot_norm) / gt_rot_norm * 100 if gt_rot_norm > 1e-6 else 0,
        'rpy_abs_err_deg': rpy_abs_err_deg(gt_euler, pred_euler),
    }


def scene_summary(scene: str, pos_error_pct: list, rpy_err_deg: np.ndarray, gt_t: np.ndarray) -> dict:
    """pos_error_pct: per-pair item-03 values; rpy_err_deg: (n, 3); gt_t: (n, 3) GT translations."""
    rpy_err_deg = np.asarray(rpy_err_deg)
    axis_mean = rpy_err_deg.mean(axis=0)
    return {
        'scene': scene,
        'pairs': len(pos_error_pct),
        'pos_mape_pct': float(np.mean(pos_error_pct)),
        'rot_roll_deg': float(axis_mean[0]),
        'rot_pitch_deg': float(axis_mean[1]),
        'rot_yaw_deg': float(axis_mean[2]),
        'rot_rpy_mean_deg': float(rpy_err_deg.mean()),
        'zero_motion_pairs': int((np.linalg.norm(np.asarray(gt_t), axis=1) <= ZERO_MOTION_M).sum()),
    }


def target-metric_items(summaries: list[dict]) -> dict:
    """Mean of per-scene means and PASS/FAIL for items 03 and 04."""
    pos = float(np.mean([s['pos_mape_pct'] for s in summaries]))
    rot = float(np.mean([s['rot_rpy_mean_deg'] for s in summaries]))
    return {'item03_mape_pct': pos, 'item03_pass': pos <= POS_MAPE_TARGET_PCT,
            'item04_rpy_mean_deg': rot, 'item04_pass': rot <= ROT_ABS_TARGET_DEG}
