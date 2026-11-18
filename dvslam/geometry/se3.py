"""SE(3) helpers shared by the evaluators and trackers. All poses are 4x4 float64, camera-to-world (T_wc)."""
import numpy as np


def relative_pose(T_0: np.ndarray, T_1: np.ndarray) -> np.ndarray:
    """Increment from frame 0 to frame 1: inv(T_wc_0) @ T_wc_1 (pose of camera 1 in camera 0's frame)."""
    return np.linalg.inv(T_0) @ T_1


def chain_increments(T_start: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """T_out_t = T_out_{t-1} @ dT_t. Returns (N+1, 4, 4) starting with T_start."""
    out = [np.asarray(T_start, dtype=np.float64)]
    for dT in increments:
        out.append(out[-1] @ dT)
    return np.stack(out)


def euler_zyx(R: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) in radians, ZYX convention (same formula as the 2026 target-metric evaluator)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy >= 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising ||dst - (s R src + t)||^2 for (N, 3) point sets, with the reflection fix."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum(1).mean()
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale and var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
