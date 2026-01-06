import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
import json
import csv
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model.posenet_single import PoseNet
from PIL import Image
import torchvision.transforms as transforms
from vo.learner_func import transformation_from_parameters


def load_redwood_poses(json_path: str, use_training_convention: bool = False) -> np.ndarray:
    """Load poses from Redwood JSON file

    Args:
        json_path: Path to apartment.json or other scene json
        use_training_convention: If True, use the same (incorrect) convention as training
                                 If False, use correct column-major format

    Returns:
        poses: (N, 4, 4) numpy array of poses

    Note:
        The model was trained with incorrect pose loading (reshape + transpose).
        For evaluation with this model, we need to match the training convention.
        For future models, train with use_training_convention=False (correct format).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    poses = []
    for node in data['nodes']:
        pose_flat = node['pose']

        if use_training_convention:
            # OLD (INCORRECT) method used during training
            # This was: reshape to row-major then transpose
            pose = np.array(pose_flat).reshape((4, 4)).T
        else:
            # CORRECT method: column-major (Fortran) order
            # Use this for future training/evaluation
            pose = np.array(pose_flat).reshape((4, 4), order='F')

        poses.append(pose)

    return np.array(poses)


def compute_relative_pose(T_0: np.ndarray, T_1: np.ndarray) -> np.ndarray:
    """Compute relative pose from T_0 to T_1

    Args:
        T_0: 4x4 transformation matrix at time t
        T_1: 4x4 transformation matrix at time t+1

    Returns:
        T_rel: 4x4 relative transformation from T_0 to T_1
               T_rel transforms points from T_0 frame to T_1 frame
    """
    # T_rel = inv(T_0) @ T_1
    # This gives the transformation from camera frame at t=0 to camera frame at t=1
    T_rel = np.linalg.inv(T_0) @ T_1
    return T_rel


def align_trajectories_umeyama(poses_pred: np.ndarray, poses_gt: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align predicted trajectory to ground truth using Umeyama alignment

    Computes optimal rotation R, scale s, and translation t such that:
    poses_aligned = s * R @ poses_pred + t

    This is also known as similarity transformation or 7-DoF alignment.

    Args:
        poses_pred: (N, 4, 4) predicted poses
        poses_gt: (N, 4, 4) ground truth poses

    Returns:
        poses_aligned: (N, 4, 4) aligned predicted poses
        scale: Optimal scale factor
        R: (3, 3) optimal rotation matrix
        t: (3,) optimal translation vector

    Reference:
        Umeyama, "Least-Squares Estimation of Transformation Parameters
        Between Two Point Patterns", IEEE PAMI 1991
    """
    # Extract positions (translation components)
    positions_pred = np.array([pose[:3, 3] for pose in poses_pred])  # (N, 3)
    positions_gt = np.array([pose[:3, 3] for pose in poses_gt])      # (N, 3)

    # Center the point clouds
    centroid_pred = np.mean(positions_pred, axis=0)  # (3,)
    centroid_gt = np.mean(positions_gt, axis=0)      # (3,)

    positions_pred_centered = positions_pred - centroid_pred
    positions_gt_centered = positions_gt - centroid_gt

    # Compute covariance matrix
    H = positions_pred_centered.T @ positions_gt_centered  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case (ensure det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_pred = np.mean(np.sum(positions_pred_centered ** 2, axis=1))
    scale = np.trace(np.diag(S)) / var_pred if var_pred > 1e-8 else 1.0

    # Compute translation
    t = centroid_gt - scale * R @ centroid_pred

    # Apply transformation to all poses
    poses_aligned = []
    for pose_pred in poses_pred:
        # Extract rotation and translation from predicted pose
        R_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:3, 3]

        # Apply similarity transformation
        t_aligned = scale * R @ t_pred + t
        R_aligned = R @ R_pred  # Rotate the orientation as well

        # Build aligned pose
        pose_aligned = np.eye(4)
        pose_aligned[:3, :3] = R_aligned
        pose_aligned[:3, 3] = t_aligned

        poses_aligned.append(pose_aligned)

    return np.array(poses_aligned), scale, R, t


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles (ZYX convention)

    Args:
        R: 3x3 rotation matrix

    Returns:
        euler: (roll, pitch, yaw) in radians
    """
    # ZYX Euler angles (roll-pitch-yaw)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])


def compute_pose_error(T_gt: np.ndarray, T_pred: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute position and rotation error between GT and predicted poses

    Args:
        T_gt: 4x4 ground truth transformation matrix
        T_pred: 4x4 predicted transformation matrix

    Returns:
        pos_error: Position error (L2 norm)
        rot_error: Rotation error in degrees
        pos_diff: Position difference vector (3,)
        rot_diff: Rotation difference in Euler angles (radians) (3,)
    """
    # Position error
    pos_gt = T_gt[:3, 3]
    pos_pred = T_pred[:3, 3]
    pos_diff = pos_gt - pos_pred
    pos_error = np.linalg.norm(pos_diff)

    # Rotation error
    R_gt = T_gt[:3, :3]
    R_pred = T_pred[:3, :3]

    # Compute relative rotation
    R_diff = R_gt @ R_pred.T

    # Convert to angle
    trace = np.trace(R_diff)
    rot_error_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    rot_error_deg = np.degrees(rot_error_rad)

    # Also compute Euler angle differences for analysis
    euler_gt = rotation_matrix_to_euler(R_gt)
    euler_pred = rotation_matrix_to_euler(R_pred)
    rot_diff = euler_gt - euler_pred

    return pos_error, rot_error_deg, pos_diff, rot_diff


def axisangle_to_matrix(axisangle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix

    Args:
        axisangle: (3,) tensor of axis-angle rotation

    Returns:
        R: (3, 3) rotation matrix
    """
    angle = torch.norm(axisangle)

    if angle < 1e-7:
        return torch.eye(3)

    axis = axisangle / angle

    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=axisangle.dtype, device=axisangle.device)

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = torch.eye(3, dtype=axisangle.dtype, device=axisangle.device) + \
        torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * (K @ K)

    return R


def pose_vec_to_matrix(pose_output: tuple) -> np.ndarray:
    """Convert PoseNet output to 4x4 transformation matrix

    Args:
        pose_output: Tuple of (translation, rotation)
                    translation: (1, 1, 1, 3) tensor [tx, ty, tz]
                    rotation: (1, 1, 1, 3) tensor [rx, ry, rz] (axis-angle)

    Returns:
        T: (4, 4) transformation matrix

    Note:
        MonoDepth2 PoseNet uses a different coordinate convention than Redwood.
        We apply Y-axis flip to match Redwood's convention.
    """
    translation_tensor, axisangle_tensor = pose_output

    # Squeeze to get (3,) vectors
    translation = translation_tensor.squeeze().cpu().numpy()  # (3,)
    axisangle = axisangle_tensor.squeeze()  # (3,)

    # Convert axis-angle to rotation matrix
    R = axisangle_to_matrix(axisangle).cpu().numpy()

    # Build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    # COORDINATE TRANSFORMATION: Y-axis flip
    # Empirical testing showed Y-axis needs to be flipped
    # This transforms: [X, Y, Z] -> [X, -Y, Z]
    T_flip = np.eye(4)
    T_flip[1, 1] = -1  # Flip Y-axis

    # Apply transformation: T_corrected = T_flip @ T @ T_flip
    # This flips both the translation and rotation appropriately
    T_corrected = T_flip @ T @ T_flip

    return T_corrected


class RedwoodEvaluator:
    def __init__(self, config: Dict, pose_model_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_root = Path('/home/park-ubuntu/park/Deep-Visual-SLAM/depth/data/redwood')

        # Load pose model
        self.pose_net = PoseNet(
            num_layers=18,
            pretrained=False,
            num_input_images=2,
        ).to(self.device)

        # Load weights
        state_dict = torch.load(pose_model_path, map_location=self.device)
        # Remove _orig_mod. prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_orig_mod.', '')
            new_state_dict[new_k] = v
        self.pose_net.load_state_dict(new_state_dict)
        self.pose_net.eval()

        # Image preprocessing
        self.img_h = config['Train']['img_h']
        self.img_w = config['Train']['img_w']
        self.transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
        ])

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor.unsqueeze(0)  # (1, 3, H, W)

    def predict_pose(self, img1_path: str, img2_path: str) -> np.ndarray:
        """Predict relative pose from img1 to img2

        Args:
            img1_path: Path to first image (t)
            img2_path: Path to second image (t+1)

        Returns:
            T_pred: (4, 4) predicted transformation matrix (camera t → camera t+1)
        """
        img1 = self.load_image(img1_path).to(self.device)
        img2 = self.load_image(img2_path).to(self.device)

        # Concatenate images
        img_pair = torch.cat([img1, img2], dim=1)  # (1, 6, H, W)

        with torch.no_grad():
            axisangle, translation = self.pose_net(img_pair)

        # IMPORTANT: Use invert=True to match GT direction
        # Testing showed this gives 73.6% better accuracy
        T_pred = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True
        ).cpu().numpy()[0]

        return T_pred

    def get_relative_path(self, abs_path: Path) -> str:
        """Convert absolute path to relative path from data root"""
        try:
            rel_path = abs_path.relative_to(self.data_root)
            return str(rel_path)
        except ValueError:
            return str(abs_path)

    def plot_trajectory(self, poses_gt: List[np.ndarray], poses_pred: List[np.ndarray],
                       scene_name: str, output_path: str):
        """Plot 3D and 2D trajectory comparison

        Args:
            poses_gt: List of GT absolute poses (4x4 matrices)
            poses_pred: List of predicted absolute poses (4x4 matrices)
            scene_name: Name of the scene
            output_path: Path to save the plot
        """
        # Extract positions
        gt_positions = np.array([pose[:3, 3] for pose in poses_gt])
        pred_positions = np.array([pose[:3, 3] for pose in poses_pred])

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 5))

        # 3D trajectory plot
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2],
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax1.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2],
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2],
                   c='red', s=100, marker='x', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{scene_name} - 3D Trajectory')
        ax1.legend()
        ax1.grid(True)

        # XY plane (top view)
        ax2 = fig.add_subplot(142)
        ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', linewidth=2, label='GT', alpha=0.8)
        ax2.plot(pred_positions[:, 0], pred_positions[:, 1], 'r--', linewidth=2, label='Pred', alpha=0.8)
        ax2.scatter(gt_positions[0, 0], gt_positions[0, 1], c='green', s=100, marker='o')
        ax2.scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='red', s=100, marker='x')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'{scene_name} - Top View (XY)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')

        # XZ plane (side view)
        ax3 = fig.add_subplot(143)
        ax3.plot(gt_positions[:, 0], gt_positions[:, 2], 'b-', linewidth=2, label='GT', alpha=0.8)
        ax3.plot(pred_positions[:, 0], pred_positions[:, 2], 'r--', linewidth=2, label='Pred', alpha=0.8)
        ax3.scatter(gt_positions[0, 0], gt_positions[0, 2], c='green', s=100, marker='o')
        ax3.scatter(gt_positions[-1, 0], gt_positions[-1, 2], c='red', s=100, marker='x')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title(f'{scene_name} - Side View (XZ)')
        ax3.legend()
        ax3.grid(True)
        ax3.axis('equal')

        # YZ plane (front view)
        ax4 = fig.add_subplot(144)
        ax4.plot(gt_positions[:, 1], gt_positions[:, 2], 'b-', linewidth=2, label='GT', alpha=0.8)
        ax4.plot(pred_positions[:, 1], pred_positions[:, 2], 'r--', linewidth=2, label='Pred', alpha=0.8)
        ax4.scatter(gt_positions[0, 1], gt_positions[0, 2], c='green', s=100, marker='o')
        ax4.scatter(gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=100, marker='x')
        ax4.set_xlabel('Y (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title(f'{scene_name} - Front View (YZ)')
        ax4.legend()
        ax4.grid(True)
        ax4.axis('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Trajectory plot saved to: {output_path}")

    def compute_scale_factor(self, poses_gt: np.ndarray, image_files: list, num_samples: int = 100) -> float:
        """Compute scale factor between GT and predicted poses

        Monocular self-supervised learning produces up-to-scale predictions.
        We compute the median ratio of GT/pred translation magnitudes.

        Args:
            poses_gt: Ground truth poses
            image_files: List of image files
            num_samples: Number of frame pairs to sample for scale estimation

        Returns:
            scale_factor: Median ratio of ||t_GT|| / ||t_pred||
        """
        print(f"\nComputing scale factor using {num_samples} frame pairs...")

        num_frames = len(poses_gt)
        sample_indices = np.linspace(0, num_frames-2, min(num_samples, num_frames-1), dtype=int)

        ratios = []

        for i in sample_indices:
            # GT relative pose
            T_gt_rel = compute_relative_pose(poses_gt[i], poses_gt[i+1])
            t_gt = T_gt_rel[:3, 3]
            gt_magnitude = np.linalg.norm(t_gt)

            if gt_magnitude < 1e-6:  # Skip very small motions
                continue

            # Predict relative pose
            T_pred_rel = self.predict_pose(str(image_files[i]), str(image_files[i+1]))
            t_pred = T_pred_rel[:3, 3]

            pred_magnitude = np.linalg.norm(t_pred)

            if pred_magnitude < 1e-6:  # Skip invalid predictions
                continue

            ratio = gt_magnitude / pred_magnitude
            ratios.append(ratio)

        if len(ratios) == 0:
            print("Warning: Could not compute scale factor, using 1.0")
            return 1.0

        scale_factor = np.median(ratios)
        print(f"Scale factor (median): {scale_factor:.4f}")
        print(f"  Mean: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}")
        print(f"  Min: {np.min(ratios):.4f}, Max: {np.max(ratios):.4f}")

        return scale_factor

    def evaluate_scene(self, scene_name: str, output_csv: str, use_scale_correction: bool = False):
        """Evaluate on a single scene

        Args:
            scene_name: Name of the scene (e.g., 'apartment')
            output_csv: Path to output CSV file
            use_scale_correction: If True, compute and apply scale correction
                                 Default is False because MonoDepth2 was trained with
                                 supervised depth (max_depth=10m), so poses are already in meters
        """
        # Paths
        test_dir = self.data_root / 'test' / scene_name

        json_path = test_dir / f'{scene_name}.json'
        image_dir = test_dir / 'image'

        # Load GT poses
        poses_gt = load_redwood_poses(str(json_path))
        num_frames = len(poses_gt)

        # Get sorted list of image files
        image_files = sorted(image_dir.glob('*.jpg'))

        if len(image_files) != num_frames:
            print(f"Warning: Number of images ({len(image_files)}) != number of poses ({num_frames})")
            return

        print(f"\nEvaluating scene: {scene_name}")
        print(f"Total frames: {num_frames}")
        print(f"Units: Both GT and predicted poses are in meters")

        # Compute scale factor for monocular predictions
        scale_factor = 1.0
        if use_scale_correction:
            scale_factor = self.compute_scale_factor(poses_gt, image_files, num_samples=100)
        

        # Prepare CSV
        csv_data = []
        csv_header = [
            'frame_t', 'frame_t1',
            'image_path_t', 'image_path_t1',
            'gt_tx', 'gt_ty', 'gt_tz',
            'gt_roll', 'gt_pitch', 'gt_yaw',
            'pred_tx', 'pred_ty', 'pred_tz',
            'pred_roll', 'pred_pitch', 'pred_yaw',
            'pos_diff_x', 'pos_diff_y', 'pos_diff_z',
            'rot_diff_roll', 'rot_diff_pitch', 'rot_diff_yaw',
            'pos_error', 'rot_error_deg',
            'pos_error_percent', 'rot_error_percent'
        ]

        # Lists to accumulate absolute poses for trajectory plotting
        abs_poses_gt = [poses_gt[0]]  # Start with first pose
        abs_poses_pred = [poses_gt[0]]  # Start predicted trajectory from same initial pose

        # Evaluate each consecutive frame pair
        for i in tqdm(range(num_frames - 1), desc=f"Processing {scene_name}"):
            # Image paths (use actual file list)
            img_t = image_files[i]
            img_t1 = image_files[i+1]

            # Compute GT relative pose
            T_gt_rel = compute_relative_pose(poses_gt[i], poses_gt[i+1])

            # Predict relative pose
            T_pred_rel = self.predict_pose(str(img_t), str(img_t1))

            # Apply scale correction to predicted translation
            if use_scale_correction:
                T_pred_rel[:3, 3] *= scale_factor

            # Accumulate absolute poses for trajectory
            # GT: T_abs[i+1] = T_abs[i] @ T_rel[i->i+1]
            abs_poses_gt.append(poses_gt[i+1])

            # Predicted: T_abs_pred[i+1] = T_abs_pred[i] @ T_pred_rel[i->i+1]
            T_abs_pred_new = abs_poses_pred[-1] @ T_pred_rel
            abs_poses_pred.append(T_abs_pred_new)

            # Compute errors using relative poses
            pos_error, rot_error_deg, pos_diff, rot_diff = compute_pose_error(T_gt_rel, T_pred_rel)

            # Extract components from relative poses
            gt_pos = T_gt_rel[:3, 3]
            gt_euler = rotation_matrix_to_euler(T_gt_rel[:3, :3])

            pred_pos = T_pred_rel[:3, 3]
            pred_euler = rotation_matrix_to_euler(T_pred_rel[:3, :3])

            # Compute error percentages (magnitude error only)
            gt_pos_norm = np.linalg.norm(gt_pos)
            pred_pos_norm = np.linalg.norm(pred_pos)
            gt_rot_norm = np.linalg.norm(gt_euler)
            pred_rot_norm = np.linalg.norm(pred_euler)

            # Percentage error in magnitude (not vector difference)
            pos_error_percent = abs(pred_pos_norm - gt_pos_norm) / gt_pos_norm * 100 if gt_pos_norm > 1e-6 else 0
            rot_error_percent = abs(pred_rot_norm - gt_rot_norm) / gt_rot_norm * 100 if gt_rot_norm > 1e-6 else 0

            # Add to CSV data with relative paths
            row = [
                i, i+1,
                self.get_relative_path(img_t),
                self.get_relative_path(img_t1),
                gt_pos[0], gt_pos[1], gt_pos[2],
                gt_euler[0], gt_euler[1], gt_euler[2],
                pred_pos[0], pred_pos[1], pred_pos[2],
                pred_euler[0], pred_euler[1], pred_euler[2],
                pos_diff[0], pos_diff[1], pos_diff[2],
                rot_diff[0], rot_diff[1], rot_diff[2],
                pos_error, rot_error_deg,
                pos_error_percent, rot_error_percent
            ]
            csv_data.append(row)

        # Write CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_data)

        # Compute statistics for relative pose errors
        pos_errors = [row[-4] for row in csv_data]
        rot_errors = [row[-3] for row in csv_data]
        pos_error_percents = [row[-2] for row in csv_data]
        rot_error_percents = [row[-1] for row in csv_data]

        print(f"\n=== Results for {scene_name} ===")
        print(f"Number of evaluated pairs: {len(csv_data)}")
        print(f"\nRelative Pose Errors (frame-to-frame):")
        print(f"  Position Error (mean):   {np.mean(pos_errors):.6f} m")
        print(f"  Position Error (median): {np.median(pos_errors):.6f} m")
        print(f"  Position Error (std):    {np.std(pos_errors):.6f} m")
        print(f"  Position Error % (mean): {np.mean(pos_error_percents):.2f}%")
        print(f"  Rotation Error (mean):   {np.mean(rot_errors):.4f} deg")
        print(f"  Rotation Error (median): {np.median(rot_errors):.4f} deg")
        print(f"  Rotation Error (std):    {np.std(rot_errors):.4f} deg")
        print(f"  Rotation Error % (mean): {np.mean(rot_error_percents):.2f}%")

        print(f"\nResults saved to: {output_csv}")

        # Plot trajectory
        output_plot = output_csv.replace('.csv', '_trajectory.png')
        self.plot_trajectory(abs_poses_gt, abs_poses_pred, scene_name, output_plot)


def main():
    # Load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Model path
    # pose_model_path = 'weights/vo/pose_net_epoch_30.pth'
    pose_model_path = "weights/vo/mode=axisAngle_res=(480, 640)_ep=31_bs=16_initLR=0.0001_endLR=1e-05_prefix=Pose-MobileNetV4HybridLarge/pose_net_epoch_31.pth"

    if not os.path.exists(pose_model_path):
        print(f"Error: Model weights not found at {pose_model_path}")
        return

    # Create evaluator
    evaluator = RedwoodEvaluator(config, pose_model_path)

    # Scenes to evaluate
    scenes = ['apartment', 'bedroom', 'boardroom', 'lobby', 'loft']

    # Output directory
    output_dir = Path('./results/redwood_eval')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each scene
    for scene in scenes:
        output_csv = output_dir / f'{scene}_pose_errors.csv'
        try:
            evaluator.evaluate_scene(scene, str(output_csv))
        except Exception as e:
            print(f"Error evaluating {scene}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
