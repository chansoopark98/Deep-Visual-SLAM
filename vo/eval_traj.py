import matplotlib
matplotlib.use('Agg')
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import yaml
from torch.amp import GradScaler, autocast
# from monodepth_learner import Learner
from learner_func import (
    disp_to_depth, 
    BackprojectDepth, 
    Project3D, 
    transformation_from_parameters,
    get_smooth_loss,
    SSIM
)

import pytransform3d.camera as pc

class EvalTrajectory():
    def __init__(self,
                 depth_model: nn.Module,
                 pose_model: nn.Module,
                 config: dict,
                 device: str = 'cuda'):
        self.depth_net = depth_model
        self.pose_net = pose_model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.batch_size = config['Train']['batch_size']
        self.image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_scales = config['Train']['num_scale']

        self.pred_pose_list = []
        self.intrinsic_list = []

    def _predict_poses(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        
        # Left -> Target (invert=True)
        concat_left_tgt = torch.cat([sample['source_left'], sample['target_image']], dim=1)
        axisangle_left, translation_left = self.pose_net(concat_left_tgt)

        # Target -> Right (invert=False)
        concat_tgt_right = torch.cat([sample['target_image'], sample['source_right']], dim=1)
        axisangle_right, translation_right = self.pose_net(concat_tgt_right)  # [B, 1, 3]

        outputs[("axisangle", 0, -1)] = axisangle_left
        outputs[("translation", 0, -1)] = translation_left
        outputs[("axisangle", 0, 1)] = axisangle_right
        outputs[("translation", 0, 1)] = translation_right


        # Invert the matrix if the frame id is negative
        outputs[("cam_T_cam", 0, -1)] = transformation_from_parameters(
            axisangle_left[:, 0], translation_left[:, 0], invert=True)
        outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
            axisangle_right[:, 0], translation_right[:, 0], invert=False)

        return outputs

    @torch.no_grad()
    def update_state(self, sample: Dict[str, torch.Tensor]):
        with autocast('cuda', dtype=torch.float16):
            outputs = self._predict_poses(sample)
            
            batch_poses = outputs[("cam_T_cam", 0, 1)]

            batch_size = batch_poses.shape[0]
            for i in range(batch_size):
                pose = batch_poses[i].detach().cpu().numpy()
                intrinsic = sample[("K", 0)].detach().cpu().numpy()[i]
                self.pred_pose_list.append(pose)
                self.intrinsic_list.append(intrinsic)

    def depth_to_pointcloud(self, depth, pose, intrinsic):
        K = intrinsic
        T_global = pose

        H, W = depth.shape
        xs = np.arange(W)
        ys = np.arange(H)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # 2D -> 3D (카메라 좌표계)
        u = grid_x.reshape(-1)
        v = grid_y.reshape(-1)
        z = depth.reshape(-1)
        
        # 유효 깊이만 사용
        valid_mask = (z > 0)
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]

        ones = np.ones_like(u)
        uv1 = np.stack([u, v, ones], axis=0)  # (3, N)

        # K^-1 @ uv1 -> 정규화 카메라 광선
        K_inv = np.linalg.inv(K)
        rays = K_inv @ uv1  # (3, N)

        # 각 픽셀 = rays * 해당 픽셀의 depth
        points_camera = rays * z  # (3, N)
        
        # 전역 좌표계로 변환
        # (4,4) @ (4, N)
        ones_3d = np.ones((1, points_camera.shape[1]))
        points_camera_h = np.concatenate([points_camera, ones_3d], axis=0)  # (4, N)

        points_world_h = T_global @ points_camera_h  # (4, N)
        points_world = points_world_h[:3, :].T       # (N, 3)
        
        # point cloud 누적 (필요시 샘플링 가능)
        num_points = points_world.shape[0]
        sampled_indices = np.random.choice(num_points, int(num_points * 0.0001), replace=False)
        points_world_sampled = points_world[sampled_indices]

        return points_world_sampled

    def eval_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 전역 좌표계에서의 카메라 위치를 기록할 리스트
        camera_centers = []
        
        # 누적 변환 행렬 (전역 기준) - 첫 프레임을 기준으로 단위행렬
        T_global = np.eye(4)
        camera_centers.append(T_global[:3, 3].copy())

        # 모든 프레임에 대해 누적
        num_frames = len(self.pred_pose_list)
        
        for i in range(num_frames):
            T_local = self.pred_pose_list[i]  # shape: (4,4)

            T_global = T_global @ T_local
            
            intrinsic = self.intrinsic_list[i]
            
            if i % int(self.batch_size * 4) == 0:
                pc.plot_camera(ax,
                               intrinsic,
                               T_global,
                               sensor_size=(self.image_shape[1], self.image_shape[0]),
                               virtual_image_distance=0.2,
                               c='g',
                               strict_check=False)

            # 이번 프레임 카메라 중심(전역 좌표계에서)
            camera_center = T_global[:3, 3]
            camera_centers.append(camera_center.copy())

        # 카메라 궤적 표시
        camera_centers = np.array(camera_centers)  # shape: (num_frames+1, 3)
        ax.plot(
            camera_centers[:, 0],
            camera_centers[:, 1],
            camera_centers[:, 2],
            c='r',
            label='Camera Trajectory'
        )

        # 카메라 진행 방향(간단히 z축 방향 벡터) 표시
        # 마지막 프레임(누적) 기준
        R_final = T_global[:3, :3]
        z_axis = R_final[:, 2]  # 카메라 z축
        origin = T_global[:3, 3]
        arrow_len = 0.5
        arrow_end = origin + z_axis * arrow_len

        ax.quiver(
            origin[0], origin[1], origin[2],
            arrow_end[0]-origin[0],
            arrow_end[1]-origin[1],
            arrow_end[2]-origin[2],
            color='blue', length=arrow_len,
            normalize=False
        )

        # 범례, 축 라벨, 시야 범위 설정
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Visual Odometry: 3D Trajectory and Pointcloud')
        fig.tight_layout()
        
        self.clear_state()

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Load from buffer
        pil_image = Image.open(buf)
        image_array = np.array(pil_image)  # [H, W, 4] RGBA or [H, W, 3] RGB
        
        # Convert RGBA to RGB if needed
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]  # Remove alpha channel
        
        # Return [H, W, 3] for visualization
        return image_array
        
    def clear_state(self):
        self.pred_pose_list.clear()
        self.intrinsic_list.clear()