import matplotlib
matplotlib.use('Agg')
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from monodepth_learner_new import Learner
from utils.projection_utils import pose_axis_angle_vec2mat, pose_vec2mat
import pytransform3d.camera as pc

def euler_to_rotation_matrix(rx, ry, rz):
    """
    간단한 Z-Y-X(roll-pitch-yaw) 순서의 Euler 각을 회전 행렬로 변환한다고 가정한 예시 함수.
    실제 환경에 맞추어 오일러 각 변환 순서를 수정하세요.
    """
    # 각각 회전 행렬
    cosx, sinx = np.cos(rx), np.sin(rx)
    cosy, siny = np.cos(ry), np.sin(ry)
    cosz, sinz = np.cos(rz), np.sin(rz)

    # Rx, Ry, Rz 순으로 곱하는 예시(roll -> pitch -> yaw)
    Rx = np.array([[1,    0,     0],
                   [0,  cosx, -sinx],
                   [0,  sinx,  cosx]])
    Ry = np.array([[ cosy, 0, siny],
                   [   0 , 1,   0 ],
                   [-siny, 0, cosy]])
    Rz = np.array([[ cosz, -sinz, 0],
                   [ sinz,  cosz, 0],
                   [   0 ,    0 , 1]])
    
    # 최종 회전 행렬
    R = Rz @ Ry @ Rx
    return R

def pose_vector_to_transform(pose_vec):
    """
    pose_vec: [tx, ty, tz, rx, ry, rz] 형태라고 가정
    4x4 변환 행렬(T)를 반환.
    """
    tx, ty, tz, rx, ry, rz = pose_vec
    T = np.eye(4)
    R = euler_to_rotation_matrix(rx, ry, rz)
    T[:3, :3] = R
    T[:3, 3]  = [tx, ty, tz]
    return T

class EvalTrajectory(Learner):
    def __init__(self, depth_model: tf.keras.Model,
                 pose_model: tf.keras.Model,
                 config: dict):
        super().__init__(depth_model, pose_model, config)
        # self.model = model
        self.depth_net = depth_model
        self.pose_net = pose_model

        self.batch_size = config['Train']['batch_size']
        self.image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        self.pred_depth_list = []
        self.pred_pose_list = []
        self.intrinsic_list = []

        if self.config['Train']['mode'] in ['axisAngle', 'euler']:
            self.pose_mode = self.config['Train']['mode']
            if self.pose_mode == 'axisAngle':
                self.is_euler = False
            else:
                raise NotImplementedError('Euler angle mode is not supported yet')
        else:
            raise ValueError('Invalid pose mode')

    def update_state(self, ref_images, tgt_image, intrinsic: tf.Tensor):
        right_image = ref_images[1]
        
        disp_raw = self.depth_net(tgt_image, training=False)

        batch_disps = []
        for s in range(self.num_scales):
            scale_h = self.image_shape[0] // (2 ** s)
            scale_w = self.image_shape[1] // (2 ** s)
            scaled_disp = tf.image.resize(disp_raw[s], [scale_h, scale_w], method=tf.image.ResizeMethod.BILINEAR)
            batch_disps.append(scaled_disp)

        cat_right = tf.concat([tgt_image, right_image], axis=3) # [B,H,W,6]

        
        pose_right = self.pose_net(cat_right, training=False)  # [B,6]

        batch_poses = tf.cast(pose_right, tf.float32) 

        # list comprehension으로 변환
        batch_disps = [tf.cast(disp, tf.float32) for disp in batch_disps]
        intrinsic = tf.cast(intrinsic, tf.float32)
        
        batch_depths = self.disp_to_depth(
            disp=batch_disps[0],
            min_depth=self.min_depth,
            max_depth=self.max_depth
        )

        if self.is_euler:
            batch_poses = pose_vec2mat(batch_poses)
        else:
            batch_poses = pose_axis_angle_vec2mat(batch_poses, invert=False)  # shape: (b, 4, 4)

        batch_size = batch_depths.shape[0]

        for i in range(batch_size):
            pred_depth = batch_depths.numpy()[i, :, :, 0]  # shape: (H, W)
            pred_pose = batch_poses[i].numpy() # shape: (4, 4)
            current_intrinsic = intrinsic.numpy()[0]  # shape: (3, 3)
    
            # 리스트에 누적
            self.pred_depth_list.append(pred_depth)
            self.pred_pose_list.append(pred_pose)
            self.intrinsic_list.append(current_intrinsic)

            del pred_depth, pred_pose, current_intrinsic

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

    def eval_plot(self, is_show: bool =False):
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

        # 마지막에 추가
        if is_show:
            plt.show()
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)  # 플롯 객체 닫기
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            buf.close()  # BytesIO 닫기
            return tf.expand_dims(image, 0)
        
    def clear_state(self):
        self.pred_depth_list.clear()
        self.pred_pose_list.clear()
        self.intrinsic_list.clear()

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from model.monodepth2 import MonoDepth2Model
    from dataset.data_loader import DataLoader
    from tqdm import tqdm

    with open('./vio/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get GPU configuration and set visible GPUs
    gpu_config = config.get('Experiment', {})
    visible_gpus = gpu_config.get('gpus', [])
    gpu_vram = gpu_config.get('gpu_vram', None)
    gpu_vram_factor = gpu_config.get('gpu_vram_factor', None)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            if visible_gpus:
                selected_gpus = [gpus[i] for i in visible_gpus]
                tf.config.set_visible_devices(selected_gpus, 'GPU')
            else:
                print("No GPUs specified in config. Using all available GPUs.")
                selected_gpus = gpus
            
            if gpu_vram and gpu_vram_factor:
                for gpu in selected_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_vram * gpu_vram_factor)]
                    )
            
            print(f"Using GPUs: {selected_gpus}")
        except RuntimeError as e:
            print(f"Error during GPU configuration: {e}")
    else:
        print('No GPU devices found')
        raise SystemExit
    
    # Load model
    config['Train']['batch_size'] = 8
    batch_size = config['Train']['batch_size']
    model = MonoDepth2Model(image_shape=(config['Train']['img_h'], config['Train']['img_w']),
                                    batch_size=config['Train']['batch_size'])
    model_input_shape = (config['Train']['batch_size'], config['Train']['img_h'], config['Train']['img_w'], 9)
    model.build(model_input_shape)
    _ = model(tf.random.normal(model_input_shape), training=False)
    # model.load_weights('./weights/resnet_mixedPrecision_Step=2/epoch_35_model.h5')

    eval_tool = EvalTrajectory(model=model, config=config)

    data_loader = DataLoader(config=config)

    valid_tqdm = tqdm(data_loader.valid_dataset, total=data_loader.num_valid_samples)
    valid_tqdm.set_description('Validation || ')
    for idx, (ref_images, target_image, imus, intrinsic) in enumerate(valid_tqdm):
        eval_tool.update_state(ref_images, target_image, intrinsic)
        # if idx > 100:
        #     break
    
    eval_tool.eval_plot()
        