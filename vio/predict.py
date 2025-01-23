import tensorflow as tf
import pyvista as pv
import numpy as np
import yaml
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.monodepth2 import DispNet, PoseImuNet
from utils.kalman_filter import SimpleEKF
from eval import EvalTrajectory, euler_to_rotation_matrix, pose_vector_to_transform, pose_axis_angle_vec2mat

import pyvista as pv
import numpy as np


def draw_camera_model(plotter, world_pose, scale=0.2, name_prefix="camera"):
    """
    Draw a camera model in the scene using PyVista.

    Parameters:
    - plotter (pv.Plotter): PyVista plotter instance to draw the camera.
    - world_pose (np.ndarray): 4x4 transformation matrix for the camera.
    - scale (float): Scale factor for the camera model size.
    - name_prefix (str): Prefix for naming the camera elements in the plot.

    Returns:
    - None
    """
    # Camera center (translation) in world coordinates
    cam_center = world_pose[:3, 3]

    # Camera axes in world coordinates
    x_axis = world_pose[:3, 0] * scale  # Red: X-axis (right)
    y_axis = world_pose[:3, 1] * scale  # Green: Y-axis (up)
    z_axis = world_pose[:3, 2] * scale  # Blue: Z-axis (forward)

    # Convert to numpy arrays with correct shape
    cam_center = np.array([cam_center])  # Shape: (1, 3)
    x_axis = np.array([x_axis])          # Shape: (1, 3)
    y_axis = np.array([y_axis])          # Shape: (1, 3)
    z_axis = np.array([z_axis])          # Shape: (1, 3)

    # Define the frustum for the camera (pyramid shape)
    frustum_vertices = np.array([
        [0, 0, 0],  # Camera center
        [1, 1, 2],  # Top-right corner of the near plane
        [-1, 1, 2],  # Top-left corner of the near plane
        [-1, -1, 2],  # Bottom-left corner of the near plane
        [1, -1, 2],  # Bottom-right corner of the near plane
    ]) * scale

    # Transform frustum vertices to world coordinates
    frustum_vertices_h = np.c_[frustum_vertices, np.ones(len(frustum_vertices))]  # Homogeneous
    frustum_vertices_world = (world_pose @ frustum_vertices_h.T).T[:, :3]

    # Define the frustum edges (camera center to corners, and edges between corners)
    frustum_edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Edges between corners
    ]

    # Create PyVista line representation of the frustum
    line_cells = []
    for edge in frustum_edges:
        line_cells.append(2)  # Two points per line
        line_cells.extend(edge)

    frustum_lines = pv.PolyData(frustum_vertices_world)
    frustum_lines.lines = np.array(line_cells)

    # Add the frustum to the plotter
    plotter.add_mesh(frustum_lines, color="cyan", line_width=2, name=f"{name_prefix}_frustum")

    # Draw camera axes
    plotter.add_arrows(cam_center, x_axis, color="red", name=f"{name_prefix}_x_axis")
    plotter.add_arrows(cam_center, y_axis, color="green", name=f"{name_prefix}_y_axis")
    plotter.add_arrows(cam_center, z_axis, color="blue", name=f"{name_prefix}_z_axis")

    # Optionally, add a sphere to represent the camera center
    plotter.add_mesh(pv.Sphere(radius=scale * 0.1, center=cam_center[0]), color="yellow", name=f"{name_prefix}_center")

if __name__ == '__main__':
    from dataset.data_loader import DataLoader
    from tqdm import tqdm

    with open('./vio/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    with tf.device('/GPU:0'):
        config['Train']['batch_size'] = 1
        num_source = config['Train']['num_source']
        image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        batch_size = config['Train']['batch_size']

        depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='disp_resnet')
        depth_net(tf.random.normal((1, *image_shape, 3)))
        depth_net.load_weights('./weights/imu_test_dataV2_poseNet_dataAug/depth_net_epoch_49_model.h5')

        pose_net = PoseImuNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
        posenet_input_shape = [(batch_size, *image_shape, 6),
                                (batch_size, config['Train']['imu_seq_len'], 6)]
        pose_net.build(posenet_input_shape)
        pose_net.load_weights('./weights/imu_test_dataV2_poseNet_dataAug/pose_net_epoch_49_model.h5')

        eval_tool = EvalTrajectory(depth_model=depth_net, pose_model=pose_net, config=config)

        data_loader = DataLoader(config=config)
        test_tqdm = tqdm(data_loader.test_dataset, total=data_loader.num_test_samples)

        # init conditions
        init_pose = np.eye(4, dtype=np.float32)
        world_pose = init_pose.copy()
        # x, y, z축 - 1미터씩 이동
        world_pose[:3, 3] -= np.array([0, 2.0, 0])
        
        trajectory_points = [world_pose[:3, 3].copy()]  # 궤적 저장

        # PyVista Plotter 초기화
        plotter = pv.Plotter()
        plotter.show_axes()
        plotter.add_axes_at_origin()

        # world_pose로부터 바닥 중심 계산
        world_center = init_pose[:3, 3]  # init_pose 중심

        # 바닥 그리드 타일 생성
        grid_size = 40  # 전체 그리드 크기 (40x40m)
        tile_size = 1   # 각 타일 크기 (1x1m)

        # Plane 생성 (world_center 기준으로 배치)
        plane = pv.Plane(
            center=(world_center[0], world_center[1], world_center[2]),  # world_pose 중심에 맞춤
            direction=(0, 1, 0),  # 법선 벡터 (z축을 기준으로 평면 생성)
            i_size=grid_size,  # x축 크기
            j_size=grid_size,  # y축 크기
            i_resolution=grid_size // tile_size,  # x축 타일 개수
            j_resolution=grid_size // tile_size,  # y축 타일 개수
        )

        # Plane에 체스보드 패턴 추가
        num_cells = (grid_size // tile_size) ** 2
        colors = np.zeros(num_cells, dtype=int)
        for i in range(grid_size // tile_size):
            for j in range(grid_size // tile_size):
                if (i + j) % 2 == 0:
                    colors[i * (grid_size // tile_size) + j] = 1  # 흰색 타일
        plane.cell_data["tile_color"] = colors

        # 바닥에 추가
        plotter.add_mesh(
            plane,
            scalars="tile_color",
            show_edges=True,  # 타일 경계선 표시
            cmap=["gray", "white"],  # 타일 색상
        )

        # plotter.show() 대신, 인터랙티브 업데이트 준비
        # plotter.open_gif("visualization.gif")  # 원하면 결과를 GIF로 저장
        plotter.open_movie("visualization.mp4", framerate=24)  # 원하면 결과를 MP4로 저장
        plotter.show(auto_close=False, interactive_update=True)  # 창 열기, 닫히지 않음

        camera_cloud = None
        trajectory_line = None
        camera_model = None

        for idx, (ref_images, target_image, imus, intrinsics) in enumerate(test_tqdm):
            left_images = ref_images[:, :num_source] # [B, num_source, H, W, 3]
            right_images = ref_images[:, num_source:] # [B, num_source, H, W, 3]

            left_imus = imus[:, :num_source] # [B, num_source, 6]
            right_imus = imus[:, num_source:] # [B, num_source, 6]

            left_image = left_images[:, 0] # [B, H, W, 3]
            right_image = right_images[:, 0] # [B, H, W, 3]

            left_imu = left_imus[:, 0] # [B, 6]
            right_imu = right_imus[:, 0] # [B, 6]

            intrinsic = intrinsics[0]
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

            disp_raw = depth_net(target_image, training=False)

            depth_map = eval_tool.disp_to_depth(disp=disp_raw[0],
                                            min_depth=config['Train']['min_depth'],
                                            max_depth=config['Train']['max_depth']) # [B, H, W]
            depth_map = depth_map[0].numpy() # [H, W]

            input_images = tf.concat([left_image, target_image], axis=3)
            pose = pose_net([input_images, left_imu], training=False) # [B, 6]
            pred_transform = pose_axis_angle_vec2mat(pose, invert=True)[0] # [4, 4]
            
            world_pose = world_pose @ pred_transform
            world_pose = world_pose.numpy()

            cam_center = world_pose[:3, 3]
            trajectory_points.append(cam_center.copy())

            # (u, v) 좌표 생성
            us, vs = np.meshgrid(range(image_shape[1]), range(image_shape[0]))  # (W, H)
            us = us.reshape(-1)
            vs = vs.reshape(-1)

            # Z 값 (depth 값)
            zs = depth_map.flatten()  # Z = depth

            # 3D 포인트 계산
            xs = (us - cx) / fx * zs
            ys = (vs - cy) / fy * zs
            points_cam = np.stack([xs, ys, zs, np.ones_like(zs)], axis=1)  # shape (N,4)

            # RGB 색상 추출
            denornalized_target = data_loader.denormalize_image(target_image[0]).numpy()  # [H, W, 3]
            rgb_image = (denornalized_target).astype(np.uint8)  # [H, W, 3], uint8 형태로 변환
            rgb_flattened = rgb_image.reshape(-1, 3)  # [N, 3], N = H*W

            # PyVista 포인트 클라우드로 변환
            points_world = (world_pose @ points_cam.T).T[:, :3]  # shape (N, 3)
            point_cloud = pv.PolyData(points_world)  # 포인트 클라우드 생성
            point_cloud["rgb"] = rgb_flattened  # RGB 색상 추가

            # PyVista에 추가
            if camera_cloud is None:
                camera_cloud = plotter.add_mesh(
                    point_cloud, scalars="rgb", rgb=True, point_size=1
                )
            else:
                camera_cloud.mapper.SetInputData(point_cloud)  # 데이터 갱신

            # (7) 카메라 궤적 라인 갱신
            traj_np = np.array(trajectory_points)

            n_pts = len(traj_np)
            line_cells = np.vstack([np.full(n_pts-1, 2),
                                    np.arange(n_pts-1),
                                    np.arange(1, n_pts)]).T.flatten()
            trajectory_pv = pv.PolyData(traj_np)
            trajectory_pv.lines = line_cells

            if trajectory_line is None:
                trajectory_line = plotter.add_mesh(trajectory_pv,
                                                color='red',
                                                line_width=2,
                                                name='trajectory')
            else:
                trajectory_line.mapper.SetInputData(trajectory_pv)
                # trajectory_line.mapper.SetInputData(trajectory_pv) (버전 호환성에 따라)

            # (8) 카메라 축(좌표계) 등 시각화
            camera_axes_length = 0.2
            # 기준 축 벡터들
            axes_local = np.array([
                [0, 0, 0, 1],
                [camera_axes_length, 0, 0, 1],
                [0, camera_axes_length, 0, 1],
                [0, 0, camera_axes_length, 1]
            ]).T  # shape (4, 4)
            axes_world = (world_pose @ axes_local).T[:, :3]  # shape (4, 3)
            
            # draw camera model
            draw_camera_model(plotter, world_pose, scale=0.5, name_prefix="camera")

            # VIO 카메라 중심 및 축 계산
            cam_center = world_pose[:3, 3]  # VIO 카메라 중심
            cam_forward = world_pose[:3, 2]  # Z축 방향 (전방)
            cam_up = world_pose[:3, 1]       # Y축 방향 (위쪽)

            # 렌더링용 카메라 위치 설정
            offset_distance_z = 5.0  # Z축으로 뒤쪽으로 이동
            offset_distance_y = -1.5  # Y축으로 위쪽으로 이동
            render_camera_position = cam_center - cam_forward * offset_distance_z + cam_up * offset_distance_y

            # `cam_up` 벡터 검증 및 보정
            # `cam_forward`와 항상 직교하도록 `cam_up` 재계산
            cam_right = np.cross(cam_up, cam_forward)  # X축 방향 (우측)
            cam_up_corrected = np.cross(cam_forward, cam_right)  # 직교 Y축 방향 (위쪽)
            cam_up_corrected /= np.linalg.norm(cam_up_corrected)  # 정규화

            # 렌더링용 카메라 설정
            plotter.camera.position = render_camera_position  # 렌더링 카메라 위치
            plotter.camera.focal_point = cam_center  # 렌더링 카메라가 VIO 카메라를 바라봄
            plotter.camera.up = -cam_up_corrected  # 수정된 카메라 위쪽 방향

            # animation
            plotter.render()
            plotter.update(force_redraw=True)
            plotter.write_frame()  # GIF 프레임 저장
        plotter.close()