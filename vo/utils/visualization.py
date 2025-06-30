import pyvista as pv
import numpy as np

class Visualizer:
    def __init__(self,
                 window_size: tuple = (1280, 480),
                 draw_plane: bool = True,
                 is_record: bool = True,
                 video_fps: int = 24,
                 video_name: str = "visualization.mp4"
                 ) -> None:
        self.window_size = window_size
        self.plotter = pv.Plotter(window_size=self.window_size)
        self.plotter.show_axes()
        self.plotter.add_axes_at_origin()
        self.plotter.show(auto_close=False, interactive_update=True)

        self.draw_plane = draw_plane
        self.is_record = is_record

        if draw_plane:
            self._draw_plane(world_center=np.array([0, 0, 0]), grid_size=10, tile_size=1)
        if is_record:
            self.plotter.open_movie(video_name, framerate=video_fps)

        # initialize camera cloud (plotter.add_mesh)
        self.camera_cloud = None
        dummy_point = np.array([[0, 0, 0]])
        dummy_data = pv.PolyData(dummy_point)

        dummy_data["rgb"] = np.array([[0, 0, 0]])
        self.camera_cloud = self.plotter.add_mesh(
            dummy_data, scalars="rgb", rgb=True, point_size=1
        )

        # initialize trajectory
        init_pose = np.eye(4, dtype=np.float32)
        self.world_pose = init_pose.copy()
        self.world_pose[:3, 3] -= np.array([0, 2.0, 0])
        self.trajectory = [self.world_pose[:3, 3].copy()]  # 궤적 저장

        # floor poisition
        self.world_center = init_pose[:3, 3]  # init_pose 중심

    def _draw_plane(self, world_center: np.ndarray, grid_size: int, tile_size: int = 1) -> None:
        """
        Draw a grid plane in the scene using PyVista.
        
        Parameters:
        - world_center (np.ndarray): 3D world coordinates for the center of the plane.
        - grid_size (int): Size of the grid plane (in meters).
        - tile_size (int): Size of each grid tile (in meters).

        Returns:
        - None
        """
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
        self.plotter.add_mesh(
            plane,
            scalars="tile_color",
            show_edges=True,  # 타일 경계선 표시
            cmap=["gray", "white"],  # 타일 색상
        )

    def draw_camera_model(self, world_pose, scale=0.2, name_prefix="camera") -> None:
        """
        Draw a camera model in the scene using PyVista.

        Parameters:
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
        self.plotter.add_mesh(frustum_lines, color="cyan", line_width=2, name=f"{name_prefix}_frustum")

        # Draw camera axes
        self.plotter.add_arrows(cam_center, x_axis, color="red", name=f"{name_prefix}_x_axis")
        self.plotter.add_arrows(cam_center, y_axis, color="green", name=f"{name_prefix}_y_axis")
        self.plotter.add_arrows(cam_center, z_axis, color="blue", name=f"{name_prefix}_z_axis")

        # Optionally, add a sphere to represent the camera center
        self.plotter.add_mesh(pv.Sphere(radius=scale * 0.1, center=cam_center[0]), color="yellow", name=f"{name_prefix}_center")

    
    def draw_pointcloud(self, rgb, depth_map, intrinsic, world_pose) -> pv.PolyData:
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

        us, vs = np.meshgrid(range(rgb.shape[1]), range(rgb.shape[0]))
        us = us.reshape(-1)
        vs = vs.reshape(-1)
        
        # Z 값 (depth 값)
        zs = depth_map.flatten()
        
        # 3D 포인트 계산
        xs = (us - cx) / fx * zs
        ys = (vs - cy) / fy * zs
        
        # SLAM에서 PyVista 좌표계로 변환 행렬 (예: Y축 반전)
        slam_to_pyvista = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Y축 반전 (아래쪽→위쪽)
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 카메라 좌표계 포인트
        points_cam = np.stack([xs, ys, zs, np.ones_like(zs)], axis=1)  # (N,4)
        
        # RGB 색상 추출
        rgb_image = (rgb).astype(np.uint8)
        rgb_flattened = rgb_image.reshape(-1, 3)
        
        # 세계 좌표계로 변환 후 PyVista 좌표계로 변환
        points_world = (world_pose @ points_cam.T).T  # SLAM 세계 좌표계
        points_pyvista = (slam_to_pyvista @ points_world.T).T[:, :3]  # PyVista 좌표계
        
        # 포인트 클라우드 생성
        point_cloud = pv.PolyData(points_pyvista)
        point_cloud["rgb"] = rgb_flattened
        
        self.camera_cloud.mapper.SetInputData(point_cloud)

    def draw_trajectory(self, world_pose: np.ndarray, color: str = "red", line_width: int = 2) -> None:
        """
        Update and visualize the camera's trajectory.

        Parameters:
        - update_point (np.ndarray): A new point (3D coordinates) to be added to the trajectory.
        - color (str): Color of the trajectory line.
        - line_width (int): Width of the trajectory line.

        Returns:
        - None
        """

        update_point = world_pose[:3, 3]  # Extract the camera center from the world pose
        # Append the new point to the trajectory
        self.trajectory.append(update_point)

        # Convert trajectory to a numpy array
        traj_np = np.array(self.trajectory)

        # Update the PolyData object for trajectory visualization
        if len(self.trajectory) > 1:
            # Create line connections between consecutive points
            n_pts = len(traj_np)
            line_cells = np.hstack([np.full((n_pts - 1, 1), 2),  # Line segment size
                                    np.arange(n_pts - 1).reshape(-1, 1),
                                    np.arange(1, n_pts).reshape(-1, 1)]).flatten()

            # Create or update the PolyData for trajectory
            trajectory_pv = pv.PolyData(traj_np)
            trajectory_pv.lines = line_cells

            # Add or update the trajectory mesh
            if not hasattr(self, "trajectory_line") or self.trajectory_line is None:
                self.trajectory_line = self.plotter.add_mesh(
                    trajectory_pv, color=color, line_width=line_width, name="trajectory"
                )
            else:
                self.trajectory_line.mapper.SetInputData(trajectory_pv)

            # Refresh the plotter
            self.plotter.render()

    def set_camera_poisition(self, world_pose):
        # 카메라 중심 위치
        cam_center = world_pose[:3, 3]
        
        # 카메라 방향 벡터 (Monodepth2 좌표계에서)
        cam_forward = world_pose[:3, 2]  # 카메라가 바라보는 방향 (Z축)
        cam_up = -world_pose[:3, 1]      # 카메라의 위쪽 방향 (Y축 반전)
        
        # 가상 카메라를 실제 카메라 뒤쪽에 위치시킴
        offset_distance_z = 5.0  # 카메라 뒤쪽으로의 거리
        offset_distance_y = 2.0  # 카메라 위쪽으로의 거리
        
        # 카메라 궤적을 따라가는 가상 카메라 위치 계산
        # cam_forward 방향의 반대로 이동 (뒤쪽)
        render_camera_position = cam_center - cam_forward * offset_distance_z + cam_up * offset_distance_y
        
        # 직교성 유지를 위한 벡터 계산
        cam_right = np.cross(cam_up, -cam_forward)  # 우측 방향
        cam_up_corrected = np.cross(-cam_forward, cam_right)  # 수정된 위쪽 방향
        cam_up_corrected /= np.linalg.norm(cam_up_corrected)  # 정규화
        
        # 렌더링 카메라 설정
        self.plotter.camera.position = render_camera_position
        self.plotter.camera.focal_point = cam_center  # 실제 카메라 위치를 바라봄
        self.plotter.camera.up = cam_up_corrected

    def render(self) -> None:
        self.plotter.render()
        self.plotter.update(force_redraw=True)
        if self.is_record:
            self.plotter.write_frame()
    
    def close(self):
        self.plotter.close()

if __name__ == '__main__':
    plotter = Visualizer(draw_plane=True, is_record=True, video_fps=24, video_name="visualization")