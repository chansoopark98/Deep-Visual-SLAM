import os
import glob
import numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import time
import pandas as pd

class DataUtils(object):    
    def extract_position_and_orientation(self, matrix):
        position = matrix[:3, 3]
        orientation = matrix[:3, :3] @ np.array([1, 0, 0])
        return position, orientation
    
    def visualize_traj(self, matrix_list: list, limit: int):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = [], [], []

        for matrix in matrix_list:
            position, orientation = self.extract_position_and_orientation(matrix)
            x.append(position[0])
            y.append(position[1])
            z.append(position[2])
        ax.plot(x, y, z, color='blue')

        # 축 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Transformation Visualization')
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.view_init(elev=20., azim=120)  # 뷰 각도 조정
        plt.show()

    def transform_matrix_to_pose(self, matrix):
        # 변위 벡터 (x, y, z) 추출
        position = matrix[:3, 3]
        
        # 회전 행렬 추출
        rotation_matrix = matrix[:3, :3]
        
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=False)
        
        return position, euler_angles

    def convert_transform_list(self, transform_list):
        camera_poses = []

        for matrix in transform_list:
            position, euler_angles = self.transform_matrix_to_pose(matrix)
            # Meter to mm
            # position *= 1000.
            camera_poses.append([*position, *euler_angles])
        
        return camera_poses

    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion to a 3x3 rotation matrix."""
        rotation = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        return rotation.as_matrix()

    def create_transformation_matrix(self, position, quaternion, divide_factor):
        """Create a 4x4 transformation matrix from position and quaternion."""
        # Convert position from mm to meters
        position = np.array(position) / divide_factor
        # Get the rotation matrix from the quaternion
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        # Create the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position
        return transformation_matrix

    def calculate_relative_pose(self, global_transform_matrix: list):
        relative_poses = []
        
        for i in range(len(global_transform_matrix) - 1):
            # 현재 행렬 및 +1 행렬
            current_transform = global_transform_matrix[i]
            next_transform = global_transform_matrix[i + 1]
            
            # Relative 포즈 계산
            relative_transform = np.dot(np.linalg.inv(current_transform), next_transform)
            relative_poses.append(relative_transform)
        
        return relative_poses
    
    def visualize_global_pose_animation(self, image_list, poses):
        """
        이미지와 포즈 (x, y, z, roll, pitch, yaw)를 받아 애니메이션을 생성합니다.

        Parameters:
        - image_list: 이미지 리스트
        - poses: 각 이미지 당 포즈 리스트 [(x, y, z, roll, pitch, yaw)]
        """

        fig = plt.figure()
        ax_image = fig.add_subplot(121)
        ax_pose = fig.add_subplot(122, projection='3d')

        # 전체 경로를 플로팅하기 위해 모든 포즈 좌표를 계산
        all_positions = np.array([pose[:3] for pose in poses])
        all_rotations = np.array([pose[3:] for pose in poses])

        def update(frame):
            ax_image.clear()
            ax_pose.clear()

            # 이미지 표시
            img = image_list[frame]
            ax_image.imshow(img)
            ax_image.axis('off')

            # 전체 경로 시각화
            ax_pose.plot(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], color='blue', linestyle='-', marker='o')

            # 현재 프레임의 포즈 강조
            current_position = all_positions[frame]
            current_rotation = all_rotations[frame]
            print(current_position, current_rotation)
            ax_pose.scatter(current_position[0], current_position[1], current_position[2], color='red', s=100, label='Current Frame')

            # 축 설정
            ax_pose.set_xlim(all_positions[:, 0].min() - 0.1, all_positions[:, 0].max() + 0.1)
            ax_pose.set_ylim(all_positions[:, 1].min() - 0.1, all_positions[:, 1].max() + 0.1)
            ax_pose.set_zlim(all_positions[:, 2].min() - 0.1, all_positions[:, 2].max() + 0.1)
            ax_pose.set_xlabel('X')
            ax_pose.set_ylabel('Y')
            ax_pose.set_zlabel('Z')
            # ax_pose.view_init(elev=20., azim=30)
            ax_pose.legend()

        ani = animation.FuncAnimation(fig, update, frames=len(image_list), repeat=True)
        plt.show()

class TspDataHandler(DataUtils):
    def __init__(self, root: str,
                 target_image_shape: tuple = (720, 1280),
                 original_image_shape: tuple = (1080, 1920),
                 imu_frequency: int = 11,
                 data_mode: dict = {
                     'image': True,
                     'depth': True,
                     'imu': True,
                     'rel_pose': True,
                     'global_pose': True,
                     'intrinsic': True,
                 }) -> None:
        super(TspDataHandler, self).__init__()
        self.root = root
        self.target_image_shape = target_image_shape
        self.original_image_shape = original_image_shape
        self.imu_freq = imu_frequency
        self.data_mode = data_mode
        self.img_data_type = 'jpg'
        self.depth_data_type = 'npy'
        self.data_len = 0.
        self.file_lists = self.parsing_file_path()
    
    def parsing_file_path(self) -> list:
        if type(self.root) == bytes:
            self.root = self.root.decode('utf-8')
        self.file_path = os.path.join(self.root, '*')
        
        file_lists = glob.glob(self.file_path)
        
        for file_list in file_lists:
            rgb_path = glob.glob(os.path.join(file_list, 'rgb') + '/*.{0}'.format(self.img_data_type))
            self.data_len += len(rgb_path)
        return file_lists
    
    def load_and_process_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.resize(size=(self.target_image_shape[1], self.target_image_shape[0]))
            return [np.array(img, dtype=np.uint8)][0]
    
    def rescale_intrinsic_matrix(self, K):
        """
        기존 intrinsic matrix를 새로운 이미지 해상도에 맞춰 조정하는 함수

        :param K: 3x3 intrinsic matrix (numpy array)
        :param original_width: 원래 이미지의 가로 해상도
        :param original_height: 원래 이미지의 세로 해상도
        :param new_width: 새로운 이미지의 가로 해상도
        :param new_height: 새로운 이미지의 세로 해상도
        :return: new 3x3 intrinsic matrix (numpy array)
        """
        
        # 가로 및 세로 스케일 비율 계산
        original_width = float(self.original_image_shape[1])
        original_height = float(self.original_image_shape[0])
        new_width = float(self.target_image_shape[1])
        new_height = float(self.target_image_shape[0])

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # 기존 intrinsic matrix에서 focal length와 principal point 추출
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        # 새로운 intrinsic matrix 계산
        K_new = np.array([
                [f_x * scale_x, 0., c_x * scale_x],
                [0., f_y * scale_y, c_y * scale_y],
                [0., 0., 1.]
            ])

        return K_new
    
    def imu_parsing_by_timestamp(self, imu_data: np.ndarray, pose_data: np.ndarray) -> np.ndarray:
        """
        imu_data(np.ndarray) : shape=(imu_samples, data) data= (timestamp, acc_x, acc_y, acc_z, velo_x, velo_y, velo_z)
        pose_data(np.ndarray) : shape=(pos_samples, data) data= (timestamp, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)

        return sampled_imu(np.ndarray) : shape=(pos_samples, seq, data)
        """
        sampled_imu_list = []

        for i in range(len(pose_data) - 1):
            t_start = pose_data[i, 0]
            t_end = pose_data[i + 1, 0]

            # 해당 구간의 imu 데이터를 가져오기
            mask = (imu_data[:, 0] >= t_start) & (imu_data[:, 0] < t_end)
            segment = imu_data[mask]

            # 시퀀스 길이를 11로 조정
            if len(segment) < self.imu_freq:
                if len(segment) > 1:
                    indices = np.linspace(0, len(segment) - 1, self.imu_freq)
                    resampled_segment = np.array([segment[int(idx)] for idx in indices])
                else:
                    resampled_segment = np.tile(segment[0], (self.imu_freq, 1))
            elif len(segment) > self.imu_freq:
                indices = np.round(np.linspace(0, len(segment) - 1, self.imu_freq)).astype(int)
                resampled_segment = segment[indices]
            else:
                resampled_segment = segment

            sampled_imu_list.append(resampled_segment)

        # 모든 구간의 데이터를 하나로 합침
        sampled_imu = np.array(sampled_imu_list)
        
        return sampled_imu

    def zed_to_kitti_format(self, sampled_imu: np.ndarray) -> np.ndarray:
        """
        sampled_imu(np.ndarray): shape=(samples, seq, 6)
        data: (timestamp, acc_x, acc_y, acc_z, velo_x, velo_y, velo_z)

        return kitti_formatted_imu(np.ndarray): shape=(samples, seq, 6)
        data: (timestamp, acc_x, acc_y, acc_z, angular_vel_x, angular_vel_y, angular_vel_z)
        """
        # Deg/s to rad/s conversion factor
        deg_to_rad = np.pi / 180.0

        # Assuming sampled_imu has shape (samples, seq, 7) where
        # 0: timestamp, 1-3: linear_acceleration, 4-6: angular_velocity
        kitti_formatted_imu = np.zeros_like(sampled_imu)

        # Copying the timestamp
        kitti_formatted_imu[:, :, 0] = sampled_imu[:, :, 0]

        # Converting angular velocity from deg/s to rad/s
        kitti_formatted_imu[:, :, 3:6] = sampled_imu[:, :, 3:6] * deg_to_rad

        return kitti_formatted_imu
        
    def parsing_files(self, file_list):
        # load data paths
        rgb_path = glob.glob(os.path.join(file_list, 'rgb') + '/*.{0}'.format(self.img_data_type))
        rgb_path.sort()
        imu_path = os.path.join(file_list, 'sensor/zed_imu.csv')
        intrinsics_path = os.path.join(file_list, 'sensor/intrinsics.npy')
        zed_path = os.path.join(file_list, 'sensor/zed_pose.csv')

        # load rgb images
        image_list = [self.load_and_process_image(rgb_sample_path) for rgb_sample_path in rgb_path]

        # load npy files(imu/intrinsic/tracker pose/zed pose)
        imu_npy = np.loadtxt(imu_path, delimiter=',', skiprows=1)
        intrinsics_npy = np.load(intrinsics_path)
        
        zed_npy = np.loadtxt(zed_path, delimiter=',', skiprows=1)

        # Extract files
        imu_npy = self.imu_parsing_by_timestamp(imu_data=imu_npy,
                                      pose_data=zed_npy)
        imu_npy = imu_npy[:, :, -6:] # acc, gyro
        zed_npy = zed_npy[:, 1:]

        # ZED IMU data convert to KITTI format
        imu_npy = self.zed_to_kitti_format(sampled_imu=imu_npy)

        # Ensure the lengths are the same as tracker_npy.shape[0]
        expected_length = zed_npy.shape[0]

        if len(image_list) != expected_length:
            print(f'Adjusting image list size from {len(image_list)} to {expected_length}')
            image_list = image_list[:expected_length]
        
        if imu_npy.shape[0] != expected_length:
            print(f'Adjusting IMU data size from {imu_npy.shape[0]} to {expected_length}')
            imu_npy = imu_npy[:expected_length]
        
        if zed_npy.shape[0] != expected_length:
            print(f'Adjusting ZED data size from {zed_npy.shape[0]} to {expected_length}')
            zed_npy = zed_npy[:expected_length]

        # IMU npy to list
        imu_list = [imu_npy[i] for i in range(imu_npy.shape[0])]

        return image_list, imu_list, intrinsics_npy, zed_npy    

    def get_sync_file(self, file_list):
        images, imus, intrinsics, zeds = self.parsing_files(file_list)

        intrinsics = self.rescale_intrinsic_matrix(K=intrinsics)
            
        global_mat_list = []

        sync_data = zeds
        # trackers convert to transform matrix
        for i in range(len(sync_data)):
            data_point = sync_data[i]
            transformation_matrix = self.create_transformation_matrix(data_point[:3],
                                                                        data_point[3:],
                                                                        1.0)
            global_mat_list.append(transformation_matrix)
        
        reference_pose_inverse = np.linalg.inv(global_mat_list[0])

        # 모든 시퀀스의 변환 행렬 조정
        adjusted_tracker_mat_list = []
        for matrix in global_mat_list:
            adjusted_matrix = reference_pose_inverse @ matrix
            adjusted_tracker_mat_list.append(adjusted_matrix)
        
        relative_tracker_pose = self.calculate_relative_pose(global_transform_matrix=adjusted_tracker_mat_list)
        
        rel_camera_pose = self.convert_transform_list(transform_list=relative_tracker_pose)
        global_camera_pose = self.convert_transform_list(transform_list=global_mat_list)
        
        return images, imus, intrinsics, rel_camera_pose, global_camera_pose

    def get_sync_all_files(self):
        for file_list in self.file_lists:
            images, imus, intrinsic, rel_camera_pose, global_camera_pose = self.get_sync_file(file_list)

            # image list to npy
            images = np.array(images, np.uint8)
            intrinsic = tf.cast(intrinsic, tf.float32)

            for idx in range(images.shape[0] - 1):
                source_image = images[idx]
                target_image = images[idx + 1]
                imu = imus[idx]
                rel_pose = rel_camera_pose[idx]
                source_global_pose = global_camera_pose[idx]
                target_global_pose = global_camera_pose[idx + 1]
                
                batch_source_image = tf.convert_to_tensor(source_image, tf.uint8)
                batch_target_image = tf.convert_to_tensor(target_image, tf.uint8)
                batch_imu = tf.convert_to_tensor(imu, tf.float32)
                batch_rel_pose = tf.convert_to_tensor(rel_pose, tf.float32)
                batch_source_pose = tf.convert_to_tensor(source_global_pose, tf.float32)
                batch_target_pose = tf.convert_to_tensor(target_global_pose, tf.float32)

                # yield {
                # 'image': image_array,
                # 'depth': depth_array,
                # 'imu': imu_array,
                # 'rel_pose': rel_pose_array,
                # 'global_pose': global_pose_array,
                # 'intrinsic': intrinsic
                # }

    def vio_generator(self):
        for file_list in self.file_lists:
            raw_images, _, intrinsic, _, _ = self.get_sync_file(file_list)

            # image list to npy
            images = np.array(raw_images, np.uint8)
            intrinsic = tf.cast(intrinsic, tf.float32)

            img_freq = 1
            for idx in range(img_freq, images.shape[0] - img_freq):
                source_left = images[idx - img_freq]
                source_right = images[idx + img_freq]
                target_image = images[idx]
                
                batch_source_left = tf.convert_to_tensor(source_left, tf.uint8)
                batch_source_right = tf.convert_to_tensor(source_right, tf.uint8)
                batch_target_image = tf.convert_to_tensor(target_image, tf.uint8)

                yield {
                'source_left': batch_source_left,
                'source_right': batch_source_right,
                'target_image': batch_target_image,
                'intrinsic': intrinsic
                }

    def create_vio_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self.vio_generator,
            output_signature={
                'source_left': tf.TensorSpec(shape=(*self.target_image_shape, 3), dtype=tf.uint8),
                'source_right': tf.TensorSpec(shape=(*self.target_image_shape, 3), dtype=tf.uint8),
                'target_image': tf.TensorSpec(shape=(*self.target_image_shape, 3), dtype=tf.uint8),
                'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32)
            }
        )
        return dataset
    
if __name__ == '__main__':
    root_path = '../data/raw/tspxr_capture/train/'
    debug = True
    dataset = TspDataHandler(root=root_path,
                             imu_frequency=11)
    if debug:
        dataset.get_sync_all_files()
    else:
        dataset.create_vio_dataset()
    # print(dataset.__len__)
    # # dataset = TspxrCaptureDataset(root='../data/tspxr_capture/train/') 
    # # dataset = dataset.batch(8)
    # # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # for batch in dataset.take(100):
    #     print()
    #     # print("Image batch shape:", batch['image'].shape)
    #     # print("IMU batch shape:", batch['imu'].shape)
    #     # print("Rel pose batch shape:", batch['rel_pose'].shape)
    #     # print("Global pose batch shape:", batch['global_pose'].shape)