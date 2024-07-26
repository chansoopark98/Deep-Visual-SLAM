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
import gc

class DataUtils(object):    
    def extract_position_and_orientation(self, matrix):
        position = matrix[:3, 3]
        orientation = matrix[:3, :3] @ np.array([1, 0, 0])
        return position, orientation

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

class TspDataHandler(DataUtils):
    def __init__(self, root: str,
                 target_image_shape: tuple = (1080, 1920),
                 original_image_shape: tuple = (1080, 1920),
                 imu_frequency: int = 11) -> None:
        super(TspDataHandler, self).__init__()
        self.root = root
        self.target_image_shape = target_image_shape
        self.original_image_shape = original_image_shape
        self.imu_freq = imu_frequency
        
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
            img = np.array(img, dtype=np.uint8)
            return img 
    
    def load_and_process_depth(self, depth_path):
        depth = np.load(depth_path)
        depth = np.expand_dims(depth, axis=-1)
        depth = np.nan_to_num(depth)
        depth = depth.astype(np.float32)
        return depth
        
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
        sampled_imu_list = np.array(sampled_imu_list)
        
        return sampled_imu_list

    def zed_to_kitti_format(self, sampled_imu: np.ndarray, is_rad: bool = True) -> np.ndarray:
        """
        sampled_imu(np.ndarray): shape=(samples, seq, 6)
        data: (timestamp, acc_x, acc_y, acc_z, velo_x, velo_y, velo_z)

        return kitti_formatted_imu(np.ndarray): shape=(samples, seq, 6)
        data: (timestamp, acc_x, acc_y, acc_z, angular_vel_x, angular_vel_y, angular_vel_z)
        """
        gravity = 9.81
        
        # Deg/s to rad/s conversion factor
        if is_rad:
            factor = np.pi / 180.0 # Degree to Radian
        else:
            factor = 1.

        # Assuming sampled_imu has shape (samples, seq, 7) where
        # 0: timestamp, 1-3: linear_acceleration, 4-6: angular_velocity
        kitti_formatted_imu = np.zeros_like(sampled_imu)

        # Copying the acceleration
        kitti_formatted_imu[:, :, 0:3] = sampled_imu[:, :, 0:3]

        # Converting angular velocity from deg/s to rad/s
        kitti_formatted_imu[:, :, 3:6] = sampled_imu[:, :, 3:6] * factor

        kitti_formatted_imu[:, :, 2] = kitti_formatted_imu[:, :, 2] - gravity

        return kitti_formatted_imu
        
    def parsing_files(self, file_list):
        # load data paths
        rgb_path = glob.glob(os.path.join(file_list, 'rgb') + '/*.{0}'.format(self.img_data_type))
        rgb_path.sort()

        depth_path = glob.glob(os.path.join(file_list, 'depth') + '/*.{0}'.format(self.depth_data_type))
        depth_path.sort()

        imu_path = os.path.join(file_list, 'sensor/zed_imu.csv')
        intrinsics_path = os.path.join(file_list, 'sensor/intrinsics.npy')
        zed_path = os.path.join(file_list, 'sensor/zed_pose.csv')

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
        imu_npy = self.zed_to_kitti_format(sampled_imu=imu_npy,
                                           is_rad=False)

        # Ensure the lengths are the same as tracker_npy.shape[0]
        expected_length = zed_npy.shape[0]

        if len(rgb_path) != expected_length:
            print(f'Adjusting image list size from {len(rgb_path)} to {expected_length}')
            rgb_path = rgb_path[:expected_length]
        
        if imu_npy.shape[0] != expected_length:
            print(f'Adjusting IMU data size from {imu_npy.shape[0]} to {expected_length}')
            imu_npy = imu_npy[:expected_length]
        
        if zed_npy.shape[0] != expected_length:
            print(f'Adjusting ZED data size from {zed_npy.shape[0]} to {expected_length}')
            zed_npy = zed_npy[:expected_length]

        # IMU npy to list
        imu_list = [imu_npy[i] for i in range(imu_npy.shape[0])]

        return rgb_path, depth_path, imu_list, intrinsics_npy, zed_npy    

    def get_sync_file(self, file_list):
        rgb_paths, depth_paths, imus, intrinsics, zeds = self.parsing_files(file_list)

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
        
        return rgb_paths, depth_paths, imus, intrinsics, rel_camera_pose, global_camera_pose

    def vio_generator(self):
        for file_list in self.file_lists:
            rgb_paths, depth_paths, _, intrinsic, _, _ = self.get_sync_file(file_list)

            # image list to npy
            intrinsic = tf.cast(intrinsic, tf.float32)

            img_freq = 1
            for idx in range(img_freq, len(rgb_paths) - img_freq):
                source_left = rgb_paths[idx - img_freq]
                source_right = rgb_paths[idx + img_freq]
                target_image = rgb_paths[idx]
                target_depth = depth_paths[idx]

                target_depth = self.load_and_process_depth(target_depth)
                target_depth = tf.convert_to_tensor(target_depth, dtype=tf.float32)

                source_left = tf.io.read_file(source_left)
                source_right = tf.io.read_file(source_right)
                target_image = tf.io.read_file(target_image)

                source_left = tf.io.decode_image(source_left, dtype=tf.uint8)
                source_right = tf.io.decode_image(source_right, dtype=tf.uint8)
                target_image = tf.io.decode_image(target_image, dtype=tf.uint8)
            

                yield {
                'source_left': source_left,
                'source_right': source_right,
                'target_image': target_image,
                'target_depth': target_depth,
                'intrinsic': intrinsic
                }
                del source_left
                del source_right
                del target_image                
            del intrinsic
            del rgb_paths
            gc.collect()

    def create_vio_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self.vio_generator,
            output_signature={
                'source_left': tf.TensorSpec(shape=(*self.target_image_shape, 3), dtype=tf.uint8),
                'source_right': tf.TensorSpec(shape=(*self.target_image_shape, 3), dtype=tf.uint8),
                'target_image': tf.TensorSpec(shape=(*self.target_image_shape, 3), dtype=tf.uint8),
                'target_depth': tf.TensorSpec(shape=(*self.target_image_shape, 1), dtype=tf.float32),
                'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32)
            }
        )
        return dataset
    
if __name__ == '__main__':
    root_path = './vio/data/raw/tspxr_capture/train/'
    debug = True
    
    dataset = TspDataHandler(root=root_path,
                   target_image_shape=(720, 1280),
                   original_image_shape=(1080, 1920),
                   imu_frequency=11)
    if debug:
        dataset.get_sync_all_files()
    else:
        dataset.create_vio_dataset()