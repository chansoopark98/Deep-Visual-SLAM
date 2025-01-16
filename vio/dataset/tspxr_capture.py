import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

def pose_to_transformation_matrix(pose):
    # 위치 추출
    position = np.array([pose['x'], pose['y'], pose['z']])
    
    # 회전을 쿼터니언에서 회전 행렬로 변환
    rotation = R.from_quat([pose['quat_x'], pose['quat_y'], pose['quat_z'], pose['quat_w']])
    rotation_matrix = rotation.as_matrix()  # 3x3 회전 행렬
    
    # 4x4 변환 행렬 생성
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    
    return transformation_matrix

def calculate_relative_pose_using_transformation(current_pose, next_pose):
    current_matrix = pose_to_transformation_matrix(current_pose)
    next_matrix = pose_to_transformation_matrix(next_pose)
    
    relative_matrix = np.linalg.inv(current_matrix) @ next_matrix

    relative_position = relative_matrix[:3, 3]
    
    relative_rotation_matrix = relative_matrix[:3, :3]
    relative_rotation = R.from_matrix(relative_rotation_matrix).as_euler('ZXY', degrees=False) 
    rel_pose = np.concatenate([relative_position, relative_rotation], axis=-1)
    return rel_pose

def transformation_matrix_to_pose(matrix):
    # 위치 추출
    position = matrix[:3, 3]
    
    # 회전 행렬을 쿼터니언으로 변환
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quat = rotation.as_quat()  # [x, y, z, w] 형태
    
    # 포즈 딕셔너리로 반환
    pose = {
        'x': position[0],
        'y': position[1],
        'z': position[2],
        'quat_x': quat[0],
        'quat_y': quat[1],
        'quat_z': quat[2],
        'quat_w': quat[3]
    }
    return pose

def test_traj(pred_rel_poses, gt_rel_poses):
    global_matrix = np.eye(4)
    global_poses = []
    
    for rel_pose, gt_pose in zip(pred_rel_poses, gt_rel_poses):
        # 상대 변환 행렬 생성
        relative_position = rel_pose[:3] + (gt_pose[:3] - rel_pose[:3]) / 1.25
        relative_rotation = R.from_euler('ZXY', gt_pose[3:], degrees=False)
        relative_matrix = np.eye(4)
        relative_matrix[:3, :3] = relative_rotation.as_matrix()
        relative_matrix[:3, 3] = relative_position
        
        # global_matrix에 상대 변환을 곱해 새로운 global_matrix 갱신
        global_matrix = global_matrix @ relative_matrix
        global_poses.append(global_matrix)
        
    return global_poses
    
def relative_vec_to_global_mat(relative_poses):
    # 초기 global pose를 변환 행렬로 변환
    global_matrix = np.eye(4)
    global_poses = []
    
    # 각 relative pose에 대해 순차적으로 global pose 계산
    for rel_pose in relative_poses:
        # 상대 변환 행렬 생성
        relative_position = rel_pose[:3]
        relative_rotation = R.from_euler('ZXY', rel_pose[3:], degrees=False)
        relative_matrix = np.eye(4)
        relative_matrix[:3, :3] = relative_rotation.as_matrix()
        relative_matrix[:3, 3] = relative_position
        
        # global_matrix에 상대 변환을 곱해 새로운 global_matrix 갱신
        global_matrix = global_matrix @ relative_matrix
        global_poses.append(global_matrix)
        
    return global_poses

class TspxrCapture(object):
    def __init__(self, config):
        self.config = config
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'tspxr_capture')
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.num_source = self.config['Train']['num_source'] # 2
        self.train_data = self.generate_datasets(scene_dirs=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(scene_dirs=self.valid_dir, shuffle=False)
        self.test_data = self.generate_datasets(scene_dirs=self.valid_dir, shuffle=False)
    
    def _load_files(self, dir):
        if type(dir) == bytes:
            dir = dir.decode('utf-8')
        file_path = os.path.join(dir, '*')
        scene_dirs = glob.glob(file_path)
        return scene_dirs

    def load_csv(self, csv_file: str):
        return pd.read_csv(csv_file)
    
    def resample_imu(self, imu_sequence: np.ndarray, new_samples: int) -> np.ndarray:
        n, features = imu_sequence.shape
        
        if n == new_samples:
            return imu_sequence
        elif n > new_samples:
            indices = np.linspace(0, n - 1, new_samples).astype(int)
            resampled_sequence = imu_sequence[indices]
        else:
            x_old = np.linspace(0, 1, n, endpoint=True)
            x_new = np.linspace(0, 1, new_samples, endpoint=True)
            
            resampled_sequence = np.zeros((new_samples, 6))

            for i in range(features):
                interp_func = interp1d(x_old, imu_sequence[:, i], kind='linear', fill_value="extrapolate")
                resampled_sequence[:, i] = interp_func(x_new)
        return resampled_sequence

    def generate_sequences(self,
                           rgb_paths: list,
                           imu_data: pd.DataFrame,
                           pose_data: pd.DataFrame,
                           intrinsic: np.ndarray,
                           step: int = 1,
                           imu_seq: int = 4):
        samples = []

        # pose_data와 rgb_paths가 같은 길이, 동일한 index에 대해 시점이 매칭된다고 가정
        length = len(rgb_paths)

        # Read rgb sample
        rgb_sample = cv2.imread(rgb_paths[0])
        new_intrinsic = self.rescale_intrinsic(rgb=rgb_sample, intrinsic=intrinsic)

        # for문 범위: step부터 length - step - 1까지
        # 예: step=1일 때, 1부터 length-2까지 순회
        #    step=2일 때, 2부터 length-3까지 순회
        for t in range(step, length - step):
            # -------------------------
            # 1) t-step, t, t+step
            # -------------------------
            t_left_idx = t - step
            t_right_idx = t + step
            
            # pose_data에서 timestamp 추출
            time_left = pose_data.loc[t_left_idx, 'Timestamp']
            time_curr = pose_data.loc[t, 'Timestamp']
            time_right = pose_data.loc[t_right_idx, 'Timestamp']

            # 2) RGB 경로
            left_rgb = rgb_paths[t_left_idx]
            curr_rgb = rgb_paths[t]
            right_rgb = rgb_paths[t_right_idx]

            # -------------------------
            # 3) IMU 데이터 슬라이싱
            #    [t-step, t) 구간 -> imu_left
            #    [t, t+step) 구간 -> imu_right
            # -------------------------
            # (1) 왼쪽 구간: [time_left, time_curr)
            mask_left = (imu_data['Timestamp'] >= time_left) & (imu_data['Timestamp'] < time_curr)
            left_imu_df = imu_data[mask_left]

            # (2) 오른쪽 구간: [time_curr, time_right)
            mask_right = (imu_data['Timestamp'] >= time_curr) & (imu_data['Timestamp'] < time_right)
            right_imu_df = imu_data[mask_right]

            # np.ndarray (n, 6) 형태로 변환한다고 가정
            left_imu_array = left_imu_df.iloc[:, 1:].values
            right_imu_array = right_imu_df.iloc[:, 1:].values

            # 3-1) 리샘플링
            left_imu_resampled = self.resample_imu(left_imu_array, imu_seq)
            right_imu_resampled = self.resample_imu(right_imu_array, imu_seq)

            # -------------------------
            # 4) Pose 데이터 (t-step, t, t+step)
            # -------------------------
            # 원하는 pose 칼럼만 추출하거나, 전체를 dict로 만들어 저장
            left_pose = pose_data.iloc[t_left_idx]
            curr_pose = pose_data.iloc[t]
            right_pose = pose_data.iloc[t_right_idx]
            
            # -------------------------
            # 5) 리스트에 저장
            # -------------------------
            sample = {
                'source_left': left_rgb,
                'target_image': curr_rgb,
                'source_right': right_rgb,
                'imu_left': left_imu_resampled,
                'imu_right': right_imu_resampled,
                'intrinsic': new_intrinsic.astype(np.float32),
                # 'pose_lists': {
                #     'left_pose': left_pose,
                #     'curr_pose': curr_pose,
                #     'right_pose': right_pose
                # }
            }
            samples.append(sample)

        return samples
    
    def rescale_intrinsic(self, rgb: np.ndarray,
                          intrinsic: np.ndarray) -> np.ndarray:
            old_h, old_w = rgb.shape[:2]
            new_h, new_w = self.image_size

            # x 방향 스케일 비율, y 방향 스케일 비율 계산
            scale_x = new_w / old_w
            scale_y = new_h / old_h

            # intrinsic matrix 복사본 생성
            new_intrinsic = intrinsic.copy()

            # Focal length, principal point 스케일 조정
            new_intrinsic[0, 0] *= scale_x  # fx
            new_intrinsic[1, 1] *= scale_y  # fy
            new_intrinsic[0, 2] *= scale_x  # cx
            new_intrinsic[1, 2] *= scale_y  # cy

            return new_intrinsic

    def generate_datasets(self, scene_dirs, shuffle: False):
        scene_dirs = self._load_files(dir=scene_dirs)
        datasets = []
        for scene_dir in scene_dirs:
            rgb_paths = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*.jpg')))
            imu_data = self.load_csv(os.path.join(scene_dir, 'sensor', 'zed_imu.csv'))
            pose_data = self.load_csv(os.path.join(scene_dir, 'sensor', 'zed_pose.csv'))
            intrinsic = np.load(os.path.join(scene_dir, 'sensor', 'intrinsics.npy'))
            """
            rgb_paths: list, imu_data: pd.DataFrame, pose_data: pd.DataFrame,
                           source_num: int = 2,
                           imu_seq_num: int = 10
            """
            dataset = self.generate_sequences(rgb_paths=rgb_paths, imu_data=imu_data, pose_data=pose_data,
                                              intrinsic=intrinsic, step=2, imu_seq=4)
            datasets.append(dataset)

        datasets = np.concatenate(datasets, axis=0)
            
        if shuffle:
            np.random.shuffle(datasets)
        return datasets

if __name__ == '__main__':
    root_dir = './vio/data/'
    tspxr_capture = TspxrCapture(root_dir)
    print(tspxr_capture.train_data.shape)

    test_dataset = tspxr_capture.valid_data
    for idx in range(test_dataset.shape[0]):
        print(test_dataset[idx]['current_rgb'])
