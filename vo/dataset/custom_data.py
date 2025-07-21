import os
import glob
from cycler import V
import pandas as pd
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import yaml
try:
    from .common import MonoDataset, StereoDataset
except:
    from common import MonoDataset, StereoDataset
from typing import Dict, Tuple

class CustomMonoDataset(MonoDataset):
    def __init__(self,
                 config,
                 fold: str = 'train',
                 shuffle: bool = True,
                 is_train: bool = True,
                 augment: bool = True):
        self.config = config
        self.fold = fold
        self.shuffle = shuffle
        self.root_dir = '/home/park-ubuntu/park/Deep-Visual-SLAM/depth/data/tspxr_capture'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source']

        self.dataset_dict = self._process_mono_scene(os.path.join(self.root_dir, fold))

        super().__init__(dataset_dict=self.dataset_dict,
                         image_size=self.image_size,
                         is_train=is_train,
                         augment=augment)

    def _rescale_intrinsic(self, intrinsic: np.ndarray, target_size: tuple, current_size: tuple) -> np.ndarray:
        """내부 파라미터를 타겟 이미지 크기에 맞게 스케일링"""
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        intrinsic_rescaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return intrinsic_rescaled
    
    def _load_calibration(self, sensor_dir: str):
        """캘리브레이션 정보 로드"""
        # 카메라 내부 파라미터
        left_intrinsic = np.load(os.path.join(sensor_dir, 'left_intrinsics.npy'))
        return left_intrinsic

    def _create_mono_samples(self, left_files, left_K):
        """개별 리스트로 반환하도록 수정"""
        source_left_paths = []
        target_image_paths = []
        source_right_paths = []
        intrinsics = []
        
        step = 1
        for t in range(self.num_source, len(left_files) - self.num_source, step):
            source_left_paths.append(left_files[t - 1])
            target_image_paths.append(left_files[t])
            source_right_paths.append(left_files[t + 1])
            intrinsics.append(left_K)
        
        return {
            'source_left': source_left_paths,
            'target_image': target_image_paths,
            'source_right': source_right_paths,
            'intrinsic': intrinsics
        }
    
    def _process_mono_scene(self, fold_dir: str):
        
        scene_dirs = sorted(glob.glob(os.path.join(fold_dir, '*')))

        source_left_images = [] # paths
        target_images = [] # paths
        source_right_images = [] # paths
        intrinsics = [] # np.ndarray

        for scene_dir in scene_dirs:
            if os.path.isdir(scene_dir):
                """모노 시퀀스 처리 - dict 반환"""
                sensor_dir = os.path.join(scene_dir, 'sensor')
                
                # 이미지 파일 로드
                extensions = ['*.png', '*.jpg', '*.jpeg']
                left_files = []

                for ext in extensions:
                    left_files.extend(glob.glob(os.path.join(scene_dir, 'rgb_left', ext)))

                left_files = sorted(left_files)
                
                # 캘리브레이션 로드
                left_K = self._load_calibration(sensor_dir)
                
                # 이미지 크기 확인 및 내부 파라미터 스케일링
                sample_img = cv2.imread(left_files[0])

                if sample_img is None:
                    raise FileNotFoundError(f"Could not read image from {left_files[0]}")
                
                original_size = (sample_img.shape[0], sample_img.shape[1])
                
                left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)
                
                dataset_dict = self._create_mono_samples(left_files, left_K)

                if dataset_dict:
                    source_left_images.extend(dataset_dict['source_left'])
                    target_images.extend(dataset_dict['target_image'])
                    source_right_images.extend(dataset_dict['source_right'])
                    intrinsics.extend(dataset_dict['intrinsic'])

        # numpy 배열로 변환
        dataset_dict = {
            'source_left': np.array(source_left_images, dtype=str),
            'target_image': np.array(target_images, dtype=str),
            'source_right': np.array(source_right_images, dtype=str),
            'intrinsic': np.array(intrinsics, dtype=np.float32)
        }
        
        # 셔플링
        if self.shuffle:
            indices = np.random.permutation(len(source_left_images))
            for key in dataset_dict:
                dataset_dict[key] = dataset_dict[key][indices]

        print(f"Generated {len(source_left_images)} samples from {fold_dir}")
        return dataset_dict

class CustomStereoDataset(StereoDataset):
    def __init__(self,
                 config,
                 fold: str = 'train',
                 shuffle: bool = True,
                 is_train: bool = True,
                 augment: bool = True):
        self.config = config
        self.fold = fold
        self.shuffle = shuffle
        self.root_dir = '/home/park-ubuntu/park/Deep-Visual-SLAM/depth/data/tspxr_capture'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source']

        self.dataset_dict = self._process_stereo_scene(os.path.join(self.root_dir, fold))

        super().__init__(dataset_dict=self.dataset_dict,
                         image_size=self.image_size,
                         is_train=is_train,
                         augment=augment)
        
    def _rescale_intrinsic(self, intrinsic: np.ndarray, target_size: tuple, current_size: tuple) -> np.ndarray:
        """내부 파라미터를 타겟 이미지 크기에 맞게 스케일링"""
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        intrinsic_rescaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return intrinsic_rescaled
    
    def _load_stereo_calibration(self, sensor_dir: str):
        """스테레오 캘리브레이션 정보 로드"""
        
        # 좌우 카메라 내부 파라미터
        left_intrinsic = np.load(os.path.join(sensor_dir, 'left_intrinsics.npy'))
        right_intrinsic = np.load(os.path.join(sensor_dir, 'right_intrinsics.npy'))

        # load json
        with open(os.path.join(sensor_dir, 'stereo_parameters.json'), 'r') as f:
            calibration_data = json.load(f)
            baseline_m = calibration_data['baseline_m'] # 미터

        # Left to Right transformation as 6DoF vector
        pose_L2R = np.array([0.0, 0.0, 0.0,  # axis-angle (no rotation)
                            baseline_m, 0.0, 0.0],  # translation (x, y, z)
                            dtype=np.float32)
        
        # Right to Left transformation as 6DoF vector
        pose_R2L = np.array([0.0, 0.0, 0.0,  # axis-angle (no rotation)
                            -baseline_m, 0.0, 0.0],  # translation (-x, y, z)
                            dtype=np.float32)
        
        baseline_m = np.float32(baseline_m)  # ensure baseline is float32
        
        return left_intrinsic, right_intrinsic, pose_L2R, pose_R2L, baseline_m
    
    def _create_stereo_samples(self, left_files, right_files, left_K, right_K,
                               stereo_T_L2R, stereo_T_R2L, indices):
        """스테레오 샘플 생성 - dict 형태로 반환"""
        source_image_paths = []
        target_image_paths = []
        intrinsics = []
        poses = []
        
        for idx in indices:
            # L2R (left to right)
            source_image_paths.append(left_files[idx])
            target_image_paths.append(right_files[idx])
            intrinsics.append(right_K)
            poses.append(stereo_T_L2R)
            
            # R2L (right to left)
            source_image_paths.append(right_files[idx])
            target_image_paths.append(left_files[idx])
            intrinsics.append(left_K)
            poses.append(stereo_T_R2L)
        
        return {
            'source_image': source_image_paths,
            'target_image': target_image_paths,
            'intrinsic': intrinsics,
            'pose': poses
        }
    
    def _process_stereo_scene(self, fold_dir: str):
        scene_dirs = sorted(glob.glob(os.path.join(fold_dir, '*')))

        source_image_paths = []
        target_image_paths = []
        intrinsics = []
        poses = []

        for scene_dir in scene_dirs:
            sensor_dir = os.path.join(scene_dir, 'sensor')
            left_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb_left', '*.png')))
            right_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb_right', '*.png')))

            if len(left_files) != len(right_files):
                raise ValueError(f"Mismatch in number of images: {len(left_files)} left, {len(right_files)} right in {scene_dir}")
            
            left_K, right_K, stereo_T_L2R, stereo_T_R2L, baseline_m = self._load_stereo_calibration(sensor_dir)

            # rescale intrinsic parameters to target image size(self.image_size)
            sample_img = cv2.imread(left_files[0])
            if sample_img is None:
                raise FileNotFoundError(f"Could not read image from {left_files[0]}")
            original_size = (sample_img.shape[0], sample_img.shape[1])

            left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)
            right_K = self._rescale_intrinsic(right_K, self.image_size, original_size)

            # Create stereo samples for the current scene
            min_length = len(left_files)
            valid_indices = list(range(0, min_length))
            stereo_samples = self._create_stereo_samples(left_files, right_files, left_K, right_K,
                                                          stereo_T_L2R, stereo_T_R2L, valid_indices)
            source_image_paths.extend(stereo_samples['source_image'])
            target_image_paths.extend(stereo_samples['target_image'])
            intrinsics.extend(stereo_samples['intrinsic'])
            poses.extend(stereo_samples['pose'])

        return {
            'source_image': source_image_paths,
            'target_image': target_image_paths,
            'intrinsic': intrinsics,
            'pose': poses
        }
    
class CustomDataHandler:
    def __init__(self, config):
        self.config = config
        self.root_dir = '/home/park-ubuntu/park/Deep-Visual-SLAM/depth/data/tspxr_capture'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        
        # 데이터셋 생성
        self.train_mono_dataset = None
        self.valid_mono_dataset = None
        self.train_stereo_dataset = None
        self.valid_stereo_dataset = None

        # Monocular Dataset
        if self.config['Dataset']['custom_data']['mono']:
            if os.path.exists(os.path.join(self.root_dir, 'train')):
                self.train_mono_dataset = CustomMonoDataset(
                    config=self.config,
                    fold='train',
                    is_train=True,
                    augment=True
                )
            
            if os.path.exists(os.path.join(self.root_dir, 'valid')):
                self.valid_mono_dataset = CustomMonoDataset(
                    config=self.config,
                    fold='valid',
                    is_train=False,
                    augment=False
                )

        # Stereo Dataset
        if self.config['Dataset']['custom_data']['stereo']:
            if os.path.exists(os.path.join(self.root_dir, 'train')):
                self.train_stereo_dataset = CustomStereoDataset(
                    config=self.config,
                    fold='train',
                    is_train=True,
                    augment=True
                )
            
            if os.path.exists(os.path.join(self.root_dir, 'valid')):
                self.valid_stereo_dataset = CustomStereoDataset(
                    config=self.config,
                    fold='valid',
                    is_train=False,
                    augment=False
                )
       
if __name__ == '__main__':
    # 설정 파일 로드
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_data = CustomMonoDataset(config=config, fold='train', is_train=True, augment=True)
    valid_data = CustomMonoDataset(config=config, fold='valid', is_train=False, augment=False)

    # iteration test
    for i in range(len(train_data)):
        sample = train_data[i]
        print(f"Sample {i}:")
        print(f"  Source Left: {sample['source_left']}")
        print(f"  Target Image: {sample['target_image']}")
        print(f"  Source Right: {sample['source_right']}")
        print(f"  Intrinsic: {sample['intrinsic']}")

        left_image = sample['source_left'].permute(1, 2, 0).numpy()
        target_image = sample['target_image'].permute(1, 2, 0).numpy()
        right_image = sample['source_right'].permute(1, 2, 0).numpy()

        # imshow
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(left_image)
        plt.title("Source Left")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(target_image)
        plt.title("Target Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(right_image)
        plt.title("Source Right")
        plt.axis("off")

        plt.show()

    # train_data = CustomStereoDataset(config=config, fold='train', is_train=True, augment=True)

    # for i in range(len(train_data)):
    #     sample = train_data[i]
    #     print(f"Sample {i}:")
    #     print(f"  Source Image: {sample['source_image']}")
    #     print(f"  Target Image: {sample['target_image']}")
    #     print(f"  Intrinsic: {sample['intrinsic']}")
    #     print(f"  Pose: {sample['pose']}")

    #     source_image = sample['source_image'].permute(1, 2, 0).numpy()
    #     target_image = sample['target_image'].permute(1, 2, 0).numpy()

    #     # imshow
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(source_image)
    #     plt.title("Source Image")
    #     plt.axis("off")

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(target_image)
    #     plt.title("Target Image")
    #     plt.axis("off")

    #     plt.show()