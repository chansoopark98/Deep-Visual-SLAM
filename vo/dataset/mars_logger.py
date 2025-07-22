import os
import glob
import pandas as pd
import numpy as np
import cv2
import json
try:
    from .common import MonoDataset, StereoDataset
except:
    from common import MonoDataset, StereoDataset
from typing import Dict, Tuple


class MarsMonoDataset(MonoDataset):
    def __init__(self,
                 config,
                 fold: str = 'train',
                 shuffle: bool = True,
                 is_train: bool = True,
                 augment: bool = True):
        self.config = config
        self.fold = fold
        self.shuffle = shuffle
        self.root_dir = '/media/park-ubuntu/park_cs/slam_data/mars_logger'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source']

        self.dataset_dict = self._process_mono_scene(fold_dir=fold)

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
    
    def _extract_video(self, scene_dir: str, camera_name) -> int:
        video_file = os.path.join(scene_dir, 'movie.mp4')
        rgb_save_path = os.path.join(scene_dir, 'rgb')

        # Ensure output directory exists
        if not os.path.exists(rgb_save_path):
            print(f'Extracting video file: {video_file}')
            os.makedirs(rgb_save_path, exist_ok=True)

            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_file}")

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_name = os.path.join(rgb_save_path, f'rgb_{str(idx).zfill(6)}.jpg')
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
                cv2.imwrite(rgb_name, frame)
                idx += 1

            cap.release()
            cv2.destroyAllWindows()

        dataset = glob.glob(os.path.join(rgb_save_path, '*.jpg'))
        data_len = len(dataset)

        # Load camera metadata
        camera_metadata_path = os.path.join(self.root_dir, camera_name, 'calibration_results')
        

        with open(os.path.join(camera_metadata_path, 'calibration_results.json'), 'r') as f:
            camera_metadata = json.load(f)
        original_image_size = (camera_metadata['image_height'], camera_metadata['image_width'])
        
        intrinsic = np.load(os.path.join(camera_metadata_path, 'camera_matrix.npy'))

        # rescale intrinsic matrix
        rescaled_intrinsic = self._rescale_intrinsic(intrinsic, self.image_size, original_image_size)

        return data_len, rescaled_intrinsic

    
    def _process(self, scene_dir: str, camera_name: str, is_test: bool=False) -> dict:
        """단일 scene 처리 - dict 반환"""
        # load video .mp4
        length, resized_intrinsic = self._extract_video(scene_dir, camera_name)
    
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*.jpg')))
        
        if is_test:
            step = 1
        else:
            step = 3
        
        source_left_paths = []
        target_image_paths = []
        source_right_paths = []
        intrinsics = []
        
        for t in range(self.num_source, length - self.num_source, step):
            source_left_paths.append(rgb_files[t - 1])
            target_image_paths.append(rgb_files[t])
            source_right_paths.append(rgb_files[t + 1])
            intrinsics.append(resized_intrinsic)
        
        return {
            'source_left': source_left_paths,
            'target_image': target_image_paths,
            'source_right': source_right_paths,
            'intrinsic': intrinsics
        }
    
    def _process_mono_scene(self, fold_dir: str):
        source_left_images = [] # paths
        target_images = [] # paths
        source_right_images = [] # paths
        intrinsics = [] # np.ndarray

        camera_types = glob.glob(os.path.join(self.root_dir, '*'))

        for camera_type in camera_types:
            current_fold = os.path.join(camera_type, fold_dir)
            camera_name = os.path.basename(camera_type)
            
            if not os.path.exists(current_fold):
                continue

            scene_files = sorted(glob.glob(os.path.join(current_fold, '*')))

            for scene in scene_files:
                if os.path.isdir(scene):
                    samples = self._process(scene, camera_name, is_test=False)
                    source_left_images.extend(samples['source_left'])
                    target_images.extend(samples['target_image'])
                    source_right_images.extend(samples['source_right'])
                    intrinsics.extend(samples['intrinsic'])

        # numpy 배열로 변환
        dataset_dict = {
            'source_left': np.array(source_left_images, dtype=str),
            'target_image': np.array(target_images, dtype=str),
            'source_right': np.array(source_right_images, dtype=str),
            'intrinsic': np.array(intrinsics, dtype=np.float32)
        }

        print('Current fold:', fold_dir)
        print(f'  -- Camera types: {[os.path.basename(ct) for ct in camera_types]}')
        print(f'  -- dataset size: {len(source_left_images)}')

        # 셔플링
        if self.shuffle and len(source_left_images) > 0:
            indices = np.random.permutation(len(source_left_images))
            for key in dataset_dict:
                dataset_dict[key] = dataset_dict[key][indices]
                
        return dataset_dict
    
class MarsDataHandler:
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/slam_data/mars_logger'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        
        # 데이터셋 생성
        self.train_mono_dataset = None
        self.valid_mono_dataset = None
        self.test_mono_dataset = None
        
        if os.path.exists(os.path.join(self.root_dir, 'train')):
            self.train_mono_dataset = MarsMonoDataset(
                config=self.config,
                fold='train',
                shuffle=True,
                is_train=True,
                augment=True
            )
        
        if os.path.exists(os.path.join(self.root_dir, 'valid')):
            self.valid_mono_dataset = MarsMonoDataset(
                config=self.config,
                fold='valid',
                shuffle=False,
                is_train=False,
                augment=False
            )
        if os.path.exists(os.path.join(self.root_dir, 'test')):
            self.test_mono_dataset = MarsMonoDataset(
                config=self.config,
                fold='test',
                shuffle=False,
                is_train=False,
                augment=False
            )

    
if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt

    # load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_data = MarsMonoDataset(config=config, fold='train', shuffle=True, is_train=True, augment=True)

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