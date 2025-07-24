import os
import glob
from matplotlib.pyplot import sca
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

        super().__init__(config=self.config,
                         dataset_dict=self.dataset_dict,
                         image_size=self.image_size,
                         is_train=is_train,
                         augment=augment)

    def _rescale_intrinsic(self,
                        intrinsic: np.ndarray,
                        target_size: tuple,
                        current_size: tuple) -> np.ndarray:
        """내부 파라미터를 타겟 이미지 크기에 맞게 스케일링하고, 4×4 homogeneous 형태로 반환"""
        # 1) 3×3 intrinsics 스케일링
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        K3 = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,   0,  1 ]], dtype=np.float32)

        # 2) 4×4 homogeneous intrinsics로 확장
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K3
        # (원하면 K4[:3, 3] = [0,0,0] 으로 translation 항목도 조정 가능)

        return K4
    
    def _load_calibration(self, sensor_dir: str):
        """캘리브레이션 정보 로드"""
        # 카메라 내부 파라미터
        left_intrinsic = np.load(os.path.join(sensor_dir, 'left_intrinsics.npy'))
        return left_intrinsic
    
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
        
        size = 2
        if is_test:
            step = 1
        else:
            step = 3
        
        source_left_paths = []
        target_image_paths = []
        source_right_paths = []
        intrinsics = []

        for t in range(step + size, length - step - size, step):
            source_left_paths.append(rgb_files[t - size])
            target_image_paths.append(rgb_files[t])
            source_right_paths.append(rgb_files[t + size])
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
                    if fold_dir == 'valid' or fold_dir == 'test':
                        is_test = True
                    else:
                        is_test = False
                    samples = self._process(scene, camera_name, is_test=is_test)
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
        
        if config['Dataset']['mars_logger']['mono']:
            self.train_mono_dataset = MarsMonoDataset(
                config=self.config,
                fold='train',
                shuffle=True,
                is_train=True,
                augment=True
            )

            self.valid_mono_dataset = MarsMonoDataset(
                config=self.config,
                fold='valid',
                shuffle=False,
                is_train=False,
                augment=False
            )
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

    train_data = MarsMonoDataset(config=config, fold='test', shuffle=False, is_train=False, augment=False)

    for i in range(len(train_data)):
        sample = train_data[i]
        
        left_images = []
        target_images = []
        right_images = []

        num_scale = train_data.num_scale
        plt.figure(figsize=(12, 4 * num_scale))

        for scale in range(num_scale):
            left_image = sample[('source_left', scale)].permute(1, 2, 0).numpy()
            target_image = sample[('target_image', scale)].permute(1, 2, 0).numpy()
            right_image = sample[('source_right', scale)].permute(1, 2, 0).numpy()

            plt.subplot(num_scale, 3, scale * 3 + 1)
            plt.imshow(left_image)
            plt.title(f"Source Left - Scale {scale}")
            plt.axis("off")

            plt.subplot(num_scale, 3, scale * 3 + 2)
            plt.imshow(target_image)
            plt.title(f"Target Image - Scale {scale}")
            plt.axis("off")

            plt.subplot(num_scale, 3, scale * 3 + 3)
            plt.imshow(right_image)
            plt.title(f"Source Right - Scale {scale}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()