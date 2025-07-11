import os
import glob
import pandas as pd
import numpy as np
import cv2
import json

class MarsLoggerHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/slam_data/mars_logger'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source'] # 1
        self.imu_seq_len = self.config['Train']['imu_seq_len'] # 10
        
        self.train_data = self.generate_datasets(fold_dir='train', shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir='valid', shuffle=False)
        self.test_dir = os.path.join(self.root_dir, 'S22', 'test')
        self.test_data = self.generate_test(test_dir=self.test_dir)

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

    def _rescale_intrinsic(self, intrinsic: np.ndarray, target_size: tuple, current_size: tuple) -> np.ndarray:
        # New shape = self.image_size (H, W)
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        intrinsic_rescaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsic_rescaled

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
            
    def generate_datasets(self, fold_dir, shuffle=False, is_test=False) -> dict:
        """데이터셋 생성 - dict 형태로 반환"""
        camera_types = glob.glob(os.path.join(self.root_dir, '*'))
        

        # 전체 데이터를 저장할 리스트
        all_source_left = []
        all_target_image = []
        all_source_right = []
        all_intrinsic = []

        for camera_type in camera_types:
            current_fold = os.path.join(camera_type, fold_dir)
            camera_name = os.path.basename(camera_type)

            if not os.path.exists(current_fold):
                continue

            scene_files = sorted(glob.glob(os.path.join(current_fold, '*')))
            
            for scene in scene_files:
                if os.path.isdir(scene):
                    samples = self._process(scene, camera_name, is_test)
                    if samples and len(samples['source_left']) > 0:
                        all_source_left.extend(samples['source_left'])
                        all_target_image.extend(samples['target_image'])
                        all_source_right.extend(samples['source_right'])
                        all_intrinsic.extend(samples['intrinsic'])
        
        # numpy 배열로 변환
        dataset_dict = {
            'source_left': np.array(all_source_left, dtype=str),
            'target_image': np.array(all_target_image, dtype=str),
            'source_right': np.array(all_source_right, dtype=str),
            'intrinsic': np.array(all_intrinsic, dtype=np.float32)
        }

        print('Current fold:', fold_dir)
        print(f'  -- Camera types: {[os.path.basename(ct) for ct in camera_types]}')
        print(f'  -- dataset size: {len(all_source_left)}')

        # 셔플링
        if shuffle and len(all_source_left) > 0:
            indices = np.random.permutation(len(all_source_left))
            for key in dataset_dict:
                dataset_dict[key] = dataset_dict[key][indices]
                
        return dataset_dict
    
    def generate_test(self, test_dir):
        """테스트 데이터셋 생성 - dict 형태로 반환"""
        if not os.path.exists(test_dir):
            print(f"Test directory {test_dir} does not exist")
            return {
                'source_left': np.array([]),
                'target_image': np.array([]),
                'source_right': np.array([]),
                'intrinsic': np.array([])
            }

        # 전체 데이터를 저장할 리스트
        all_source_left = []
        all_target_image = []
        all_source_right = []
        all_intrinsic = []

        scene_files = sorted(glob.glob(os.path.join(test_dir, '*')))
        
        for scene in scene_files:
            if os.path.isdir(scene):
                samples = self._process(scene, 'S22', is_test=True)
                if samples and len(samples['source_left']) > 0:
                    all_source_left.extend(samples['source_left'])
                    all_target_image.extend(samples['target_image'])
                    all_source_right.extend(samples['source_right'])
                    all_intrinsic.extend(samples['intrinsic'])

        # numpy 배열로 변환
        dataset_dict = {
            'source_left': np.array(all_source_left, dtype=str),
            'target_image': np.array(all_target_image, dtype=str),
            'source_right': np.array(all_source_right, dtype=str),
            'intrinsic': np.array(all_intrinsic, dtype=np.float32)
        }
        
        print(f'Test dataset size: {len(all_source_left)}')
        
        return dataset_dict
    
if __name__ == '__main__':
    import yaml
    
    # load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = MarsLoggerHandler(config)

    print(f"Train samples: {dataset.train_data['source_left'].shape}")
    print(f"Valid samples: {dataset.valid_data['source_left'].shape}")
    print(f"Test samples: {dataset.test_data['source_left'].shape}")