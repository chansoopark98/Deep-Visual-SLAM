import os
import glob
import pandas as pd
import numpy as np
import cv2
import json
import yaml

class CustomDataHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/tspxr_capture'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source']  # temporal frames
        
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.test_dir = os.path.join(self.root_dir, 'valid')
        
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)
        self.test_data = self.generate_datasets(fold_dir=self.test_dir, shuffle=False)

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
        
        # 스테레오 파라미터 로드
        baseline_m = 0.0497779  # 5cm
        baseline_y = -0.0000153
        baseline_z = 0.0001807
        
        # Left to Right transformation as 6DoF vector
        # 스테레오는 일반적으로 x축 방향 translation만 있고 회전은 없음
        pose_L2R = np.array([0.0, 0.0, 0.0,  # axis-angle (no rotation)
                            baseline_m, baseline_y, baseline_z],  # translation (x, y, z)
                            dtype=np.float32)
        
        # Right to Left transformation as 6DoF vector
        pose_R2L = np.array([0.0, 0.0, 0.0,  # axis-angle (no rotation)
                            -baseline_m, -baseline_y, -baseline_z],  # translation (-x, y, z)
                            dtype=np.float32)
        
        return left_intrinsic, right_intrinsic, pose_L2R, pose_R2L, baseline_m

    def _create_stereo_samples(self, left_files, right_files, left_K, right_K, 
                            stereo_T_L2R, stereo_T_R2L, indices):
        """스테레오 샘플 생성 (양방향 warping)"""
        samples = []
        
        for idx in indices:
            sample_L2R = {
                'source_image': left_files[idx],
                'target_image': right_files[idx],
                'intrinsic': right_K,
                'pose': stereo_T_L2R,
            }
            samples.append(sample_L2R)

            sample_R2L = {
                'source_image': right_files[idx],
                'target_image': left_files[idx],
                'intrinsic': left_K,
                'pose': stereo_T_R2L,
            }
            samples.append(sample_R2L)
            
        return samples

    def _process_stereo_scene(self, scene_dir: str):
        """스테레오 시퀀스 처리"""
        sensor_dir = os.path.join(scene_dir, 'sensor')
        
        # 이미지 파일 로드
        # load both PNG and JPG images
        extensions = ['*.png', '*.jpg', '*.jpeg']
        left_files = []
        right_files = []

        for ext in extensions:
            left_files.extend(glob.glob(os.path.join(scene_dir, 'rgb_left', ext)))
            right_files.extend(glob.glob(os.path.join(scene_dir, 'rgb_right', ext)))

        left_files = sorted(left_files)
        right_files = sorted(right_files)
        
        if not left_files or not right_files:
            print(f"No stereo images found in {scene_dir}")
            return []
        
        # 파일 개수 맞추기
        min_length = min(len(left_files), len(right_files))
        left_files = left_files[:min_length]
        right_files = right_files[:min_length]
        
        # 캘리브레이션 로드
        left_K, right_K, pose_L2R, pose_R2L, baseline = self._load_stereo_calibration(sensor_dir)
        if left_K is None:
            return []
        
        # 이미지 크기 확인 및 내부 파라미터 스케일링
        sample_img = cv2.imread(left_files[0])
        if sample_img is None:
            return []
        original_size = (sample_img.shape[0], sample_img.shape[1])
        
        left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)
        right_K = self._rescale_intrinsic(right_K, self.image_size, original_size)
        
        samples = []
        
        # 유효한 인덱스 범위 계산
        valid_indices = list(range(1, min_length - 1))  # 첫/마지막 프레임 제외
        
        stereo_samples = self._create_stereo_samples(
            left_files, right_files, left_K, right_K, 
            pose_L2R, pose_R2L, valid_indices
        )
        samples.extend(stereo_samples)
        
        return samples

    def _process(self, scene_dir: str):
        """씬 디렉토리 처리 - 스테레오/모노 자동 감지"""
        # 디렉토리 구조 확인
        has_stereo = (os.path.exists(os.path.join(scene_dir, 'rgb_left')) and 
                     os.path.exists(os.path.join(scene_dir, 'rgb_right')))
        
        if has_stereo:
            print(f"Processing stereo scene: {os.path.basename(scene_dir)}")
            return self._process_stereo_scene(scene_dir)
        else:
            print(f"No valid image directory found in {scene_dir}")
            return []

    def generate_datasets(self, fold_dir, shuffle=False):
        """데이터셋 생성"""
        if not os.path.exists(fold_dir):
            print(f"Directory {fold_dir} does not exist")
            return np.array([])
        
        scene_dirs = sorted(glob.glob(os.path.join(fold_dir, '*')))
        all_samples = []
        
        for scene_dir in scene_dirs:
            if os.path.isdir(scene_dir):
                samples = self._process(scene_dir)
                if samples:
                    all_samples.extend(samples)
        
        if not all_samples:
            print(f"No samples generated from {fold_dir}")
            return np.array([])
        
        all_samples = np.array(all_samples)
        
        if shuffle:
            np.random.shuffle(all_samples)
        
        print(f"Generated {len(all_samples)} samples from {fold_dir}")
        return all_samples

       
if __name__ == '__main__':
    # 설정 파일 로드
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    dataset = CustomDataHandler(config)