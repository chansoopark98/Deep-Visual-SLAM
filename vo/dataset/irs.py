import os
import glob
import pandas as pd
import numpy as np
import cv2
import json
import yaml

class IrsDataHandler(object):
    def __init__(self, config, mode='stereo'):
        self.config = config
        self.mode = mode  # 'stereo' or 'mono'
        self.root_dir = '/media/park-ubuntu/park_file/IRS/'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source']  # temporal frames
        
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.test_dir = os.path.join(self.root_dir, 'test')
        
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

    def _load_stereo_calibration(self):
        """스테레오 캘리브레이션 정보 로드"""
        # make camera intrinsic
        # Focal Length: 480 for both the x-axis and y-axis
        # Resolution: 540x960(H x W)
        left_intrinsic = np.array([[480.0, 0.0, 480.0],
                                   [0.0, 480.0, 270.0],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)
        right_intrinsic = np.array([[480.0, 0.0, 480.0],
                                    [0.0, 480.0, 270.0],    
                                    [0.0, 0.0, 1.0]], dtype=np.float32)
        
        # 스테레오 파라미터 로드
        baseline_m = 0.100000 # 10cm
        
        # Left to Right transformation as 6DoF vector
        # 스테레오는 일반적으로 x축 방향 translation만 있고 회전은 없음
        pose_L2R = np.array([0.0, 0.0, 0.0,  # axis-angle (no rotation)
                            baseline_m, 0.0, 0.0],  # translation (x, y, z)
                            dtype=np.float32)
        
        # Right to Left transformation as 6DoF vector
        pose_R2L = np.array([0.0, 0.0, 0.0,  # axis-angle (no rotation)
                            -baseline_m, 0.0, 0.0],  # translation (-x, y, z)
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
    
    def _create_mono_samples(self, left_files, left_K):
        samples = []
        step = 3

        for t in range(self.num_source, len(left_files) - self.num_source, step):
            sample = {
                'source_left': left_files[t - 1],
                'target_image': left_files[t],
                'source_right': left_files[t + 1],
                'intrinsic': left_K,
            }
            samples.append(sample)
        return samples

    def _process_stereo_scene(self, scene_dir: str):
        """스테레오 시퀀스 처리"""
        
        # 모든 RGB 파일 로드
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, '*.png')))

        # left-right 분리
        """
        left image = l_00000.png ...
        right image = r_00000.png ...
        """
        left_files = [f for f in rgb_files if 'l_' in os.path.basename(f)]
        right_files = [f for f in rgb_files if 'r_' in os.path.basename(f)]
        
        if not left_files or not right_files:
            print(f"No stereo images found in {scene_dir}. "
                  f"Left files: {len(left_files)}, Right files: {len(right_files)}")
    
        if len(left_files) != len(right_files):
            print(f"Left and right image counts do not match in {scene_dir}: "
                  f"{len(left_files)} left images, {len(right_files)} right images.")
            return []
        
        # 캘리브레이션 로드
        left_K, right_K, pose_L2R, pose_R2L, baseline = self._load_stereo_calibration()
        
        # 이미지 크기 확인 및 내부 파라미터 스케일링
        sample_img = cv2.imread(left_files[0])
        if sample_img is None:
            return []
        original_size = (sample_img.shape[0], sample_img.shape[1])
        
        left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)
        right_K = self._rescale_intrinsic(right_K, self.image_size, original_size)
        
        samples = []
        
        # 유효한 인덱스 범위 계산
        dataset_length = len(left_files)
        valid_indices = list(range(1, dataset_length - 1))  # 첫/마지막 프레임 제외
        
        stereo_samples = self._create_stereo_samples(
            left_files, right_files, left_K, right_K, 
            pose_L2R, pose_R2L, valid_indices
        )
        samples.extend(stereo_samples)
        
        return samples
    
    def _process_mono_scene(self, scene_dir: str):
        """스테레오 시퀀스 처리"""
        
        # 모든 RGB 파일 로드
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, '*.png')))

        # left-right 분리
        """
        left image = l_00000.png ...
        right image = r_00000.png ...
        """
        left_files = [f for f in rgb_files if 'l_' in os.path.basename(f)]
        right_files = [f for f in rgb_files if 'r_' in os.path.basename(f)]
        
        if not left_files or not right_files:
            print(f"No mono images found in {scene_dir}. "
                  f"Left files: {len(left_files)}, Right files: {len(right_files)}")
    
        if len(left_files) < 1:
            print(f"Not enough left images found in {scene_dir}: {len(left_files)} left images.")
            return []
        
        # 캘리브레이션 로드
        left_K, right_K, pose_L2R, pose_R2L, baseline = self._load_stereo_calibration()
        
        # 이미지 크기 확인 및 내부 파라미터 스케일링
        sample_img = cv2.imread(left_files[0])
        if sample_img is None:
            return []
        original_size = (sample_img.shape[0], sample_img.shape[1])
        
        left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)
        right_K = self._rescale_intrinsic(right_K, self.image_size, original_size)
        
        samples = []
        
        # 유효한 인덱스 범위 계산
        dataset_length = len(left_files)
        valid_indices = list(range(1, dataset_length - 1))  # 첫/마지막 프레임 제외

        mono_samples = self._create_mono_samples(left_files, left_K)
        samples.extend(mono_samples)
        
        return samples
    
    def _process(self, scene_dir: str):
        """씬 디렉토리 처리 - 스테레오/모노 자동 감지"""
        # 디렉토리 구조 확인
        has_stereo = (os.path.exists(os.path.join(scene_dir, 'rgb_left')) and 
                     os.path.exists(os.path.join(scene_dir, 'rgb_right')))
        
        if self.mode == 'stereo':
            return self._process_stereo_scene(scene_dir)
        elif self.mode == 'mono':
            return self._process_mono_scene(scene_dir)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'stereo' or 'mono'.")


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
    dataset = IrsDataHandler(config)