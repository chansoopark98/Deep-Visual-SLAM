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
        
        # 스테레오 설정
        self.use_stereo = self.config['Train'].get('use_stereo', True)
        self.stereo_ratio = self.config['Train'].get('stereo_ratio', 0.5)  # 스테레오 샘플 비율
        self.use_temporal = self.config['Train'].get('use_temporal', True)
        self.temporal_ratio = self.config['Train'].get('temporal_ratio', 0.5)  # temporal 샘플 비율
        
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
        try:
            # 좌우 카메라 내부 파라미터
            left_intrinsic = np.load(os.path.join(sensor_dir, 'left_intrinsics.npy'))
            right_intrinsic = np.load(os.path.join(sensor_dir, 'right_intrinsics.npy'))
            
            # 스테레오 파라미터 로드
            stereo_params_file = os.path.join(sensor_dir, 'stereo_parameters.json')
            if os.path.exists(stereo_params_file):
                with open(stereo_params_file, 'r') as f:
                    stereo_params = json.load(f)
                    baseline_m = stereo_params.get('baseline_m', 0.05)  # ZED X Mini 기본값
            else:
                baseline_m = 0.05  # 5cm
            
            # 좌->우 변환 행렬 (4x4)
            stereo_T_L2R = np.eye(4, dtype=np.float32)
            stereo_T_L2R[0, 3] = baseline_m  # x축으로 이동
            
            # 우->좌 변환 행렬 (역변환)
            stereo_T_R2L = np.linalg.inv(stereo_T_L2R)
            
            return left_intrinsic, right_intrinsic, stereo_T_L2R, stereo_T_R2L, baseline_m
            
        except Exception as e:
            print(f"Error loading stereo calibration: {e}")
            return None, None, None, None, None

    def _load_mono_calibration(self, sensor_dir: str):
        """모노 카메라 캘리브레이션 정보 로드"""
        try:
            intrinsic = np.load(os.path.join(sensor_dir, 'intrinsics.npy'))
            return intrinsic
        except Exception as e:
            print(f"Error loading mono calibration: {e}")
            return None

    def _create_stereo_samples(self, left_files, right_files, left_K, right_K, 
                            stereo_T_L2R, stereo_T_R2L, baseline, indices):
        """스테레오 샘플 생성 (양방향 warping)"""
        samples = []
        
        for idx in indices:
            # 1. 좌->우 warping 샘플 (left가 source, right가 target)
            sample_stereo = {
                'source_left': left_files[idx],
                'target_image': right_files[idx],
                'source_right': right_files[idx],
                'intrinsic': left_K,
                'poses': np.array([stereo_T_L2R, stereo_T_R2L], dtype=np.float32),
                'baseline': baseline,
                'data_type': 'stereo',
                'use_pose_net': False,  # 단일 bool 값으로 변경
            }
            samples.append(sample_stereo)
   
        return samples

    def _create_temporal_samples(self, rgb_files, intrinsic, indices, camera_side='mono'):
        """Temporal sequence 샘플 생성"""
        samples = []
        dummy_pose = np.eye(4, dtype=np.float32)
        for idx in indices:
            sample = {
                'source_left': rgb_files[idx - 1],
                'target_image': rgb_files[idx],
                'source_right': rgb_files[idx + 1],
                'intrinsic': intrinsic,
                'poses': np.array([dummy_pose, dummy_pose], dtype=np.float32),  # 두 개의 더미 pose
                'baseline': 0.0,
                'data_type': 'temporal',
                'use_pose_net': True,
            }
            samples.append(sample)
        
        return samples


    def _process_stereo_scene(self, scene_dir: str):
        """스테레오 시퀀스 처리"""
        sensor_dir = os.path.join(scene_dir, 'sensor')
        
        # 이미지 파일 로드
        left_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb_left', '*.jpg')))
        right_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb_right', '*.jpg')))
        
        if not left_files or not right_files:
            print(f"No stereo images found in {scene_dir}")
            return []
        
        # 파일 개수 맞추기
        min_length = min(len(left_files), len(right_files))
        left_files = left_files[:min_length]
        right_files = right_files[:min_length]
        
        # 캘리브레이션 로드
        left_K, right_K, stereo_T_L2R, stereo_T_R2L, baseline = self._load_stereo_calibration(sensor_dir)
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
        
        # 스테레오 샘플 생성 - 모든 유효한 프레임 사용
        if self.use_stereo:
            stereo_samples = self._create_stereo_samples(
                left_files, right_files, left_K, right_K, 
                stereo_T_L2R, stereo_T_R2L, baseline, valid_indices
            )
            samples.extend(stereo_samples)
        
        # Temporal 샘플 생성 - 모든 유효한 프레임 사용
        if self.use_temporal:
            # 좌측 카메라 temporal - 모든 프레임
            left_temporal_samples = self._create_temporal_samples(
                left_files, left_K, valid_indices, camera_side='left'
            )
            samples.extend(left_temporal_samples)
            
            # 우측 카메라 temporal - 모든 프레임
            right_temporal_samples = self._create_temporal_samples(
                right_files, right_K, valid_indices, camera_side='right'
            )
            samples.extend(right_temporal_samples)
        
        return samples

    def _process_mono_scene(self, scene_dir: str):
        """모노큘러 시퀀스 처리"""
        sensor_dir = os.path.join(scene_dir, 'sensor')
        
        # 이미지 파일 로드
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*.jpg')))
        if not rgb_files:
            print(f"No mono images found in {scene_dir}")
            return []
        
        # 캘리브레이션 로드
        intrinsic = self._load_mono_calibration(sensor_dir)
        if intrinsic is None:
            return []
        
        # 이미지 크기 확인 및 내부 파라미터 스케일링
        sample_img = cv2.imread(rgb_files[0])
        if sample_img is None:
            return []
        original_size = (sample_img.shape[0], sample_img.shape[1])
        
        intrinsic = self._rescale_intrinsic(intrinsic, self.image_size, original_size)
        
        # 유효한 인덱스 범위 - 모든 프레임 사용
        valid_indices = list(range(1, len(rgb_files) - 1))
        
        # Temporal 샘플 생성 - 모든 프레임 사용
        samples = self._create_temporal_samples(
            rgb_files, intrinsic, valid_indices, camera_side='mono'
        )
        
        return samples

    def _process(self, scene_dir: str):
        """씬 디렉토리 처리 - 스테레오/모노 자동 감지"""
        # 디렉토리 구조 확인
        has_stereo = (os.path.exists(os.path.join(scene_dir, 'rgb_left')) and 
                     os.path.exists(os.path.join(scene_dir, 'rgb_right')))
        has_mono = os.path.exists(os.path.join(scene_dir, 'rgb'))
        
        if has_stereo and self.use_stereo:
            print(f"Processing stereo scene: {os.path.basename(scene_dir)}")
            return self._process_stereo_scene(scene_dir)
        elif has_mono:
            print(f"Processing mono scene: {os.path.basename(scene_dir)}")
            return self._process_mono_scene(scene_dir)
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

    def get_data_statistics(self):
        """데이터셋 통계 정보 출력"""
        def analyze_samples(samples):
            if len(samples) == 0:
                return {
                    'total': 0,
                    'stereo': 0,
                    'temporal': {'left': 0, 'right': 0, 'mono': 0}
                }
            
            stats = {
                'total': len(samples),
                'stereo': 0,
                'temporal': {'left': 0, 'right': 0, 'mono': 0}
            }
            
            for sample in samples:
                if sample['data_type'] == 'stereo':
                    stats['stereo'] += 1
                elif sample['data_type'] == 'temporal':
                    camera_side = sample.get('camera_side', 'mono')
                    stats['temporal'][camera_side] += 1
            
            return stats
        
        train_stats = analyze_samples(self.train_data)
        valid_stats = analyze_samples(self.valid_data)
        test_stats = analyze_samples(self.test_data)
        
        print("\n=== Dataset Statistics ===")
        print(f"Training set: {train_stats['total']} samples")
        print(f"  - Stereo: {train_stats['stereo']}")
        print(f"  - Temporal: left={train_stats['temporal']['left']}, right={train_stats['temporal']['right']}, mono={train_stats['temporal']['mono']}")
        
        print(f"\nValidation set: {valid_stats['total']} samples")
        print(f"  - Stereo: {valid_stats['stereo']}")
        print(f"  - Temporal: left={valid_stats['temporal']['left']}, right={valid_stats['temporal']['right']}, mono={valid_stats['temporal']['mono']}")

        print(f"\nTest set: {test_stats['total']} samples")
        print(f"  - Stereo: {test_stats['stereo']}")
        print(f"  - Temporal: left={test_stats['temporal']['left']}, right={test_stats['temporal']['right']}, mono={test_stats['temporal']['mono']}")
        print("\n=== End of Statistics ===")

if __name__ == '__main__':
    # 설정 파일 로드
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    dataset = CustomDataHandler(config)
    dataset.get_data_statistics()
    
    # 샘플 데이터 확인
    if len(dataset.train_data) > 0:
        print("\n=== Sample Data ===")
        for i in range(min(5, len(dataset.train_data))):
            sample = dataset.train_data[i]
            print(f"\nSample {i}:")
            print(f"  Type: {sample['data_type']}")
            print(f"  Target: {os.path.basename(sample['target_image'])}")

            print(f"  Use pose net: {sample['use_pose_net']}")
            print(f"  Baseline: {sample['baseline']:.3f}m")
            
            if sample['data_type'] == 'stereo':
                print(f"  Stereo direction: {sample.get('stereo_direction', 'N/A')}")
            elif sample['data_type'] == 'temporal':
                print(f"  Camera side: {sample.get('camera_side', 'N/A')}")