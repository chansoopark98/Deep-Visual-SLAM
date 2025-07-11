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
    
    def _create_mono_samples(self, left_files, left_K):
        """모노 샘플 생성 - dict 형태로 반환"""
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

    def _process_stereo_scene(self, scene_dir: str):
        """스테레오 시퀀스 처리 - dict 반환"""
        
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
            return None
    
        if len(left_files) != len(right_files):
            print(f"Left and right image counts do not match in {scene_dir}: "
                  f"{len(left_files)} left images, {len(right_files)} right images.")
            return None
        
        # 캘리브레이션 로드
        left_K, right_K, pose_L2R, pose_R2L, baseline = self._load_stereo_calibration()
        
        # 이미지 크기 확인 및 내부 파라미터 스케일링
        sample_img = cv2.imread(left_files[0])
        if sample_img is None:
            return None
        original_size = (sample_img.shape[0], sample_img.shape[1])
        
        left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)
        right_K = self._rescale_intrinsic(right_K, self.image_size, original_size)
        
        # 유효한 인덱스 범위 계산
        dataset_length = len(left_files)
        valid_indices = list(range(0, dataset_length))  # 모든 프레임 사용
        
        return self._create_stereo_samples(
            left_files, right_files, left_K, right_K, 
            pose_L2R, pose_R2L, valid_indices
        )
    
    def _process_mono_scene(self, scene_dir: str):
        """모노 시퀀스 처리 - dict 반환"""
        
        # 모든 RGB 파일 로드
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, '*.png')))

        # left-right 분리
        """
        left image = l_00000.png ...
        right image = r_00000.png ...
        """
        left_files = [f for f in rgb_files if 'l_' in os.path.basename(f)]
        
        if not left_files:
            print(f"No mono images found in {scene_dir}. "
                  f"Left files: {len(left_files)}")
            return None
    
        if len(left_files) < 3:  # 최소 3개 프레임 필요 (t-1, t, t+1)
            print(f"Not enough left images found in {scene_dir}: {len(left_files)} left images.")
            return None
        
        # 캘리브레이션 로드
        left_K, _, _, _, _ = self._load_stereo_calibration()
        
        # 이미지 크기 확인 및 내부 파라미터 스케일링
        sample_img = cv2.imread(left_files[0])
        if sample_img is None:
            return None
        original_size = (sample_img.shape[0], sample_img.shape[1])
        
        left_K = self._rescale_intrinsic(left_K, self.image_size, original_size)

        return self._create_mono_samples(left_files, left_K)
    
    def _process(self, scene_dir: str):
        """씬 디렉토리 처리 - 스테레오/모노 자동 감지"""
        if self.mode == 'stereo':
            return self._process_stereo_scene(scene_dir)
        elif self.mode == 'mono':
            return self._process_mono_scene(scene_dir)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'stereo' or 'mono'.")


    def generate_datasets(self, fold_dir, shuffle=False):
        """데이터셋 생성 - dict 형태로 반환"""
        if not os.path.exists(fold_dir):
            print(f"Directory {fold_dir} does not exist")
            
            if self.mode == 'mono':
                return {
                    'source_left': np.array([]),
                    'target_image': np.array([]),
                    'source_right': np.array([]),
                    'intrinsic': np.array([])
                }
            else:  # stereo
                return {
                    'source_image': np.array([]),
                    'target_image': np.array([]),
                    'intrinsic': np.array([]),
                    'pose': np.array([])
                }
        
        scene_dirs = sorted(glob.glob(os.path.join(fold_dir, '*')))
        
        if self.mode == 'mono':
            # 전체 데이터를 저장할 리스트
            all_source_left = []
            all_target_image = []
            all_source_right = []
            all_intrinsic = []
            
            for scene_dir in scene_dirs:
                if os.path.isdir(scene_dir):
                    samples = self._process(scene_dir)
                    if samples:
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
            
            # 셔플링
            if shuffle and len(all_source_left) > 0:
                indices = np.random.permutation(len(all_source_left))
                for key in dataset_dict:
                    dataset_dict[key] = dataset_dict[key][indices]
            
            print(f"Generated {len(all_source_left)} mono samples from {fold_dir}")
            return dataset_dict
            
        elif self.mode == 'stereo':
            # 전체 데이터를 저장할 리스트
            all_source_image = []
            all_target_image = []
            all_intrinsic = []
            all_pose = []
            
            for scene_dir in scene_dirs:
                if os.path.isdir(scene_dir):
                    samples = self._process(scene_dir)
                    if samples:
                        all_source_image.extend(samples['source_image'])
                        all_target_image.extend(samples['target_image'])
                        all_intrinsic.extend(samples['intrinsic'])
                        all_pose.extend(samples['pose'])
            
            # numpy 배열로 변환
            dataset_dict = {
                'source_image': np.array(all_source_image, dtype=str),
                'target_image': np.array(all_target_image, dtype=str),
                'intrinsic': np.array(all_intrinsic, dtype=np.float32),
                'pose': np.array(all_pose, dtype=np.float32)
            }
            
            # 셔플링
            if shuffle and len(all_source_image) > 0:
                indices = np.random.permutation(len(all_source_image))
                for key in dataset_dict:
                    dataset_dict[key] = dataset_dict[key][indices]
            
            print(f"Generated {len(all_source_image)} stereo samples from {fold_dir}")
            return dataset_dict
        
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'stereo' or 'mono'.")

       
if __name__ == '__main__':
    # 설정 파일 로드
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    mono_dataset = IrsDataHandler(config, mode='mono')  # or mode='stereo'
    print(f"Train samples: {mono_dataset.train_data['source_left'].shape}")
    print(f"Valid samples: {mono_dataset.valid_data['source_left'].shape}")

    stereo_dataset = IrsDataHandler(config, mode='stereo')
    print(f"Train samples: {stereo_dataset.train_data['source_image'].shape}")
    print(f"Valid samples: {stereo_dataset.valid_data['source_image'].shape}")
