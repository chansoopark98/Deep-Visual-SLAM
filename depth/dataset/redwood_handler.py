import os
import glob
import numpy as np
import yaml
from .common import BaseDepthDataset


class RedwoodDataset(BaseDepthDataset):
    """Redwood RGBD Dataset"""
    
    def __init__(self, root_dir: str, fold: str, image_size: tuple, is_train: bool = True, augment: bool = True):
        super().__init__(image_size, is_train, augment)
        
        self.root_dir = root_dir
        self.fold = fold
        self.depth_factor = 1000.0  # Redwood depth is in mm
        
        # intrinsic 파일 로드 (있는 경우)
        intrinsic_path = os.path.join(self.root_dir, 'intrinsic.npy')
        if os.path.exists(intrinsic_path):
            self.intrinsic = np.load(intrinsic_path)
        else:
            self.intrinsic = None
        
        # 데이터 경로 로드
        self._load_file_paths()
        
    def _load_file_paths(self):
        """파일 경로 로드"""
        fold_dir = os.path.join(self.root_dir, self.fold)
        
        all_rgb_files = []
        all_depth_files = []
        
        # 각 scene 디렉토리 처리
        scene_dirs = sorted(glob.glob(os.path.join(fold_dir, '*')))
        
        for scene_dir in scene_dirs:
            if os.path.isdir(scene_dir):
                # RGB와 Depth 파일 경로
                rgb_dir = os.path.join(scene_dir, 'image')
                depth_dir = os.path.join(scene_dir, 'depth')
                
                if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
                    # 파일 리스트 가져오기
                    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
                    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
                    
                    # 파일 개수 확인
                    if len(rgb_files) == len(depth_files):
                        all_rgb_files.extend(rgb_files)
                        all_depth_files.extend(depth_files)
                    else:
                        print(f"Warning: Mismatch in {scene_dir} - RGB: {len(rgb_files)}, Depth: {len(depth_files)}")
        
        self.image_paths = all_rgb_files
        self.depth_paths = all_depth_files
        
        # 셔플링 (훈련 데이터의 경우)
        if self.is_train and len(self.image_paths) > 0:
            indices = np.random.permutation(len(self.image_paths))
            self.image_paths = [self.image_paths[i] for i in indices]
            self.depth_paths = [self.depth_paths[i] for i in indices]
        
        print(f"Redwood {self.fold}: Found {len(self.image_paths)} RGB-Depth pairs")


class RedwoodHandler:
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'redwood')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        
        # 데이터셋 생성
        self.train_dataset = None
        self.valid_dataset = None
        
        if os.path.exists(os.path.join(self.root_dir, 'train')):
            self.train_dataset = RedwoodDataset(
                root_dir=self.root_dir,
                fold='train',
                image_size=self.image_size,
                is_train=True,
                augment=True
            )
        
        if os.path.exists(os.path.join(self.root_dir, 'valid')):
            self.valid_dataset = RedwoodDataset(
                root_dir=self.root_dir,
                fold='valid',
                image_size=self.image_size,
                is_train=False,
                augment=False
            )


if __name__ == '__main__':
    # 설정 파일 로드
    with open('./depth/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    redwood_handler = RedwoodHandler(config)
    
    # 데이터 확인
    if redwood_handler.train_dataset:
        print(f"Train samples: {len(redwood_handler.train_dataset)}")
    if redwood_handler.valid_dataset:
        print(f"Valid samples: {len(redwood_handler.valid_dataset)}")