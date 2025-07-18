import os
import glob
import numpy as np
import yaml
from common import BaseDepthDataset

class NyuDepthDataset(BaseDepthDataset):
    """NYU Depth V2 Dataset"""
    
    def __init__(self, root_dir: str, fold: str, image_size: tuple, is_train: bool = True, augment: bool = True):
        super().__init__(image_size, is_train, augment)
        
        self.root_dir = root_dir
        self.fold = fold
        self.depth_factor = 1000.0  # NYU depth is in mm
        
        # 데이터 경로 로드
        self._load_file_paths()
        
    def _load_file_paths(self):
        """파일 경로 로드"""
        fold_dir = os.path.join(self.root_dir, self.fold)
        rgb_dir = os.path.join(fold_dir, 'rgb')
        depth_dir = os.path.join(fold_dir, 'depth')
        
        # 파일 리스트 가져오기
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        
        # 파일 개수 확인
        if len(rgb_files) != len(depth_files):
            print(f"Warning: Mismatch - RGB: {len(rgb_files)}, Depth: {len(depth_files)}")
            min_len = min(len(rgb_files), len(depth_files))
            rgb_files = rgb_files[:min_len]
            depth_files = depth_files[:min_len]
        
        self.image_paths = rgb_files
        self.depth_paths = depth_files
        
        print(f"NYU Depth V2 {self.fold}: Found {len(self.image_paths)} RGB-Depth pairs")


class NyuDepthHandler:
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'nyu_depth_v2_raw')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        
        # 데이터셋 생성
        self.train_dataset = None
        self.valid_dataset = None
        
        if os.path.exists(os.path.join(self.root_dir, 'train')):
            self.train_dataset = NyuDepthDataset(
                root_dir=self.root_dir,
                fold='train',
                image_size=self.image_size,
                is_train=True,
                augment=True
            )
        
        if os.path.exists(os.path.join(self.root_dir, 'valid')):
            self.valid_dataset = NyuDepthDataset(
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
    nyu_handler = NyuDepthHandler(config)
    
    if nyu_handler.train_dataset:
        print(f"Train samples: {len(nyu_handler.train_dataset)}")
    if nyu_handler.valid_dataset:
        print(f"Valid samples: {len(nyu_handler.valid_dataset)}")