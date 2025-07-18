import os
import glob
import numpy as np
import yaml

class CustomDataHandler:
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'redwood')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.depth_factor = 1000.
        
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.intrinsic = np.load(os.path.join(self.root_dir, 'intrinsic.npy'))
        
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)

    def generate_datasets(self, fold_dir, shuffle=False):
        # 전체 데이터를 저장할 리스트
        all_rgb_files = []
        all_depth_files = []
        
        # 각 scene 디렉토리 처리
        scene_dirs = sorted(glob.glob(os.path.join(fold_dir, '*')))
        
        for scene_dir in scene_dirs:
            if os.path.isdir(scene_dir):
                # RGB와 Depth 파일 경로
                rgb_dir = os.path.join(scene_dir, 'rgb_left')
                depth_dir = os.path.join(scene_dir, 'depth')
                
                if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
                    # 파일 리스트 가져오기
                    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
                    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
                    
                    # 파일 개수 확인
                    if len(rgb_files) == len(depth_files):
                        all_rgb_files.extend(rgb_files)
                        all_depth_files.extend(depth_files)
                    else:
                        print(f"Warning: Mismatch in {scene_dir} - RGB: {len(rgb_files)}, Depth: {len(depth_files)}")
        
        print(f"Found {len(all_rgb_files)} RGB-Depth pairs in {fold_dir}")
        
        dataset_dict = {
            'image': np.array(all_rgb_files, dtype=str),
            'depth': np.array(all_depth_files, dtype=str),
        }
        
        # 셔플링
        if shuffle and len(all_rgb_files) > 0:
            indices = np.random.permutation(len(all_rgb_files))
            for key in dataset_dict:
                dataset_dict[key] = dataset_dict[key][indices]
        return dataset_dict

if __name__ == '__main__':
    # 설정 파일 로드
    with open('./depth/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    redwood_dataset = CustomDataHandler(config)
    
    # 데이터 확인
    print(f"Train samples: {len(redwood_dataset.train_data['image'])}")
    print(f"Valid samples: {len(redwood_dataset.valid_data['image'])}")
    
    # 첫 번째 샘플 확인
    if len(redwood_dataset.train_data['image']) > 0:
        print(f"\nFirst train sample:")
        print(f"RGB: {redwood_dataset.train_data['image'][0]}")
        print(f"Depth: {redwood_dataset.train_data['depth'][0]}")