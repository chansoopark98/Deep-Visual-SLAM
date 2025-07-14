import os
import glob
import numpy as np
import yaml

class NyuDepthHandler:
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'nyu_depth_v2_raw')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.depth_factor = 1000.
        
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)

    def generate_datasets(self, fold_dir, shuffle=False):
        # 전체 데이터를 저장할 리스트
        
        rgb_dir = os.path.join(fold_dir, 'rgb')
        depth_dir = os.path.join(fold_dir, 'depth')
        
        # 파일 리스트 가져오기
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        
        print(f"Found {len(rgb_files)} RGB files and {len(depth_files)} depth files")

        dataset_dict = {
            'image': np.array(rgb_files, dtype=str),
            'depth': np.array(depth_files, dtype=str),
        }

        # 셔플링
        if shuffle and len(rgb_files) > 0:
            indices = np.random.permutation(len(rgb_files))
            for key in dataset_dict:
                dataset_dict[key] = dataset_dict[key][indices]
                
        return dataset_dict
            
       
if __name__ == '__main__':
    # 설정 파일 로드
    with open('./depth/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    nyu_dataset = NyuDepthHandler(config)
    