import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import os
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed

output_path = './depth/data/nyu_depth_v2_raw/'
os.makedirs(output_path, exist_ok=True)

# nyu_dataset_v2
nyu_path = './depth/data/'

def save_data_cv2(data_dict):
    """OpenCV를 사용한 더 빠른 저장"""
    try:
        image = data_dict['image']
        depth = data_dict['depth']
        image_path = data_dict['image_path']
        depth_path = data_dict['depth_path']
        
        # 디렉토리 확인
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        
        # RGB 저장 (OpenCV는 BGR 형식이므로 변환)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 뎁스 저장
        depth = np.clip(depth, 0., 10.0)
        depth_mm = (depth * 1000).astype(np.uint16)
        cv2.imwrite(depth_path, depth_mm)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# joblib를 사용한 병렬 처리 (가장 간단한 방법)
def process_with_joblib(dataset, split, output_path):
    # 데이터 준비
    data_dicts = []
    for i, item in enumerate(tqdm(dataset, desc=f"Preparing {split} data")):
        # 7자리 제로 패딩 적용
        file_id = str(i+1).zfill(7)  # 0000001, 0000002, ..., 0010000
        
        data_dict = {
            'image': item['image'].numpy(),
            'depth': item['depth'].numpy(),
            # 폴더 구조: train/rgb/0000001.png, train/depth/0000001.png
            'image_path': os.path.join(output_path, split, 'rgb', f'{file_id}.png'),
            'depth_path': os.path.join(output_path, split, 'depth', f'{file_id}.png')
        }
        data_dicts.append(data_dict)
    
    # 병렬 처리
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(save_data_cv2)(data_dict) 
        for data_dict in tqdm(data_dicts, desc=f"Saving {split} data")
    )
    
    print(f"Successfully saved {sum(results)} out of {len(results)} files")
    
    # 저장 결과 확인
    rgb_dir = os.path.join(output_path, split, 'rgb')
    depth_dir = os.path.join(output_path, split, 'depth')
    
    if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
        rgb_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        depth_count = len([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        print(f"Saved {rgb_count} RGB images and {depth_count} depth images in {split} folder")

if __name__ == "__main__":
    # 데이터셋 로드
    nyu_dataset_train = tfds.load('nyu_depth_v2', data_dir=nyu_path, split='train', shuffle_files=False)
    nyu_dataset_valid = tfds.load('nyu_depth_v2', data_dir=nyu_path, split='validation', shuffle_files=False)
    
    # 최종 폴더 구조 출력
    print("Data will be saved in the following structure:")
    print("nyu_depth_v2_raw/")
    print("├── train/")
    print("│   ├── rgb/")
    print("│   │   ├── 0000001.png")
    print("│   │   └── ...")
    print("│   └── depth/")
    print("│       ├── 0000001.png")
    print("│       └── ...")
    print("└── valid/")
    print("    ├── rgb/")
    print("    └── depth/")
    print()
    
    # joblib 방식으로 처리
    process_with_joblib(nyu_dataset_train, 'train', output_path)
    process_with_joblib(nyu_dataset_valid, 'valid', output_path)