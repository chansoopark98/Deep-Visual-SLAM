import os
import glob
import pandas as pd
import numpy as np
import cv2
try:
    from .utils import resample_imu
except:
    from utils import resample_imu

class RedwoodHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/redwood'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source'] # 2
        self.imu_seq_len = self.config['Train']['imu_seq_len'] # 10
        self.original_image_size = (480, 640)
        self.train_dir = os.path.join(self.root_dir, 'train')
        # self.valid_dir = os.path.join(self.root_dir, 'valid')
        # self.test_dir = os.path.join(self.root_dir, 'test')
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        # self.valid_data = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)
        # self.test_data = self.generate_datasets(fold_dir=self.test_dir, shuffle=False)

    def _process(self, scene_dir: str):
        # load camera metadata
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'image', '*.jpg')))
        length = len(rgb_files)
        
        samples = []
        for t in range(self.num_source, length - self.num_source, 4):
            left_images = []
            right_images = []

            for step in range(1, self.num_source + 1):
                left_images.append(rgb_files[t - step])
                right_images.append(rgb_files[t + step])

            sample = {
                'source_left': left_images, # List [str, str]]
                'target_image': rgb_files[t], # str
                'source_right': right_images, # List [str, str]]
                'intrinsic': intrinsic # np.ndarray (3, 3)
            }
            samples.append(sample)
        return samples
            
    def generate_datasets(self, fold_dir, shuffle=False):
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        datasets = []
        for scene in scene_files:
            dataset = self._process(scene)
            datasets.append(dataset)
        datasets = np.concatenate(datasets, axis=0)

        if shuffle:
            np.random.shuffle(datasets)
        return datasets

if __name__ == '__main__':
    root_dir = './vo/data/'
    config = {
        'Directory': {
            'data_dir': root_dir
        },
        'Dataset':{
            'tspxr_capture': False,
            'mars_logger': True,
        },
        'Train': {
            'batch_size': 1,
            'use_shuffle': True,
            'num_source': 2,
            'imu_seq_len': 10,
            'img_h': 720,
            'img_w': 1280
        }
    }
    dataset = RedwoodHandler(config)
    data_len = dataset.train_data.shape[0]
    for idx in range(data_len):
        print(dataset.train_data[idx])
    