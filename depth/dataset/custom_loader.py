import os
import glob
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

class CustomLoader(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'capture_dataset')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        
        self.train_dataset, self.train_samples = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        self.valid_dataset, self.valid_samples = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)

    def _process(self, scene_dir: str):
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*')))
        depth_files = sorted(glob.glob(os.path.join(scene_dir, 'depth', '*')))
        samples = []
        for idx in range(len(rgb_files)):
            sample = {
                'image': rgb_files[idx],
                'depth': depth_files[idx]
            }
            samples.append(sample)
        return samples
           
    def generate_datasets(self, fold_dir, shuffle=False):
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        raw_datasets = []
        for scene in scene_files:
            dataset = self._process(scene)
            raw_datasets.append(dataset)
        raw_datasets = np.concatenate(raw_datasets, axis=0)

        if shuffle:
            np.random.shuffle(raw_datasets)

        data_len = raw_datasets.shape[0]

        tf_dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: raw_datasets,
            output_signature={
                'image': tf.TensorSpec(shape=(), dtype=tf.string),
                'depth': tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )
        tf_dataset = tf_dataset.map(self.parse_function)
        return tf_dataset, data_len
    
    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        rgb_image = tf.cast(rgb_image, tf.uint8)
        return rgb_image

    def load_npy_file(self, path):
        # path는 EagerTensor이므로 numpy().decode()로 Python 문자열 변환
        data = np.load(path.numpy().decode('utf-8'))  # `.npy` 파일 로드
        return data.astype(np.float32)    

    @tf.function()
    def _read_depth(self, depth_path):
        depth_image = tf.io.read_file(depth_path)
        depth_image = tf.io.decode_png(depth_image, channels=1)
        depth_image = tf.image.resize(depth_image, self.image_size)
        depth_image = tf.cast(depth_image, tf.float32)
        depth_image /= 255.0
        depth_image *= 15.0
        return depth_image
    
    @tf.function()
    def parse_function(self, sample):
        image = self._read_image(sample['image'])
        depth = self._read_depth(sample['depth'])
        return (image, depth)

if __name__ == '__main__':
    root_dir = './depth/data/'
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
    dataset = CustomLoader(config)
    data_len = dataset.train_samples
    data = dataset.train_dataset
    for rgb, depth in dataset.train_dataset.take(100):
        print(rgb.shape, depth.shape)
    