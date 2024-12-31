import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
try:
    from .nyu_handler import NyuDepthLoader
    from .tfrecord_loader import TFRecordLoader
except:
    from nyu_handler import NyuDepthLoader
    from tfrecord_loader import TFRecordLoader

class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = self.config['Train']['batch_size']
        self.use_shuffle = self.config['Train']['use_shuffle']
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.max_depth = self.config['Train']['max_depth']
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self.train_datasets, self.valid_datasets,\
              self.num_train_samples, self.num_valid_samples = self._load_dataset()
        
        self.num_train_samples = self.num_train_samples // self.batch_size
        self.num_valid_samples = self.num_valid_samples // self.batch_size

        self.train_dataset = self._compile_dataset(self.train_datasets, batch_size=self.batch_size, use_shuffle=True)
        self.valid_dataset = self._compile_dataset(self.valid_datasets, batch_size=self.batch_size, use_shuffle=False)
        
    def _load_dataset(self) -> list:
        train_datasets = []
        valid_datasets = []
        train_samples = 0
        valid_samples = 0

        if self.config['Dataset']['Nyu_depth_v2']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'nyu_depth_v2_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name)
            train_datasets.append(dataset.train_dataset)
            valid_datasets.append(dataset.valid_dataset)

            train_samples += dataset.train_samples
            valid_samples += dataset.valid_samples

        if self.config['Dataset']['Diode']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'diode_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name)
            train_datasets.append(dataset.train_dataset)
            valid_datasets.append(dataset.valid_dataset)

            train_samples += dataset.train_samples
            valid_samples += dataset.valid_samples
        return train_datasets, valid_datasets, train_samples, valid_samples

    @tf.function(jit_compile=True)
    def preprocess_image(self, rgb: tf.Tensor):
        rgb = tf.image.resize(rgb,
                              self.image_size,
                              method=tf.image.ResizeMethod.BILINEAR)
        rgb = tf.cast(rgb, tf.float32)
        rgb = self.normalize_image(rgb)
        return rgb

    @tf.function(jit_compile=True)
    def preprocess_depth(self, depth: tf.Tensor):
        depth = tf.cast(depth, tf.float32)
        depth = tf.image.resize(depth,
                                self.image_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth = tf.where(depth > self.max_depth, 0., depth)
        return depth
        
    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image /= 255.0
        image = (image - self.mean) / self.std
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        image = (image * self.std) + self.mean
        image *= 255.0
        image = tf.cast(image, tf.uint8)
        return image
    
    @tf.function(jit_compile=True)
    def preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        # 1. Augmentation
        rgb, depth = self.augment(rgb, depth)

        # 2. Resize
        rgb = tf.image.resize(rgb, self.image_size,
                              method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, self.image_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 3. Normalize
        rgb = self.normalize_image(rgb)

        return rgb, depth
    
    @tf.function(jit_compile=True)
    def augment(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        rgb: RGB image tensor (H, W, 3) [0, 255]
        depth: Depth image tensor (H, W, 1) [0, max_depth]
        """
        # rgb augmentations
        # rgb = tf.cast(rgb, tf.float32) / 255.0

        # if tf.random.uniform([]) > 0.5:
        #     delta_brightness = tf.random.uniform([], -0.2, 0.2)
        #     rgb = tf.image.adjust_brightness(rgb, delta_brightness)
        
        # if tf.random.uniform([]) > 0.5:
        #     contrast_factor = tf.random.uniform([], 0.7, 1.3)
        #     rgb = tf.image.adjust_contrast(rgb, contrast_factor)
        
        # if tf.random.uniform([]) > 0.5:
        #     gamma = tf.random.uniform([], 0.8, 1.2)
        #     rgb = tf.image.adjust_gamma(rgb, gamma)
        
        # if tf.random.uniform([]) > 0.5:
        #     max_delta = 0.1
        #     rgb = tf.image.adjust_hue(rgb, tf.random.uniform([], -max_delta, max_delta))

        # flip left-right
        if tf.random.uniform([]) > 0.5:
            rgb = tf.image.flip_left_right(rgb)
            depth = tf.image.flip_left_right(depth)

        # back to [0, 255]
        # rgb = tf.clip_by_value(rgb, 0., 255.)
        # rgb = tf.cast(rgb * 255.0, tf.uint8)

        return rgb, depth
        

    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool) -> tf.data.Dataset:
        combined_dataset: tf.data.Dataset = datasets[0]
        
        for ds in datasets[1:]:
            combined_dataset = combined_dataset.concatenate(ds)
            
        if use_shuffle:
            combined_dataset = combined_dataset.shuffle(buffer_size=batch_size * 128, reshuffle_each_iteration=True)
        combined_dataset = combined_dataset.map(self.preprocess, num_parallel_calls=self.auto_opt)
        combined_dataset = combined_dataset.batch(batch_size, drop_remainder=True)
        combined_dataset = combined_dataset.prefetch(self.auto_opt)
        return combined_dataset

if __name__ == '__main__':
    root_dir = './depth/data/'
    config = {
        'Directory': {
            'data_dir': root_dir
        },
        'Dataset':{
            'Nyu_depth_v2': False,
            'Diode': True
        },
        'Train': {
            'batch_size': 1,
            'max_depth': 10.,
            'use_shuffle': True,
            'img_h': 256,
            'img_w': 256
        }
    }
    data_loader = DataLoader(config)
    
    for rgb, depth in data_loader.train_dataset.take(100):
        print(rgb.shape)
        print(depth.shape)
        rgb = data_loader.denormalize_image(rgb)
        plt.imshow(rgb[0])
        plt.show()
        plt.imshow(depth[0], cmap='plasma')
        plt.show()