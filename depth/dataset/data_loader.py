import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
try:
    from .nyu_handler import NyuDepthLoader
except:
    from nyu_handler import NyuDepthLoader

class DataLoader(object):
    def __init__(self, config) -> None:
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self.dataset = self._load_dataset(root_dir=config['Directory']['data_dir'],
                                          data_type='nyu_depth_v2')

        self.train_dataset = self._compile_dataset(self.dataset.train_data, batch_size=self.batch_size, use_shuffle=True)
        self.valid_dataset = self._compile_dataset(self.dataset.valid_data, batch_size=self.batch_size, use_shuffle=False)
        
        self.num_train_samples = self.dataset.train_data.shape[0] // self.batch_size
        self.num_valid_samples = self.dataset.valid_data.shape[0] // self.batch_size
    
    def _load_dataset(self, root_dir: str, data_type: str):
        if data_type == 'nyu_depth_v2':
            return NyuDepthLoader(root_dir=os.path.join(root_dir, 'nyu_depth_v2_raw'))
        else:
            raise ValueError('Invalid data type.')

    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        return rgb_image
    
    @tf.function()
    def _read_depth(self, depth_path):
        image = tf.io.read_file(depth_path)
        depth = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
        return depth

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
        depth = (depth / 65535.0) * 10.0
        depth = tf.image.resize(depth,
                                self.image_size,
                                method=tf.image.ResizeMethod.BILINEAR)
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
    
    @tf.function()
    def load_data(self, sample: dict) -> tuple:
        rgb = self._read_image(sample['rgb'])
        depth = self._read_depth(sample['depth'])
        return rgb, depth
    
    @tf.function(jit_compile=True)
    def preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        rgb = self.preprocess_image(rgb)
        depth = self.preprocess_depth(depth)
        return rgb, depth

    def _compile_dataset(self, np_samples: np.ndarray, batch_size: int, use_shuffle: bool) -> tf.data.Dataset:
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: np_samples,
            output_signature={
                'rgb': tf.TensorSpec(shape=(), dtype=tf.string),
                'depth': tf.TensorSpec(shape=(), dtype=tf.string),
            }
        )
        if use_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 128, reshuffle_each_iteration=True)
        dataset = dataset.map(self.load_data, num_parallel_calls=self.auto_opt)
        dataset = dataset.map(self.preprocess, num_parallel_calls=self.auto_opt)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.auto_opt)
        return dataset

if __name__ == '__main__':
    root_dir = './depth/data/'
    config = {
        'Directory': {
            'data_dir': root_dir
        },
        'Train': {
            'batch_size': 1,
            'use_shuffle': True,
            'img_h': 256,
            'img_w': 256
        }
    }
    data_loader = DataLoader(config)
    
    for rgb, depth in data_loader.train_dataset.take(10):
        print(rgb.shape)
        print(depth.shape)
        plt.imshow(depth[0], cmap='plasma')
        plt.show()