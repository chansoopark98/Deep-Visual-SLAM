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
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        rgb_image = tf.cast(rgb_image, tf.float32)
        rgb_image = self.normalize_image(rgb_image)
        return rgb_image
    
    @tf.function()
    def _read_depth(self, depth_path):
        image = tf.io.read_file(depth_path)
        depth = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
        depth = tf.cast(depth, tf.float32)
        depth = (depth / 65535.0) * 10.0
        depth = tf.image.resize(depth, self.image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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
    
    def preprocess(self, sample: dict) -> tuple:
        rgb = self._read_image(sample['rgb'])
        depth = self._read_depth(sample['depth'])
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