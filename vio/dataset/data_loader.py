import numpy as np
import tensorflow as tf
try:
    from .data_handler import TspxrCapture
except:
    from data_handler import TspxrCapture

class DataLoader(TspxrCapture):
    def __init__(self, config) -> None:
        super().__init__(root_dir=config['Directory']['data_dir'])
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self.train_dataset = self._compile_dataset(self.train_data, batch_size=self.batch_size, use_shuffle=True)
        self.valid_dataset = self._compile_dataset(self.valid_data, batch_size=self.batch_size, use_shuffle=False)
        self.test_dataset = self._compile_dataset(self.test_data, batch_size=self.batch_size, use_shuffle=False)
        self.num_train_samples = self.train_data.shape[0] // self.batch_size
        self.num_valid_samples = self.valid_data.shape[0] // self.batch_size
        self.num_test_samples = self.valid_data.shape[0] // self.batch_size

    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        rgb_image = tf.cast(rgb_image, tf.float32)
        rgb_image = self.normalize_image(rgb_image)
        return rgb_image

    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image /= 255.0
        image = (image - self.mean) / self.std
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        image *= 255.0
        image = (image * self.std) + self.mean
        return image
    
    def preprocess(self, sample: dict) -> tuple:
        source_left = self._read_image(sample['source_left'])
        target_image = self._read_image(sample['target_image'])
        source_right = self._read_image(sample['source_right'])
        imu_left = tf.cast(sample['imu_left'], tf.float32)
        imu_right = tf.cast(sample['imu_right'], tf.float32)
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)
        
        images = tf.concat([source_left, target_image, source_right], axis=-1)
        imus = tf.concat([imu_left, imu_right], axis=-1)
        return images, imus, intrinsic

    def _compile_dataset(self, np_samples: np.ndarray, batch_size: int, use_shuffle: bool) -> tf.data.Dataset:
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: np_samples,
            output_signature={
                'source_left': tf.TensorSpec(shape=(), dtype=tf.string),
                'target_image': tf.TensorSpec(shape=(), dtype=tf.string),
                'source_right': tf.TensorSpec(shape=(), dtype=tf.string),
                'imu_left': tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
                'imu_right': tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
                'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32)
            }
        )
        if use_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 128, reshuffle_each_iteration=True)
        dataset = dataset.map(self.preprocess, num_parallel_calls=self.auto_opt)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.auto_opt)
        return dataset

if __name__ == '__main__':
    root_dir = './vio/data/'
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
    
    for sample in data_loader.train_dataset.take(10):
        print(sample)

    # Test Files