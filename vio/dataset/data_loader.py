import numpy as np
import tensorflow as tf
try:
    from .tspxr_capture import TspxrCapture
    from .mars_logger import MarsLoggerHandler
except:
    from dataset.tspxr_capture import TspxrCapture
    from dataset.mars_logger import MarsLoggerHandler

class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self.train_dataset, self.valid_dataset, self.test_dataset = self._load_dataset()
        self.num_train_samples = self.num_train_samples // self.batch_size
        self.num_valid_samples = self.num_valid_samples // self.batch_size
        self.num_test_samples = self.num_test_samples // self.batch_size

        if self.num_train_samples > 0:
            self.train_dataset = self._compile_dataset(self.train_dataset, batch_size=self.batch_size, use_shuffle=True, is_train=True)
        if self.num_valid_samples > 0:
            self.valid_dataset = self._compile_dataset(self.valid_dataset, batch_size=self.batch_size, use_shuffle=False, is_train=False)
        if self.num_test_samples > 0:
            self.test_dataset = self._compile_dataset(self.test_dataset, batch_size=self.batch_size, use_shuffle=False, is_train=False)
        
    def _load_dataset(self):
        train_datasets = []
        valid_datasets = []
        test_datasets = []

        self.num_train_samples = 0
        self.num_valid_samples = 0
        self.num_test_samples = 0
        
        if self.config['Dataset']['tspxr_capture']:
            dataset = TspxrCapture(config=self.config)
            train_dataset  = self._build_generator(np_samples=dataset.train_data)
            valid_dataset = self._build_generator(np_samples=dataset.valid_data)
            test_dataset = self._build_generator(np_samples=dataset.test_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)

            self.num_train_samples += dataset.train_data.shape[0]
            self.num_valid_samples += dataset.valid_data.shape[0]
            self.num_test_samples += dataset.test_data.shape[0]
        
        if self.config['Dataset']['mars_logger']:
            dataset = MarsLoggerHandler(config=self.config)
            train_dataset = self._build_generator(np_samples=dataset.train_data)
            valid_dataset = self._build_generator(np_samples=dataset.valid_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)

            self.num_train_samples += dataset.train_data.shape[0]
            self.num_valid_samples += dataset.valid_data.shape[0]
        return train_datasets, valid_datasets, test_datasets


    def _build_generator(self, np_samples: np.ndarray) -> tf.data.Dataset:
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
        return dataset

    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        return rgb_image

    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        image /= 255.0
        image = (image - self.mean) / self.std
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        image = (image * self.std) + self.mean
        image *= 255.0
        image = tf.cast(image, tf.uint8)
        return image
    
    def train_preprocess(self, sample: dict) -> tuple:
        # read images
        source_left = self._read_image(sample['source_left'])
        target_image = self._read_image(sample['target_image'])
        source_right = self._read_image(sample['source_right'])

        # normalize images
        source_left = self.normalize_image(source_left)
        target_image = self.normalize_image(target_image)
        source_right = self.normalize_image(source_right)

        # Augmentation
        # TODO

        imu_left = tf.cast(sample['imu_left'], tf.float32)
        imu_right = tf.cast(sample['imu_right'], tf.float32)
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)

        images = tf.concat([source_left, target_image, source_right], axis=-1)
        imus = tf.concat([imu_left, imu_right], axis=-1)
        return images, imus, intrinsic
    
    def valid_preprocess(self, sample: dict) -> tuple:
        source_left = self._read_image(sample['source_left'])
        target_image = self._read_image(sample['target_image'])
        source_right = self._read_image(sample['source_right'])

        # normalize images
        source_left = self.normalize_image(source_left)
        target_image = self.normalize_image(target_image)
        source_right = self.normalize_image(source_right)
        
        imu_left = tf.cast(sample['imu_left'], tf.float32)
        imu_right = tf.cast(sample['imu_right'], tf.float32)
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)

        images = tf.concat([source_left, target_image, source_right], axis=-1)
        imus = tf.concat([imu_left, imu_right], axis=-1)
        return images, imus, intrinsic

    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool, is_train: bool = True) -> tf.data.Dataset:
        combined_dataset = tf.data.Dataset.sample_from_datasets(datasets, rerandomize_each_iteration=True)
        if use_shuffle:
            combined_dataset = combined_dataset.shuffle(buffer_size=batch_size * 256, reshuffle_each_iteration=True)
        if is_train:
            combined_dataset = combined_dataset.map(self.train_preprocess, num_parallel_calls=self.auto_opt)
        else:
            combined_dataset = combined_dataset.map(self.valid_preprocess, num_parallel_calls=self.auto_opt)
        combined_dataset = combined_dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=self.auto_opt)
        combined_dataset = combined_dataset.prefetch(self.auto_opt)
        return combined_dataset

if __name__ == '__main__':
    root_dir = './vio/data/'
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
            'img_h': 256,
            'img_w': 256
        }
    }
    data_loader = DataLoader(config)
    
    for image, imu, intrinsic in data_loader.train_dataset.take(10):
        print(image.shape, imu.shape, intrinsic.shape)
        print(intrinsic)

    # Test Files