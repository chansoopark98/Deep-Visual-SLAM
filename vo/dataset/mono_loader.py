import numpy as np
import tensorflow as tf
try:
    from .mars_logger import MarsLoggerHandler
    from .redwood import RedwoodHandler
    from .custom_data import CustomDataHandler
    from .irs import IrsDataHandler
    from .augmentation_tool import Augmentations
except:
    from mars_logger import MarsLoggerHandler
    from redwood import RedwoodHandler
    from custom_data import CustomDataHandler
    from irs import IrsDataHandler
    from augmentation_tool import Augmentations

class MonoLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_source = config['Train']['num_source']
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self._load_dataset()
        self.num_mono_train = self.num_mono_train // self.batch_size
        self.num_mono_valid = self.num_mono_valid // self.batch_size
        self.num_mono_test = self.num_mono_test // self.batch_size

        self.augmentor = Augmentations(image_size=self.image_size)

        if self.num_mono_train > 0:
            self.train_mono_datasets = self._compile_dataset(self.train_mono_datasets, batch_size=self.batch_size, use_shuffle=True, is_train=True)
        if self.num_mono_valid > 0:
            self.valid_mono_datasets = self._compile_dataset(self.valid_mono_datasets, batch_size=self.batch_size, use_shuffle=False, is_train=False)
        if self.num_mono_test > 0:
            self.test_mono_datasets = self._compile_dataset(self.test_mono_datasets, batch_size=self.batch_size, use_shuffle=False, is_train=False)
        
    def _load_dataset(self):
        train_datasets = []
        valid_datasets = []
        test_datasets = []

        self.num_mono_train = 0
        self.num_mono_valid = 0
        self.num_mono_test = 0
        
        if self.config['Dataset']['mars_logger']:
            dataset = MarsLoggerHandler(config=self.config)
            train_dataset = self._build_generator(np_samples=dataset.train_data)
            valid_dataset = self._build_generator(np_samples=dataset.valid_data)
            test_dataset = self._build_generator(np_samples=dataset.test_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)

            self.num_mono_train += dataset.train_data.shape[0]
            self.num_mono_valid += dataset.valid_data.shape[0]
            self.num_mono_test += dataset.test_data.shape[0]
        
        # if self.config['Dataset']['custom_data']:
        #     dataset = CustomDataHandler(config=self.config, mode='mono')
        #     train_dataset = self._build_generator(np_samples=dataset.train_data)
        #     valid_dataset = self._build_generator(np_samples=dataset.valid_data)
            
        #     train_datasets.append(train_dataset)
        #     valid_datasets.append(valid_dataset)

        #     self.num_mono_train += dataset.train_data.shape[0]
        #     self.num_mono_valid += dataset.valid_data.shape[0]
        
        if self.config['Dataset']['irs']:
            dataset = IrsDataHandler(config=self.config, mode='mono')
            train_dataset = self._build_generator(np_samples=dataset.train_data)
            valid_dataset = self._build_generator(np_samples=dataset.valid_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)

            self.num_mono_train += dataset.train_data.shape[0]
            self.num_mono_valid += dataset.valid_data.shape[0]
        
        self.train_mono_datasets = train_datasets
        self.valid_mono_datasets = valid_datasets
        self.test_mono_datasets = test_datasets
        

    def _build_generator(self, np_samples: np.ndarray) -> tf.data.Dataset:
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: np_samples,
            output_signature = {
            'source_left': tf.TensorSpec(shape=(), dtype=tf.string),
            'target_image': tf.TensorSpec(shape=(), dtype=tf.string),
            'source_right': tf.TensorSpec(shape=(), dtype=tf.string),
            'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32),  # 3x3 행렬
            }
        )
        return dataset

    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        rgb_image = tf.cast(rgb_image, tf.uint8)
        return rgb_image

    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        image /= 255.0
        # image = (image - self.mean) / self.std
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        # image = (image * self.std) + self.mean
        image *= 255.0
        image = tf.cast(image, tf.uint8)
        return image
    
    @tf.function()
    def train_preprocess(self, sample: dict) -> tuple:
        # read images
        left_image = self._read_image(sample['source_left'])
        target_image = self._read_image(sample['target_image'])
        right_image = self._read_image(sample['source_right'])

        # read intrinsic
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)

        # augmentation
        left_image, right_image, target_image = self.augmentation(left_image, target_image, right_image)

        # normalize images
        left_image = self.normalize_image(left_image)
        target_image = self.normalize_image(target_image)
        right_image = self.normalize_image(right_image)
        
        processed_sample = {
            'source_left': left_image,
            'target_image': target_image,
            'source_right': right_image,
            'intrinsic': intrinsic,
        }
        return processed_sample
    
    @tf.function()
    def valid_process(self, sample: dict) -> tuple:
        # read images
        left_image = self._read_image(sample['source_left'])
        target_image = self._read_image(sample['target_image'])
        right_image = self._read_image(sample['source_right'])

        # read intrinsic
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)

        # normalize images
        left_image = self.normalize_image(left_image)
        target_image = self.normalize_image(target_image)
        right_image = self.normalize_image(right_image)
        
        processed_sample = {
            'source_left': left_image,
            'target_image': target_image,
            'source_right': right_image,
            'intrinsic': intrinsic,
        }
        return processed_sample
    
    
    @tf.function(jit_compile=True)
    def augmentation(self, left_image, target_image, right_image):
        left_image = tf.cast(left_image, tf.float32) / 255.0
        target_image = tf.cast(target_image, tf.float32) / 255.0
        right_image = tf.cast(right_image, tf.float32) / 255.0
        
        if tf.random.uniform([]) > 0.5:
            left_image = self.augmentor.image_left_right_flip(left_image)
            right_image = self.augmentor.image_left_right_flip(right_image)
            target_image = self.augmentor.image_left_right_flip(target_image)

        if tf.random.uniform([]) > 0.5:
            delta_brightness = tf.random.uniform([], -0.2, 0.2)
            left_image = tf.image.adjust_brightness(left_image, delta_brightness)
            right_image = tf.image.adjust_brightness(right_image, delta_brightness)
            target_image = tf.image.adjust_brightness(target_image, delta_brightness)
        
        if tf.random.uniform([]) > 0.5:
            contrast_factor = tf.random.uniform([], 0.8, 1.2)
            left_image = tf.image.adjust_contrast(left_image, contrast_factor)
            right_image = tf.image.adjust_contrast(right_image, contrast_factor)
            target_image = tf.image.adjust_contrast(target_image, contrast_factor)
        
        if tf.random.uniform([]) > 0.5:
            saturation_factor = tf.random.uniform([], 0.8, 1.2)
            left_image = tf.image.adjust_saturation(left_image, saturation_factor)
            right_image = tf.image.adjust_saturation(right_image, saturation_factor)
            target_image = tf.image.adjust_saturation(target_image, saturation_factor)
        
        if tf.random.uniform([]) > 0.5:
            hue_factor = tf.random.uniform([], -0.1, 0.1)
            left_image = tf.image.adjust_hue(left_image, hue_factor)
            right_image = tf.image.adjust_hue(right_image, hue_factor)
            target_image = tf.image.adjust_hue(target_image, hue_factor)

        left_image *= 255.
        right_image *= 255.
        target_image *= 255.

        left_image = tf.clip_by_value(left_image, 0., 255.)
        right_image = tf.clip_by_value(right_image, 0., 255.)
        target_image = tf.clip_by_value(target_image, 0., 255.)

        return left_image, right_image, target_image
    
    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool, is_train: bool = True) -> tf.data.Dataset:
        combined_dataset = tf.data.Dataset.sample_from_datasets(datasets)
        if use_shuffle:
            combined_dataset = combined_dataset.shuffle(buffer_size=batch_size * 256, reshuffle_each_iteration=True)
        if is_train:
            combined_dataset = combined_dataset.map(self.train_preprocess, num_parallel_calls=self.auto_opt)
        else:
            combined_dataset = combined_dataset.map(self.valid_process, num_parallel_calls=self.auto_opt)
        combined_dataset = combined_dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=self.auto_opt)
        combined_dataset = combined_dataset.prefetch(self.auto_opt)
        return combined_dataset

if __name__ == '__main__':
    import yaml   
    import matplotlib.pyplot as plt
    
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_loader = MonoLoader(config)
    
    # 샘플 시각화
    for i, sample in enumerate(data_loader.train_dataset.take(5)):
        left_image = sample['source_left'][0]
        right_image = sample['source_right'][0]
        target_image = sample['target_image'][0]
        
        left_image = data_loader.denormalize_image(left_image)
        right_image = data_loader.denormalize_image(right_image)
        target_image = data_loader.denormalize_image(target_image)   

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(left_image)
        axes[0].set_title('Source Left')
        axes[1].imshow(target_image)
        axes[1].set_title('Target')
        axes[2].imshow(right_image)
        axes[2].set_title('Source Right')
        
        plt.suptitle(f'Sample {i}')
        plt.tight_layout()
        plt.show()