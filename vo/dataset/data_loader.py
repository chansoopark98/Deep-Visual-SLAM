import numpy as np
import tensorflow as tf
try:
    from .custom_data import CustomDataHandler
    from .mars_logger import MarsLoggerHandler
    from .augmentation_tool import Augmentations
except:
    from custom_data import CustomDataHandler
    from mars_logger import MarsLoggerHandler
    from augmentation_tool import Augmentations

class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_source = config['Train']['num_source']
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self.train_dataset, self.valid_dataset, self.test_dataset = self._load_dataset()
        self.num_train_samples = self.num_train_samples // self.batch_size
        self.num_valid_samples = self.num_valid_samples // self.batch_size
        self.num_test_samples = self.num_test_samples // self.batch_size

        self.augmentor = Augmentations(image_size=self.image_size)

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
        
        # if self.config['Dataset']['custom_data']:
        #     dataset = CustomDataHandler(config=self.config)
        #     train_dataset = self._build_generator(np_samples=dataset.train_data)
        #     valid_dataset = self._build_generator(np_samples=dataset.valid_data)

        #     train_datasets.append(train_dataset)
        #     valid_datasets.append(valid_dataset)

        #     self.num_train_samples += dataset.train_data.shape[0]
        #     self.num_valid_samples += dataset.valid_data.shape[0]

        if self.config['Dataset']['mars_logger']:
            dataset = MarsLoggerHandler(config=self.config)
            train_dataset = self._build_generator(np_samples=dataset.train_data)
            valid_dataset = self._build_generator(np_samples=dataset.valid_data)
            test_dataset = self._build_generator(np_samples=dataset.test_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)

            self.num_train_samples += dataset.train_data.shape[0]
            self.num_valid_samples += dataset.valid_data.shape[0]
            self.num_test_samples += dataset.test_data.shape[0]
        return train_datasets, valid_datasets, test_datasets

    def _build_generator(self, np_samples: np.ndarray) -> tf.data.Dataset:
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: np_samples,
            output_signature={
                'source_left': tf.TensorSpec(shape=(None,), dtype=tf.string),  # 리스트로 가변 길이
                'target_image': tf.TensorSpec(shape=(), dtype=tf.string),  # 단일 스트링
                'source_right': tf.TensorSpec(shape=(None,), dtype=tf.string),  # 리스트로 가변 길이
                'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32)  # 고정 크기 (3, 3)
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
        image = (image - self.mean) / self.std
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        image = (image * self.std) + self.mean
        image *= 255.0
        image = tf.cast(image, tf.uint8)
        return image
    
    @tf.function()
    def train_preprocess(self, sample: dict) -> tuple:
        # read images
        left_images = tf.map_fn(self._read_image, sample['source_left'], fn_output_signature=tf.uint8)
        right_images = tf.map_fn(self._read_image, sample['source_right'], fn_output_signature=tf.uint8)
        target_image = self._read_image(sample['target_image'])

        # read intrinsic
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)

        # augmentation
        left_images, right_images, target_image, intrinsic = self.augmentation(left_images, right_images, target_image, intrinsic)

        # normalize images
        left_images = tf.map_fn(self.normalize_image, left_images, fn_output_signature=tf.float32)
        right_images = tf.map_fn(self.normalize_image, right_images, fn_output_signature=tf.float32)
        
        target_image = self.normalize_image(target_image)

        ref_images = tf.concat([left_images, right_images], axis=0)

        return ref_images, target_image, intrinsic
    
    @tf.function()
    def valid_process(self, sample: dict) -> tuple:
        # read images
        left_images = tf.map_fn(self._read_image, sample['source_left'], fn_output_signature=tf.uint8)
        right_images = tf.map_fn(self._read_image, sample['source_right'], fn_output_signature=tf.uint8)
        target_image = self._read_image(sample['target_image'])

        # read intrinsic
        intrinsic = tf.cast(sample['intrinsic'], tf.float32)

        # normalize images
        left_images = tf.map_fn(self.normalize_image, left_images, fn_output_signature=tf.float32)
        right_images = tf.map_fn(self.normalize_image, right_images, fn_output_signature=tf.float32)
        
        target_image = self.normalize_image(target_image)

        ref_images = tf.concat([left_images, right_images], axis=0)

        return ref_images, target_image, intrinsic
    
    @tf.function(jit_compile=True)
    def augmentation(self, left_image, right_image, target_image, intrinsic):
        left_image = tf.cast(left_image, tf.float32) / 255.0
        right_image = tf.cast(right_image, tf.float32) / 255.0
        target_image = tf.cast(target_image, tf.float32) / 255.0

        if tf.random.uniform([]) > 0.5:
            left_image = tf.map_fn(self.augmentor.image_left_right_flip, left_image)
            right_image = tf.map_fn(self.augmentor.image_left_right_flip, right_image)
            target_image = self.augmentor.image_left_right_flip(target_image)

        if tf.random.uniform([]) > 0.5:
            delta_brightness = tf.random.uniform([], -0.2, 0.2)
            left_image = tf.map_fn(lambda x: tf.image.adjust_brightness(x, delta_brightness), left_image)
            right_image = tf.map_fn(lambda x: tf.image.adjust_brightness(x, delta_brightness), right_image)
            target_image = tf.image.adjust_brightness(target_image, delta_brightness)
        
        if tf.random.uniform([]) > 0.5:
            contrast_factor = tf.random.uniform([], 0.8, 1.2)
            left_image = tf.map_fn(lambda x: tf.image.adjust_contrast(x, contrast_factor), left_image)
            right_image = tf.map_fn(lambda x: tf.image.adjust_contrast(x, contrast_factor), right_image)
            target_image = tf.image.adjust_contrast(target_image, contrast_factor)
        
        if tf.random.uniform([]) > 0.5:
            saturation_factor = tf.random.uniform([], 0.8, 1.2)
            left_image = tf.map_fn(lambda x: tf.image.adjust_saturation(x, saturation_factor), left_image)
            right_image = tf.map_fn(lambda x: tf.image.adjust_saturation(x, saturation_factor), right_image)
            target_image = tf.image.adjust_saturation(target_image, saturation_factor)
        
        if tf.random.uniform([]) > 0.5:
            hue_factor = tf.random.uniform([], -0.1, 0.1)
            left_image = tf.map_fn(lambda x: tf.image.adjust_hue(x, hue_factor), left_image)
            right_image = tf.map_fn(lambda x: tf.image.adjust_hue(x, hue_factor), right_image)
            target_image = tf.image.adjust_hue(target_image, hue_factor)

        left_image *= 255.
        right_image *= 255.
        target_image *= 255.

        left_image = tf.clip_by_value(left_image, 0., 255.)
        right_image = tf.clip_by_value(right_image, 0., 255.)
        target_image = tf.clip_by_value(target_image, 0., 255.)

        return left_image, right_image, target_image, intrinsic
    
    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool, is_train: bool = True) -> tf.data.Dataset:
        combined_dataset = tf.data.Dataset.sample_from_datasets(datasets, rerandomize_each_iteration=True)
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
    with open('./vio/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_loader = DataLoader(config)
    
    for sample in data_loader.train_dataset.take(100):
        ref_images, target_image, intrinsic = sample

        """
        ref_images: (batch_size, num_source * 2, img_h, img_w, 3)
            left_images: ref_images[:, :num_source]
            right_images: ref_images[:, num_source:]

        target_image: (batch_size, img_h, img_w, 3)

        imus: (batch_size, imu_seq_len, 12)
           left_imus: imug[:, :num_source]
           right_imus: imug[:, :, num_source:]

        intrinsic: (batch_size, 3, 3)
        """

        left_images = data_loader.denormalize_image(ref_images[:, :data_loader.num_source])
        right_images = data_loader.denormalize_image(ref_images[:, data_loader.num_source:])
        print(left_images.shape, right_images.shape)

        # show all images
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(left_images[0, 0])
        axes[1].imshow(left_images[0, 1])
        axes[2].imshow(right_images[0, 0])
        axes[3].imshow(right_images[0, 1])
        axes[4].imshow(data_loader.denormalize_image(target_image[0]))
        plt.show()
