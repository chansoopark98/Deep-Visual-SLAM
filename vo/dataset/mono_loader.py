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
            train_dataset = self._build_generator(samples=dataset.train_data)
            valid_dataset = self._build_generator(samples=dataset.valid_data)
            test_dataset = self._build_generator(samples=dataset.test_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)

            self.num_mono_train += dataset.train_data['source_left'].shape[0]
            self.num_mono_valid += dataset.valid_data['source_left'].shape[0]
            self.num_mono_test += dataset.test_data['source_left'].shape[0]
        
        if self.config['Dataset']['custom_data']:
            dataset = CustomDataHandler(config=self.config, mode='mono')
            train_dataset = self._build_generator(samples=dataset.train_data)
            valid_dataset = self._build_generator(samples=dataset.valid_data)
            
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)

            self.num_mono_train += dataset.train_data['source_left'].shape[0]
            self.num_mono_valid += dataset.valid_data['source_left'].shape[0]
        
        if self.config['Dataset']['irs']:
            dataset = IrsDataHandler(config=self.config, mode='mono')
            train_dataset = self._build_generator(samples=dataset.train_data)
            valid_dataset = self._build_generator(samples=dataset.valid_data)

            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)

            self.num_mono_train += dataset.train_data['source_left'].shape[0]
            self.num_mono_valid += dataset.valid_data['source_left'].shape[0]
        
        self.train_mono_datasets = train_datasets
        self.valid_mono_datasets = valid_datasets
        self.test_mono_datasets = test_datasets
        

    # def _build_generator(self, np_samples: np.ndarray) -> tf.data.Dataset:
    #     dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
    #         lambda: np_samples,
    #         output_signature = {
    #         'source_left': tf.TensorSpec(shape=(), dtype=tf.string),
    #         'target_image': tf.TensorSpec(shape=(), dtype=tf.string),
    #         'source_right': tf.TensorSpec(shape=(), dtype=tf.string),
    #         'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32),  # 3x3 행렬
    #         }
    #     )
    #     return dataset

    def _build_generator(self, samples: dict) -> tf.data.Dataset:
        # from_tensor_slices 사용
        dataset = tf.data.Dataset.from_tensor_slices({
            'source_left': samples['source_left'],
            'target_image': samples['target_image'],
            'source_right': samples['source_right'],
            'intrinsic': samples['intrinsic']
        })
        
        return dataset

    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        
        is_png = tf.strings.regex_full_match(rgb_path, ".*\\.png$")
        
        rgb_image = tf.cond(
            is_png,
            lambda: tf.io.decode_png(rgb_image, channels=3),
            lambda: tf.io.decode_jpeg(rgb_image, channels=3, dct_method='INTEGER_FAST')
        )
        
        rgb_image = tf.image.resize(rgb_image, self.image_size, method='bilinear')
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
        # uint8 -> float32 (0-1 범위)로 변환
        left_image = tf.image.convert_image_dtype(left_image, tf.float32)
        target_image = tf.image.convert_image_dtype(target_image, tf.float32)
        right_image = tf.image.convert_image_dtype(right_image, tf.float32)
        
        # 스택하여 한번에 처리
        images = tf.stack([left_image, right_image, target_image], axis=0)
        
        # 랜덤 값들을 미리 생성
        do_flip = tf.random.uniform([]) > 0.5
        do_brightness = tf.random.uniform([]) > 0.5
        do_contrast = tf.random.uniform([]) > 0.5
        do_saturation = tf.random.uniform([]) > 0.5
        do_hue = tf.random.uniform([]) > 0.5
        
        if do_flip:
            images = tf.image.flip_left_right(images)
        
        if do_brightness:
            delta_brightness = tf.random.uniform([], -0.2, 0.2)
            images = tf.image.adjust_brightness(images, delta_brightness)
        
        if do_contrast:
            contrast_factor = tf.random.uniform([], 0.8, 1.2)
            images = tf.image.adjust_contrast(images, contrast_factor)
        
        if do_saturation:
            saturation_factor = tf.random.uniform([], 0.8, 1.2)
            images = tf.image.adjust_saturation(images, saturation_factor)
        
        if do_hue:
            hue_factor = tf.random.uniform([], -0.2, 0.2)
            images = tf.image.adjust_hue(images, hue_factor)
        
        # 클리핑 (0-1 범위 유지)
        images = tf.clip_by_value(images, 0.0, 1.0)
        
        # float32 -> uint8로 변환
        images = tf.image.convert_image_dtype(images, tf.uint8)
        
        # 언스택
        left_image, right_image, target_image = tf.unstack(images, axis=0)
        
        return left_image, right_image, target_image
    
    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool, is_train: bool = True) -> tf.data.Dataset:
        if len(datasets) == 0:
            raise ValueError("No datasets provided")
        
        # 단일 데이터셋인 경우
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        else:
            # 여러 데이터셋을 interleave로 결합
            weights = [1.0] * len(datasets)
            combined_dataset = tf.data.Dataset.sample_from_datasets(
                datasets, 
                weights=weights,
                stop_on_empty_dataset=True
            )
        
        # 셔플링
        if use_shuffle:
            combined_dataset = combined_dataset.shuffle(
                buffer_size=min(10000, batch_size * 512),
                reshuffle_each_iteration=True,
                seed=None  # 매번 다른 시드
            )
        
        # 전처리 맵핑
        map_func = self.train_preprocess if is_train else self.valid_process
        combined_dataset = combined_dataset.map(
            map_func,
            num_parallel_calls=self.auto_opt,
            deterministic=not is_train  # 훈련 시에는 비결정적으로
        )
        
        # 배치 처리
        combined_dataset = combined_dataset.batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=self.auto_opt
        )
        
        # 프리페칭
        combined_dataset = combined_dataset.prefetch(buffer_size=self.auto_opt)
        
        return combined_dataset

if __name__ == '__main__':
    import yaml   
    import matplotlib.pyplot as plt
    
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_loader = MonoLoader(config)
    
    import time
    avg_time = 0.0
    for i, sample in enumerate(data_loader.train_mono_datasets.take(1000)):
        start_time = time.time()
        left_image = sample['source_left'][0]
        right_image = sample['source_right'][0]
        target_image = sample['target_image'][0]
        intrinsic = sample['intrinsic'][0]
        
        left_image = data_loader.denormalize_image(left_image)
        right_image = data_loader.denormalize_image(right_image)
        target_image = data_loader.denormalize_image(target_image)

        avg_time += time.time() - start_time

        if i % 100 == 0:
            print(f"Processed {i} samples, average time: {avg_time / (i + 1):.4f} seconds per sample")


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(left_image)
        axes[0].set_title('Source Left')
        axes[1].imshow(target_image)
        axes[1].set_title('Target')
        axes[2].imshow(right_image)
        axes[2].set_title('Source Right')

        print(f"Intrinsic: {intrinsic.numpy()}")
        print(f'image shape: {left_image.shape}, {target_image.shape}, {right_image.shape}')
        
        plt.suptitle(f'Sample {i}')
        plt.tight_layout()
        plt.show()