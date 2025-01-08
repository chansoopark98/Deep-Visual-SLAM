import os
import tensorflow as tf
try:
    from .tfrecord_loader import TFRecordLoader
    from .nyu_handler import NyuHandler
    from .diode_handler import DiodeHandler
    from .diml_handler import DimlHandler
except:
    from tfrecord_loader import TFRecordLoader
    from nyu_handler import NyuHandler
    from diode_handler import DiodeHandler
    from diml_handler import DimlHandler

class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = self.config['Train']['batch_size']
        self.use_shuffle = self.config['Train']['use_shuffle']
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        self.min_depth = self.config['Train']['min_depth']
        self.max_depth = self.config['Train']['max_depth']
        self.num_train_samples = 0
        self.num_valid_samples = 0

        self.train_datasets, self.valid_datasets = self._load_dataset()
        
        self.num_train_samples = self.num_train_samples // self.batch_size
        self.num_valid_samples = self.num_valid_samples // self.batch_size

        self.train_dataset = self._compile_dataset(self.train_datasets, batch_size=self.batch_size, use_shuffle=True, is_train=True)
        if self.valid_datasets:
            self.valid_dataset = self._compile_dataset(self.valid_datasets, batch_size=self.batch_size, use_shuffle=False, is_train=False)
        
    def _load_dataset(self) -> list:
        train_datasets = []
        valid_datasets = []
        
        if self.config['Dataset']['Nyu_depth_v2']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'nyu_depth_v2_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float32)
            handler = NyuHandler(target_size=self.image_size)
            dataset.train_dataset = dataset.train_dataset.map(handler.nyu_crop_resize,
                                                              num_parallel_calls=self.auto_opt)
            train_datasets.append(dataset.train_dataset)
            valid_datasets.append(dataset.valid_dataset)

            self.num_train_samples += dataset.train_samples
            self.num_valid_samples += dataset.valid_samples

        if self.config['Dataset']['Diode']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'diode_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float32)
            # handler = DiodeHandler(target_size=self.image_size)
            # dataset.train_dataset = dataset.train_dataset.map(handler.preprocess,
            #                                                   num_parallel_calls=self.auto_opt)
            train_datasets.append(dataset.train_dataset)
            # valid_datasets.append(dataset.valid_dataset)

            self.num_train_samples += dataset.train_samples
            # self.num_valid_samples += dataset.valid_samples
        
        if self.config['Dataset']['DIML']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'diml_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float16)
            # handler = DimlHandler(image_size=self.image_size)
            # dataset.train_dataset = dataset.train_dataset.map(handler.preprocess,
            #                                                     num_parallel_calls=self.auto_opt)
            train_datasets.append(dataset.train_dataset)
            # valid_datasets.append(dataset.valid_dataset)

            self.num_train_samples += dataset.train_samples
            # self.num_valid_samples += dataset.valid_samples
        return train_datasets, valid_datasets

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
        depth = tf.clip_by_value(depth, 0., self.max_depth)
        depth = tf.image.resize(depth,
                                self.image_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return depth
        
    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        # image /= 255.0
        # image = (image - self.mean) / self.std
        # image = image * (1.0 / 128.0) - 1.0
        # x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
        image = (image * (1.0 / 127.5)) - 1.0
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        # image = (image * self.std) + self.mean
        # image *= 255.0
        # image = (image + 1.0) * 128.0
        image = (image + 1.0) * 127.5
        image = tf.cast(image, tf.uint8)
        return image
    
    @tf.function(jit_compile=True)
    def train_preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        rgb, depth = self.augment(rgb, depth)

        rgb = self.preprocess_image(rgb)
        depth = self.preprocess_depth(depth)
        return rgb, depth

    @tf.function(jit_compile=True)
    def valid_preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        rgb = self.preprocess_image(rgb)
        depth = self.preprocess_depth(depth)
        return rgb, depth

    @tf.function(jit_compile=True)
    def augment(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        rgb: RGB image tensor (H, W, 3) [0, 255]
        depth: Depth image tensor (H, W, 1) [0, max_depth]
        """
        # rgb augmentations
        rgb = tf.cast(rgb, tf.float32) / 255.0

        if tf.random.uniform([]) > 0.5:
            delta_brightness = tf.random.uniform([], -0.2, 0.2)
            rgb = tf.image.adjust_brightness(rgb, delta_brightness)
        
        if tf.random.uniform([]) > 0.5:
            contrast_factor = tf.random.uniform([], 0.7, 1.3)
            rgb = tf.image.adjust_contrast(rgb, contrast_factor)
        
        if tf.random.uniform([]) > 0.5:
            gamma = tf.random.uniform([], 0.8, 1.2)
            rgb = tf.image.adjust_gamma(rgb, gamma)
        
        if tf.random.uniform([]) > 0.5:
            max_delta = 0.1
            rgb = tf.image.adjust_hue(rgb, tf.random.uniform([], -max_delta, max_delta))

        # flip left-right
        if tf.random.uniform([]) > 0.5:
            rgb = tf.image.flip_left_right(rgb)
            depth = tf.image.flip_left_right(depth)

        # back to [0, 255]
        rgb = tf.clip_by_value(rgb, 0., 255.)
        rgb = tf.cast(rgb * 255.0, tf.uint8)
        return rgb, depth

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
    import matplotlib.pyplot as plt
    root_dir = './depth/data/'
    config = {
        'Directory': {
            'data_dir': root_dir
        },
        'Dataset':{
            'Nyu_depth_v2': True,
            'Diode': True,
            'DIML': True,
        },
        'Train': {
            'batch_size': 128,
            'max_depth': 10.,
            'min_depth': 0.1,
            'use_shuffle': True,
            'img_h': 480, # 480
            'img_w': 720 # 720
        }
    }
    data_loader = DataLoader(config)
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from depth_learner import DepthLearner
    learner = DepthLearner(None, None)

    for idx, samples in enumerate(data_loader.train_dataset.take(data_loader.num_train_samples)):
        rgb, depth, intrinsic = samples
        print(rgb.shape)
        print(depth.shape)
        print(intrinsic)
        rgb = data_loader.denormalize_image(rgb)
        plt.imshow(rgb[0])
        plt.show()
        plt.imshow(depth[0], cmap='plasma')
        plt.show()

    
        mask = depth > 0
        disp, mask = learner.depth_to_disparity(depth[0], mask=mask)

        plt.imshow(disp, cmap='plasma')
        plt.show()
        

        



        