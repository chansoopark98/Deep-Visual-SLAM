import os
import tensorflow as tf
try:
    from .tfrecord_loader import TFRecordLoader
    from .flyingchair_handler import FlyingChairHandler
except:
    from tfrecord_loader import TFRecordLoader
    from flyingchair_handler import FlyingChairHandler

class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = self.config['Train']['batch_size']
        self.use_shuffle = self.config['Train']['use_shuffle']
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.auto_opt = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
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
        
        if self.config['Dataset']['FlyingChairs']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'flyingChairs_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), flow_dtype=tf.float32)
            handler = FlyingChairHandler(target_size=self.image_size)
            dataset.train_dataset = dataset.train_dataset.map(handler.preprocess, num_parallel_calls=self.auto_opt)
            dataset.valid_dataset = dataset.valid_dataset.map(handler.preprocess, num_parallel_calls=self.auto_opt)
            
            train_datasets.append(dataset.train_dataset)
            valid_datasets.append(dataset.valid_dataset)

            self.num_train_samples += dataset.train_samples
            self.num_valid_samples += dataset.valid_samples

        return train_datasets, valid_datasets

    @tf.function(jit_compile=True)
    def preprocess_image(self, rgb: tf.Tensor):
        # rgb = tf.image.resize(rgb,
        #                       self.image_size,
        #                       method=tf.image.ResizeMethod.BILINEAR)
        rgb = self.normalize_image(rgb)
        return rgb

    @tf.function(jit_compile=True)
    def preprocess_flow(self, flow: tf.Tensor):
        flow = tf.cast(flow, tf.float32)
        # flow = tf.image.resize(flow,
        #                         self.image_size,
        #                         method=tf.image.ResizeMethod.BILINEAR)
        return flow
        
    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        # image /= 255.0
        # image = (image - self.mean) / self.std
        image /= 255.0
        image = (2 * image) - 1.0
        # image = image * (1.0 / 127.5) - 1.0
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        # image = (image * self.std) + self.mean
        # image *= 255.0
        image = (image + 1.0) / 2.0
        image = image * 255.0
        # image = (image + 1.0) * 127.5
        image = tf.cast(image, tf.uint8)
        return image

    @tf.function(jit_compile=True)
    def train_preprocess(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor) -> tuple:
        left = self.preprocess_image(left)
        right = self.preprocess_image(right)
        flow = self.preprocess_flow(flow)
        return left, right, flow

    @tf.function(jit_compile=True)
    def valid_preprocess(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor) -> tuple:
        left = self.preprocess_image(left)
        right = self.preprocess_image(right)
        flow = self.preprocess_flow(flow)
        return left, right, flow
    
    @tf.function(jit_compile=True)
    def augment(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor) -> tuple:
        """
        rgb: RGB image tensor (H, W, 3) [0, 255]
        depth: Depth image tensor (H, W, 1) [0, max_depth]
        """
        # rgb augmentations
        left = tf.cast(left, tf.float32) / 255.0
        right = tf.cast(right, tf.float32) / 255.0
        

        if tf.random.uniform([]) > 0.5:
            delta_brightness = tf.random.uniform([], -0.2, 0.2)
            left = tf.image.adjust_brightness(left, delta_brightness)
            right = tf.image.adjust_brightness(right, delta_brightness)
        
        if tf.random.uniform([]) > 0.5:
            contrast_factor = tf.random.uniform([], 0.7, 1.3)
            left = tf.image.adjust_contrast(left, contrast_factor)
            right = tf.image.adjust_contrast(right, contrast_factor)
        
        if tf.random.uniform([]) > 0.5:
            gamma = tf.random.uniform([], 0.8, 1.2)
            left = tf.image.adjust_gamma(left, gamma)
            right = tf.image.adjust_gamma(right, gamma)
        
        if tf.random.uniform([]) > 0.5:
            max_delta = 0.1
            delta = tf.random.uniform([], -max_delta, max_delta)
            left = tf.image.adjust_hue(left, delta)
            right = tf.image.adjust_hue(right, delta)

        # flip left-right
        if tf.random.uniform([]) > 0.5:
            left = tf.image.flip_left_right(left)
            right = tf.image.flip_left_right(right)
            flow = tf.image.flip_left_right(flow)

            flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)  
            flow_x = -flow_x
            flow = tf.concat([flow_x, flow_y], axis=-1)

        # flip up-down
        if tf.random.uniform([]) > 0.5:
            left = tf.image.flip_up_down(left)
            right = tf.image.flip_up_down(right)
            flow = tf.image.flip_up_down(flow)
    
            flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)
            flow_y = -flow_y

            # 3) 다시 합치기
            flow = tf.concat([flow_x, flow_y], axis=-1)  # shape [H, W, 2]

        left = tf.clip_by_value(left, 0., 255.)
        right = tf.clip_by_value(right, 0., 255.)
        
        return left, right, flow

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
    root_dir = './flow/data/'
    config = {
        'Directory': {
            'data_dir': root_dir
        },
        'Dataset':{
            'FlyingChairs': True,

        },
        'Train': {
            'batch_size': 8,
            
            'use_shuffle': True,
            'img_h': 480, # 480
            'img_w': 720 # 720
        }
    }
    data_loader = DataLoader(config)
    for left, right, flow in data_loader.train_dataset.take(data_loader.num_train_samples):
        left = data_loader.denormalize_image(left)
        right = data_loader.denormalize_image(right)
        print(left.shape, right.shape, flow.shape)
        plt.imshow(left[0])
        plt.show()
        plt.imshow(right[0])
        plt.show()
        plt.imshow(flow[0, :, :, 0], cmap='plasma')
        plt.show()
        plt.imshow(flow[0, :, :, 1], cmap='plasma')
        plt.show()