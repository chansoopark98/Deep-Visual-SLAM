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
        self.crop_size = 100
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
    def random_crop(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor) -> tuple:
        """
        left, right : [H, W, 3] (uint8 또는 float32 등)
        flow        : [H, W, 2]
        최종적으로 (H_out, W_out) 크기 Tensor로 리턴
        """
        concat = tf.concat([left, right, flow], axis=-1)
        concat = tf.image.random_crop(concat, size=(self.image_size[0] - self.crop_size,
                                                    self.image_size[1] - self.crop_size, 8))

        left_cropped = concat[:, :, :3]
        right_cropped = concat[:, :, 3:6]
        flow_cropped = concat[:, :, 6:]

        # 4) 리사이즈 (BILINEAR 등 필요에 따라 선택)
        left_cropped = tf.image.resize(left_cropped, size=self.image_size,
                                       method=tf.image.ResizeMethod.BILINEAR)
        right_cropped = tf.image.resize(right_cropped, size=self.image_size,
                                        method=tf.image.ResizeMethod.BILINEAR)
        flow_cropped = tf.image.resize(flow_cropped, size=self.image_size,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 5) Flow 값 스케일링
        cropped_shape = tf.shape(concat)
        
        scale_x = tf.cast(self.image_size[1], tf.float32) / tf.cast(cropped_shape[1], tf.float32)
        scale_y = tf.cast(self.image_size[0], tf.float32) / tf.cast(cropped_shape[0], tf.float32)

        # flow_resized는 shape = [H_out, W_out, 2]
        flow_x, flow_y = tf.split(flow_cropped, num_or_size_splits=2, axis=-1)
        flow_x = flow_x * scale_x
        flow_y = flow_y * scale_y

        flow_resized = tf.concat([flow_x, flow_y], axis=-1)

        return left_cropped, right_cropped, flow_resized
    
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

    @tf.function(jit_compile=True)
    def train_preprocess(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor) -> tuple:
        left, right, flow = self.augment(left, right, flow)
        left = self.normalize_image(left)
        right = self.normalize_image(right)
        return left, right, flow

    @tf.function(jit_compile=True)
    def valid_preprocess(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor) -> tuple:
        left = self.normalize_image(left)
        right = self.normalize_image(right)
        return left, right, flow
    
    @tf.function(jit_compile=True)
    def salt_and_pepper_noise(self, left: tf.Tensor, right: tf.Tensor, flow: tf.Tensor, prob: float = 0.05) -> tuple:
        """
        Apply identical Salt-and-Pepper noise to left and right images.

        Args:
            left (tf.Tensor): Left image tensor of shape [H, W, 3], range [0, 1].
            right (tf.Tensor): Right image tensor of shape [H, W, 3], range [0, 1].
            flow (tf.Tensor): Flow tensor of shape [H, W, 2].
            prob (float): Probability of noise (default: 0.05).

        Returns:
            tuple: (noised_left, noised_right, flow)
        """
        # Generate Salt-and-Pepper noise mask
        noise = tf.random.uniform(shape=(self.image_size[0], self.image_size[1], 1))  # Single channel for consistency
        salt_mask = tf.cast(noise < (prob / 2), tf.float32)  # Salt: white pixels
        pepper_mask = tf.cast(noise > (1 - prob / 2), tf.float32)  # Pepper: black pixels

        # Create the noisy image for both left and right
        noised_left = left * (1 - salt_mask - pepper_mask) + salt_mask + pepper_mask * 0
        noised_right = right * (1 - salt_mask - pepper_mask) + salt_mask + pepper_mask * 0

        # Clip to valid range [0, 1]
        noised_left = tf.clip_by_value(noised_left, 0.0, 1.0)
        noised_right = tf.clip_by_value(noised_right, 0.0, 1.0)

        return noised_left, noised_right, flow

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
            contrast_factor = tf.random.uniform([], 0.2, 1.2)
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

        # Salt-and-Pepper noise
        # if tf.random.uniform([]) > 0.1:
        #     left, right, flow = self.salt_and_pepper_noise(left, right, flow)

        # random crop
        # if tf.random.uniform([]) > 0.5:
        #     left, right, flow = self.random_crop(left, right, flow)

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

        left *= 255.
        right *= 255.

        left = tf.clip_by_value(left, 0., 255.)
        right = tf.clip_by_value(right, 0., 255.)
        
        return left, right, flow

    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool, is_train: bool = True) -> tf.data.Dataset:
        combined_dataset = tf.data.Dataset.sample_from_datasets(datasets, rerandomize_each_iteration=True)
            
        if use_shuffle and is_train:
            combined_dataset = combined_dataset.shuffle(buffer_size=batch_size * 128, reshuffle_each_iteration=True)
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
        # left, right, flow = data_loader.random_crop(left[0], right[0], flow[0])
        left, right, flow = left[0], right[0], flow[0]
        left = data_loader.denormalize_image(left)
        right = data_loader.denormalize_image(right)
        print(left.shape, right.shape, flow.shape)
        plt.imshow(left)
        plt.show()
        plt.imshow(right)
        plt.show()
        plt.imshow(flow[:, :, 0], cmap='plasma')
        plt.show()
        plt.imshow(flow[:, :, 1], cmap='plasma')
        plt.show()