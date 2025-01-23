import os
import tensorflow as tf
try:
    from .tfrecord_loader import TFRecordLoader
    from .nyu_handler import NyuHandler
except:
    from tfrecord_loader import TFRecordLoader
    from nyu_handler import NyuHandler

class DataLoader(object):
    def __init__(self, config: dict) -> None:
        """
        Initializes the DataLoader class.

        Args:
            config (dict): Configuration dictionary containing dataset paths, preprocessing parameters, etc.
        """
        self.config = config
        self.train_mode = self.config['Train']['mode']
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
        self.crop_size = 100

        self.train_datasets, self.valid_datasets = self._load_dataset()
        
        self.num_train_samples = self.num_train_samples // self.batch_size
        self.num_valid_samples = self.num_valid_samples // self.batch_size

        self.train_dataset = self._compile_dataset(self.train_datasets, batch_size=self.batch_size, use_shuffle=True, is_train=True)
        if self.valid_datasets:
            self.valid_dataset = self._compile_dataset(self.valid_datasets, batch_size=self.batch_size, use_shuffle=False, is_train=False)
        
    def _load_dataset(self) -> list:
        """
        Loads datasets based on the configuration.

        Returns:
            tuple: Lists of train and validation datasets.
        """
        train_datasets = []
        valid_datasets = []
        
        if self.config['Dataset']['Nyu_depth_v2']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'nyu_depth_v2_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float32)
            handler = NyuHandler()

            if self.config['Dataset']['Nyu_depth_v2']['train']:
                dataset.train_dataset = dataset.train_dataset.map(handler.nyu_crop_resize,
                                                                num_parallel_calls=self.auto_opt)
                train_datasets.append(dataset.train_dataset)
                self.num_train_samples += dataset.train_samples
            
            if self.config['Dataset']['Nyu_depth_v2']['valid']:
                dataset.valid_dataset = dataset.valid_dataset.map(handler.nyu_crop_resize,
                                                                num_parallel_calls=self.auto_opt)
                valid_datasets.append(dataset.valid_dataset)
                self.num_valid_samples += dataset.valid_samples

        if self.config['Dataset']['Diode']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'diode_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float32)
            if self.config['Dataset']['Diode']['train']:
                train_datasets.append(dataset.train_dataset)
                self.num_train_samples += dataset.train_samples
            if self.config['Dataset']['Diode']['valid']:
                valid_datasets.append(dataset.valid_dataset)
                self.num_valid_samples += dataset.valid_samples
            
        if self.config['Dataset']['DIML']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'diml_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float16)
            if self.config['Dataset']['DIML']['train']:
                train_datasets.append(dataset.train_dataset)
                self.num_train_samples += dataset.train_samples
            if self.config['Dataset']['DIML']['valid']:
                valid_datasets.append(dataset.valid_dataset)
                self.num_valid_samples += dataset.valid_samples

        if self.config['Dataset']['Hypersim']:
            dataset_name = os.path.join(self.config['Directory']['data_dir'], 'hypersim_tfrecord')
            dataset = TFRecordLoader(root_dir=dataset_name, is_train=True,
                                     is_valid=True, image_size=(None, None), depth_dtype=tf.float16)
            if self.config['Dataset']['Hypersim']['train']:
                train_datasets.append(dataset.train_dataset)
                self.num_train_samples += dataset.train_samples
            if self.config['Dataset']['Hypersim']['valid']:
                valid_datasets.append(dataset.valid_dataset)
                self.num_valid_samples += dataset.valid_samples   
        return train_datasets, valid_datasets

    @tf.function(jit_compile=True)
    def preprocess_image(self, rgb: tf.Tensor):
        """
        Preprocesses an input RGB image tensor.

        - Resizes the image to the target size.
        - Casts the image to float32 for further processing.
        - Normalizes the image using mean and standard deviation.

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [H, W, 3].

        Returns:
            tf.Tensor: Preprocessed RGB image tensor of shape [self.image_size[0], self.image_size[1], 3].
        """
        rgb = tf.image.resize(rgb,
                              self.image_size,
                              method=tf.image.ResizeMethod.BILINEAR)
        rgb = tf.cast(rgb, tf.float32)
        return rgb

    @tf.function(jit_compile=True)
    def preprocess_depth(self, depth: tf.Tensor):
        """
        Preprocesses an input depth map tensor.

        - Converts the depth map to float32.
        - Resizes the depth map to the target size.
        - Clips depth values to the range [0, max_depth].

        Args:
            depth (tf.Tensor): Input depth tensor of shape [H, W, 1].

        Returns:
            tf.Tensor: Preprocessed depth tensor of shape [self.image_size[0], self.image_size[1], 1].
        """
        depth = tf.cast(depth, tf.float32)
        depth = tf.image.resize(depth,
                                self.image_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth = tf.clip_by_value(depth, 0., self.max_depth)
        depth = tf.where(depth > self.max_depth, 0., depth)
        return depth
        
    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Normalizes an input image tensor using PyTorch style normalization.

        Args:
            image (tf.Tensor): Input image tensor of shape [H, W, 3] with dtype uint8 or float32.

        Returns:
            tf.Tensor: Normalized image tensor of shape [H, W, 3] with dtype float32.
        """
        # Normalize image pytorch style
        image = tf.cast(image, tf.float32)
        image /= 255.0
        image = (image - self.mean) / self.std
        return image
    
    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        """
        Denormalizes an input image tensor back to its original scale.

        Args:
            image (tf.Tensor): Normalized image tensor of shape [H, W, 3] with dtype float32.

        Returns:
            tf.Tensor: Denormalized image tensor of shape [H, W, 3] with dtype uint8.
        """
        image = (image * self.std) + self.mean
        image *= 255.0
        image = tf.cast(image, tf.uint8)
        return image

    @tf.function(jit_compile=True)
    def get_relative_depth(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Normalizes the input depth map to a relative range [0, 1].

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [H, W, 3].
            depth (tf.Tensor): Input depth map tensor of shape [H, W, 1].

        Returns:
            tuple: Tuple containing:
                - rgb (tf.Tensor): Unchanged input RGB tensor.
                - normalized_depth (tf.Tensor): Depth tensor normalized to the range [0, 1].
        """
        normalized_depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        normalized_depth = tf.clip_by_value(normalized_depth, 0., 1.0)
        return rgb, normalized_depth

    @tf.function(jit_compile=True)
    def train_preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Preprocesses training data by applying data augmentation and normalization.

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [H, W, 3].
            depth (tf.Tensor): Input depth map tensor of shape [H, W, 1].

        Returns:
            tuple: Tuple containing:
                - rgb (tf.Tensor): Augmented and preprocessed RGB tensor.
                - depth (tf.Tensor): Augmented and preprocessed depth tensor.
        """

        rgb = self.preprocess_image(rgb)
        depth = self.preprocess_depth(depth)
        
        rgb, depth = self.augment(rgb, depth)
    
        rgb = self.normalize_image(rgb)
        return rgb, depth

    @tf.function(jit_compile=True)
    def valid_preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Preprocesses validation data by applying normalization without augmentation.

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [H, W, 3].
            depth (tf.Tensor): Input depth map tensor of shape [H, W, 1].

        Returns:
            tuple: Tuple containing:
                - rgb (tf.Tensor): Preprocessed RGB tensor.
                - depth (tf.Tensor): Preprocessed depth tensor.
        """
        rgb = self.preprocess_image(rgb)
        depth = self.preprocess_depth(depth)

        rgb = self.normalize_image(rgb)
        return rgb, depth
    
    @tf.function(jit_compile=True)
    def salt_and_pepper_noise(self, image: tf.Tensor, prob: float = 0.05) -> tf.Tensor:
        """
        Applies Salt-and-Pepper noise to an input image.

        Args:
            image (tf.Tensor): Input image tensor of shape [H, W, 3] with values in the range [0, 1].
            prob (float): Probability of applying noise to each pixel (default: 0.05).

        Returns:
            tf.Tensor: Noised image tensor of shape [H, W, 3] with values in the range [0, 1].
        """
        # Ensure input is in range [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Generate noise masks
        noise = tf.random.uniform(shape=(self.image_size[0], self.image_size[1], 1))  # Single channel
        salt_mask = tf.cast(noise < (prob / 2), tf.float32)  # White pixels
        pepper_mask = tf.cast(noise > (1 - prob / 2), tf.float32)  # Black pixels

        # Expand masks to match image channels
        salt_mask = tf.broadcast_to(salt_mask, tf.shape(image))
        pepper_mask = tf.broadcast_to(pepper_mask, tf.shape(image))

        # Apply noise
        noised_image = image * (1 - salt_mask - pepper_mask) + salt_mask + pepper_mask * 0

        # Clip to valid range [0, 1]
        noised_image = tf.clip_by_value(noised_image, 0.0, 1.0)

        return noised_image

    @tf.function(jit_compile=True)
    def augment(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Applies data augmentation to the input RGB image and depth map.

        Args:
            rgb (tf.Tensor): RGB image tensor of shape [H, W, 3] with values in [0, 255].
            depth (tf.Tensor): Depth map tensor of shape [H, W, 1] with values in [0, max_depth].

        Returns:
            tuple: Tuple containing:
                - Augmented RGB image tensor of shape [H, W, 3] with values in [0, 255].
                - Augmented depth map tensor of shape [H, W, 1].
        """
        # rgb augmentations
        rgb = tf.cast(rgb, tf.float32) / 255.0

        if tf.random.uniform([]) > 0.5:
            delta_brightness = tf.random.uniform([], -0.3, 0.3)
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

        # random crop
        if tf.random.uniform([]) > 0.5:
            concat = tf.concat([rgb, depth], axis=-1)
            concat = tf.image.random_crop(concat, size=(self.image_size[0] - self.crop_size,
                                                    self.image_size[1] - self.crop_size, 4))
            
            cropped_rgb = concat[:, :, :3]
            cropped_depth = concat[:, :, 3:]

            rgb = tf.image.resize(cropped_rgb, self.image_size, method=tf.image.ResizeMethod.BILINEAR)
            depth = tf.image.resize(cropped_depth, self.image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # flip left-right
        if tf.random.uniform([]) > 0.5:
            rgb = tf.image.flip_left_right(rgb)
            depth = tf.image.flip_left_right(depth)

        # back to [0, 255]
        rgb = tf.clip_by_value(rgb, 0., 255.)
        rgb = tf.cast(rgb * 255.0, tf.uint8)
        return rgb, depth

    def _compile_dataset(self, datasets: list, batch_size: int, use_shuffle: bool, is_train: bool = True) -> tf.data.Dataset:
        """
        Compiles a dataset from multiple sources with preprocessing and batching.

        Args:
            datasets (list): List of TensorFlow datasets to combine.
            batch_size (int): Batch size for the dataset.
            use_shuffle (bool): Whether to shuffle the dataset.
            is_train (bool): Whether the dataset is for training (default: True).

        Returns:
            tf.data.Dataset: Compiled and optimized dataset ready for training or validation.
        """
        combined_dataset = tf.data.Dataset.sample_from_datasets(datasets, rerandomize_each_iteration=True)
            
        if use_shuffle:
            combined_dataset = combined_dataset.shuffle(buffer_size=batch_size * 256, reshuffle_each_iteration=True)
        if is_train:
            combined_dataset = combined_dataset.map(self.train_preprocess, num_parallel_calls=self.auto_opt)
        else:
            combined_dataset = combined_dataset.map(self.valid_preprocess, num_parallel_calls=self.auto_opt)
        
        if self.train_mode == 'relative':
            combined_dataset = combined_dataset.map(self.get_relative_depth, num_parallel_calls=self.auto_opt)

        combined_dataset = combined_dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=self.auto_opt)
        combined_dataset = combined_dataset.prefetch(self.auto_opt)
        return combined_dataset

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import yaml
    root_dir = './depth/data/'
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_loader = DataLoader(config)
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    # from depth_learner import DepthLearner
    # learner = DepthLearner(None, None)

    for idx, samples in enumerate(data_loader.train_dataset.take(data_loader.num_train_samples)):
        rgb, depth = samples
        print(rgb.shape)
        print(depth.shape)
        
        rgb = data_loader.denormalize_image(rgb)
        plt.imshow(rgb[0])
        plt.show()
        plt.imshow(depth[0], cmap='plasma')
        plt.show()
    
        mask = depth > 0
        # disp, mask = learner.depth_to_disparity(depth[0], mask=mask)

        # plt.imshow(disp, cmap='plasma')
        # plt.show()