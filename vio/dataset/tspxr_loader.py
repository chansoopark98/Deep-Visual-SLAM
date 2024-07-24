import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow_datasets as tfds
import tensorflow as tf
import math
from typing import Union
import os
import sys
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from vio.utils.tspxr_capture_utils import TspDataHandler

AUTO = tf.data.experimental.AUTOTUNE

class DataGenerator(object):
    def __init__(self, data_dir) -> None:
        self.train_handler = TspDataHandler(root=data_dir + '/train/',
                                            imu_frequency=11)
        self.valid_handler = TspDataHandler(root=data_dir + '/valid/',
                                            imu_frequency=11)
        self.create_dataset()
        self.num_train = self.train_handler.data_len
        self.num_valid = self.valid_handler.data_len

    def create_dataset(self):
        self.train_data = self.train_handler.create_vio_dataset()
        self.valid_data = self.valid_handler.create_vio_dataset()

class DataLoadHandler(object):
    def __init__(self, data_dir):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            dataset_name (str)   : Tensorflow dataset name (e.g: 'citiscapes')
        
        """
        self.dataset_name = 'tspxr_capture'
        self.mode = 'tf'
        self.data_dir = data_dir

        self.__select_dataset()

    def __select_dataset(self):
        self.train_data, self.valid_data = self.__load_custom_dataset()

    def __load_custom_dataset(self):
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train')
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='validation')
        return train_data, valid_data

class TspxrTFDSGenerator(DataGenerator):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int):
        """
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            image_size   (tuple) : Model input image resolution 
            batch_size   (int)   : Batch size
            dataset_name (str)   : Tensorflow dataset name (e.g: 'cityscapes')
            norm_type    (str)   : Set input image normalization type (e.g: 'torch')
        """
        # Configuration
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_depth = 10.
        super().__init__(data_dir=self.data_dir)
        
        self.number_train_iters = math.ceil(self.num_train / self.batch_size)
        self.number_test_iters = math.ceil(self.num_train / self.batch_size)

    @tf.function(jit_compile=True)
    def prepare_data(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Load RGB images and segmentation labels from the dataset.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().

            Returns:
                (img, labels) (dict) : Returns the image and label extracted from sample as a key value.
        """
        """
        'source_left'
        'source_right'
        'target_image'
        'intrinsic'
        """
        # Load samples
        source_left = tf.cast(sample['source_left'], dtype=tf.uint8)
        source_right = tf.cast(sample['source_right'], dtype=tf.uint8)
        target_image = tf.cast(sample['target_image'], dtype=tf.uint8)
        intrinsic = tf.cast(sample['intrinsic'], dtype=tf.float32)


        # Resize image
        source_left = tf.image.resize(source_left, self.image_size,
                                    method=tf.image.ResizeMethod.BILINEAR)
        
        source_right = tf.image.resize(source_right, self.image_size,
                                    method=tf.image.ResizeMethod.BILINEAR)
        
        target_image = tf.image.resize(target_image, self.image_size,
                                    method=tf.image.ResizeMethod.BILINEAR)
        
        # Rescale intrinsic
        intrinsic = self.rescale_intrinsic_matrix(intrinsic,
                                                  self.train_handler.target_image_shape[0],
                                                  self.train_handler.target_image_shape[1],
                                                  self.image_size[0],
                                                  self.image_size[1])


        return (source_left, source_right, target_image, intrinsic)

    @tf.function(jit_compile=True)
    def preprocess(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Dataset mapping function to apply to the train dataset.
            Various methods can be applied here, such as image resizing, random cropping, etc.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().
            
            Returns:
                (img, labels) (dict) : tf.Tensor
        """
        source_left, source_right, target_image, intrinsic = self.prepare_data(sample)
        return (source_left, source_right, target_image, intrinsic)
    
    @tf.function(jit_compile=True)
    def rescale_intrinsic_matrix(self, K, original_height, original_width, new_height, new_width):
        """
        기존 intrinsic matrix를 새로운 이미지 해상도에 맞춰 조정하는 함수

        :param K: 3x3 intrinsic matrix (Tensor)
        :param original_width: 원래 이미지의 가로 해상도
        :param original_height: 원래 이미지의 세로 해상도
        :param new_width: 새로운 이미지의 가로 해상도
        :param new_height: 새로운 이미지의 세로 해상도
        :return: new 3x3 intrinsic matrix (Tensor)
        """
    
        # 가로 및 세로 스케일 비율 계산
        original_width = tf.cast(original_width, tf.float32)
        original_height = tf.cast(original_height, tf.float32)
        new_width = tf.cast(new_width, tf.float32)
        new_height = tf.cast(new_height, tf.float32)

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # 기존 intrinsic matrix에서 focal length와 principal point 추출
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        # 새로운 intrinsic matrix 계산
        K_new = tf.stack([
                [f_x * scale_x, 0., c_x * scale_x],
                [0., f_y * scale_y, c_y * scale_y],
                [0., 0., 1.]
            ])

        return K_new

    @tf.function(jit_compile=True)
    def normalize_images(self, source_left, source_right, target_image, intrinsic):
        source_left = keras.applications.imagenet_utils.preprocess_input(source_left, mode='torch')
        source_right = keras.applications.imagenet_utils.preprocess_input(source_right, mode='torch')
        target_image = keras.applications.imagenet_utils.preprocess_input(target_image, mode='torch')
        
        return (source_left, source_right, target_image, intrinsic)

    @tf.function(jit_compile=True)
    def augment_brightness(self, image):
        brightness_factor = tf.random.uniform([], minval=-0.3, maxval=0.3)
        image = tf.image.adjust_brightness(image, brightness_factor)
        image = tf.clip_by_value(image, 0., 255.)
        return image

    @tf.function(jit_compile=True)
    def augment_hue(self, image):
        hue_factor = tf.random.uniform([], minval=-0.5, maxval=0.5)
        image = tf.image.adjust_hue(image, hue_factor)
        image = tf.clip_by_value(image, 0., 255.)
        return image
    
    @tf.function(jit_compile=True)
    def augment_saturation(self, image):
        saturation_factor = tf.random.uniform([], minval=0.5, maxval=3.)
        image = tf.image.adjust_saturation(image, saturation_factor)
        image = tf.clip_by_value(image, 0., 255.)
        return image
    
    @tf.function(jit_compile=True)
    def augment_contrast(self, image):
        contrast_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
        image = tf.image.adjust_contrast(image, contrast_factor)
        image = tf.clip_by_value(image, 0., 255.)
        return image
    
    @tf.function(jit_compile=True)
    def augment_gamma(self, image):
        gamma = tf.random.uniform([], minval=0.8, maxval=1.2)
        image = tf.image.adjust_gamma(image, gamma)
        return image

    @tf.function(jit_compile=True)
    def color_aug(self, image):
        if tf.random.uniform([]) > 0.2:
            image = tf.map_fn(
                lambda x: self.augment_hue(x), image)

        if tf.random.uniform([]) > 0.2:
            image = tf.map_fn(
                lambda x: self.augment_saturation(x), image)
            
        if tf.random.uniform([]) > 0.2:
            image = tf.map_fn(
                lambda x: self.augment_contrast(x), image)
            
        if tf.random.uniform([]) > 0.2:
            image = tf.map_fn(
                lambda x: self.augment_brightness(x), image)
        
        return image
    
    @tf.function(jit_compile=True)
    def augmentation(self, image_seq, depth_seq, imu, rel_pose, global_pose, intrinsic):
        # if tf.random.uniform([]) > 0.5:
        #     image_seq = self.augment_hue(image_seq)
        if tf.random.uniform([]) > 0.5:
            image_seq = self.augment_saturation(image_seq)
        if tf.random.uniform([]) > 0.5:
            image_seq = self.augment_contrast(image_seq)
        if tf.random.uniform([]) > 0.5:
            image_seq = self.augment_brightness(image_seq)
        # if tf.random.uniform([]) > 0.5:
        #     image_seq = self.augment_gamma(image_seq)

        return (image_seq, depth_seq, imu, rel_pose, global_pose, intrinsic)
   
    @tf.function(jit_compile=True)
    def decode_image(self, image):
        # torch 모드에서 사용된 mean과 std 값
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
    
        # 채널별 역정규화
        image *= std
        image += mean
        
        # 픽셀 값의 역스케일링
        image *= 255.0
        image = tf.cast(image, dtype=tf.uint8)
        return image 
    
    def get_trainData(self, train_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                train_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                train_data    (tf.data.Dataset)  : Apply data augmentation, batch, and shuffling
        """    
        # train_data = train_data.shuffle(128)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        # train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.map(self.normalize_images, num_parallel_calls=AUTO)
        
        # train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        train_data = train_data.prefetch(AUTO)
        
        # train_data = train_data.repeat()
        return train_data

    def get_testData(self, valid_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                valid_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                valid_data    (tf.data.Dataset)  : Apply data resize, batch, and shuffling
        """    
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        valid_data = valid_data.map(self.normalize_images, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size, drop_remainder=True)
        # valid_data = valid_data.prefetch(AUTO)
        return valid_data
    
if __name__ == '__main__':
    root_path = '../data/raw/tspxr_capture/'
    dataset = TspxrTFDSGenerator(data_dir=root_path,
                       image_size=(360, 640),
                       batch_size=1)
    
    train_data = dataset.get_trainData(dataset.train_data)
    
    for source_img, target_img, imu, rel_pose, source_global_pose, target_global_pose, intrinsic in train_data.take(dataset.number_train_iters):
        print('imu_data', imu.shape)
        print('rel_pose', rel_pose.shape, rel_pose)
        print('source_global_pose', source_global_pose.shape, source_global_pose)
        print('target_global_pose', target_global_pose.shape, target_global_pose)
        print('intrinsic', intrinsic.shape, intrinsic)
        plt.imshow(source_img[0])
        plt.show()
        plt.imshow(target_img[0])
        plt.show()
