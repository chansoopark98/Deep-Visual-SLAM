import tensorflow as tf
import keras
import math
import tensorflow_datasets as tfds
from typing import Union
AUTO = tf.data.experimental.AUTOTUNE

class DataLoadHandler(object):
    def __init__(self, data_dir: str, batch_size: int):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.__select_dataset()

    def __select_dataset(self):
        self.dataset_list = self.__load_custom_dataset()

    def __load_custom_dataset(self):
        """
            Loads a custom dataset specified by the user.
            NyuConverted : 
                    train : 47584
                    valid : 654
            DiodeDataset :
                    train : 8574
                    valid : 325
            CustomDataset :
                    train : 656
            nyu_depth_v2
        """
        # nyu_converted nyu_depth_v2
        print(self.data_dir)
        self.nyu_train = tfds.load(name='nyu_depth_v2', data_dir=self.data_dir, split='train')
        self.nyu_valid = tfds.load(name='nyu_depth_v2', data_dir=self.data_dir, split='validation')
        
        self.train_data = self.nyu_train
        self.valid_data = self.nyu_valid
        self.test_data = self.valid_data
        
        self.number_train = math.ceil(47584 / self.batch_size)
        self.number_valid = math.ceil(654 / self.batch_size)
        self.number_test = self.number_valid
        
        self.train_data.shuffle(self.number_train)

        # Print  dataset meta data
        print("Number of train dataset = {0}".format(self.number_train))
        print("Number of validation dataset = {0}".format(self.number_valid))
        print("Number of test dataset = {0}".format(self.number_test))

class GenerateDatasets(DataLoadHandler):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int):
        """
        Args:
            data_dir         (str)    : Dataset relative path (default : './datasets/').
            image_size       (tuple)  : Model input image resolution.
            batch_size       (int)    : Batch size.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_depth = 10.
        super().__init__(data_dir=self.data_dir, batch_size=self.batch_size)

    @tf.function(jit_compile=True)
    def preprocess(self, sample) -> Union[tf.Tensor, tf.Tensor]:
        """
        preprocessing image
        :return:
            RGB image(H,W,3), Depth map(H,W,1)
        """
        image = tf.cast(sample['image'], tf.float32)
        depth = tf.cast(sample['depth'], tf.float32)
        depth = tf.expand_dims(depth, axis=-1)

        image = tf.image.resize(image, size=(self.image_size[0], self.image_size[1]),
                                method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, size=(self.image_size[0], self.image_size[1]),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (image, depth)
    
    @tf.function(jit_compile=True)
    def augmentations(self, image, depth):        
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, 0.3)

        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, 0.6, 1.4,)

        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_saturation(image, 0.4, 2)

        # if tf.random.uniform([]) > 0.5:
        #     image = tf.image.random_hue(image, 0.3)
        
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            depth = tf.image.flip_left_right(depth)
        
        image = tf.clip_by_value(image, 0., 255.)
        return (image, depth)

    @tf.function(jit_compile=True)
    def encode_image(self, image: tf.Tensor) -> tf.Tensor:
        image = keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
        return image
    
    @tf.function(jit_compile=True)
    def encode_depth(self, depth: tf.Tensor) -> tf.Tensor:
        depth = tf.clip_by_value(depth, 0.0, 10.0)
        depth = tf.where(tf.math.is_nan(depth), 0., depth)
        depth = tf.where(tf.math.is_inf(depth), 0., depth)
        return depth
    
    @tf.function(jit_compile=True)
    def restore_from_torch_mode(self, image):
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
    
    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        image = self.encode_image(image)
        depth = self.encode_depth(depth)
        return image, depth
    
    @tf.function(jit_compile=True)
    def decode_image(self, image) -> tf.Tensor:
        image = self.restore_from_torch_mode(image)
        image = tf.cast(image, dtype=tf.uint8)
        return image
    
    def get_trainData(self, train_data: tf.data.Dataset):
        train_data = train_data.shuffle(self.batch_size * 64, reshuffle_each_iteration=True)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentations, num_parallel_calls=AUTO)
        train_data = train_data.map(self.normalize_image, num_parallel_calls=AUTO)
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        train_data = train_data.prefetch(AUTO)
        return train_data

    def get_validData(self, valid_data: tf.data.Dataset):
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        valid_data = valid_data.map(self.normalize_image, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size, drop_remainder=True)
        valid_data = valid_data.prefetch(AUTO)
        return valid_data

    def get_testData(self, test_data: tf.data.Dataset):
        test_data = test_data.map(self.preprocess)
        test_data = test_data.map(self.normalize_image)
        test_data = test_data.batch(self.batch_size)
        test_data = test_data.prefetch(AUTO)
        return test_data
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = GenerateDatasets(data_dir='./depth/dataset/',
                               image_size=(480, 640),
                               batch_size=1)
    train_dataset = dataset.get_trainData(train_data=dataset.train_data)

    for image, depth in train_dataset.take(10):
        image = image[0]
        depth = depth[0]
        image = dataset.decode_image(image)

        plt.imshow(image)
        plt.show()

        plt.imshow(depth, cmap='plasma', vmin=0., vmax=10.)
        plt.show()
