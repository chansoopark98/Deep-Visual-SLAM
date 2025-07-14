import tensorflow as tf
import json
import os
import numpy as np

class TFRecordLoader:
    def __init__(self, root_dir: str,
                 is_train: bool = True,
                 is_valid: bool = True,
                 is_test: bool = False,
                 image_size: tuple = (None, None),
                 depth_dtype: tf.dtypes.DType = tf.float32,
                 use_intrinsic: bool = False) -> None:
        """
        Initializes the TFRecordLoader class for loading TFRecord datasets.

        Args:
            root_dir (str): Root directory containing TFRecord files and metadata.
            is_train (bool): Whether to load the training dataset (default: True).
            is_valid (bool): Whether to load the validation dataset (default: True).
            is_test (bool): Whether to load the test dataset (default: False).
            image_size (tuple): Target size (height, width) of the images and depth maps.
            depth_dtype (tf.dtypes.DType): Data type of the depth maps (default: tf.float32).
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.is_valid = is_valid
        self.is_test = is_test
        self.image_size = image_size
        self.depth_dtype = depth_dtype
        self.use_intrinsic = use_intrinsic

        # metadata_path = os.path.join(self.root_dir, 'metadata.json')
        # if not os.path.exists(metadata_path):
        #     raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        if self.use_intrinsic:
            self.intrinsic = np.load(os.path.join(self.root_dir, 'intrinsic.npy'))
            self.intrinsic = tf.convert_to_tensor(self.intrinsic, dtype=tf.float32)

        # self.train_samples, self.valid_samples, self.test_samples = self._load_metadata(metadata_path)

        if self.is_train:
            train_path = os.path.join(self.root_dir, 'train.tfrecord')
            self._validate_file(train_path)
            self.train_dataset = self._load_tfrecord(train_path)

        if self.is_valid:
            valid_path = os.path.join(self.root_dir, 'valid.tfrecord')
            self._validate_file(valid_path)
            self.valid_dataset = self._load_tfrecord(valid_path)

        if self.is_test:
            test_path = os.path.join(self.root_dir, 'test.tfrecord')
            self._validate_file(test_path)
            self.test_dataset = self._load_tfrecord(test_path)

    def _load_metadata(self, metadata_path: str) -> tuple:
        """
        Loads metadata containing sample counts for train, validation, and test datasets.

        Args:
            metadata_path (str): Path to the metadata JSON file.

        Returns:
            Tuple[int, int, int]: Counts of train, validation, and test samples.
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        train_samples = metadata.get('train_count', 0)
        valid_samples = metadata.get('valid_count', 0)
        test_samples = metadata.get('test_count', 0)
        return train_samples, valid_samples, test_samples

    def _validate_file(self, path: str) -> None:
        """Validates if a file exists."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"TFRecord file not found: {path}")

    def _load_tfrecord(self, path: str) -> tf.data.Dataset:
        """Loads and parses a TFRecord file."""
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(self._parse_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def _parse_data(self, example_proto: tf.train.Example) -> tuple:
        """
        Parses a single TFRecord example into RGB and depth tensors.

        Args:
            example_proto (tf.train.Example): Serialized TFRecord example.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Decoded RGB image and depth map tensors.
        """
        feature_description = {
            'rgb': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.string),
        }
        # if self.use_intrinsic:
        #     # self.intrinsic == tf.float32 (3, 3)
        #     feature_description['intrinsic'] = tf.io.FixedLenFeature([], tf.string)

        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        rgb = tf.image.decode_jpeg(parsed_features['rgb'], channels=3)
        depth = tf.io.parse_tensor(parsed_features['depth'], out_type=self.depth_dtype)
        depth = tf.cast(depth, tf.float32)
        depth = tf.expand_dims(depth, axis=-1)

        rgb = tf.ensure_shape(rgb, [self.image_size[0], self.image_size[1], 3])
        depth = tf.ensure_shape(depth, [self.image_size[0], self.image_size[1], 1])
        
        if self.use_intrinsic:
            intrinsic = self.intrinsic
            # intrinsic = tf.io.parse_tensor(parsed_features['intrinsic'], out_type=tf.float32)
            intrinsic = tf.reshape(intrinsic, (3, 3))
            return rgb, depth, intrinsic
        return rgb, depth
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diode_path = './depth/data/diode_tfrecord'
    diode_dataset = TFRecordLoader(diode_path,
                                   is_train=True,
                                   is_valid=True,
                                   depth_dtype=tf.float32,
                                   use_intrinsic=True)
    print(diode_dataset.train_dataset)
    print(diode_dataset.valid_dataset)

    nyu_path = './depth/data/nyu_depth_v2_tfrecord'
    nyu_dataset = TFRecordLoader(nyu_path,
                                 is_train=True,
                                 is_valid=True,
                                 depth_dtype=tf.float32)
    print(nyu_dataset.train_dataset)
    print(nyu_dataset.valid_dataset)

    # diml_path = './depth/data/diml_tfrecord'
    # diml_dataset = TFRecordLoader(diml_path,
    #                               is_train=True,
    #                               is_valid=False,
    #                               depth_dtype=tf.float16)
    # print(diml_dataset.train_dataset)

    for rgb, depth, intrinsic in diode_dataset.train_dataset:
        print(depth.shape)
        plt.imshow(depth)
        plt.show()
        break