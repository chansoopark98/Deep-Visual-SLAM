import tensorflow as tf, tf_keras
import json

class TFRecordLoader(object):
    def __init__(self, root_dir: str,
                 is_train: bool = True,
                 is_valid: bool = True,
                 is_test: bool = False,
                 image_size: tuple = (None, None),
                 depth_dtype: tf.dtypes.DType = tf.float32) -> None:
        self.root_dir = root_dir
        self.is_train = is_train
        self.is_valid = is_valid
        self.is_test = is_test
        self.image_size = image_size
        self.depth_dtype = depth_dtype
        if self.is_train:
            self.train_samples, self.valid_samples, self.test_samples = self._load_metadata(f'{self.root_dir}/metadata.json')
            self.train_dataset = tf.data.TFRecordDataset(f'{self.root_dir}/train.tfrecord')
            self.train_dataset = self.train_dataset.map(self._parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_valid:
            self.valid_dataset = tf.data.TFRecordDataset(f'{self.root_dir}/valid.tfrecord')
            self.valid_dataset = self.valid_dataset.map(self._parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_test:
            self.test_dataset = tf.data.TFRecordDataset(f'{self.root_dir}/test.tfrecord')
            self.test_dataset = self.test_dataset.map(self._parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    def _load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if self.is_train:
            train_samples = metadata.get('train_count', None)
        if self.is_valid:
            valid_samples = metadata.get('valid_count', None)
        else:
            valid_samples = 0
        if self.is_test:
            test_samples = metadata.get('test_count', None)
        else:
            test_samples = 0
        return train_samples, valid_samples, test_samples

    def _parse_data(self, example_proto):
        feature_description = {
            'rgb': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.string)
        }

        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        rgb = tf.image.decode_jpeg(parsed_features['rgb'], channels=3)
        depth = tf.io.parse_tensor(parsed_features['depth'], out_type=self.depth_dtype)
        depth = tf.cast(depth, tf.float32)
        depth = tf.expand_dims(depth, axis=-1)

        rgb = tf.ensure_shape(rgb, [self.image_size[0], self.image_size[1], 3])
        depth = tf.ensure_shape(depth, [self.image_size[0], self.image_size[1], 1])
        return rgb, depth

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diode_path = './depth/data/diode_tfrecord'
    diode_dataset = TFRecordLoader(diode_path,
                                   is_train=True,
                                   is_valid=True,
                                   depth_dtype=tf.float32)
    print(diode_dataset.train_dataset)
    print(diode_dataset.valid_dataset)

    nyu_path = './depth/data/nyu_depth_v2_tfrecord'
    nyu_dataset = TFRecordLoader(nyu_path,
                                 is_train=True,
                                 is_valid=True,
                                 depth_dtype=tf.float32)
    print(nyu_dataset.train_dataset)
    print(nyu_dataset.valid_dataset)

    diml_path = './depth/data/diml_tfrecord'
    diml_dataset = TFRecordLoader(diml_path,
                                  is_train=True,
                                  is_valid=False,
                                  depth_dtype=tf.float16)
    print(diml_dataset.train_dataset)

    for rgb, depth in diml_dataset.train_dataset:
        print(depth.shape)
        plt.imshow(depth)
        plt.show()
        break