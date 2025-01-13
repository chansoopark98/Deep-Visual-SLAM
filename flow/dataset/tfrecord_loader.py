import tensorflow as tf
import json

class TFRecordLoader(object):
    def __init__(self, root_dir: str,
                 is_train: bool = True,
                 is_valid: bool = True,
                 is_test: bool = False,
                 image_size: tuple = (None, None),
                 flow_dtype: tf.dtypes.DType = tf.float32) -> None:
        self.root_dir = root_dir
        self.is_train = is_train
        self.is_valid = is_valid
        self.is_test = is_test
        self.image_size = image_size
        self.flow_dtype = flow_dtype

        self.feature_description = {
            'left': tf.io.FixedLenFeature([], tf.string),
            'right': tf.io.FixedLenFeature([], tf.string),
            'flow': tf.io.FixedLenFeature([], tf.string)
        }

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
        parsed_features = tf.io.parse_single_example(example_proto, self.feature_description)

        left = tf.image.decode_jpeg(parsed_features['left'], channels=3)
        right = tf.image.decode_jpeg(parsed_features['right'], channels=3)
        flow = tf.io.parse_tensor(parsed_features['flow'], out_type=self.flow_dtype)

        left = tf.cast(left, tf.uint8)
        right = tf.cast(right, tf.uint8)
        flow = tf.cast(flow, tf.float32)

        left = tf.ensure_shape(left, [self.image_size[0], self.image_size[1], 3])
        right = tf.ensure_shape(right, [self.image_size[0], self.image_size[1], 3])
        flow = tf.ensure_shape(flow, [self.image_size[0], self.image_size[1], 2])
        return left, right, flow

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    diml_path = './flow/data/flyingChairs_tfrecord'
    diml_dataset = TFRecordLoader(diml_path,
                                  is_train=True,
                                  is_valid=False,
                                  flow_dtype=tf.float16)
    print(diml_dataset.train_dataset)

    for left, right, flow in diml_dataset.train_dataset:
        print(left.shape)
        print(right.shape)
        plt.imshow(flow[:, :, 0])
        plt.show()
        break