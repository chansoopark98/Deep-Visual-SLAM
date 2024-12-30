import tensorflow as tf

class TFRecordLoader(object):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.train_dataset = tf.data.TFRecordDataset(f'{self.root_dir}/train.tfrecord')
        self.train_dataset = self.train_dataset.map(self._parse_depth, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.valid_dataset = tf.data.TFRecordDataset(f'{self.root_dir}/valid.tfrecord')
        self.valid_dataset = self.valid_dataset.map(self._parse_depth, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _parse_depth(self, example_proto):
        feature_description = {
            'rgb': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.string)
            }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        rgb = tf.image.decode_jpeg(parsed_features['rgb'])
        depth = tf.io.parse_tensor(parsed_features['depth'], out_type=tf.float32)
        return rgb, depth

    
if __name__ == '__main__':
    diode_path = './depth/data/diode_tfrecord'

    diode_dataset = TFRecordLoader(diode_path)
    print(diode_dataset.train_dataset)
    print(diode_dataset.valid_dataset)

    nyu_path = './depth/data/nyu_depth_v2_tfrecord'
    nyu_dataset = TFRecordLoader(nyu_path)
    print(nyu_dataset.train_dataset)
    print(nyu_dataset.valid_dataset)

    for sample in diode_dataset.train_dataset:
        print(sample)
        break