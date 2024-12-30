import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import os
import tensorflow as tf

def serialize_example(rgb, depth):
    """Serialize a single RGB and depth pair into a TFRecord example."""
    rgb_bytes = tf.io.encode_jpeg(tf.convert_to_tensor(rgb), quality=100).numpy()
    depth_bytes = tf.io.serialize_tensor(tf.convert_to_tensor(depth)).numpy()

    feature = {
        'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb_bytes])),
        'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_bytes]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

if __name__ == '__main__':
    data_dir = './depth/data/'
    output_dir = './depth/data/nyu_depth_v2_tfrecord/'

    train_tfrecord_path = os.path.join(output_dir, 'train.tfrecord')
    valid_tfrecord_path = os.path.join(output_dir, 'valid.tfrecord')

    os.makedirs(output_dir, exist_ok=True)

    nyu_train = tfds.load(name='nyu_depth_v2', data_dir=data_dir, split='train')
    nyu_valid = tfds.load(name='nyu_depth_v2', data_dir=data_dir, split='validation')

    all_datasets = [(nyu_train, train_tfrecord_path), 
                    (nyu_valid, valid_tfrecord_path)]

    # Iterate over the datasets
    for dataset, tfrecord_path in all_datasets:
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for data in dataset:
                # Extract RGB and depth data
                rgb = data['image'].numpy()
                depth = data['depth'].numpy()

                # Serialize and write to TFRecord
                serialized_example = serialize_example(rgb, depth)
                writer.write(serialized_example)

    print(f"TFRecord files saved to {output_dir}")
