import tensorflow_datasets as tfds
import tensorflow as tf

import os
import json
import argparse


def serialize_example(rgb, depth):
    """Serialize a single RGB and depth pair into a TFRecord example."""
    rgb_bytes = tf.io.encode_jpeg(tf.convert_to_tensor(rgb, tf.uint8), quality=100).numpy()
    depth_bytes = tf.io.serialize_tensor(tf.convert_to_tensor(depth, tf.float32)).numpy()

    feature = {
        'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb_bytes])),
        'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_bytes]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def default_parser():
    parser = argparse.ArgumentParser("TF Dataset base Depth estimate Dataset")
    parser.add_argument('--root-path', type=str, default="depth/data/", help='dist dataset root-path')
    args = parser.parse_args()
    return args


def nyu_main_process(args): 
    # dist /{root-path}/nyu_depth_v2_tfrecord
    dist_path = os.path.join(args.root_path, "nyu_depth_v2_tfrecord")
    metadata = {}
    all_datasets = []

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Load NYU Depth v2 dataset
    for dataset_name in ['train', 'validation']:
        all_datasets.append({'dataset': tfds.load(name='nyu_depth_v2', data_dir=args.root_path, split=dataset_name),
                             'path': os.path.join(dist_path, f'{dataset_name}.tfrecord'),
                             'split': dataset_name})

    # Iterate over the datasets
    for dataset_info in all_datasets:
        with tf.io.TFRecordWriter(dataset_info['path']) as writer:
            for idx, data in enumerate(dataset_info['dataset'], 1):
                # Extract RGB and depth data
                rgb = data['image'].numpy()
                depth = data['depth'].numpy()

                # Serialize and write to TFRecord
                serialized_example = serialize_example(rgb, depth)
                writer.write(serialized_example)

        # Save count to metadata
        # Dictionary to store dataset sample counts
        metadata[f'{dataset_info["split"]}_count'] = idx

    # Save metadata to JSON
    metadata_path = os.path.join(args.output_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print(f"TFRecord files saved to {dist_path}")
    print(f"Metadata saved to {metadata_path}")
    return dist_path


if __name__ == '__main__':
    args = default_parser()
    nyu_main_process(args)