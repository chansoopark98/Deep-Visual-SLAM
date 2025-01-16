import os
import glob
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image
import json

class DiodeConverterTFRecord(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.save_dir = os.path.join(self.root_dir, 'diode_tfrecord')

        self.train_dir = os.path.join(self.root_dir, 'train', 'indoors')
        self.valid_dir = os.path.join(self.root_dir, 'val', 'indoors')

        self.train_tfrecord_path = os.path.join(self.save_dir, 'train.tfrecord')
        self.valid_tfrecord_path = os.path.join(self.save_dir, 'valid.tfrecord')

        os.makedirs(self.save_dir, exist_ok=True)

        self.train_count = self.convert(self.train_dir, self.train_tfrecord_path, is_train=True)
        self.valid_count = self.convert(self.valid_dir, self.valid_tfrecord_path, is_train=False)

        # Save metadata with sample counts
        metadata = {
            "train_count": self.train_count,
            "valid_count": self.valid_count
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def serialize_example(self, rgb, depth):
        """Serialize a single RGB and depth pair into a TFRecord example."""
        rgb_bytes = tf.io.encode_jpeg(tf.convert_to_tensor(rgb, tf.uint8), quality=100).numpy()
        depth_bytes = tf.io.serialize_tensor(tf.convert_to_tensor(depth, tf.float32)).numpy()

        feature = {
            'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb_bytes])),
            'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_bytes]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def convert(self, root_dir, tfrecord_path, is_train: bool = True) -> int:
        """Convert the dataset to TFRecord format and return the sample count."""
        scenes = glob.glob(os.path.join(root_dir, 'indoors', '*'))
        count = 0

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for scene in scenes:
                scans = glob.glob(os.path.join(scene, '*'))

                for scan in scans:
                    print(scan)
                    rgb_files = sorted(glob.glob(os.path.join(scan, '*.png')))

                    for rgb_file in rgb_files:
                        # RGB file name
                        rgb_file_name = os.path.basename(rgb_file)
                        # Depth file name
                        depth_file_name = rgb_file_name.replace('.png', '_depth.npy')

                        # Depth mask file name
                        depth_mask_file_name = rgb_file_name.replace('.png', '_depth_mask.npy')

                        # Read RGB
                        rgb = np.array(Image.open(rgb_file))

                        # Read depth and mask
                        depth = np.load(os.path.join(scan, depth_file_name))
                        depth_mask = np.load(os.path.join(scan, depth_mask_file_name))

                        depth = np.squeeze(depth, axis=-1)
                        depth = depth * depth_mask

                        # Serialize example and write to TFRecord
                        serialized_example = self.serialize_example(rgb, depth)
                        writer.write(serialized_example)

                        count += 1  # Increment sample count

        return count

if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./depth/data/diode/')
    args = parser.parse_args()

    converter = DiodeConverterTFRecord(args.root_dir)