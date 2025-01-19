import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import requests

import argparse
import json
import os


DIODE_DATASET_URL_DICT = {'train': "http://diode-dataset.s3.amazonaws.com/train.tar.gz",
                          'valid': "http://diode-dataset.s3.amazonaws.com/val.tar.gz"}


class DiodeConverterTFRecord(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.save_dir = os.path.join(self.root_dir, 'diode_tfrecord')

        self.train_dir = os.path.join(self.root_dir, 'train', 'indoors')
        self.valid_dir = os.path.join(self.root_dir, 'val', 'indoors')

        self.train_tfrecord_path = os.path.join(self.save_dir, 'train.tfrecord')
        self.valid_tfrecord_path = os.path.join(self.save_dir, 'valid.tfrecord')

    def __call__(self):
        if not os.path.exists(self.save_dir):
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
        # count = 0

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for scene in scenes:
                scans = glob.glob(os.path.join(scene, '*'))

                for scan in scans:
                    print(scan)
                    rgb_files = sorted(glob.glob(os.path.join(scan, '*.png')))

                    for count, rgb_file in enumerate(rgb_files, 1):
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

                        # count += 1  # Increment sample count

        return count


def diode_main_process(args):
    if not os.path.exists(args.root_dir):
        import requests
        import tarfile
        compress_filename = []
        for dataset, url in DIODE_DATASET_URL_DICT.items():
            local_filename = url.split('/')[-1]
            with requests.get(os.path.join(args.root_dir, url), stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            compress_filename.append(local_filename)
        save_path = os.path.join(args.root_dir, 'diode')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        for download_file_path in map(lambda x: os.path.join(args.root_dir, x), compress_filename):
            with tarfile.open(download_file_path, 'r') as tar:
                tar.add(save_path)
        del compress_filename, local_filename, save_path, requests, tarfile

    converter = DiodeConverterTFRecord(os.path.join(args.root_dir, "diode"))
    converter()


if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./depth/data')
    args = parser.parse_args()
    diode_main_process(args)