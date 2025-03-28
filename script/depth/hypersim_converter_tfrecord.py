import os
import glob
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import argparse

class HypersimTFRecord(object):
    def __init__(self, root_dir, save_dir):
        """
        Hypersim dataset max depth : 65.5625
        """
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.max_depth = 60.0

        self.raw_train_dir = os.path.join(self.root_dir, 'train')
        self.raw_valid_dir = os.path.join(self.root_dir, 'val')
        self.raw_test_dir = os.path.join(self.root_dir, 'test')

        self.save_dir = os.path.join(self.save_dir, 'hypersim_tfrecord')
        self.train_tfrecord_path = os.path.join(self.save_dir, 'train.tfrecord')
        self.valid_tfrecord_path = os.path.join(self.save_dir, 'valid.tfrecord')
        self.test_tfrecord_path = os.path.join(self.save_dir, 'test.tfrecord')

        os.makedirs(self.save_dir, exist_ok=True)
        self.removed_count = 0
        self.train_count = self.convert(raw_file_dir=self.raw_train_dir, 
                                        tfrecord_path=self.train_tfrecord_path)
        self.valid_count = self.convert(raw_file_dir=self.raw_valid_dir, 
                                        tfrecord_path=self.valid_tfrecord_path)
        self.test_count = self.convert(raw_file_dir=self.raw_test_dir,
                                        tfrecord_path=self.test_tfrecord_path)
        
        # Save metadata with sample counts
        metadata = {
            'train_count': self.train_count,
            'valid_count': self.valid_count,
            'test_count': self.test_count,
            'removed_count': self.removed_count
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def convert(self, raw_file_dir, tfrecord_path):
        count = 0

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            scenes = sorted(glob.glob(os.path.join(raw_file_dir, '*')))

            for scene in tqdm(scenes, desc="[Raw Scenes] Generating TFRecord files"):
                all_files = glob.glob(os.path.join(scene, '*'))

                 # RGB 파일 필터링
                rgb_files = sorted([f for f in all_files if os.path.basename(f).startswith('rgb_cam_') and f.endswith('.png')])

                # Depth 파일 필터링
                depth_files = sorted([f for f in all_files if os.path.basename(f).startswith('depth_plane_cam_') and f.endswith('.png')])

                assert len(rgb_files) == len(depth_files), f'RGB and depth file count mismatch in {scene}'

                for rgb_name, depth_name in zip(rgb_files, depth_files):
                    rgb = np.array(Image.open(rgb_name))
                    depth = np.array(Image.open(depth_name)) * 0.001

                    depth = np.where(depth >= self.max_depth, 0., depth)

                    serialized_example = self.serialize_example(rgb, depth)
                    writer.write(serialized_example)

                    count += 1
        return count

    def serialize_example(self, rgb, depth):
        """Serialize a single RGB and depth pair into a TFRecord example."""
        rgb = tf.convert_to_tensor(rgb, tf.uint8)
        depth = tf.convert_to_tensor(depth, tf.float16)
        
        rgb = tf.cast(rgb, tf.uint8)
        depth = tf.cast(depth, tf.float16)

        rgb_bytes = tf.io.encode_jpeg(rgb, quality=100).numpy()
        depth_bytes = tf.io.serialize_tensor(depth).numpy()

        feature = {
            'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb_bytes])),
            'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_bytes]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./depth/data/hypersim_output')
    parser.add_argument('--save_dir', type=str, default='./depth/data/')
    args = parser.parse_args()
    
    converter = HypersimTFRecord(args.root_dir, save_dir=args.save_dir)