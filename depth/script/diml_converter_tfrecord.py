import os
import glob
import subprocess
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image
import json

class DimlConverterTFRecord(object):
    def __init__(self, root_dir, save_dir):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.target_shape = (480, 720)

        self.raw_train_dir = os.path.join(self.root_dir, 'train')
        self.raw_valid_dir = os.path.join(self.root_dir, 'valid')

        self.save_dir = os.path.join(self.save_dir, 'diml_tfrecord')
        self.train_tfrecord_path = os.path.join(self.save_dir, 'train.tfrecord')
        self.valid_tfrecord_path = os.path.join(self.save_dir, 'valid.tfrecord')

        os.makedirs(self.save_dir, exist_ok=True)

        train_unzip_dir = os.path.join(self.root_dir, 'train_raw')
        valid_unzip_dir = os.path.join(self.root_dir, 'valid_raw')

        self.unzip_files(self.raw_train_dir, train_unzip_dir)
        self.unzip_files(self.raw_valid_dir, valid_unzip_dir)

        self.train_count = self.convert(raw_file_dir=train_unzip_dir, 
                                        tfrecord_path=self.train_tfrecord_path) # 110599
        self.valid_count = self.convert(raw_file_dir=valid_unzip_dir, 
                                        tfrecord_path=self.valid_tfrecord_path)

        # Save metadata with sample counts
        metadata = {
            "train_count": self.train_count,
            "valid_count": self.valid_count
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def unzip_files(self, root_dir, save_dir):
        """Unzip all files in the root directory to the save directory."""
        zip_files = glob.glob(os.path.join(root_dir, '*.zip'))

        for zip_file in tqdm(zip_files, desc="Unzipping files"):
            zip_file_name = os.path.basename(zip_file).replace('.zip', '')
            save_zip_dir = os.path.join(save_dir, zip_file_name)

            is_path = os.path.exists(save_zip_dir)
            if  is_path == False:
                os.makedirs(save_zip_dir, exist_ok=True)
                os.system(f'cd {save_zip_dir} && ls ')
                command = (
                    f"unzip -l {zip_file} | awk '{{print $NF}}' | "
                    "tail -n +4 | head -n -2 | "
                    f"xargs -P 4 -I {{}} unzip {zip_file} {{}} -d {save_zip_dir}"
                )

                # Suppress stdout and stderr
                subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def convert(self, raw_file_dir, tfrecord_path):
        count = 0

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            raw_files = glob.glob(os.path.join(raw_file_dir, '*'))
            for raw_file in tqdm(raw_files, desc="[Raw Files] Generating TFRecord files"):
                raw_folds = glob.glob(os.path.join(raw_file, '*'))

                for raw_fold in tqdm(raw_folds, desc="[Raw Folds] Generating TFRecord files"):
                    raw_scenes = glob.glob(os.path.join(raw_fold, '*'))

                    for raw_scene in raw_scenes:
                        rgb_path = os.path.join(raw_scene, 'col')                        
                        rgb_files = sorted(glob.glob(os.path.join(rgb_path, '*.png')))

                        for idx in range(0, len(rgb_files), 3):
                            rgb_name = rgb_files[idx]
                            depth_name = rgb_files[idx].replace('col', 'up_png')
                            depth_name = depth_name.replace('c.png', 'ud.png')

                            # depth 파일 존재 확인
                            if not os.path.exists(depth_name):
                                continue

                            rgb = np.array(Image.open(rgb_name))
                            depth = np.array(Image.open(depth_name)) * 0.001

                            serialized_example = self.serialize_example(rgb, depth)
                            writer.write(serialized_example)

                            count += 1
        return count

    def serialize_example(self, rgb, depth):
        """Serialize a single RGB and depth pair into a TFRecord example."""
        rgb = tf.convert_to_tensor(rgb, tf.uint8)
        depth = tf.convert_to_tensor(depth, tf.float16)
        depth = tf.expand_dims(depth, axis=-1)

        rgb = tf.image.resize(rgb, self.target_shape, method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, self.target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        depth = tf.squeeze(depth, axis=-1)

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
    root_dir = '/media/park-ubuntu/park_file/depth_data/'
    save_dir = './depth/data/'
    converter = DimlConverterTFRecord(root_dir, save_dir=save_dir)