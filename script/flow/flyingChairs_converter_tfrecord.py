import os
import glob
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def load_flow(uri):
    """
    Function to load optical flow data
    Args: str uri: target flow path
    Returns: np.ndarray: extracted optical flow
    """
    with open(uri, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None
    
class FlyingChairsTFRecord(object):
    def __init__(self, root_dir, save_dir, split_txt):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.split_txt = split_txt

        self.save_dir = os.path.join(self.save_dir, 'flyingChairs_tfrecord')
        self.train_tfrecord_path = os.path.join(self.save_dir, 'train.tfrecord')
        self.valid_tfrecord_path = os.path.join(self.save_dir, 'valid.tfrecord')
        os.makedirs(self.save_dir, exist_ok=True)

        # load .txt file
        with open(self.split_txt, 'r') as f:
            self.split = f.read().splitlines()
                
        self.train_list = []
        self.valid_list = []

        self.preprocess(self.root_dir)
        self.train_count = self.convert(raw_file_dir=self.root_dir,
                                        tfrecord_path=self.train_tfrecord_path,
                                        file_list=self.train_list)
        self.valid_count = self.convert(raw_file_dir=self.root_dir,
                                        tfrecord_path=self.valid_tfrecord_path,
                                        file_list=self.valid_list)
        
        # Save metadata with sample counts
        metadata = {
            'train_count': self.train_count,
            'valid_count': self.valid_count,
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def preprocess(self, file_dir):
        files = sorted(glob.glob(os.path.join(file_dir, '*')))
            
        for i in range(0, len(files), 3):
            # 00001_flow.flo
            flow_image = os.path.basename(files[i])
            
            # flo to ppm
            left_image = flow_image.replace('_flow.flo', '_img1.ppm')
            right_image = flow_image.replace('_flow.flo', '_img2.ppm')

            sample = {
                'flow': flow_image,
                'left': left_image,
                'right': right_image
            }

            # 이름에서 숫자번호 추출
            num = int(flow_image.split('_')[0])

            fold_type = self.split[num-1]

            if fold_type == '1':
                self.train_list.append(sample)
            elif fold_type == '2':
                self.valid_list.append(sample)


    def convert(self, raw_file_dir, tfrecord_path, file_list):
        count = 0

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for file in tqdm(file_list):
                flow_image = file['flow']
                left_image = file['left']
                right_image = file['right']

                flow = load_flow(os.path.join(raw_file_dir, flow_image))
                left = np.array(Image.open(os.path.join(raw_file_dir, left_image)))
                right = np.array(Image.open(os.path.join(raw_file_dir, right_image)))

                serialized_example = self.serialize_example(left, right, flow)
                writer.write(serialized_example)

                count += 1
        return count

            
    def serialize_example(self, left, right, flow):
        """Serialize a single RGB and depth pair into a TFRecord example."""
        left = tf.convert_to_tensor(left, tf.uint8)
        right = tf.convert_to_tensor(right, tf.uint8)
        flow = tf.convert_to_tensor(flow, tf.float32)

        left_bytes = tf.io.encode_jpeg(left, quality=100).numpy()
        right_bytes = tf.io.encode_jpeg(right, quality=100).numpy()
        flow_bytes = tf.io.serialize_tensor(flow).numpy()

        feature = {
            'left': tf.train.Feature(bytes_list=tf.train.BytesList(value=[left_bytes])),
            'right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[right_bytes])),
            'flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[flow_bytes]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

if __name__ == '__main__':
    root_dir = '/media/park-ubuntu/park_file/FlyingChairs_release/data/'
    split_txt = '/media/park-ubuntu/park_file/FlyingChairs_release/FlyingChairs_train_val.txt'
    save_dir = './flow/data/'
    converter = FlyingChairsTFRecord(root_dir, save_dir=save_dir, split_txt=split_txt)