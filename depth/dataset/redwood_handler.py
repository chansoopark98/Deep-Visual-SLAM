import os
import glob
import numpy as np
import tensorflow as tf

class RedwoodLoader(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/redwood'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.intrinsic = np.load(os.path.join(self.root_dir, 'intrinsic.npy'))
        self.intrinsic = tf.convert_to_tensor(self.intrinsic, dtype=tf.float32)
        
        self.train_dataset, self.train_samples = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        # self.valid_dataset, self.valid_samples = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)

    def _process(self, scene_dir: str):
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'image', '*')))
        depth_files = sorted(glob.glob(os.path.join(scene_dir, 'depth', '*')))
        samples = []
        for idx in range(len(rgb_files)):
            sample = {
                'image': rgb_files[idx],
                'depth': depth_files[idx],
                'intrinsic': self.intrinsic
            }
            samples.append(sample)
        return samples
           
    def generate_datasets(self, fold_dir, shuffle=False):
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        raw_datasets = []
        for scene in scene_files:
            dataset = self._process(scene)
            raw_datasets.append(dataset)
        raw_datasets = np.concatenate(raw_datasets, axis=0)

        if shuffle:
            np.random.shuffle(raw_datasets)

        data_len = raw_datasets.shape[0]

        tf_dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: raw_datasets,
            output_signature={
                'image': tf.TensorSpec(shape=(), dtype=tf.string),
                'depth': tf.TensorSpec(shape=(), dtype=tf.string),
                'intrinsic': tf.TensorSpec(shape=(3, 3), dtype=tf.float32)
            }
        )
        tf_dataset = tf_dataset.map(self.parse_function)
        return tf_dataset, data_len
    
    @tf.function()
    def _read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        rgb_image = tf.cast(rgb_image, tf.uint8)
        return rgb_image
    
    @tf.function()
    def _read_depth(self, depth_path):
        depth_image = tf.io.read_file(depth_path)
        depth_image = tf.io.decode_image(depth_image, channels=1, dtype=tf.uint16)
        depth_image /= 1000
        # depth_image = depth_image * 5000.0
        # depth_image = tf.io.decode_png(depth_image, channels=1)
        # depth_image = tf.io.decode_raw(depth_image, tf.float32)
        depth_image = tf.ensure_shape(depth_image, (480, 640, 1))
        depth_image = tf.image.resize(depth_image, self.image_size, method='nearest')
        depth_image = tf.cast(depth_image, tf.float32)
        return depth_image 

    @tf.function()
    def resize_intrinsic(self, intrinsic: tf.Tensor, original_size: tuple, new_size: tuple) -> tf.Tensor:
        """
        original_size: (H, W)
        new_size: (H, W)
        """
        intrinsic = tf.cast(intrinsic, tf.float64)

        scale_x = new_size[1] / original_size[1]
        scale_y = new_size[0] / original_size[0]

        intrinsic_scaled = tf.identity(intrinsic)

        # focal lengths scaling
        intrinsic_scaled = tf.tensor_scatter_nd_update(
            intrinsic_scaled,
            indices=[[0,0], [1,1]],
            updates=[
                intrinsic[0,0] * scale_x,  # fx'
                intrinsic[1,1] * scale_y   # fy'
            ]
        )

        # principal point scaling
        intrinsic_scaled = tf.tensor_scatter_nd_update(
            intrinsic_scaled,
            indices=[[0,2], [1,2]],
            updates=[
                intrinsic[0,2] * scale_x,  # cx'
                intrinsic[1,2] * scale_y   # cy'
            ]
        )
        intrinsic_scaled = tf.cast(intrinsic_scaled, tf.float32)
        return intrinsic_scaled

    @tf.function()
    def parse_function(self, sample):
        image = self._read_image(sample['image'])
        depth = self._read_depth(sample['depth'])
        intrinsic = self.resize_intrinsic(sample['intrinsic'], (480, 640), self.image_size)
        return (image, depth, intrinsic)

if __name__ == '__main__':
    root_dir = './depth/data/'
    config = {
        'Directory': {
            'data_dir': root_dir
        },
        'Dataset':{
            'tspxr_capture': False,
            'mars_logger': True,
        },
        'Train': {
            'batch_size': 1,
            'use_shuffle': True,
            'num_source': 2,
            'imu_seq_len': 10,
            'img_h': 480,
            'img_w': 640
        }
    }
    dataset = RedwoodLoader(config)
    data_len = dataset.train_samples
    data = dataset.train_dataset
    import matplotlib.pyplot as plt
    for rgb, depth, intrinsic in dataset.train_dataset.take(100):
        plt.imshow(rgb)
        plt.show()
        plt.imshow(depth)
        plt.show()
        print(rgb.shape, depth.shape, intrinsic.shape)
        print(intrinsic)