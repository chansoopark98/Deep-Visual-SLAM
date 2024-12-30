import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import os
import cv2

def save_depth_as_png(depth_array, max_depth, file_path):
    """
    Save depth data as a 16-bit PNG file.
    Args:
        depth_array: The depth data as a numpy array.
        file_path: The file path to save the PNG file.
    """
    # Normalize depth to range [0, 65535] and convert to uint16
    depth_normalized = (depth_array / max_depth * 65535).astype(np.uint16)
    cv2.imwrite(file_path, depth_normalized)

if __name__ == '__main__':
    data_dir = './depth/data/'
    output_dir = './depth/data/nyu_depth_v2_raw/'    
    train_rgb_path = os.path.join(output_dir, 'train/rgb')
    train_depth_path = os.path.join(output_dir, 'train/depth')
    valid_rgb_path = os.path.join(output_dir, 'valid/rgb')
    valid_depth_path = os.path.join(output_dir, 'valid/depth')
    os.makedirs(train_rgb_path, exist_ok=True)
    os.makedirs(train_depth_path, exist_ok=True)
    os.makedirs(valid_rgb_path, exist_ok=True)
    os.makedirs(valid_depth_path, exist_ok=True)

    nyu_train = tfds.load(name='nyu_depth_v2', data_dir=data_dir, split='train')
    nyu_valid = tfds.load(name='nyu_depth_v2', data_dir=data_dir, split='validation')

    all_datasets = [(nyu_train, train_rgb_path, train_depth_path), 
                    (nyu_valid, valid_rgb_path, valid_depth_path)]

    # Iterate over the datasets
    for dataset, rgb_path, depth_path in all_datasets:
        idx = 0
        for data in dataset:
            # Extract RGB and depth data
            rgb = data['image'].numpy()
            depth = data['depth'].numpy()

            # Save RGB image
            rgb_image = Image.fromarray(rgb, mode='RGB')
            rgb_filename = os.path.join(rgb_path, f'rgb_{str(idx).zfill(6)}.jpg')
            rgb_image.save(rgb_filename, 'jpeg', quality=100)

            # Save depth image as 16-bit PNG
            depth_filename = os.path.join(depth_path, f'depth_{str(idx).zfill(6)}.png')
            save_depth_as_png(depth, 10., depth_filename)

            idx += 1
