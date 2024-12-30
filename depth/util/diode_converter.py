import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DiodeConverter(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.save_dir = os.path.join(self.root_dir, 'diode_converted')

        self.train_rgb_path = os.path.join(self.save_dir, 'train/rgb')
        self.train_depth_path = os.path.join(self.save_dir, 'train/depth')
        self.valid_rgb_path = os.path.join(self.save_dir, 'valid/rgb')
        self.valid_depth_path = os.path.join(self.save_dir, 'valid/depth')

        os.makedirs(self.train_rgb_path, exist_ok=True)
        os.makedirs(self.train_depth_path, exist_ok=True)
        os.makedirs(self.valid_rgb_path, exist_ok=True)
        os.makedirs(self.valid_depth_path, exist_ok=True)

        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'val')
        self.train_data = self.convert(root_dir=self.train_dir)
        self.valid_data = self.convert(root_dir=self.valid_dir)

    def convert(self, root_dir, is_train: bool = True) -> None:
        if is_train:
            rgb_save_path = self.train_rgb_path
            depth_save_path = self.train_depth_path
        else:
            rgb_save_path = self.valid_rgb_path
            depth_save_path = self.valid_depth_path

        scenes = glob.glob(os.path.join(root_dir, 'indoors', '*'))

        idx = 0
        for scene in scenes:
            scans = glob.glob(os.path.join(scene, '*'))

            for scan in scans:
                rgb_files = sorted(glob.glob(os.path.join(scan, '*.png')))
            
                for rgb_file in rgb_files:
                    # rgb file name
                    rgb_file_name = os.path.basename(rgb_file)
                    # depth file name .png -> .npy
                    depth_file_name = rgb_file_name.replace('.png', '_depth.npy')

                    # depth mask file name depth name + _mask.npy
                    depth_mask_file_name = rgb_file_name.replace('.png', '_depth_mask.npy')

                    # read rgb
                    rgb = Image.open(rgb_file)

                    # depth
                    depth = np.load(os.path.join(scan, depth_file_name))
                    depth_mask = np.load(os.path.join(scan, depth_mask_file_name))
                    depth = np.squeeze(depth, axis=-1)
                    depth = depth * depth_mask

                    rgb_filename = os.path.join(rgb_save_path, f'rgb_{str(idx).zfill(6)}.jpg')
                    depth_file_name = os.path.join(depth_save_path, f'depth_{str(idx).zfill(6)}.npy')

                    # save rgb
                    rgb.save(rgb_filename, 'jpeg', quality=100)

                    # save depth
                    np.save(depth_file_name, depth)  

                    idx += 1

if __name__ == '__main__':
    root_dir = './depth/data/diode/'
    tspxr_capture = DiodeConverter(root_dir)