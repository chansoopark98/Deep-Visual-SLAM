import os
import glob
import numpy as np

class NyuDepthLoader(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.train_data = self.generate_datasets(root_dir=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(root_dir=self.valid_dir, shuffle=False)

    def generate_datasets(self, root_dir, shuffle: False):
        rgb_files = sorted(glob.glob(os.path.join(root_dir, 'rgb', '*.jpg')))
        depth_files = sorted(glob.glob(os.path.join(root_dir, 'depth', '*.png')))
        
        # check the number of files
        assert len(rgb_files) == len(depth_files), 'The number of rgb and depth files are not matched.'

        samples = []

        for idx in range(len(rgb_files)):
            sample = {
                'rgb': rgb_files[idx],
                'depth': depth_files[idx]
            }
            samples.append(sample)

        samples = np.array(samples)
        
        if shuffle:
            np.random.shuffle(samples)
        return samples

if __name__ == '__main__':
    root_dir = './depth/data/nyu_depth_v2_raw'
    tspxr_capture = NyuDepthLoader(root_dir)
    
    for idx in range(tspxr_capture.train_data.shape[0]):
        print(idx)