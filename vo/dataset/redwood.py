import os
import glob
import numpy as np
import numpy.linalg

class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat
    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

def write_trajectory(traj, filename):
    with open(filename, 'w') as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(' '.join(map(str, x.metadata)) + '\n')
            f.write('\n'.join(' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
            f.write('\n')

class RedwoodHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/slam_data/redwood'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source'] # 1
        self.imu_seq_len = self.config['Train']['imu_seq_len'] # 10
        self.original_image_size = (480, 640)
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.test_dir = os.path.join(self.root_dir, 'test')
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True, is_test=False)

         # share valid data with test data
        self.valid_data = self.generate_datasets(fold_dir=self.test_dir, shuffle=False, is_test=True)
        self.test_data = self.generate_datasets(fold_dir=self.test_dir, shuffle=False, is_test=True)

    def _process(self, scene_dir: str, is_test: bool=False):
        # load camera metadata
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'image', '*.jpg')))
        length = len(rgb_files)

        if is_test:
            step = 1
        else:
            step = 2
        
        samples = []

        for t in range(self.num_source, length - self.num_source, step):
            sample = {
                'source_left': rgb_files[t - 1], # str
                'target_image': rgb_files[t], # str
                'source_right': rgb_files[t + 1], # str
                'intrinsic': intrinsic # np.ndarray (3, 3)
            }
            samples.append(sample)
        return samples
            
    def generate_datasets(self, fold_dir, shuffle=False, is_test=False):
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        datasets = []
        for scene in scene_files:
            dataset = self._process(scene, is_test)
            datasets.append(dataset)
        datasets = np.concatenate(datasets, axis=0)

        if shuffle:
            np.random.shuffle(datasets)
        if is_test:
            # pick 1000 samples for test
            datasets = datasets[:1000]
        return datasets

if __name__ == '__main__':
    import yaml

    # load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = RedwoodHandler(config)
    data_len = dataset.train_data.shape[0]
    for idx in range(data_len):
        sample = dataset.train_data[idx]
        print(sample)