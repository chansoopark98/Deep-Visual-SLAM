import os
import glob
import pandas as pd
import numpy as np
import cv2
try:
    from .utils import resample_imu
except:
    from utils import resample_imu

class MarsLoggerHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'mars_logger')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source'] # 1
        self.imu_seq_len = self.config['Train']['imu_seq_len'] # 10
        self.original_image_size = (3000, 4000)
        self.save_image_size = (3000 // 4, 4000 // 4)
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.test_dir = os.path.join(self.root_dir, 'test')
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)
        self.test_data = self.generate_datasets(fold_dir=self.test_dir, shuffle=False, is_test=True)

    def _extract_video(self, scene_dir: str, current_intrinsic: np.ndarray, camera_data: pd.DataFrame) -> int:
        video_file = os.path.join(scene_dir, 'movie.mp4')
        rgb_save_path = os.path.join(scene_dir, 'rgb')

        # Rescale intrinsic matrix
        resized_intrinsic = self._rescale_intrinsic(current_intrinsic, self.save_image_size, self.original_image_size)

        # Read metadata
        # timestamps_ns = camera_data['Timestamp[nanosec]'].values  # Extract timestamps in nanoseconds
        # timestamps_s = timestamps_ns / 1e9  # Convert to seconds

        # Ensure output directory exists
        if not os.path.exists(rgb_save_path):
            print(f'Extracting video file: {video_file}')
            os.makedirs(rgb_save_path, exist_ok=True)

            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_file}")

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_name = os.path.join(rgb_save_path, f'rgb_{str(idx).zfill(6)}.jpg')
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.resize(frame, (self.save_image_size[1], self.save_image_size[0]))
                cv2.imwrite(rgb_name, frame)
                idx += 1

            cap.release()
            cv2.destroyAllWindows()

        # Count and return the number of saved frames
        return len(glob.glob(os.path.join(rgb_save_path, '*.jpg'))), resized_intrinsic

    def _rescale_intrinsic(self, intrinsic: np.ndarray, target_size: tuple, current_size: tuple) -> np.ndarray:
        # New shape = self.image_size (H, W)
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        intrinsic_rescaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsic_rescaled

    def _process(self, scene_dir: str, is_test: bool=False) -> list:
        # load camera metadata
        camera_file = os.path.join(scene_dir, 'movie_metadata.csv')
        camera_data = pd.read_csv(camera_file)
        fx = 2.66908046e+03
        fy = 2.67550677e+03
        cx = 2.05566387e+03
        cy = 1.44153479e+03
        raw_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
        # load video .mp4
        length, resized_intrinsic = self._extract_video(scene_dir, raw_intrinsic, camera_data)

        intrinsic = self._rescale_intrinsic(resized_intrinsic, self.image_size, self.save_image_size)
    
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*.jpg')))
        
        if is_test:
            step = 1
        else:
            step = 2

        samples = []
        for t in range(self.num_source, length - self.num_source, step):
            sample = {
                'source_left': rgb_files[t-1], # str
                'target_image': rgb_files[t], # str
                'source_right': rgb_files[t+1], # str
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
        return datasets

if __name__ == '__main__':
    import yaml
    
    # load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = MarsLoggerHandler(config)
    data_len = dataset.train_data.shape[0]
    for idx in range(data_len):
        print(dataset.train_data[idx])