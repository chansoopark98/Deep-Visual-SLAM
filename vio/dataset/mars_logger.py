import os
import glob
import pandas as pd
import numpy as np
import cv2
from .utils import resample_imu

class MarsLoggerHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = os.path.join(self.config['Directory']['data_dir'], 'mars_logger')
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.original_image_size = (3000, 4000)
        self.save_image_size = (3000 // 4, 4000 // 4)
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.valid_dir = os.path.join(self.root_dir, 'valid')
        self.train_data = self.generate_datasets(fold_dir=self.train_dir, shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir=self.valid_dir, shuffle=False)

    def _extract_video(self, scene_dir: str, current_intrinsic: np.ndarray) -> int:
        video_file = os.path.join(scene_dir, 'movie.mp4')
        rgb_save_path = os.path.join(scene_dir, 'rgb')
        
        resized_intrinsic = self._rescale_intrinsic(current_intrinsic, self.save_image_size, self.original_image_size)
        
        if not os.path.exists(rgb_save_path):
            print(f'Extracting video file: {video_file}')
            os.makedirs(rgb_save_path, exist_ok=True)
            idx = 0
            cap = cv2.VideoCapture(video_file)    

            while cap.isOpened():
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
        return len(glob.glob(os.path.join(rgb_save_path, '*.jpg'))), resized_intrinsic

    def _rescale_intrinsic(self, intrinsic: np.ndarray, target_size: tuple, current_size: tuple) -> np.ndarray:
        # New shape = self.image_size (H, W)
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        intrinsic_rescaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsic_rescaled

    def _process(self, scene_dir: str):
        # load csv imu file
        imu_file = os.path.join(scene_dir, 'gyro_accel.csv')
        imu_data = pd.read_csv(imu_file)

        # load camera metadata
        camera_file = os.path.join(scene_dir, 'movie_metadata.csv')
        camera_data = pd.read_csv(camera_file)
        fx = camera_data['fx[px]'].values[0]
        fy = camera_data['fy[px]'].values[0]
        cx = 4000 / 2
        cy = 3000 / 2
        raw_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
        # load video .mp4
        length, resized_intrinsic = self._extract_video(scene_dir, raw_intrinsic)

        intrinsic = self._rescale_intrinsic(resized_intrinsic, self.image_size, self.save_image_size)
        step = 1
        imu_seq = 8
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*.jpg')))
        
        samples = []
        for t in range(step, length - step):
            # -------------------------
            # 1) t-step, t, t+step
            # -------------------------
            t_left_idx = t - step
            t_right_idx = t + step

            time_left = camera_data.loc[t_left_idx, 'Timestamp[nanosec]']
            time_curr = camera_data.loc[t, 'Timestamp[nanosec]']
            time_right = camera_data.loc[t_right_idx, 'Timestamp[nanosec]']

            mask_left = (imu_data['Timestamp[nanosec]'] >= time_left) & (imu_data['Timestamp[nanosec]'] < time_curr)
            left_imu_df = imu_data[mask_left]

            mask_right = (imu_data['Timestamp[nanosec]'] >= time_curr) & (imu_data['Timestamp[nanosec]'] < time_right)
            right_imu_df = imu_data[mask_right]

            left_imu_array = left_imu_df.iloc[:, 1:].values
            right_imu_array = right_imu_df.iloc[:, 1:].values

            if left_imu_array.shape[0] < 4:
                continue
            if right_imu_array.shape[0] < 4:
                continue

            left_imu_resampled = resample_imu(left_imu_array, imu_seq)
            right_imu_resampled = resample_imu(right_imu_array, imu_seq)

            sample = {
                'source_left': rgb_files[t_left_idx],
                'target_image': rgb_files[t],
                'source_right': rgb_files[t_right_idx],
                'imu_left': left_imu_resampled,
                'imu_right': right_imu_resampled,
                'intrinsic': intrinsic
            }
            samples.append(sample)
        return samples
            
    def generate_datasets(self, fold_dir, shuffle=False):
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        datasets = []
        for scene in scene_files:
            dataset = self._process(scene)
            datasets.append(dataset)
        datasets = np.concatenate(datasets, axis=0)

        if shuffle:
            np.random.shuffle(datasets)
        return datasets

if __name__ == '__main__':
    root_dir = './vio/data/'
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
            'img_h': 720,
            'img_w': 1280
        }
    }
    tspxr_capture = MarsLoggerHandler(config)
    