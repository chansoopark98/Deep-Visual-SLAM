import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

def resample_imu(imu_sequence: np.ndarray, new_samples: int) -> np.ndarray:
    n, features = imu_sequence.shape
    
    if n == new_samples:
        return imu_sequence
    elif n > new_samples:
        indices = np.linspace(0, n - 1, new_samples).astype(int)
        resampled_sequence = imu_sequence[indices]
    else:
        x_old = np.linspace(0, 1, n, endpoint=True)
        x_new = np.linspace(0, 1, new_samples, endpoint=True)
        
        resampled_sequence = np.zeros((new_samples, 6))

        for i in range(features):
            interp_func = interp1d(x_old, imu_sequence[:, i], kind='linear', fill_value="extrapolate")
            resampled_sequence[:, i] = interp_func(x_new)
    return resampled_sequence