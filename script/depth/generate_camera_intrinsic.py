import numpy as np

# NYU Depth v2 dataset camera intrinsic parameters
fx = 518.8579
fy = 519.4696
cx = 325.5824
cy = 253.7362

# generate camera intrinsic matrix
intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
np.save('./nyu_intrinsic.npy', intrinsic)


# DIODE indoor dataset camera intrinsic parameters
fx = 886.81
fy = 927.06
cx = 512
cy = 384

intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
np.save('./diode_intrinsic.npy', intrinsic)

# REDWOOD indoor dataset camera intrinsic parameters
# FOCAL LENGTH: 525.0
# OPTICAL CENTER: (319.5, 239.5)
# IMAGE SIZE: H,W (480, 640)
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
np.save('./redwood_intrinsic.npy', intrinsic)