import matplotlib.pyplot as plt
import os

# /media/park-ubuntu/park_file/depth_data/diml_raw/01_warehouse_1/16.01.20/5
path = '/media/park-ubuntu/park_file/depth_data/1'

# rgb
rgb_path = os.path.join(path, 'col/in_k_03_160225_000001_c.png')
depth_raw_path = os.path.join(path, 'raw_png/in_k_03_160225_000001_rd.png')
depth_up_path = os.path.join(path, 'up_png/in_k_03_160225_000001_ud.png')
depth_warp_path = os.path.join(path, 'warp_png/in_k_03_160225_000001_wd.png')

plt.imshow(plt.imread(rgb_path))
plt.show()

plt.imshow(plt.imread(depth_raw_path))
plt.show()

plt.imshow(plt.imread(depth_up_path))
plt.show()

plt.imshow(plt.imread(depth_warp_path))
plt.show()