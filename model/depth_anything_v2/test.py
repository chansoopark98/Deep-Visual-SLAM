import cv2
import torch
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# Now use absolute imports
from model.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'
root = './assets/weights/depth'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'{root}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('./assets/depth_test.png')
# depth = model.infer_image(raw_img) # HxW raw depth map in numpy
# rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) / 255.0
# normalize rgb (imagenet mean/std)
rgb, res = model.image2tensor(raw_img)
output = model(rgb)
print(f"Output shape: {output.shape}")  # Should be [1, 1, H, W] for depth map