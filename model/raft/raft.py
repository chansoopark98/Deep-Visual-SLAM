import torch
import argparse
import numpy as np
import cv2
from PIL import Image

try:
    from .core.raft import RAFT, SmallRAFT
    from .core.utils.utils import InputPadder
    from .core.utils import flow_viz
except:
    from core.raft import RAFT, SmallRAFT
    from core.utils.utils import InputPadder
    from core.utils import flow_viz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(imfile):
    img = Image.open(imfile).convert("RGB")
    img = img.resize((640, 480), Image.BILINEAR)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img /= 255.0  # Normalize to [0, 1]
    return img[None].to(DEVICE)

def viz(img, flo):
    img = img[0].permute(1,2,0).detach().cpu().numpy()
    flo = flo[0].permute(1,2,0).detach().cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.small = True
args.alternate_corr = False
args.mixed_precision = False

model = SmallRAFT(args)
# load pretrained weights if available
path = './raft-small.pth'
state_dict = torch.load(path)
# 'module.' prefix 제거
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.to(DEVICE)
model.eval()

source_left = load_image('./model/raft/prev.jpg')
target_image = load_image('./model/raft/current.jpg')
padder = InputPadder(source_left.shape)

flow_low, flow_up = model(source_left, target_image, iters=20, test_mode=True)

viz(source_left * 255, flow_up)

