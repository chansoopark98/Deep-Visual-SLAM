import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple

class MonoDataset(Dataset):
    """Monodepth2용 모노 데이터셋"""
    def __init__(self,
                 config: Dict,
                 dataset_dict: Dict[str, list],
                 image_size: Tuple[int, int],
                 is_train: bool = True, 
                 augment: bool = True):
        self.config = config
        self.image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_scale = config['Train']['num_scale']
        self.rgb_samples = dataset_dict['rgb_samples']
        self.intrinsic = dataset_dict['intrinsic']
        self.image_size = image_size
        self.is_train = is_train
        if self.is_train:
            self.max_size = 3 # t-max_size, t, t+max_size 
        else:
            self.max_size = 1
        self.augment = augment and is_train
        self._setup_transforms()

    def _setup_transforms(self):
        self.to_tensor = transforms.ToTensor()
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
            )

    def _read_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.rgb_samples) - (self.max_size * 2)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs = {}

        imgs = []

        # get random size
        size_1 = random.randint(1, self.max_size)
        size_2 = random.randint(1, self.max_size)

        source_left = self.rgb_samples[idx]
        target_image = self.rgb_samples[idx + size_1]
        source_right = self.rgb_samples[idx + size_1 + size_2]

        imgs.append(self._read_image(source_left))
        imgs.append(self._read_image(target_image))
        imgs.append(self._read_image(source_right))

        for scale in range(4):
            Wnew = self.image_shape[1] // (2**scale)
            Hnew = self.image_shape[0] // (2**scale)

            K = self.intrinsic[idx].copy()  # 절대 픽셀 단위
            K[0, :] *= (Wnew / self.image_shape[1])
            K[1, :] *= (Hnew / self.image_shape[0])

            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)]     = torch.from_numpy(K).float()
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).float()
        
        batch = torch.stack([self.to_tensor(im) for im in imgs], dim=0)

        if self.augment:
            if random.random() < 0.5:
                batch = self.color_jitter(batch)

        # batch to images
        source_left = batch[0]
        target_image = batch[1]
        source_right = batch[2]

        inputs[("source_left", 0)] = source_left
        inputs[("target_image", 0)] = target_image
        inputs[("source_right", 0)] = source_right

        return inputs
    
class StereoDataset(Dataset):
    """Monodepth2용 스테레오 데이터셋 (baseline 기반 pose 사용)"""
    def __init__(self,
                 dataset_dict: Dict[str, list],
                 image_size: Tuple[int, int], 
                 is_train: bool = True, 
                 augment: bool = True):
        self.source_image = dataset_dict['source_image']
        self.target_image = dataset_dict['target_image']
        self.intrinsic = dataset_dict['intrinsic']
        self.pose = dataset_dict['pose']
        self.image_size = image_size
        self.is_train = is_train
        self.augment = augment and is_train
        self._setup_transforms()

    def _setup_transforms(self):
        self.to_tensor = transforms.ToTensor()
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )

    def _read_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.source_image)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        imgs = []
        for path in (self.source_image[idx], self.target_image[idx]):
            imgs.append(self._read_image(path))

        batch = torch.stack([self.to_tensor(im) for im in imgs], dim=0)
        pose = torch.from_numpy(self.pose[idx].copy()).float()
        K = torch.from_numpy(self.intrinsic[idx].copy())

        if self.augment:
            # if random.random() < 0.5:
                # batch = torch.flip(batch, dims=[-1])
                # principal point c_x = width - old_cx
                # image_size: (H, W)
                # W = self.image_size[1]
                # K[0, 2] = W - K[0, 2]
                # pose[3] = -pose[3]

            if random.random() < 0.5:
                batch = self.color_jitter(batch)

        return {
            'source_image': batch[0],
            'target_image': batch[1],
            'intrinsic': K,
            'pose': pose
        }