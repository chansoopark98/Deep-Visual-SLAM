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
                 dataset_dict: Dict[str, list],
                 image_size: Tuple[int, int],
                 is_train: bool = True, 
                 augment: bool = True):
        self.source_left = dataset_dict['source_left']
        self.target_image = dataset_dict['target_image']
        self.source_right = dataset_dict['source_right']
        self.intrinsic = dataset_dict['intrinsic']
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
        return len(self.target_image)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        imgs = []
        for path in (self.source_left[idx],
                     self.target_image[idx],
                     self.source_right[idx]):
            imgs.append(self._read_image(path))

        batch = torch.stack([self.to_tensor(im) for im in imgs], dim=0)

        K = torch.from_numpy(self.intrinsic[idx].copy())

        if self.augment:
            if random.random() < 0.5:
                batch = torch.flip(batch, dims=[-1])
                W = self.image_size[1]
                K[0, 2] = W - K[0, 2]
            
            if random.random() < 0.5:
                batch = self.color_jitter(batch)

        return {
            'source_left':  batch[0],
            'target_image': batch[1],
            'source_right': batch[2],
            'intrinsic':    K
        }

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
            if random.random() < 0.5:
                batch = torch.flip(batch, dims=[-1])
                # principal point c_x = width - old_cx
                # image_size: (H, W)
                W = self.image_size[1]
                K[0, 2] = W - K[0, 2]
                pose[3] = -pose[3]

            if random.random() < 0.5:
                batch = self.color_jitter(batch)

        return {
            'source_image': batch[0],
            'target_image': batch[1],
            'intrinsic': K,
            'pose': pose
        }