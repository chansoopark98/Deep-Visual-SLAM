import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple
import random


class BaseDepthDataset(Dataset):
    """Base PyTorch Dataset for depth estimation"""
    
    def __init__(self, image_size: Tuple[int, int], is_train: bool = True, augment: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.augment = augment and is_train
        
        # ImageNet 정규화 값
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # 데이터 증강을 위한 transform 설정
        self._setup_transforms()
        
        # 서브클래스에서 구현해야 할 속성들
        self.image_paths = []
        self.depth_paths = []
        self.depth_factor = 1000.0  # mm to meter conversion factor
        
    def _setup_transforms(self):
        """데이터 증강 및 전처리 transform 설정"""
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2
            )
        
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)

    def _read_image(self, path: str) -> Image.Image:
        """이미지를 PIL Image로 읽기"""
        image = Image.open(path)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        
        # PIL Image로 리사이즈
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        # RGB로 변환 (필요한 경우)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image

    def _read_depth(self, path: str) -> Image.Image:
        """16-bit 깊이 이미지를 PIL Image로 읽기"""
        depth = Image.open(path)
        if depth is None:
            raise ValueError(f"Failed to read depth: {path}")
        
        # PIL Image로 리사이즈 (16-bit 유지)
        depth = depth.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        
        return depth
    
    def _apply_augmentation(self, image, depth) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image(PIL Image): RGB 이미지
        depth(PIL Image): 16bit 깊이 이미지(mm 단위)
        """
    
        if self.augment:
            # 랜덤 좌우 반전
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # Color jitter 적용
            image = self.color_jitter(image)
        
        image = self.to_tensor(image)
        
        # mm depth to meter depth
        depth_array = np.array(depth, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0) / self.depth_factor

        return image, depth_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 샘플 반환"""
        # 경로 읽기
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        
        # 이미지 읽기
        image = self._read_image(image_path)
        depth = self._read_depth(depth_path)

        # 증강 적용
        image, depth = self._apply_augmentation(image, depth)

        return {
            'image': image,
            'depth': depth,
        }