from matplotlib.pylab import f
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple
import random

try:
    from .nyu_depth_v2 import NyuDepthHandler
except:
    from nyu_depth_v2 import NyuDepthHandler

class DepthDataset(Dataset):
    """PyTorch Dataset for depth estimation"""
    
    def __init__(self, samples: Dict[str, np.ndarray], image_size: Tuple[int, int], 
                 is_train: bool = True, augment: bool = True):
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train
        self.augment = augment and is_train
        
        # ImageNet 정규화 값
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # 데이터 증강을 위한 transform 설정
        self._setup_transforms()
        
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
        # self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
    def __len__(self):
        return len(self.samples['image'])

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
    
    def _apply_augmentation(self, image, depth) -> List[torch.Tensor]:
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
            # seed = random.randint(0, 2**32)
            image = self.color_jitter(image)
        
        image = self.to_tensor(image)
        
        # mm depth to meter depth
        depth_array = np.array(depth, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0) / 1000.0  # mm to m

        return image, depth_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 샘플 반환"""
        # 경로 읽기
        image_path = self.samples['image'][idx]
        depth_path = self.samples['depth'][idx]

        
        # 이미지 읽기
        image = self._read_image(image_path)
        depth = self._read_depth(depth_path)

        # 증강 적용
        image, depth = self._apply_augmentation(image, depth)

        return {
            'image': image,
            'depth': depth,
        }

class DepthLoader:
    """PyTorch DataLoader wrapper for depth estimation"""

    def __init__(self, config):
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_workers = config.get('Train', {}).get('num_workers', 16)
        
        # 데이터셋 로드
        self._load_dataset()
        
        # DataLoader 생성
        if self.num_depth_train > 0:
            self.train_depth_loader = self._create_dataloader(
                self.train_depth_data,
                batch_size=self.batch_size,
                shuffle=True,
                is_train=True
            )
            self.num_depth_train = len(self.train_depth_loader)

        if self.num_depth_valid > 0:
            self.valid_depth_loader = self._create_dataloader(
                self.valid_depth_data,
                batch_size=self.batch_size,
                shuffle=False,
                is_train=False
            )
            self.num_depth_valid = len(self.valid_depth_loader)

        # if self.num_depth_test > 0:
        #     self.test_depth_loader = self._create_dataloader(
        #         self.test_depth_data,
        #         batch_size=self.batch_size,
        #         shuffle=False,
        #         is_train=False
        #     )
        #     self.num_depth_test = len(self.test_depth_loader)
    
    def _load_dataset(self):
        """데이터셋 로드"""
        train_data_list = []
        valid_data_list = []

        self.num_depth_train = 0
        self.num_depth_valid = 0

        # Nyu Depth V2 데이터
        if self.config['Dataset']['Nyu_depth_v2']:
            dataset = NyuDepthHandler(config=self.config)

            if self.config['Dataset']['Nyu_depth_v2']['train']:
                train_data_list.append(dataset.train_data)
                self.num_depth_train += len(dataset.train_data['image'])

            if self.config['Dataset']['Nyu_depth_v2']['valid']:
                valid_data_list.append(dataset.valid_data)
                self.num_depth_valid += len(dataset.valid_data['image'])

        # 데이터 결합
        self.train_depth_data = self._combine_datasets(train_data_list)
        self.valid_depth_data = self._combine_datasets(valid_data_list)

    def _combine_datasets(self, data_list: List[Dict]) -> Dict[str, np.ndarray]:
        """여러 데이터셋을 하나로 결합"""
        if not data_list:
            return {
                'image': np.array([]),
                'depth': np.array([]),
            }
        
        combined = {
            'image': [],
            'depth': [],
        }
        
        for data in data_list:
            if len(data['image']) > 0:
                combined['image'].extend(data['image'])
            if len(data['depth']) > 0:
                combined['depth'].extend(data['depth'])
        
        # numpy 배열로 변환
        for key in combined:
            combined[key] = np.array(combined[key])
        
        return combined
    
    def _create_dataloader(self, data: Dict, batch_size: int, shuffle: bool, 
                          is_train: bool) -> DataLoader:
        """DataLoader 생성"""
        dataset = DepthDataset(
            samples=data,
            image_size=self.image_size,
            is_train=is_train,
            augment=is_train
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
            # persistent_workers=False
        )
        
        return dataloader
    
    @staticmethod
    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        """이미지 역정규화 (시각화용)"""
        # image = image * std + mean  # ImageNet 역정규화
        image = image * 255.0
        image = image.clamp(0, 255).byte()
        return image


# 사용 예시
if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    import time
    
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # PyTorch DataLoader 생성
    data_loader = DepthLoader(config)
    
    # 데이터 로드 테스트
    debug = False
    avg_time = 0.0
    for i, batch in enumerate(data_loader.train_depth_loader):
        start_time = time.time()

        images = batch['image']
        target_images = batch['depth']

        if debug:
            if i % 10 == 0:
                print(f"Batch {i}: images shape: {images.shape}, target_images shape: {target_images.shape}")
                # 이미지 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(data_loader.denormalize_image(images[0]).permute(1, 2, 0).numpy())
                axs[0].set_title('Image')
                axs[1].imshow(target_images[0][0].numpy(), cmap='gray')
                axs[1].set_title('Depth')
                plt.show()

        avg_time += time.time() - start_time

        if i > 100:
            break

    print(f"Average time per batch: {avg_time / (i + 1):.4f} seconds")