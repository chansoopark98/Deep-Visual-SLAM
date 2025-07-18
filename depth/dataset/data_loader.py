import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import List
import yaml

try:
    from .nyu_depth_v2 import NyuDepthHandler
    from .redwood_handler import RedwoodHandler
    from .custom_loader import CustomDataHandler
except:
    from nyu_depth_v2 import NyuDepthHandler
    from redwood_handler import RedwoodHandler
    from custom_loader import CustomDataHandler


class DepthLoader:
    """PyTorch DataLoader wrapper for depth estimation"""

    def __init__(self, config):
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_workers = config.get('Train', {}).get('num_workers', 16)
        
        # 데이터셋 로드
        self._load_datasets()
        
        # DataLoader 생성
        if self.train_datasets:
            self.train_depth_loader = self._create_dataloader(
                self.train_datasets,
                batch_size=self.batch_size,
                shuffle=True
            )
            self.num_depth_train = len(self.train_depth_loader)
        else:
            self.train_depth_loader = None
            self.num_depth_train = 0

        if self.valid_datasets:
            self.valid_depth_loader = self._create_dataloader(
                self.valid_datasets,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.num_depth_valid = len(self.valid_depth_loader)
        else:
            self.valid_depth_loader = None
            self.num_depth_valid = 0
    
    def _load_datasets(self):
        """데이터셋 로드"""
        self.train_datasets = []
        self.valid_datasets = []
        
        # NYU Depth V2 데이터
        if self.config['Dataset']['Nyu_depth_v2'].get('train', False):
            nyu_handler = NyuDepthHandler(config=self.config)
            
            if nyu_handler.train_dataset:
                self.train_datasets.append(nyu_handler.train_dataset)
                print(f"Added NYU train dataset: {len(nyu_handler.train_dataset)} samples")
            
            if self.config['Dataset'].get('Nyu_depth_v2', {}).get('valid', False) and nyu_handler.valid_dataset:
                self.valid_datasets.append(nyu_handler.valid_dataset)
                print(f"Added NYU valid dataset: {len(nyu_handler.valid_dataset)} samples")
        
        # Redwood 데이터
        if self.config['Dataset']['Redwood'].get('train', False):
            redwood_handler = RedwoodHandler(config=self.config)
            
            if redwood_handler.train_dataset:
                self.train_datasets.append(redwood_handler.train_dataset)
                print(f"Added Redwood train dataset: {len(redwood_handler.train_dataset)} samples")
            
            if self.config['Dataset'].get('Redwood', {}).get('valid', False) and redwood_handler.valid_dataset:
                self.valid_datasets.append(redwood_handler.valid_dataset)
                print(f"Added Redwood valid dataset: {len(redwood_handler.valid_dataset)} samples")
        
        # Custom 데이터
        if self.config['Dataset']['Custom'].get('train', False):
            custom_handler = CustomDataHandler(config=self.config)
            
            if custom_handler.train_dataset:
                self.train_datasets.append(custom_handler.train_dataset)
                print(f"Added Custom train dataset: {len(custom_handler.train_dataset)} samples")
            
            if self.config['Dataset'].get('Custom', {}).get('valid', False) and custom_handler.valid_dataset:
                self.valid_datasets.append(custom_handler.valid_dataset)
                print(f"Added Custom valid dataset: {len(custom_handler.valid_dataset)} samples")
        
        # 전체 샘플 수 출력
        total_train = sum(len(d) for d in self.train_datasets)
        total_valid = sum(len(d) for d in self.valid_datasets)
        print(f"\nTotal train samples: {total_train}")
        print(f"Total valid samples: {total_valid}")
    
    def _create_dataloader(self, datasets: List, batch_size: int, shuffle: bool) -> DataLoader:
        """DataLoader 생성"""
        # 여러 데이터셋을 하나로 결합
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        else:
            combined_dataset = ConcatDataset(datasets)
        
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return dataloader
    
    @staticmethod
    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        """이미지 역정규화 (시각화용)"""
        image = image * 255.0
        image = image.clamp(0, 255).byte()
        return image


# 사용 예시
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # PyTorch DataLoader 생성
    data_loader = DepthLoader(config)
    
    if data_loader.train_depth_loader is None:
        print("No training data available")
        exit()
    
    # 데이터 로드 테스트
    debug = True
    avg_time = 0.0
    
    for i, batch in enumerate(data_loader.train_depth_loader):
        start_time = time.time()

        images = batch['image']
        depths = batch['depth']

        if debug:
            if i % 10 == 0:
                print(f"Batch {i}: images shape: {images.shape}, depths shape: {depths.shape}")
                # 이미지 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(data_loader.denormalize_image(images[0]).permute(1, 2, 0).numpy())
                axs[0].set_title('Image')
                axs[1].imshow(depths[0][0].numpy(), cmap='gray')
                axs[1].set_title('Depth')
                plt.show()

        avg_time += time.time() - start_time

        if i > 100:
            break

    print(f"Average time per batch: {avg_time / (i + 1):.4f} seconds")