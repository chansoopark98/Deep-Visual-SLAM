import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import List
import yaml

try:
    from .custom_data import CustomDataHandler
    
except:
    from custom_data import CustomDataHandler
    

class VoDataLoader:
    """PyTorch DataLoader wrapper for depth estimation"""

    def __init__(self, config):
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_workers = config['Train']['num_workers']
        print(f"Using {self.num_workers} workers for data loading")

        # Datasets (Mono, Stereo)
        self.train_mono_datasets = []
        self.valid_mono_datasets = []
        self.test_mono_datasets = []

        self.train_stereo_datasets = []
        self.valid_stereo_datasets = []
        self.test_stereo_datasets = []

        # 데이터셋 로드
        self._load_datasets()
        
        # DataLoader 생성
        if self.train_mono_datasets:
            self.train_mono_loader = self._create_dataloader(
                self.train_mono_datasets,
                batch_size=self.batch_size,
                shuffle=True
            )
            self.num_train_mono = len(self.train_mono_loader)
        else:
            self.train_mono_loader = None
            self.num_train_mono = 0

        if self.valid_mono_datasets:
            self.valid_mono_loader = self._create_dataloader(
                self.valid_mono_datasets,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.num_valid_mono = len(self.valid_mono_loader)
        else:
            self.valid_mono_loader = None
            self.num_valid_mono = 0
        
        if self.test_mono_datasets:
            self.test_mono_loader = self._create_dataloader(
                self.test_mono_datasets,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.num_test_mono = len(self.test_mono_loader)
        else:
            self.test_mono_loader = None
            self.num_test_mono = 0

        if self.train_stereo_datasets:
            self.train_stereo_loader = self._create_dataloader(
                self.train_stereo_datasets,
                batch_size=self.batch_size,
                shuffle=True
            )
            self.num_train_stereo = len(self.train_stereo_loader)
        else:
            self.train_stereo_loader = None
            self.num_train_stereo = 0

        if self.valid_stereo_datasets:
            self.valid_stereo_loader = self._create_dataloader(
                self.valid_stereo_datasets,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.num_valid_stereo = len(self.valid_stereo_loader)
        else:
            self.valid_stereo_loader = None
            self.num_valid_stereo = 0

        if self.test_stereo_datasets:
            self.test_stereo_loader = self._create_dataloader(
                self.test_stereo_datasets,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.num_test_stereo = len(self.test_stereo_loader)
        else:
            self.test_stereo_loader = None
            self.num_test_stereo = 0

    def _load_datasets(self):
        """데이터셋 로드"""

        # Custom 데이터셋 로드
        if self.config['Dataset']['custom_data']:
            custom_handler = CustomDataHandler(config=self.config)

            if custom_handler.train_mono_dataset:
                self.train_mono_datasets.append(custom_handler.train_mono_dataset)
                print(f"Added Custom train dataset: {len(custom_handler.train_mono_dataset)} samples")
            
            if custom_handler.valid_mono_dataset:
                self.valid_mono_datasets.append(custom_handler.valid_mono_dataset)
                print(f"Added Custom valid dataset: {len(custom_handler.valid_mono_dataset)} samples")
            
            if custom_handler.train_stereo_dataset:
                self.train_stereo_datasets.append(custom_handler.train_stereo_dataset)
                print(f"Added Custom Stereo train dataset: {len(custom_handler.train_stereo_dataset)} samples")
    
            if custom_handler.valid_stereo_dataset:
                self.valid_stereo_datasets.append(custom_handler.valid_stereo_dataset)
                print(f"Added Custom Stereo valid dataset: {len(custom_handler.valid_stereo_dataset)} samples")

        # 전체 샘플 수 출력
        total_train_mono = sum(len(d) for d in self.train_mono_datasets)
        total_valid_mono = sum(len(d) for d in self.valid_mono_datasets)
        print(f"\nTotal train samples: {total_train_mono}")
        print(f"Total valid samples: {total_valid_mono}")

        total_train_stereo = sum(len(d) for d in self.train_stereo_datasets)
        total_valid_stereo = sum(len(d) for d in self.valid_stereo_datasets)
        print(f"Total Stereo train samples: {total_train_stereo}")
        print(f"Total Stereo valid samples: {total_valid_stereo}")

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
    
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # PyTorch DataLoader 생성
    data_loader = VoDataLoader(config)
    
    if data_loader.train_mono_loader is None:
        print("No training data available")
        exit()
    
    # 데이터 로드 테스트
    debug = True
    avg_time = 0.0
    
    for i, batch in enumerate(data_loader.train_mono_loader):
        start_time = time.time()

        source_left = batch['source_left']
        target_image = batch['target_image']
        source_right = batch['source_right']
        intrinsic = batch['intrinsic']

        if debug:
            if i % 10 == 0:
                print(f"Batch {i}: images shape: {source_left.shape}, intrinsic: {intrinsic.shape}")
                print(f'Intrinsic matrix:\n{intrinsic[0]}')
                # 이미지 시각화
                fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                axs[0].imshow(data_loader.denormalize_image(source_left[0]).permute(1, 2, 0).numpy())
                axs[0].set_title('Source Left')
                axs[1].imshow(data_loader.denormalize_image(target_image[0]).permute(1, 2, 0).numpy())
                axs[1].set_title('Target Image')
                axs[2].imshow(data_loader.denormalize_image(source_right[0]).permute(1, 2, 0).numpy())
                axs[2].set_title('Source Right')
                plt.show()


        avg_time += time.time() - start_time

        if i > 100:
            break

    print(f"Average time per batch: {avg_time / (i + 1):.4f} seconds")