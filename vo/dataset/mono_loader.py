import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Dict, List, Tuple
import random

try:
    from .mars_logger import MarsLoggerHandler
    from .redwood import RedwoodHandler
    from .custom_data import CustomDataHandler
    from .irs import IrsDataHandler
except:
    from mars_logger import MarsLoggerHandler
    from redwood import RedwoodHandler
    from custom_data import CustomDataHandler
    from irs import IrsDataHandler


class MonoDataset(Dataset):
    """PyTorch Dataset for mono visual odometry"""
    
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
        return len(self.samples['source_left'])

    def _read_image(self, path: str) -> np.ndarray:
        """이미지 읽기 및 리사이즈 - 최적화 버전"""
        # OpenCV로 읽기 (더 빠름)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        
        # 리사이즈 먼저, 그 다음 색상 변환 (메모리 효율적)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), 
                          interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _apply_augmentation(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """이미지 증강 적용"""
        # numpy to PIL
        pil_images = [Image.fromarray(img) for img in images]
        
        # 동일한 증강 적용을 위한 seed 설정
        if self.augment:
            # 랜덤 좌우 반전
            if random.random() > 0.5:
                pil_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in pil_images]
            
            # Color jitter 적용
            seed = random.randint(0, 2**32)
            augmented_images = []
            for img in pil_images:
                random.seed(seed)
                torch.manual_seed(seed)
                augmented_images.append(self.color_jitter(img))
            pil_images = augmented_images
        
        # PIL to tensor
        tensor_images = [self.to_tensor(img) for img in pil_images]
        
        return tensor_images
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 샘플 반환"""
        # 경로 읽기
        source_left_path = self.samples['source_left'][idx]
        target_path = self.samples['target_image'][idx]
        source_right_path = self.samples['source_right'][idx]
        
        # 이미지 읽기
        source_left = self._read_image(source_left_path)
        target = self._read_image(target_path)
        source_right = self._read_image(source_right_path)
        
        # 증강 적용
        images = self._apply_augmentation([source_left, target, source_right])
        source_left, target, source_right = images
        
        # intrinsic 변환
        intrinsic = torch.from_numpy(self.samples['intrinsic'][idx]).float()
        
        return {
            'source_left': source_left,
            'target_image': target,
            'source_right': source_right,
            'intrinsic': intrinsic
        }


class MonoLoader:
    """PyTorch DataLoader wrapper for mono visual odometry"""
    
    def __init__(self, config):
        self.config = config
        self.batch_size = config['Train']['batch_size']
        self.use_shuffle = config['Train']['use_shuffle']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_source = config['Train']['num_source']
        self.num_workers = config.get('Train', {}).get('num_workers', 16)
        
        # 데이터셋 로드
        self._load_dataset()
        
        # DataLoader 생성
        if self.num_mono_train > 0:
            self.train_mono_loader = self._create_dataloader(
                self.train_mono_data, 
                batch_size=self.batch_size,
                shuffle=True,
                is_train=True
            )
            self.num_mono_train = len(self.train_mono_loader)
            
        if self.num_mono_valid > 0:
            self.valid_mono_loader = self._create_dataloader(
                self.valid_mono_data,
                batch_size=self.batch_size,
                shuffle=False,
                is_train=False
            )
            self.num_mono_valid = len(self.valid_mono_loader)
            
        if self.num_mono_test > 0:
            self.test_mono_loader = self._create_dataloader(
                self.test_mono_data,
                batch_size=self.batch_size,
                shuffle=False,
                is_train=False
            )
            self.num_mono_test = len(self.test_mono_loader)
    
    def _load_dataset(self):
        """데이터셋 로드"""
        train_data_list = []
        valid_data_list = []
        test_data_list = []
        
        self.num_mono_train = 0
        self.num_mono_valid = 0
        self.num_mono_test = 0
        
        # Mars Logger 데이터
        if self.config['Dataset']['mars_logger']:
            dataset = MarsLoggerHandler(config=self.config)
            train_data_list.append(dataset.train_data)
            valid_data_list.append(dataset.valid_data)
            test_data_list.append(dataset.test_data)
            
            self.num_mono_train += len(dataset.train_data['source_left'])
            self.num_mono_valid += len(dataset.valid_data['source_left'])
            self.num_mono_test += len(dataset.test_data['source_left'])
        
        # Custom 데이터
        if self.config['Dataset']['custom_data']:
            dataset = CustomDataHandler(config=self.config, mode='mono')
            train_data_list.append(dataset.train_data)
            valid_data_list.append(dataset.valid_data)
            
            self.num_mono_train += len(dataset.train_data['source_left'])
            self.num_mono_valid += len(dataset.valid_data['source_left'])
        
        # IRS 데이터
        if self.config['Dataset']['irs']:
            dataset = IrsDataHandler(config=self.config, mode='mono')
            train_data_list.append(dataset.train_data)
            valid_data_list.append(dataset.valid_data)
            
            self.num_mono_train += len(dataset.train_data['source_left'])
            self.num_mono_valid += len(dataset.valid_data['source_left'])
        
        # 데이터 결합
        self.train_mono_data = self._combine_datasets(train_data_list)
        self.valid_mono_data = self._combine_datasets(valid_data_list)
        self.test_mono_data = self._combine_datasets(test_data_list)
    
    def _combine_datasets(self, data_list: List[Dict]) -> Dict[str, np.ndarray]:
        """여러 데이터셋을 하나로 결합"""
        if not data_list:
            return {
                'source_left': np.array([]),
                'target_image': np.array([]),
                'source_right': np.array([]),
                'intrinsic': np.array([])
            }
        
        combined = {
            'source_left': [],
            'target_image': [],
            'source_right': [],
            'intrinsic': []
        }
        
        for data in data_list:
            if len(data['source_left']) > 0:
                combined['source_left'].extend(data['source_left'])
                combined['target_image'].extend(data['target_image'])
                combined['source_right'].extend(data['source_right'])
                combined['intrinsic'].extend(data['intrinsic'])
        
        # numpy 배열로 변환
        for key in combined:
            combined[key] = np.array(combined[key])
        
        return combined
    
    def _create_dataloader(self, data: Dict, batch_size: int, shuffle: bool, 
                          is_train: bool) -> DataLoader:
        """DataLoader 생성"""
        dataset = MonoDataset(
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
    
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # PyTorch DataLoader 생성
    data_loader = MonoLoader(config)
    
    # 데이터 로드 테스트
    avg_time = 0.0
    for i, batch in enumerate(data_loader.train_mono_loader):
        # if i >= 10:  # 10 배치만 테스트
        #     break
        # print(i)
        start_time = time.time()
        
        left_images = batch['source_left']
        target_images = batch['target_image']
        right_images = batch['source_right']
        intrinsics = batch['intrinsic']
        
        avg_time += time.time() - start_time
        
        # # 첫 번째 샘플 시각화
        # if i == 0:
        #     # 역정규화
        #     left_img = MonoLoader.denormalize_image(left_images[0])
        #     target_img = MonoLoader.denormalize_image(target_images[0])
        #     right_img = MonoLoader.denormalize_image(right_images[0])
            
        #     # CHW to HWC
        #     left_img = left_img.permute(1, 2, 0).numpy()
        #     target_img = target_img.permute(1, 2, 0).numpy()
        #     right_img = right_img.permute(1, 2, 0).numpy()
            
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #     axes[0].imshow(left_img)
        #     axes[0].set_title('Source Left')
        #     axes[1].imshow(target_img)
        #     axes[1].set_title('Target')
        #     axes[2].imshow(right_img)
        #     axes[2].set_title('Source Right')
            
        #     print(f"Batch size: {left_images.shape[0]}")
        #     print(f"Image shape: {left_images.shape}")
        #     print(f"Intrinsic shape: {intrinsics.shape}")
        #     print(f"Intrinsic[0]: \n{intrinsics[0].numpy()}")
            
        #     plt.tight_layout()
        #     plt.show()

        if i > 100:
            break

    print(f"\nAverage loading time per batch: {avg_time / 10:.4f} seconds")