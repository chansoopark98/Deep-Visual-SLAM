import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Dict, List, Tuple
import random

try:
    from .custom_data import CustomDataHandler
    from .irs import IrsDataHandler
except:
    from custom_data import CustomDataHandler
    from irs import IrsDataHandler


class StereoDataset(Dataset):
    """PyTorch Dataset for stereo visual odometry"""
    
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
        
        # 자주 사용하는 값들 미리 계산
        self.num_samples = len(self.samples['source_image'])
        
    def _setup_transforms(self):
        """데이터 증강 및 전처리 transform 설정"""
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return self.num_samples
    
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
        """이미지 증강 적용 - 스테레오 이미지에 동일한 증강 적용"""
        if self.augment:
            # 랜덤 파라미터 미리 계산
            do_flip = random.random() > 0.5
            
            # NumPy 레벨에서 증강 적용 (더 빠름)
            if do_flip:
                images = [np.fliplr(img) for img in images]
            
            # PIL로 변환하여 color jitter 적용
            pil_images = [Image.fromarray(img) for img in images]
            
            # 동일한 seed로 color jitter 적용
            seed = random.randint(0, 2**32)
            augmented_images = []
            for img in pil_images:
                random.seed(seed)
                torch.manual_seed(seed)
                augmented_images.append(self.color_jitter(img))
            
            # Tensor로 변환
            tensor_images = [self.to_tensor(img) for img in augmented_images]
        else:
            # 증강 없이 바로 tensor 변환
            tensor_images = []
            for img in images:
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                tensor_images.append(img)
        
        return tensor_images
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 샘플 반환"""
        try:
            # 경로 읽기
            source_path = self.samples['source_image'][idx]
            target_path = self.samples['target_image'][idx]
            
            # 이미지 읽기
            source_image = self._read_image(source_path)
            target_image = self._read_image(target_path)
            
            # 증강 적용 (스테레오 이미지에 동일한 증강)
            images = self._apply_augmentation([source_image, target_image])
            source_image, target_image = images
            
            # intrinsic과 pose 변환
            intrinsic = torch.from_numpy(self.samples['intrinsic'][idx].astype(np.float32, copy=False))
            pose = torch.from_numpy(self.samples['pose'][idx].astype(np.float32, copy=False))
            
            return {
                'source_image': source_image,
                'target_image': target_image,
                'intrinsic': intrinsic,
                'pose': pose
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # 에러 발생 시 다음 샘플 반환
            return self.__getitem__((idx + 1) % self.num_samples)


class StereoLoader:
    """PyTorch DataLoader wrapper for stereo visual odometry"""
    
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
        if self.num_stereo_train > 0:
            self.train_stereo_loader = self._create_dataloader(
                self.train_stereo_data,
                batch_size=self.batch_size,
                shuffle=True,
                is_train=True
            )
            self.num_stereo_train = len(self.train_stereo_loader)
            
        if self.num_stereo_valid > 0:
            self.valid_stereo_loader = self._create_dataloader(
                self.valid_stereo_data,
                batch_size=self.batch_size,
                shuffle=False,
                is_train=False
            )
            self.num_stereo_valid = len(self.valid_stereo_loader)
    
    def _load_dataset(self):
        """데이터셋 로드"""
        train_data_list = []
        valid_data_list = []
        
        self.num_stereo_train = 0
        self.num_stereo_valid = 0
        
        # Custom 데이터
        if self.config['Dataset']['custom_data']:
            dataset = CustomDataHandler(config=self.config, mode='stereo')
            train_data_list.append(dataset.train_data)
            valid_data_list.append(dataset.valid_data)
            
            self.num_stereo_train += len(dataset.train_data['source_image'])
            self.num_stereo_valid += len(dataset.valid_data['source_image'])
        
        # IRS 데이터
        if self.config['Dataset']['irs']:
            dataset = IrsDataHandler(config=self.config, mode='stereo')
            train_data_list.append(dataset.train_data)
            valid_data_list.append(dataset.valid_data)
            
            self.num_stereo_train += len(dataset.train_data['source_image'])
            self.num_stereo_valid += len(dataset.valid_data['source_image'])
        
        # 데이터 결합
        self.train_stereo_data = self._combine_datasets(train_data_list)
        self.valid_stereo_data = self._combine_datasets(valid_data_list)
    
    def _combine_datasets(self, data_list: List[Dict]) -> Dict[str, np.ndarray]:
        """여러 데이터셋을 하나로 결합"""
        if not data_list:
            return {
                'source_image': np.array([]),
                'target_image': np.array([]),
                'intrinsic': np.array([]),
                'pose': np.array([])
            }
        
        combined = {
            'source_image': [],
            'target_image': [],
            'intrinsic': [],
            'pose': []
        }
        
        for data in data_list:
            if len(data['source_image']) > 0:
                combined['source_image'].extend(data['source_image'])
                combined['target_image'].extend(data['target_image'])
                combined['intrinsic'].extend(data['intrinsic'])
                combined['pose'].extend(data['pose'])
        
        # numpy 배열로 변환
        for key in combined:
            combined[key] = np.array(combined[key])
        
        return combined
    
    def _create_dataloader(self, data: Dict, batch_size: int, shuffle: bool, 
                          is_train: bool) -> DataLoader:
        """DataLoader 생성 - 최적화 버전"""
        dataset = StereoDataset(
            samples=data,
            image_size=self.image_size,
            is_train=is_train,
            augment=is_train
        )
        
        # 최적의 num_workers 자동 설정
        import multiprocessing
        optimal_workers = min(self.num_workers, multiprocessing.cpu_count() - 1)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=optimal_workers,
            pin_memory=True,
            drop_last=True if is_train else False,
            persistent_workers=True if optimal_workers > 0 else False,
            # persistent_workers=False,
            prefetch_factor=2 if optimal_workers > 0 else 2,
            worker_init_fn=self._worker_init_fn
        )
        
        return dataloader
    
    @staticmethod
    def _worker_init_fn(worker_id):
        """DataLoader 워커 초기화 - 랜덤 시드 설정"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    @staticmethod
    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        """이미지 역정규화 (시각화용)"""
        image = image * 255.0
        image = image.clamp(0, 255).byte()
        return image
    
    @staticmethod
    def normalize_batch(images: torch.Tensor) -> torch.Tensor:
        """배치 단위 정규화 (GPU에서 수행 시 더 빠름)"""
        # ImageNet 정규화
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if images.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        
        return (images - mean) / std


# 사용 예시
if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    import time
    
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # PyTorch DataLoader 생성
    data_loader = StereoLoader(config)
    
    print(f"Train batches: {data_loader.num_stereo_train}")
    print(f"Valid batches: {data_loader.num_stereo_valid}")
    
    # 데이터 로드 테스트
    avg_time = 0.0
    for i, batch in enumerate(data_loader.train_stereo_loader):
        start_time = time.time()
        
        source_images = batch['source_image']
        target_images = batch['target_image']
        intrinsics = batch['intrinsic']
        poses = batch['pose']
        
        avg_time += time.time() - start_time
        
        # 첫 번째 샘플 시각화
        if i == 0:
            # 역정규화
            source_img = StereoLoader.denormalize_image(source_images[0])
            target_img = StereoLoader.denormalize_image(target_images[0])
            
            # CHW to HWC
            source_img = source_img.permute(1, 2, 0).numpy()
            target_img = target_img.permute(1, 2, 0).numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].imshow(source_img)
            axes[0].set_title('Source Image')
            axes[1].imshow(target_img)
            axes[1].set_title('Target Image')
            
            print(f"Batch size: {source_images.shape[0]}")
            print(f"Image shape: {source_images.shape}")
            print(f"Intrinsic shape: {intrinsics.shape}")
            print(f"Intrinsic[0]: \n{intrinsics[0].numpy()}")
            print(f"Pose[0]: {poses[0].numpy()}")
            
            plt.tight_layout()
            plt.show()
        
        if i >= 100:
            break
    
    print(f"\nAverage loading time per batch: {avg_time / min(i+1, 100):.4f} seconds")