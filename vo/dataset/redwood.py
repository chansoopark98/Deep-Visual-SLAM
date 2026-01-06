import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple, List

try:
    from .common import MonoDataset
except:
    from common import MonoDataset


class RedwoodSceneDataset(MonoDataset):
    """단일 Redwood 씬을 위한 데이터셋 (MonoDataset 상속)"""

    def __init__(self,
                 config: Dict,
                 scene_dir: str,
                 scene_name: str,
                 intrinsic: np.ndarray,
                 is_train: bool = True,
                 augment: bool = True):
        self.scene_dir = scene_dir
        self.scene_name = scene_name
        self.poses = None  # GT 포즈 저장용

        # 데이터 로드
        dataset_dict = self._load_scene_data(scene_dir, scene_name, intrinsic)

        # 부모 클래스 초기화
        image_size = (config['Train']['img_h'], config['Train']['img_w'])
        super().__init__(
            config=config,
            dataset_dict=dataset_dict,
            image_size=image_size,
            is_train=is_train,
            augment=augment
        )

    def _load_poses_from_json(self, json_path: str) -> List[np.ndarray]:
        """JSON 파일에서 포즈 행렬들을 로드

        Open3D PoseGraph JSON은 column-major (Fortran order)로 저장됨
        즉, 16개 값이 [col0, col1, col2, col3] 순서로 저장
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        poses = []
        nodes = data.get('nodes', [])

        for node in nodes:
            pose_flat = node['pose']  # 16개 값 (column-major)
            # Column-major (Fortran order)로 4x4 행렬로 변환
            # order='F'를 사용하여 올바른 SE(3) 행렬 복원 (마지막 열이 translation)
            pose_matrix = np.array(pose_flat, dtype=np.float32).reshape(4, 4, order='F')
            poses.append(pose_matrix)

        return poses

    def _load_scene_data(self, scene_dir: str, scene_name: str, intrinsic: np.ndarray) -> Dict:
        """씬 데이터 로드"""
        image_dir = os.path.join(scene_dir, 'image')
        json_path = os.path.join(scene_dir, f'{scene_name}.json')

        # 이미지 파일 목록
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

        # 포즈 로드
        poses = self._load_poses_from_json(json_path)

        # 이미지와 포즈 개수 맞추기
        min_len = min(len(image_files), len(poses))
        image_files = image_files[:min_len]
        poses = poses[:min_len]

        # GT 포즈 저장
        self.poses = poses

        # intrinsic 복사
        intrinsics = [intrinsic.copy() for _ in range(min_len)]

        return {
            'rgb_samples': np.array(image_files, dtype=str),
            'intrinsic': np.array(intrinsics, dtype=np.float32)
        }

    def _get_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """두 절대 포즈에서 상대 포즈 계산

        Redwood poses는 T_wc (camera-to-world)
        상대 포즈 T_rel은 camera1 frame에서 camera2 frame으로의 변환
        T_rel = T_c2w @ T_wc1 = inv(T_wc2) @ T_wc1

        하지만 PoseNet은 source->target 방향을 예측하므로:
        T_rel = inv(T_wc1) @ T_wc2
        """
        T_rel = np.linalg.inv(pose1) @ pose2
        return T_rel

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """MonoDataset의 __getitem__을 오버라이드하여 GT 포즈 추가 (test mode only)"""
        # 부모 클래스의 __getitem__ 호출
        inputs = super().__getitem__(idx)

        # Test mode에서만 GT 포즈 추가 (self-supervised learning이므로 train에서는 불필요)
        if not self.is_train and self.poses is not None:
            # 현재 인덱스에 대한 프레임 인덱스 계산 (부모 클래스와 동일한 로직)
            size_1 = random.randint(1, self.max_size)
            size_2 = random.randint(1, self.max_size)

            left_idx = idx
            target_idx = idx + size_1
            right_idx = idx + size_1 + size_2

            # 범위 체크
            max_idx = len(self.poses) - 1
            target_idx = min(target_idx, max_idx)
            right_idx = min(right_idx, max_idx)

            # GT 상대 포즈 계산
            pose_left = self.poses[left_idx]
            pose_target = self.poses[target_idx]
            pose_right = self.poses[right_idx]

            rel_pose_left_to_target = self._get_relative_pose(pose_left, pose_target)
            rel_pose_target_to_right = self._get_relative_pose(pose_target, pose_right)

            inputs["gt_pose_left_to_target"] = torch.from_numpy(rel_pose_left_to_target).float()
            inputs["gt_pose_target_to_right"] = torch.from_numpy(rel_pose_target_to_right).float()
            inputs["frame_indices"] = torch.tensor([left_idx, target_idx, right_idx])

        return inputs


class RedwoodMonoDataset(Dataset):
    """Redwood 전체 데이터셋 (여러 씬을 결합)"""

    def __init__(self,
                 config: Dict,
                 fold: str = 'train',
                 is_train: bool = True,
                 augment: bool = True):
        self.config = config
        self.fold = fold
        self.root_dir = '/home/park-ubuntu/park/Deep-Visual-SLAM/depth/data/redwood'
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])

        # 각 씬별로 데이터셋 생성
        self.scene_datasets = self._create_scene_datasets(is_train, augment)

        # ConcatDataset으로 결합
        if self.scene_datasets:
            self.combined_dataset = ConcatDataset(self.scene_datasets)
        else:
            self.combined_dataset = None

    def _load_intrinsic(self) -> np.ndarray:
        """카메라 내부 파라미터 로드 및 스케일링"""
        intrinsic_path = os.path.join(self.root_dir, 'intrinsic.npy')
        K3 = np.load(intrinsic_path)  # 3x3

        # 원본 해상도 (640x480)에서 타겟 해상도로 스케일링
        original_size = (480, 640)  # (H, W)
        target_size = self.image_size  # (H, W)

        fx = K3[0, 0] * target_size[1] / original_size[1]
        fy = K3[1, 1] * target_size[0] / original_size[0]
        cx = K3[0, 2] * target_size[1] / original_size[1]
        cy = K3[1, 2] * target_size[0] / original_size[0]

        # 4x4 homogeneous 행렬로 확장
        K4 = np.eye(4, dtype=np.float32)
        K4[0, 0] = fx
        K4[1, 1] = fy
        K4[0, 2] = cx
        K4[1, 2] = cy

        return K4

    def _create_scene_datasets(self, is_train: bool, augment: bool) -> List[RedwoodSceneDataset]:
        """각 씬별로 데이터셋 생성"""
        fold_dir = os.path.join(self.root_dir, self.fold)

        if not os.path.exists(fold_dir):
            print(f"Warning: {fold_dir} does not exist")
            return []

        scenes = sorted([d for d in os.listdir(fold_dir)
                        if os.path.isdir(os.path.join(fold_dir, d))])

        intrinsic = self._load_intrinsic()
        datasets = []

        for scene in scenes:
            scene_dir = os.path.join(fold_dir, scene)
            image_dir = os.path.join(scene_dir, 'image')
            json_path = os.path.join(scene_dir, f'{scene}.json')

            if not os.path.exists(image_dir) or not os.path.exists(json_path):
                print(f"Warning: Skipping {scene} - missing image folder or json")
                continue

            try:
                scene_dataset = RedwoodSceneDataset(
                    config=self.config,
                    scene_dir=scene_dir,
                    scene_name=scene,
                    intrinsic=intrinsic,
                    is_train=is_train,
                    augment=augment
                )

                if len(scene_dataset) > 0:
                    datasets.append(scene_dataset)
                    print(f"  Loaded scene '{scene}': {len(scene_dataset)} samples")

            except Exception as e:
                print(f"Warning: Failed to load scene '{scene}': {e}")
                continue

        total_samples = sum(len(d) for d in datasets)
        print(f"Redwood {self.fold}: Total {total_samples} samples from {len(datasets)} scenes")

        return datasets

    def __len__(self):
        if self.combined_dataset is None:
            return 0
        return len(self.combined_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.combined_dataset is None:
            raise IndexError("No data available")
        return self.combined_dataset[idx]


class RedwoodDataHandler:
    """Redwood 데이터셋 핸들러"""

    def __init__(self, config: Dict):
        self.config = config
        self.root_dir = '/home/park-ubuntu/park/Deep-Visual-SLAM/depth/data/redwood'

        self.train_mono_dataset = None
        self.valid_mono_dataset = None
        self.test_mono_dataset = None

        if config['Dataset']['redwood'].get('mono', False):
            print("\nLoading Redwood dataset...")

            # Train 데이터셋
            train_dir = os.path.join(self.root_dir, 'train')
            if os.path.exists(train_dir) and os.listdir(train_dir):
                self.train_mono_dataset = RedwoodMonoDataset(
                    config=config,
                    fold='train',
                    is_train=True,
                    augment=True
                )

            # Validation 데이터셋
            valid_dir = os.path.join(self.root_dir, 'validation')
            if os.path.exists(valid_dir) and os.listdir(valid_dir):
                self.valid_mono_dataset = RedwoodMonoDataset(
                    config=config,
                    fold='validation',
                    is_train=False,
                    augment=False
                )

            # Test 데이터셋
            test_dir = os.path.join(self.root_dir, 'test')
            if os.path.exists(test_dir) and os.listdir(test_dir):
                self.test_mono_dataset = RedwoodMonoDataset(
                    config=config,
                    fold='test',
                    is_train=False,
                    augment=False
                )


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt

    # Load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Test dataset
    print("Testing RedwoodMonoDataset...")
    dataset = RedwoodMonoDataset(config=config, fold='train', is_train=False, augment=False)

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]

        print("\nSample keys:", list(sample.keys()))
        print(f"Source left shape: {sample[('source_left', 0)].shape}")
        print(f"Target image shape: {sample[('target_image', 0)].shape}")
        print(f"Source right shape: {sample[('source_right', 0)].shape}")
        print(f"K shape: {sample[('K', 0)].shape}")

        if 'gt_pose_left_to_target' in sample:
            print(f"\nGT pose left->target:\n{sample['gt_pose_left_to_target']}")
            print(f"Frame indices: {sample['frame_indices']}")

            # Extract translation from GT pose
            gt_pose = sample['gt_pose_left_to_target'].numpy()
            translation = gt_pose[:3, 3]
            print(f"\nTranslation (x, y, z): {translation}")
            print(f"Translation magnitude: {np.linalg.norm(translation):.6f} m")

        # Visualize
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(sample[('source_left', 0)].permute(1, 2, 0).numpy())
        axs[0].set_title('Source Left')
        axs[0].axis('off')

        axs[1].imshow(sample[('target_image', 0)].permute(1, 2, 0).numpy())
        axs[1].set_title('Target')
        axs[1].axis('off')

        axs[2].imshow(sample[('source_right', 0)].permute(1, 2, 0).numpy())
        axs[2].set_title('Source Right')
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig('redwood_sample.png')
        print("\nSaved sample visualization to redwood_sample.png")
