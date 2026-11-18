"""Redwood indoor RGB-D splits as prepared under <root>/{train,validation,test}/<scene>/ (plan v3 item 0.4).

Layout per scene: <scene>.json (Open3D PoseGraph, column-major 4x4 camera-to-world), image/NNNNNN.jpg,
depth/NNNNNN.png (uint16, millimetres). Intrinsics: <root>/intrinsic.npy (fx = fy = 525, 640x480).
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path(os.environ.get('DVSLAM_REDWOOD_ROOT', '/mnt/backup/SLAM/redwood'))
SCENES = ('apartment', 'bedroom', 'boardroom', 'lobby', 'loft')
# UNVERIFIED: the data has no timestamps; 30 Hz is the usual rate of the PrimeSense-class sensor used for
# Redwood, not yet checked against the dataset documentation (plan v3 §2.2). Timestamps below are index / 30.
FRAME_RATE_HZ = 30.0
FRAME_RATE_VERIFIED = False
DEPTH_SCALE = 1000.0  # uint16 PNG units per metre


@dataclass
class RedwoodSequence:
    scene: str
    split: str
    frame_ids: np.ndarray        # (N,) int, from file names; consecutive
    image_paths: list[Path]
    depth_paths: list[Path]
    T_wc: np.ndarray             # (N, 4, 4) float64 camera-to-world
    K: np.ndarray                # (3, 3) float64
    width: int = 640
    height: int = 480

    @property
    def timestamps(self) -> np.ndarray:
        return np.arange(len(self.frame_ids), dtype=np.float64) / FRAME_RATE_HZ

    def __len__(self) -> int:
        return len(self.frame_ids)


def load_poses(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        nodes = json.load(f)['nodes']
    return np.array([np.array(n['pose'], dtype=np.float64).reshape((4, 4), order='F') for n in nodes])


def load_sequence(scene: str, split: str = 'test', root: Path | str | None = None) -> RedwoodSequence:
    root = Path(root) if root else DEFAULT_ROOT
    d = root / split / scene
    T_wc = load_poses(d / f'{scene}.json')
    images = sorted((d / 'image').glob('*.jpg'))
    if len(images) != len(T_wc):
        raise ValueError(f'{scene}/{split}: {len(images)} images but {len(T_wc)} poses')
    ids = np.array([int(p.stem) for p in images])
    if not np.array_equal(np.diff(ids), np.ones(len(ids) - 1, dtype=ids.dtype)):
        raise ValueError(f'{scene}/{split}: image indices are not consecutive')
    depths = [d / 'depth' / f'{p.stem}.png' for p in images]
    missing = [p for p in depths if not p.exists()]
    if missing:
        raise ValueError(f'{scene}/{split}: {len(missing)} depth PNGs missing, e.g. {missing[0]}')
    K = np.load(root / 'intrinsic.npy').astype(np.float64)
    return RedwoodSequence(scene, split, ids, images, depths, T_wc, K)


def load_depth(path: Path) -> np.ndarray:
    """Depth in metres (float32), 0 = invalid."""
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
