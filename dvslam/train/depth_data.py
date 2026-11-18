"""Phase 3.1 data pipeline for DAv2-S metric fine-tuning (plan v3 §4.5).

Deployment camera: Redwood K (fx = fy = 525, 640x480, f/W = 0.8203). Network input: fixed 476x630 (H x W).
Focal canonicalisation (Metric3D-style canonical camera): an image from a source camera with focal f_s (pixels at
width W_s) is fed unchanged except for the resize to 476x630, and its GT depth is rescaled to what the deployment
camera would see for the same image content:  d_canon = d_true * (f_dep / W_dep) / (f_s / W_s).
At inference on a camera with K, d_true = d_pred / canon_factor(K, W). On Redwood the factor is exactly 1.

Leakage rules (task brief): fold `loso-<room>` trains on the other 4 rooms (all splits) plus external data and is
evaluated on <room> validation+test; fold `single` trains on external data only (0 Redwood frames) and is
evaluated on all 5 rooms. External data: TUM RGB-D (CC BY 4.0), see reports/phase3/DATA.md.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dvslam.data.redwood import DEFAULT_ROOT as REDWOOD_ROOT, DEPTH_SCALE as REDWOOD_DEPTH_SCALE, SCENES

TUM_ROOT = Path(os.environ.get('DVSLAM_TUM_ROOT', '/home/park-ubuntu/park/datasets/tum_rgbd'))

DEPLOY_K = np.array([[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]])
DEPLOY_W, DEPLOY_H = 640, 480
NET_H, NET_W = 476, 630          # 34 x 45 ViT-S/14 patches
D_MIN, D_MAX = 0.1, 10.0
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# TUM calibrated RGB intrinsics (file_formats page). Depth is pre-registered to RGB and pre-scaled (ds); the TUM
# page recommends not undistorting the registered depth, so images are used as recorded (fr3 is already undistorted).
TUM_K = {'freiburg1': (517.3, 516.5, 318.6, 255.3), 'freiburg2': (520.9, 521.0, 325.1, 249.7),
         'freiburg3': (535.4, 539.2, 320.1, 247.6)}
TUM_DEPTH_SCALE = 5000.0
TUM_HOLDOUT = ('rgbd_dataset_freiburg1_room', 'rgbd_dataset_freiburg2_desk',
               'rgbd_dataset_freiburg3_long_office_household')
FOLDS = ('single',) + tuple(f'loso-{s}' for s in SCENES)


@dataclass(frozen=True)
class FrameRec:
    rgb: str
    depth: str
    depth_scale: float           # PNG units per metre
    fx: float
    fy: float
    width: int
    height: int
    source: str                  # 'redwood' | 'tum'
    seq: str                     # redwood: '<split>/<scene>'; tum: sequence dir name

    @property
    def canon_factor(self) -> float:
        return canon_factor(self.fx, self.fy, self.width)


Clip = tuple[FrameRec, ...]


def canon_factor(fx: float, fy: float, width: int) -> float:
    """Multiply true depth by this to get deployment-canonical depth (1.0 for the Redwood camera)."""
    return (DEPLOY_K[0, 0] / DEPLOY_W) / (0.5 * (fx + fy) / width)


# ----------------------------------------------------------------------------------------------- listing

def redwood_frames(scene: str, split: str, root: Path | None = None) -> list[FrameRec]:
    d = Path(root or REDWOOD_ROOT) / split / scene
    stems = sorted(p[:-4] for p in os.listdir(d / 'image') if p.endswith('.jpg'))
    K = DEPLOY_K  # Redwood intrinsic.npy is the deployment K
    return [FrameRec(str(d / 'image' / f'{s}.jpg'), str(d / 'depth' / f'{s}.png'), REDWOOD_DEPTH_SCALE,
                     K[0, 0], K[1, 1], DEPLOY_W, DEPLOY_H, 'redwood', f'{split}/{scene}') for s in stems]


def _read_tum_list(path: Path) -> list[tuple[float, str]]:
    out = []
    for line in path.read_text().splitlines():
        if line and not line.startswith('#'):
            t, f = line.split()[:2]
            out.append((float(t), f))
    return out


def tum_frames(seq_dir: Path, max_dt: float = 0.02) -> list[FrameRec]:
    """RGB frames in time order, each associated with the nearest depth frame within max_dt seconds."""
    seq_dir = Path(seq_dir)
    fr = next(k for k in TUM_K if k in seq_dir.name)
    fx, fy, _, _ = TUM_K[fr]
    rgb, dep = _read_tum_list(seq_dir / 'rgb.txt'), _read_tum_list(seq_dir / 'depth.txt')
    td = np.array([t for t, _ in dep])
    out = []
    for t, f in rgb:
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) <= max_dt:
            out.append(FrameRec(str(seq_dir / f), str(seq_dir / dep[j][1]), TUM_DEPTH_SCALE, fx, fy, 640, 480,
                                'tum', seq_dir.name))
    return out


def tum_sequences(root: Path | None = None, holdout: bool = False) -> list[Path]:
    root = Path(root or TUM_ROOT)
    seqs = sorted(p for p in root.glob('rgbd_dataset_freiburg*') if (p / 'rgb.txt').exists())
    return [p for p in seqs if (p.name in TUM_HOLDOUT) == holdout]


def make_clips(frames: list[FrameRec], T: int, stride: int, gap: int = 1) -> list[Clip]:
    """Clips of T frames spaced `gap` apart, starting every `stride` frames (frames of one sequence, in order)."""
    span = (T - 1) * gap
    return [tuple(frames[s + k * gap] for k in range(T)) for s in range(0, len(frames) - span, stride)]


def fold_lists(fold: str, T: int = 2, redwood_stride: int = 10, tum_stride: int = 4,
               redwood_root: Path | None = None, tum_root: Path | None = None) -> dict:
    """Training clips and evaluation targets for one fold.

    Returns dict(external=[Clip], redwood=[Clip], eval=[(scene, split)], heldout_scenes=[scene]).
    """
    if fold not in FOLDS:
        raise ValueError(f'unknown fold {fold!r}, expected one of {FOLDS}')
    external = [c for seq in tum_sequences(tum_root) for c in make_clips(tum_frames(seq), T, tum_stride)]
    if fold == 'single':
        train_rooms, held = (), SCENES
    else:
        k = fold.removeprefix('loso-')
        train_rooms, held = tuple(s for s in SCENES if s != k), (k,)
    redwood = [c for sc in train_rooms for sp in ('train', 'validation', 'test')
               for c in make_clips(redwood_frames(sc, sp, redwood_root), T, redwood_stride)]
    return dict(external=external, redwood=redwood, heldout_scenes=list(held),
                eval=[(sc, sp) for sc in held for sp in ('validation', 'test')])


def holdout_clips(T: int = 2, stride: int = 30, tum_root: Path | None = None) -> list[Clip]:
    """External hold-out (item 3.3): whole TUM sequences never used for training."""
    return [c for seq in tum_sequences(tum_root, holdout=True) for c in make_clips(tum_frames(seq), T, stride)]


def clip_paths(clips: list[Clip]) -> list[str]:
    return [p for c in clips for f in c for p in (f.rgb, f.depth)]


# ----------------------------------------------------------------------------------------------- loading

def prepare_image(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR (any size, 4:3) -> float32 (3, NET_H, NET_W) ImageNet-normalised RGB. Shared with dav2_ft."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)


def load_frame(rec: FrameRec) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 BGR at native res, canonical depth in metres at NET res, 0 = invalid)."""
    bgr = cv2.imread(rec.rgb, cv2.IMREAD_COLOR)
    raw = cv2.imread(rec.depth, cv2.IMREAD_UNCHANGED)
    if bgr is None or raw is None:
        raise FileNotFoundError(rec.rgb if bgr is None else rec.depth)
    d = raw.astype(np.float32) / rec.depth_scale
    d = cv2.resize(d, (NET_W, NET_H), interpolation=cv2.INTER_NEAREST) * rec.canon_factor
    return bgr, d


class ClipDataset(Dataset):
    """Returns image (T, 3, NET_H, NET_W) float32 and canonical depth (T, NET_H, NET_W) float32 (0 = invalid).
    Augmentation (same for every frame of a clip): horizontal flip, brightness/contrast/saturation jitter.
    No geometric scale augmentation: it would change the focal and break the metric target."""

    def __init__(self, clips: list[Clip], augment: bool = True):
        self.clips, self.augment = clips, augment

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i):
        rng = np.random.default_rng((i, int(torch.initial_seed()) % (2 ** 31))) if self.augment else None
        flip = bool(rng.random() < 0.5) if rng is not None else False
        jit = rng.uniform([0.8, 0.8, 0.8], [1.2, 1.2, 1.2]) if rng is not None else None
        imgs, deps = [], []
        for rec in self.clips[i]:
            bgr, d = load_frame(rec)
            if jit is not None:
                x = bgr.astype(np.float32)
                x = x * jit[0]                                                 # brightness
                x = (x - x.mean()) * jit[1] + x.mean()                         # contrast
                g = x.mean(axis=2, keepdims=True)
                x = (x - g) * jit[2] + g                                       # saturation
                bgr = np.clip(x, 0, 255).astype(np.uint8)
            img = prepare_image(bgr)
            if flip:
                img, d = img[:, :, ::-1], d[:, ::-1]
            imgs.append(np.ascontiguousarray(img))
            deps.append(np.ascontiguousarray(d))
        return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(deps))
