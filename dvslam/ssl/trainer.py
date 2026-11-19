"""Training loop for the self-supervised metric depth + pose model on TUM-VI.

Config (configs/ssl_tumvi.yaml): sections data / model / loss / train. Optimiser AdamW (no weight decay on norm layers
and biases, i.e. parameters with ndim <= 1), linear warmup then cosine to train.min_lr over the run, gradient clipping,
EMA of the weights (AveragedModel, buffers included). Seeds: python, numpy, torch, DataLoader shuffling (seed + epoch)
and workers (one torch thread each).
Validation (every val_every epochs, EMA weights, eval mode) on data.val_scene: the increment of every consecutive pair
through dvslam.ssl.predict.Predictor, scored with dvslam.eval.metrics.pair_metrics on pairs with seq.pair_valid. Mocap
GT is used here only. best.pt = lowest validation item 03 (MAPE %).
Outputs in out_dir: train_log.jsonl (step, epoch, lr, loss, each term), val_log.jsonl (also the epoch's training
steps_per_s and, on CUDA, peak_mem_gb so far), last.pt, best.pt. A checkpoint holds config, model, ema, optimizer,
scheduler, epoch (completed), step, best, git_commit, the training K / size and the AMP grad-scaler state.
Multi-GPU: started with torchrun, every process trains on its own GPU (DistributedDataParallel, gradients averaged).
train.batch_size is per GPU, so the global batch is batch_size x GPUs and an epoch has 1 / GPUs as many steps;
train.lr is used as written. Rank 0 alone keeps the EMA, validates, logs (its own loss values) and saves.
"""
import contextlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dvslam.data import tumvi
from dvslam.eval.metrics import pair_metrics
from dvslam.geometry.se3 import relative_pose
from dvslam.ssl.losses import SSLLoss
from dvslam.ssl.predict import Predictor, build_model
from dvslam.ssl.tumvi_data import TumviSSLDataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """'section.key=value' (value parsed as YAML) into cfg in place; the key must already exist."""
    for item in items:
        key, value = item.split('=', 1)
        *path, last = key.split('.')
        node = cfg
        for p in path:
            node = node[p]
        if last not in node:
            raise KeyError(f'unknown config key {key!r}')
        node[last] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init(worker_id: int) -> None:
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def init_distributed() -> tuple[int, int]:
    """(rank, world size). Under torchrun (WORLD_SIZE > 1): bind this process to GPU LOCAL_RANK and join the group."""
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world == 1:
        return 0, 1
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    # the other ranks wait on rank 0 while it builds caches and validates: longer than NCCL's default 10 minutes
    torch.distributed.init_process_group('nccl', timeout=timedelta(hours=6), device_id=torch.cuda.current_device())
    return int(os.environ['RANK']), world


def make_loader(cfg: dict, rank: int = 0, world: int = 1) -> DataLoader:
    """Training loader of this process. Building the dataset writes missing rectified caches, so with several
    processes rank 0 goes first and the others wait for it."""
    if world > 1 and rank > 0:
        torch.distributed.barrier()
    ds = TumviSSLDataset(cfg['data']['train_scenes'], train=True, root=cfg['data']['root'],
                         hints_dir=cfg['data'].get('stereo_hints'))
    if world > 1 and rank == 0:
        torch.distributed.barrier()
    workers = cfg['train']['num_workers']
    sampler = None
    if world > 1:
        sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=cfg['train']['seed'], drop_last=True)
    # spawn, not fork: forked workers deadlock in OpenCV once the parent has used its thread pool (cache build, remap)
    return DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=sampler is None, sampler=sampler,
                      drop_last=True, num_workers=workers, pin_memory=True, worker_init_fn=_worker_init,
                      generator=torch.Generator(), persistent_workers=workers > 0,
                      multiprocessing_context='spawn' if workers > 0 else None)


def git_commit() -> str:
    """HEAD, with '-dirty' when the work tree outside .omc/ differs from it (modified or untracked files)."""
    def git(*args: str) -> str:
        return subprocess.run(['git', '-C', str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=False).stdout.strip()
    return git('rev-parse', 'HEAD') + ('-dirty' if git('status', '--porcelain', '--', '.', ':(exclude).omc') else '')


class _Step(torch.nn.Module):
    """Model + loss as one forward: DistributedDataParallel only synchronises gradients of calls made through it."""

    def __init__(self, model: torch.nn.Module, loss: SSLLoss):
        super().__init__()
        self.model, self.loss = model, loss

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss(self.model, batch)


class Trainer:
    def __init__(self, cfg: dict, out_dir: Path, loader, teacher: torch.nn.Module | None = None, rank: int = 0,
                 world: int = 1):
        """loader: an iterable of batches with len() (a DataLoader from make_loader in the app).
        rank / world: from init_distributed(); the model starts from rank 0's weights on every process."""
        tc = cfg['train']
        seed_everything(tc['seed'])
        self.cfg, self.out, self.loader = cfg, Path(out_dir), loader
        self.rank, self.world = rank, world
        self.device = torch.device(tc['device'])
        self.model = build_model(cfg['model'], cfg['model']['pretrained']).to(self.device)
        self.loss = SSLLoss(cfg['loss'], teacher).to(self.device)
        self.step_fn = _Step(self.model, self.loss)
        if world > 1:
            # find_unused_parameters: the torchvision encoders carry an fc layer that never gets a gradient
            self.step_fn = DistributedDataParallel(self.step_fn, device_ids=[torch.cuda.current_device()],
                                                   find_unused_parameters=True)
        seed_everything(tc['seed'] + rank)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW([{'params': [p for p in params if p.ndim > 1], 'weight_decay': tc['weight_decay']},
                                      {'params': [p for p in params if p.ndim <= 1], 'weight_decay': 0.0}], lr=tc['lr'])
        total = max(tc['epochs'] * len(loader), 1)
        warmup, floor = tc['warmup_steps'], tc['min_lr'] / tc['lr']

        def lr_factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / max(total - warmup, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_factor)
        self.ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(tc['ema_decay']), use_buffers=True)
        # train.amp: autocast on the network forwards only (models.py docstring); the EMA copy made above and every
        # checkpoint stay float32. float16 + GradScaler on CUDA, bfloat16 (no scaler needed) on CPU.
        amp = bool(tc.get('amp', False))
        if amp:
            dtype = torch.float16 if self.device.type == 'cuda' else torch.bfloat16
            self.model.set_amp(dtype)
            if self.loss.teacher is not None:
                self.loss.teacher.amp_dtype = dtype
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=amp and self.device.type == 'cuda')
        self.epoch, self.step, self.best = 0, 0, math.inf
        self.K, self.size = None, None

    def train_step(self, batch: dict) -> dict[str, float]:
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        self.K, self.size = batch['K'][0].double().cpu().numpy(), tuple(batch['net'].shape[-2:])
        self.model.train()
        total, terms = self.step_fn(batch)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()          # disabled scaler (default): plain backward / step
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if self.rank == 0:
            self.ema.update_parameters(self.model)
        self.step += 1
        return {'loss': float(total.detach()), **{k: float(v.detach()) for k, v in terms.items()}}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        seq = tumvi.load_sequence(self.cfg['data']['val_scene'], root=self.cfg['data']['root'])
        pred = Predictor(self.ema.module, seq.K, (seq.height, seq.width), self.device)
        pairs = np.flatnonzero(seq.pair_valid) + 1          # frame t of each scored pair (t-1, t)
        pos, rpy, ratio = [], [], []
        bs = self.cfg['train']['batch_size']

        def read(idx: np.ndarray) -> np.ndarray:
            return np.stack([cv2.imread(str(seq.image_paths[j]), cv2.IMREAD_GRAYSCALE) for j in idx])

        for i in range(0, len(pairs), bs):
            t = pairs[i:i + bs]
            for k, T in zip(t, pred.increment(read(t - 1), read(t))):
                m = pair_metrics(relative_pose(seq.T_wc[k - 1], seq.T_wc[k]), T)
                pos.append(m['pos_error_percent'])
                rpy.append(m['rpy_abs_err_deg'])
                gt_norm = np.linalg.norm(m['gt_t'])
                if gt_norm > 1e-6:
                    ratio.append(np.linalg.norm(m['pred_t']) / gt_norm)
        return {'item03_mape_pct': float(np.mean(pos)), 'item04_rpy_mean_deg': float(np.mean(rpy)),
                'scale_median': float(np.median(ratio)), 'pairs': len(pos)}

    def save(self, path: Path) -> None:
        torch.save({'config': self.cfg, 'model': self.model.state_dict(), 'ema': self.ema.state_dict(),
                    'optimizer': self.opt.state_dict(), 'scheduler': self.sched.state_dict(), 'epoch': self.epoch,
                    'step': self.step, 'best': self.best, 'git_commit': git_commit(), 'K': self.K,
                    'size': self.size, 'scaler': self.scaler.state_dict()}, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.ema.load_state_dict(ckpt['ema'])
        self.opt.load_state_dict(ckpt['optimizer'])
        self.sched.load_state_dict(ckpt['scheduler'])
        if ckpt.get('scaler'):
            self.scaler.load_state_dict(ckpt['scaler'])
        self.epoch, self.step, self.best = ckpt['epoch'], ckpt['step'], ckpt['best']
        self.K, self.size = ckpt['K'], ckpt['size']

    def fit(self, resume: bool = False) -> None:
        tc = self.cfg['train']
        if resume:
            self.load(self.out / 'last.pt')
        main = self.rank == 0
        with contextlib.ExitStack() as files:
            if main:
                # line-buffered: a running job can be followed step by step
                log = files.enter_context(open(self.out / 'train_log.jsonl', 'a', buffering=1))
                vlog = files.enter_context(open(self.out / 'val_log.jsonl', 'a'))
            while self.epoch < tc['epochs']:
                if getattr(self.loader, 'generator', None) is not None:
                    self.loader.generator.manual_seed(tc['seed'] + self.epoch + 1000 * self.rank)
                if isinstance(getattr(self.loader, 'sampler', None), DistributedSampler):
                    self.loader.sampler.set_epoch(self.epoch)
                t0, step0 = time.perf_counter(), self.step
                for batch in self.loader:
                    terms = self.train_step(batch)
                    if main:
                        log.write(json.dumps({'step': self.step, 'epoch': self.epoch,
                                              'lr': self.sched.get_last_lr()[0], **terms}) + '\n')
                self.epoch += 1
                if not main:
                    continue
                speed = {'steps_per_s': (self.step - step0) / (time.perf_counter() - t0)}
                if self.device.type == 'cuda':
                    speed['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
                log.flush()
                improved = False
                if self.epoch % tc['val_every'] == 0 or self.epoch == tc['epochs']:
                    val = self.validate()
                    vlog.write(json.dumps({'step': self.step, 'epoch': self.epoch, **val, **speed}) + '\n')
                    vlog.flush()
                    print(f'epoch {self.epoch}: {val}')
                    improved = val['item03_mape_pct'] < self.best
                    self.best = min(self.best, val['item03_mape_pct'])
                self.save(self.out / 'last.pt')
                if improved:
                    shutil.copyfile(self.out / 'last.pt', self.out / 'best.pt')
