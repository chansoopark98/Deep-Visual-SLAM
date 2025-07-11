import sys
import os
import gc
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import PolynomialLR
from torch.amp import GradScaler, autocast

from vo.dataset.stereo_loader import StereoLoader
from vo.dataset.mono_loader import MonoLoader
from vo.monodepth_learner_torch import MonodepthLearner
from model.depthnet import DepthNet
from model.posenet import PoseNet
from utils.plot_utils import PlotTool

from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from itertools import cycle

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision 설정
        self.use_amp = config.get('Train', {}).get('use_amp', True) and self.device.type == 'cuda'
        
        # Update experiment name
        original_name = self.config['Directory']['exp_name']
        self.config['Directory']['exp_name'] = 'mode={0}_res={1}_ep={2}_bs={3}_initLR={4}_endLR={5}_prefix={6}'.format(
            self.config['Train']['mode'],
            (self.config['Train']['img_h'], self.config['Train']['img_w']),
            self.config['Train']['epoch'],
            self.config['Train']['batch_size'],
            self.config['Train']['init_lr'],
            self.config['Train']['final_lr'],
            original_name
        )
        
        self.configure_train_ops()
        print(f'Trainer initialized (AMP: {self.use_amp})')
    
    def configure_train_ops(self) -> None:
        """Configure training operations"""
        # 1. Models
        self.batch_size = self.config['Train']['batch_size']
        
        # Depth network
        self.depth_net = DepthNet(
            num_layers=18,
            pretrained=True,
            num_input_images=1,
            scales=range(4),
            num_output_channels=1,
            use_skips=True
        ).to(self.device)
        
        # Pose network  
        self.pose_net = PoseNet(
            num_layers=18,
            pretrained=True,
            num_input_images=2,
            num_frames_to_predict_for=1
        ).to(self.device)

        # torch.compile 제거 또는 옵션화 (메모리 누수 가능성)
        if self.config.get('Train', {}).get('use_compile', False):
            self.depth_net = torch.compile(self.depth_net, fullgraph=True)
            self.pose_net = torch.compile(self.pose_net, fullgraph=True)

        # 2. Data loaders
        self.stereo_loader = StereoLoader(config=self.config)
        self.mono_loader = MonoLoader(config=self.config)
        
        # 3. Optimizers
        self.stereo_optimizer = optim.Adam(
            self.depth_net.parameters(),
            lr=self.config['Train']['init_lr'],
            betas=(self.config['Train']['beta1'], 0.999),
            weight_decay=self.config['Train']['weight_decay'] if self.config['Train']['weight_decay'] > 0 else 0
        )
        
        self.mono_optimizer = optim.Adam(
            list(self.depth_net.parameters()) + list(self.pose_net.parameters()),
            lr=self.config['Train']['init_lr'],
            betas=(self.config['Train']['beta1'], 0.999),
            weight_decay=self.config['Train']['weight_decay'] if self.config['Train']['weight_decay'] > 0 else 0
        )
        
        # 4. Learning rate schedulers
        self.stereo_scheduler = PolynomialLR(
            self.stereo_optimizer,
            total_iters=self.config['Train']['epoch'],
            power=0.9
        )
        
        self.mono_scheduler = PolynomialLR(
            self.mono_optimizer,
            total_iters=self.config['Train']['epoch'],
            power=0.9
        )
        
        # 5. GradScaler for AMP
        if self.use_amp:
            self.stereo_scaler = GradScaler('cuda')
            self.mono_scaler = GradScaler('cuda')
        
        # 6. Learner
        self.learner = MonodepthLearner(
            depth_net=self.depth_net,
            pose_net=self.pose_net,
            config=self.config,
            device=str(self.device)
        )
        
        # 7. Plot tool
        self.plot_tool = PlotTool(config=self.config)
        
        # 8. Metrics
        self.metrics = {
            'train': {
                'total_loss': [],
                'pixel_loss': [],
                'smooth_loss': []
            },
            'valid': {
                'total_loss': [],
                'pixel_loss': [],
                'smooth_loss': []
            }
        }
        
        # 9. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join(
            'vo',
            self.config['Directory']['log_dir'],
            f"{current_time}_{self.config['Directory']['exp_name']}"
        )
        self.writer = SummaryWriter(tensorboard_path)
        
        # 10. Save path
        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        self.save_path = os.path.join(
            self.config['Directory']['weights'],
            'vo',
            self.config['Directory']['exp_name']
        )
        os.makedirs(self.save_path, exist_ok=True)
    
    def train_stereo_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single training step for stereo with AMP support"""
        self.stereo_optimizer.zero_grad(set_to_none=True)  # 더 효율적인 메모리 정리
        
        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_stereo(
                    sample, training=True
                )
            
            self.stereo_scaler.scale(total_loss).backward()
            self.stereo_scaler.step(self.stereo_optimizer)
            self.stereo_scaler.update()
        else:
            total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_stereo(
                sample, training=True
            )
            total_loss.backward()
            self.stereo_optimizer.step()
        
        # 메모리 정리를 위해 detach
        total_loss = total_loss.detach()
        pixel_loss = pixel_loss.detach()
        smooth_loss = smooth_loss.detach()
        pred_depths = [d.detach() for d in pred_depths]
        
        return total_loss, pixel_loss, smooth_loss, pred_depths
    
    def train_mono_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single training step for mono with AMP support"""
        self.mono_optimizer.zero_grad(set_to_none=True)  # 더 효율적인 메모리 정리
        
        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_mono(
                    sample, training=True
                )
            
            self.mono_scaler.scale(total_loss).backward()
            self.mono_scaler.step(self.mono_optimizer)
            self.mono_scaler.update()
        else:
            total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_mono(
                sample, training=True
            )
            total_loss.backward()
            self.mono_optimizer.step()
        
        # 메모리 정리를 위해 detach
        total_loss = total_loss.detach()
        pixel_loss = pixel_loss.detach()
        smooth_loss = smooth_loss.detach()
        pred_depths = [d.detach() for d in pred_depths]
        
        return total_loss, pixel_loss, smooth_loss, pred_depths
    
    @torch.no_grad()
    def valid_stereo_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single validation step for stereo with AMP support"""
        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_stereo(
                    sample, training=False
                )
        else:
            total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_stereo(
                sample, training=False
            )
        return total_loss, pixel_loss, smooth_loss, pred_depths
    
    @torch.no_grad()
    def valid_mono_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single validation step for mono with AMP support"""
        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_mono(
                    sample, training=False
                )
        else:
            total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_mono(
                sample, training=False
            )
        return total_loss, pixel_loss, smooth_loss, pred_depths
    
    def train(self) -> None:
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Stereo batches: {self.stereo_loader.num_stereo_train}")
        print(f"Mono batches: {self.mono_loader.num_mono_train}")
        
        global_step = 0
        
        for epoch in range(1, self.config['Train']['epoch'] + 1):
            # Training phase
            self.depth_net.train()
            self.pose_net.train()
            
            # Reset metrics
            train_metrics = {
                'total_loss': 0.0,
                'pixel_loss': 0.0,
                'smooth_loss': 0.0,
                'count': 0
            }
            
            # Create iterators
            # stereo_iter = cycle(self.stereo_loader.train_stereo_loader)
            # mono_iter = cycle(self.mono_loader.train_mono_loader)
            stereo_iter = iter(self.stereo_loader.train_stereo_loader)
            mono_iter = iter(self.mono_loader.train_mono_loader)

            
            # Use minimum of both dataset sizes
            min_batches = min(
                self.stereo_loader.num_stereo_train,
                self.mono_loader.num_mono_train
            )
            
            print(f"\nEpoch {epoch}/{self.config['Train']['epoch']}")
            print(f"Learning Rate: {self.mono_optimizer.param_groups[0]['lr']:.6f}")
            
            # Training loop
            train_pbar = tqdm(range(min_batches), desc=f'Training Epoch {epoch}')
            
            for batch_idx in train_pbar:
                # Stereo training    
                stereo_sample = next(stereo_iter)
                t_loss_s, p_loss_s, s_loss_s, pred_depths_s = self.train_stereo_step(stereo_sample)
            
                # Mono training
                mono_sample = next(mono_iter)
                t_loss_m, p_loss_m, s_loss_m, pred_depths_m = self.train_mono_step(mono_sample)
                
                # Average losses - use .item() to prevent graph retention
                avg_total = (t_loss_s.item() + t_loss_m.item()) / 2.0
                avg_pixel = (p_loss_s.item() + p_loss_m.item()) / 2.0
                avg_smooth = (s_loss_s.item() + s_loss_m.item()) / 2.0
                
                train_metrics['total_loss'] += avg_total
                train_metrics['pixel_loss'] += avg_pixel
                train_metrics['smooth_loss'] += avg_smooth
                train_metrics['count'] += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'total': f"{avg_total:.4f}",
                    'pixel': f"{avg_pixel:.4f}",
                    'smooth': f"{avg_smooth:.4f}"
                })
                
                # Log images periodically
                if batch_idx % self.config['Train']['train_plot_interval'] == 0:
                    with torch.no_grad():  # 추가 보호
                        depth_plot = self.plot_tool.plot_images(
                            images=mono_sample['target_image'].detach(),
                            pred_depths=[d.detach() for d in pred_depths_m],
                            denorm_func=self.mono_loader.denormalize_image
                        )
                        self.writer.add_image(
                            'Train/Depth_Result',
                            depth_plot.transpose(2, 0, 1),
                            global_step
                        )
                        del depth_plot  # 명시적 삭제
                
                # 주기적으로 메모리 정리
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 명시적으로 불필요한 변수 삭제
                del stereo_sample, mono_sample
                del t_loss_s, p_loss_s, s_loss_s, pred_depths_s
                del t_loss_m, p_loss_m, s_loss_m, pred_depths_m
                
                global_step += 1
            
            # Average training metrics
            for key in ['total_loss', 'pixel_loss', 'smooth_loss']:
                train_metrics[key] /= train_metrics['count']
                self.metrics['train'][key].append(train_metrics[key])
            
            # Log training metrics
            self.writer.add_scalar('Train/total_loss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Train/pixel_loss', train_metrics['pixel_loss'], epoch)
            self.writer.add_scalar('Train/smooth_loss', train_metrics['smooth_loss'], epoch)
            self.writer.add_scalar('Train/learning_rate', self.mono_optimizer.param_groups[0]['lr'], epoch)
            
            # Log memory usage
            if torch.cuda.is_available():
                self.writer.add_scalar('Memory/allocated_gb', 
                                     torch.cuda.memory_allocated() / 1024**3, epoch)
                self.writer.add_scalar('Memory/reserved_gb', 
                                     torch.cuda.memory_reserved() / 1024**3, epoch)
            
            # Log scale factors if using AMP
            if self.use_amp:
                self.writer.add_scalar('Train/stereo_scale', self.stereo_scaler.get_scale(), epoch)
                self.writer.add_scalar('Train/mono_scale', self.mono_scaler.get_scale(), epoch)
            
            # Validation phase
            if epoch % self.config['Train']['valid_freq'] == 0:
                self.validate(epoch)
            
            # Save checkpoint
            if epoch % self.config['Train']['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Update schedulers
            self.mono_scheduler.step()
            self.stereo_scheduler.step()
            
            # 에폭 끝에서 철저한 메모리 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

    @torch.no_grad()
    def validate(self, epoch: int) -> None:
        """Validation loop"""
        self.depth_net.eval()
        self.pose_net.eval()
        
        valid_metrics = {
            'total_loss': 0.0,
            'pixel_loss': 0.0,
            'smooth_loss': 0.0,
            'count': 0
        }
        
        # Create iterators
        stereo_iter = iter(self.stereo_loader.valid_stereo_loader)
        mono_iter = iter(self.mono_loader.valid_mono_loader)
        
        min_batches = min(
            self.stereo_loader.num_stereo_valid,
            self.mono_loader.num_mono_valid
        )
        
        valid_pbar = tqdm(range(min_batches), desc=f'Validation Epoch {epoch}')
        
        for batch_idx in valid_pbar:
            # Stereo validation
            stereo_sample = next(stereo_iter)
            t_loss_s, p_loss_s, s_loss_s, _ = self.valid_stereo_step(stereo_sample)
            
            # Mono validation
            mono_sample = next(mono_iter)
            t_loss_m, p_loss_m, s_loss_m, pred_depths_m = self.valid_mono_step(mono_sample)
            
            # Average losses
            avg_total = (t_loss_s.item() + t_loss_m.item()) / 2.0
            avg_pixel = (p_loss_s.item() + p_loss_m.item()) / 2.0
            avg_smooth = (s_loss_s.item() + s_loss_m.item()) / 2.0
            
            valid_metrics['total_loss'] += avg_total
            valid_metrics['pixel_loss'] += avg_pixel
            valid_metrics['smooth_loss'] += avg_smooth
            valid_metrics['count'] += 1
            
            # Update progress bar
            valid_pbar.set_postfix({
                'total': f"{avg_total:.4f}",
                'pixel': f"{avg_pixel:.4f}",
                'smooth': f"{avg_smooth:.4f}"
            })
            
            # Log validation images
            if batch_idx % self.config['Train']['valid_plot_interval'] == 0:
                depth_plot = self.plot_tool.plot_images(
                    images=mono_sample['target_image'].detach(),
                    pred_depths=[d.detach() for d in pred_depths_m],
                    denorm_func=self.mono_loader.denormalize_image
                )
                self.writer.add_image(
                    f'Valid/Depth_Result_epoch{epoch}',
                    depth_plot.transpose(2, 0, 1),
                    batch_idx
                )
                del depth_plot
            
            # 명시적 메모리 정리
            del stereo_sample, mono_sample
            del t_loss_s, p_loss_s, s_loss_s
            del t_loss_m, p_loss_m, s_loss_m, pred_depths_m
        
        # Average validation metrics
        for key in ['total_loss', 'pixel_loss', 'smooth_loss']:
            valid_metrics[key] /= valid_metrics['count']
            self.metrics['valid'][key].append(valid_metrics[key])
        
        # Log validation metrics
        self.writer.add_scalar('Valid/total_loss', valid_metrics['total_loss'], epoch)
        self.writer.add_scalar('Valid/pixel_loss', valid_metrics['pixel_loss'], epoch)
        self.writer.add_scalar('Valid/smooth_loss', valid_metrics['smooth_loss'], epoch)
        
        print(f"\nValidation - Total: {valid_metrics['total_loss']:.4f}, "
              f"Pixel: {valid_metrics['pixel_loss']:.4f}, "
              f"Smooth: {valid_metrics['smooth_loss']:.4f}")
        
        # 검증 후 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'depth_net_state_dict': self.depth_net.state_dict(),
            'pose_net_state_dict': self.pose_net.state_dict(),
            'stereo_optimizer_state_dict': self.stereo_optimizer.state_dict(),
            'mono_optimizer_state_dict': self.mono_optimizer.state_dict(),
            'stereo_scheduler_state_dict': self.stereo_scheduler.state_dict(),
            'mono_scheduler_state_dict': self.mono_scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        # Save AMP state if using
        if self.use_amp:
            checkpoint['stereo_scaler_state_dict'] = self.stereo_scaler.state_dict()
            checkpoint['mono_scaler_state_dict'] = self.mono_scaler.state_dict()
        
        checkpoint_path = os.path.join(
            self.save_path,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Also save model weights separately
        torch.save(
            self.depth_net.state_dict(),
            os.path.join(self.save_path, f'depth_net_epoch_{epoch}.pth')
        )
        torch.save(
            self.pose_net.state_dict(),
            os.path.join(self.save_path, f'pose_net_epoch_{epoch}.pth')
        )
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint with AMP support"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.depth_net.load_state_dict(checkpoint['depth_net_state_dict'])
        self.pose_net.load_state_dict(checkpoint['pose_net_state_dict'])
        self.stereo_optimizer.load_state_dict(checkpoint['stereo_optimizer_state_dict'])
        self.mono_optimizer.load_state_dict(checkpoint['mono_optimizer_state_dict'])
        self.stereo_scheduler.load_state_dict(checkpoint['stereo_scheduler_state_dict'])
        self.mono_scheduler.load_state_dict(checkpoint['mono_scheduler_state_dict'])
        
        # Load AMP state if available and using AMP
        if self.use_amp and 'stereo_scaler_state_dict' in checkpoint:
            self.stereo_scaler.load_state_dict(checkpoint['stereo_scaler_state_dict'])
            self.mono_scaler.load_state_dict(checkpoint['mono_scaler_state_dict'])
        
        self.metrics = checkpoint['metrics']
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint['epoch']


def main():
    # Load configuration
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # GPU configuration
    gpu_config = config.get('Experiment', {})
    visible_gpus = gpu_config.get('gpus', [])
    
    if visible_gpus and torch.cuda.is_available():
        # Set visible GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_gpus))
        print(f"Using GPUs: {visible_gpus}")
    else:
        print("Using CPU or all available GPUs")
    
    # Create trainer and start training
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == '__main__':
    main()