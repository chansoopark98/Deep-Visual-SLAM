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

from depth.dataset.data_loader import DepthLoader
from depth.depth_learner import DepthLearner
from model.depthnet import DepthNet
from depth.util.plot import PlotTool

from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Tuple

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

        if self.config.get('Train', {}).get('use_compile', False):
            self.depth_net = torch.compile(self.depth_net)

        # 2. Data loaders
        self.depth_loader = DepthLoader(config=self.config)

        # 3. Optimizers
        self.optimizer = optim.Adam(
            self.depth_net.parameters(),
            lr=self.config['Train']['init_lr'],
            betas=(self.config['Train']['beta1'], 0.999),
            weight_decay=self.config['Train']['weight_decay'] if self.config['Train']['weight_decay'] > 0 else 0
        )
        
        # 4. Learning rate schedulers
        self.scheduler = PolynomialLR(
            self.optimizer,
            total_iters=self.config['Train']['epoch'],
            power=0.9
        )

        # 5. GradScaler for AMP
        if self.use_amp:
            self.grad_scaler = GradScaler('cuda')
        
        # 6. Learner
        self.learner = DepthLearner(
            model=self.depth_net,
            config=self.config,
            device=self.device
        )
        
        # 7. Plot tool
        self.plot_tool = PlotTool(config=self.config)

        # 8. Metrics
        self.metrics = {
            'train': {
                'total_loss': [],
                'depth_loss': [],
                'smooth_loss': []
            },
            'valid': {
                'total_loss': [],
                'depth_loss': [],
                'smooth_loss': []
            }
        }
        
        # 9. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join(
            'depth',
            self.config['Directory']['log_dir'],
            f"{current_time}_{self.config['Directory']['exp_name']}"
        )
        self.writer = SummaryWriter(tensorboard_path)
        
        # 10. Save path
        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        self.save_path = os.path.join(
            self.config['Directory']['weights'],
            'depth',
            self.config['Directory']['exp_name']
        )
        os.makedirs(self.save_path, exist_ok=True)
    
    def train_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single training step for mono with AMP support"""
        self.optimizer.zero_grad(set_to_none=True)  # 더 효율적인 메모리 정리

        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                total_loss, depth_loss, smooth_loss, pred_depths = self.learner.forward_step(
                    sample
                )

            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss, depth_loss, smooth_loss, pred_depths = self.learner.forward_step(
                sample
            )
            total_loss.backward()
            self.optimizer.step()

        # 메모리 정리를 위해 detach
        total_loss = total_loss.detach()
        depth_loss = depth_loss.detach()
        smooth_loss = smooth_loss.detach()
        pred_depths = [d.detach() for d in pred_depths]

        return total_loss, depth_loss, smooth_loss, pred_depths
    
    @torch.no_grad()
    def valid_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single validation step for mono with AMP support"""
        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                total_loss, depth_loss, smooth_loss, pred_depths = self.learner.forward_step(
                    sample
                )
        else:
            total_loss, depth_loss, smooth_loss, pred_depths = self.learner.forward_step(
                sample
            )
        return total_loss, depth_loss, smooth_loss, pred_depths

    def train(self) -> None:
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Depth batches: {self.depth_loader.num_depth_train}")

        global_step = 0
        
        for epoch in range(1, self.config['Train']['epoch'] + 1):
            # Training phase
            self.depth_net.train()
            
            # Reset metrics
            train_metrics = {
                'total_loss': 0.0,
                'depth_loss': 0.0,
                'smooth_loss': 0.0,
                'count': 0
            }
            
            # Create iterators
            # depth_dataset = iter(self.depth_loader.train_depth_loader)

            print(f"\nEpoch {epoch}/{self.config['Train']['epoch']}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Training loop
            # train_pbar = tqdm(range(len(depth_dataset)), desc=f'Training Epoch {epoch}')
            train_pbar = tqdm(self.depth_loader.train_depth_loader, desc=f'Training Epoch {epoch}')

            for batch_idx, depth_sample in enumerate(train_pbar):
                # Depth training
                t_loss_d, p_loss_d, s_loss_d, pred_depths_d = self.train_step(depth_sample)
                
                # Average losses - use .item() to prevent graph retention
                avg_total = t_loss_d.item()
                avg_depth = p_loss_d.item()
                avg_smooth = s_loss_d.item()
                
                train_metrics['total_loss'] += avg_total
                train_metrics['depth_loss'] += avg_depth
                train_metrics['smooth_loss'] += avg_smooth
                train_metrics['count'] += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'total': f"{avg_total:.4f}",
                    'depth': f"{avg_depth:.4f}",
                    'smooth': f"{avg_smooth:.4f}"
                })
                
                # Log images periodically
                if batch_idx % self.config['Train']['train_plot_interval'] == 0:
                    with torch.no_grad():  # 추가 보호
                        depth_plot = self.plot_tool.plot_images(
                            images=depth_sample['image'].detach(),
                            pred_depths=[d.detach() for d in pred_depths_d],
                            denorm_func=self.depth_loader.denormalize_image
                        )
                        self.writer.add_image(
                            'Train/Depth_Result',
                            depth_plot.transpose(2, 0, 1),
                            global_step
                        )
                        del depth_plot  # 명시적 삭제
                
                # 주기적으로 메모리 정리
                # if batch_idx % 50 == 0:
                #     torch.cuda.empty_cache()
                #     gc.collect()
                
                # 명시적으로 불필요한 변수 삭제
                # del depth_sample
                # del t_loss_d, p_loss_d, s_loss_d, pred_depths_d

                global_step += 1
            
            # Average training metrics
            for key in ['total_loss', 'depth_loss', 'smooth_loss']:
                train_metrics[key] /= train_metrics['count']
                self.metrics['train'][key].append(train_metrics[key])
            
            # Log training metrics
            self.writer.add_scalar('Train/total_loss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Train/depth_loss', train_metrics['depth_loss'], epoch)
            self.writer.add_scalar('Train/smooth_loss', train_metrics['smooth_loss'], epoch)
            self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Log memory usage
            if torch.cuda.is_available():
                self.writer.add_scalar('Memory/allocated_gb', 
                                     torch.cuda.memory_allocated() / 1024**3, epoch)
                self.writer.add_scalar('Memory/reserved_gb', 
                                     torch.cuda.memory_reserved() / 1024**3, epoch)
            
            # Log scale factors if using AMP
            if self.use_amp:
                self.writer.add_scalar('Train/gradient_scale', self.grad_scaler.get_scale(), epoch)
            
            # Validation phase
            if epoch % self.config['Train']['valid_freq'] == 0:
                self.validate(epoch)
            
            # Save checkpoint
            if epoch % self.config['Train']['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Update schedulers
            self.scheduler.step()
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

    @torch.no_grad()
    def validate(self, epoch: int) -> None:
        """Validation loop"""
        self.depth_net.eval()
        
        valid_metrics = {
            'total_loss': 0.0,
            'depth_loss': 0.0,
            'smooth_loss': 0.0,
            'count': 0
        }
        
        # Create iterators
        # valid_dataset = iter(self.depth_loader.valid_depth_loader)

        # valid_pbar = tqdm(range(len(valid_dataset)), desc=f'Validation Epoch {epoch}')
        valid_pbar = tqdm(self.depth_loader.valid_depth_loader, desc=f'Validation Epoch {epoch}')
        
        for batch_idx, depth_sample in enumerate(valid_pbar):
            # Depth validation
            # depth_sample = next(valid_dataset)
            t_loss_d, p_loss_d, s_loss_d, pred_depths_d = self.valid_step(depth_sample)

            # Average losses
            avg_total = (t_loss_d.item()) / 1.0
            avg_depth = (p_loss_d.item()) / 1.0
            avg_smooth = (s_loss_d.item()) / 1.0

            valid_metrics['total_loss'] += avg_total
            valid_metrics['depth_loss'] += avg_depth
            valid_metrics['smooth_loss'] += avg_smooth
            valid_metrics['count'] += 1
            
            # Update progress bar
            valid_pbar.set_postfix({
                'total': f"{avg_total:.4f}",
                'depth': f"{avg_depth:.4f}",
                'smooth': f"{avg_smooth:.4f}"
            })
            
            # Log validation images
            if batch_idx % self.config['Train']['valid_plot_interval'] == 0:
                depth_plot = self.plot_tool.plot_images(
                    images=depth_sample['image'].detach(),
                    pred_depths=[d.detach() for d in pred_depths_d],
                    denorm_func=self.depth_loader.denormalize_image
                )
                self.writer.add_image(
                    f'Valid/Depth_Result_epoch{epoch}',
                    depth_plot.transpose(2, 0, 1),
                    batch_idx
                )
                del depth_plot
            
            # 명시적 메모리 정리
            del depth_sample
            del t_loss_d, p_loss_d, s_loss_d
            del pred_depths_d

        # Average validation metrics
        for key in ['total_loss', 'depth_loss', 'smooth_loss']:
            valid_metrics[key] /= valid_metrics['count']
            self.metrics['valid'][key].append(valid_metrics[key])
        
        # Log validation metrics
        self.writer.add_scalar('Valid/total_loss', valid_metrics['total_loss'], epoch)
        self.writer.add_scalar('Valid/depth_loss', valid_metrics['depth_loss'], epoch)
        self.writer.add_scalar('Valid/smooth_loss', valid_metrics['smooth_loss'], epoch)
        
        print(f"\nValidation - Total: {valid_metrics['total_loss']:.4f}, "
              f"Depth: {valid_metrics['depth_loss']:.4f}, "
              f"Smooth: {valid_metrics['smooth_loss']:.4f}")
        
        # 검증 후 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()

    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'depth_net_state_dict': self.depth_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        # Save AMP state if using
        if self.use_amp:
            checkpoint['grad_scaler_state_dict'] = self.grad_scaler.state_dict()
        
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
        
        print(f"Checkpoint saved: {checkpoint_path}")

def main():
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # GPU configuration - compile 전에 먼저 설정
    gpu_config = config.get('Experiment', {})
    visible_gpus = gpu_config.get('gpus', [])
    
    if visible_gpus and torch.cuda.is_available():
        # Set visible GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_gpus))
        print(f"Using GPUs: {visible_gpus}")
        
        # CUDA 디바이스 재초기화
        torch.cuda.init()
        torch.cuda.synchronize()
    else:
        print("Using CPU or all available GPUs")
    
    # Create trainer and start training
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == '__main__':
    main()