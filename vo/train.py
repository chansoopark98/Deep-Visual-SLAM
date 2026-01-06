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

from vo.dataset.vo_loader import VoDataLoader
from vo.learner_new import MonodepthTrainer
from eval_traj import EvalTrajectory
from model.depthnet import DepthNet
# from model.posenet import PoseNet
from model.posenet_single import PoseNet, FlowPoseNet
from utils.plot_utils import PlotTool

from tqdm import tqdm
from datetime import datetime
import yaml
from typing import Dict, Tuple

# torch.autograd.set_detect_anomaly(True)

def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

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

        # load pretrained weights(depth_net)
        """
        torch.save(
            self.depth_net.state_dict(),
            os.path.join(self.save_path, f'depth_net_epoch_{epoch}.pth')
        )
        """
        path = 'weights/vo/depth_net_epoch_30.pth'
        state_dict = torch.load(path, map_location=self.device)
        state_dict = remove_prefix_from_state_dict(state_dict, prefix="_orig_mod.")
        self.depth_net.load_state_dict(state_dict)

        # Pose network
        self.pose_net = PoseNet(
            num_layers=18,
            pretrained=True,
            num_input_images=2,
        ).to(self.device)

        path = 'weights/vo/pose_net_epoch_30.pth'
        state_dict = torch.load(path, map_location=self.device)
        state_dict = remove_prefix_from_state_dict(state_dict, prefix="_orig_mod.")
        self.pose_net.load_state_dict(state_dict)

        if self.config['Train']['use_compile']:
            self.depth_net = torch.compile(self.depth_net, fullgraph=True)
            self.pose_net = torch.compile(self.pose_net, fullgraph=True)

        # 2. Data loader
        self.data_loader = VoDataLoader(config=self.config)
        self.eval_tool = EvalTrajectory(
            depth_model= self.depth_net,
            pose_model=self.pose_net,
            config=self.config,
            device=self.device
        )
        
        # 3. Optimizers
        self.optimizer = optim.Adam(
            list(self.depth_net.parameters()) + list(self.pose_net.parameters()),
            lr=self.config['Train']['init_lr'],
        )
        
        # 4. Learning rate schedulers
        self.scheduler = PolynomialLR(
            self.optimizer,
            total_iters=self.config['Train']['epoch'],
            power=0.9
        )
        
        # 5. GradScaler for AMP
        if self.use_amp:
            self.mono_scaler = GradScaler()

        # 6. Learner
        self.learner = MonodepthTrainer(
            depth_net=self.depth_net,
            pose_net=self.pose_net,
            config=self.config,
            device=self.device
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
    
    def train_mono_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single training step for mono with AMP support"""
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs, losses = self.learner.process_batch(
                    sample
                )
            total_loss = losses['loss']
            self.mono_scaler.scale(total_loss).backward()
            self.mono_scaler.step(self.optimizer)
            self.mono_scaler.update()
        else:
            outputs, losses = self.learner.process_batch(
                sample
            )
            total_loss = losses['loss']
            total_loss.backward()
            self.optimizer.step()
        
        total_loss = total_loss.detach()
        # detach all losses
        for key in losses:
            losses[key] = losses[key].detach().cpu()

        return total_loss, outputs, losses
    
    @torch.no_grad()
    def valid_mono_step(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Single validation step for mono with AMP support"""
        if self.use_amp:
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs, losses = self.learner.process_batch(
                    sample
                )
        else:
            outputs, losses = self.learner.process_batch(
                sample
            )
        total_loss = losses['loss']
        total_loss = total_loss.detach()
        # pred_depths = [outputs[("depth", scale)].detach() for scale in range(self.learner.num_scales)]
        
        return total_loss, outputs

    def train(self) -> None:
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Stereo batches: {self.data_loader.num_train_stereo}")
        print(f"Mono batches: {self.data_loader.num_train_mono}")

        global_step = 0
        
        for epoch in range(1, self.config['Train']['epoch'] + 1):
            # Training phase
            self.depth_net.train()
            self.pose_net.train()
            
            # Reset metrics
            train_metrics = {
                'total_loss': 0.0,
                'count': 0
            }
            
            mono_iter = iter(self.data_loader.train_mono_loader)
            min_batches = self.data_loader.num_train_mono
            
            print(f"\nEpoch {epoch}/{self.config['Train']['epoch']}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Training loop
            train_pbar = tqdm(range(min_batches), desc=f'Training Epoch {epoch}')
            
            for batch_idx in train_pbar:
                # Mono training
                mono_sample = next(mono_iter)
        
                t_loss_m, outputs_m, losses_m = self.train_mono_step(mono_sample)

                avg_total = t_loss_m.item()

                train_metrics['total_loss'] += avg_total
                train_metrics['count'] += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'total': f"{avg_total:.4f}",
                    'loss/0': f"{losses_m['loss/0']:.4f}",
                    'loss/1': f"{losses_m['loss/1']:.4f}",
                    'loss/2': f"{losses_m['loss/2']:.4f}",
                    'loss/3': f"{losses_m['loss/3']:.4f}",
                })
                
                # Log images periodically
                if batch_idx % self.config['Train']['train_plot_interval'] == 0:
                    with torch.no_grad():  # 추가 보호
                        depth_plot = self.plot_tool.plot_result(
                            inputs=mono_sample,
                            outputs=outputs_m,
                            denorm_func=self.data_loader.denormalize_image
                        )
                        self.writer.add_image(
                            'Train/Depth_Result',
                            depth_plot.transpose(2, 0, 1),
                            global_step
                        )
                        del depth_plot  # 명시적 삭제
                
                del t_loss_m, outputs_m
                
                global_step += 1
            
            # Average training metrics
            for key in ['total_loss']:
                train_metrics[key] /= train_metrics['count']
                self.metrics['train'][key].append(train_metrics[key])
            
            # Log training metrics
            self.writer.add_scalar('Train/total_loss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Validation phase
            if epoch % self.config['Train']['valid_freq'] == 0:
                self.validate(epoch)
            
            # Save checkpoint
            if epoch % self.config['Train']['save_freq'] == 0:
                self.save_checkpoint(epoch)
                
            # Update schedulers
            self.scheduler.step()
            
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
            'count': 0
        }

        mono_iter = iter(self.data_loader.valid_mono_loader)
        min_batches = self.data_loader.num_valid_mono
        
        valid_pbar = tqdm(range(min_batches), desc=f'Validation Epoch {epoch}')
        
        for batch_idx in valid_pbar:
            # Mono validation
            mono_sample = next(mono_iter)

            t_loss_m, outputs_m = self.valid_mono_step(mono_sample)
            self.eval_tool.update_state(sample=mono_sample)

            avg_total = t_loss_m.item()
            
            valid_metrics['total_loss'] += avg_total
            valid_metrics['count'] += 1
            
            # Update progress bar
            valid_pbar.set_postfix({
                'total': f"{avg_total:.4f}",
            })
            
            # Log validation images
            if batch_idx % self.config['Train']['valid_plot_interval'] == 0:
                depth_plot = self.plot_tool.plot_result(
                    inputs=mono_sample,
                    outputs=outputs_m,
                    denorm_func=self.data_loader.denormalize_image
                )
                self.writer.add_image(
                    f'Valid/Depth_Result',
                    depth_plot.transpose(2, 0, 1),
                    batch_idx
                )
                del depth_plot

            del t_loss_m, outputs_m

        # Average validation metrics
        for key in ['total_loss']:
            valid_metrics[key] /= valid_metrics['count']
            self.metrics['valid'][key].append(valid_metrics[key])
        
        # Log validation metrics
        self.writer.add_scalar('Valid/total_loss', valid_metrics['total_loss'], epoch)

        # Evaluate trajectory
        eval_plot_image = self.eval_tool.eval_plot()
        self.writer.add_image(
            f'Valid/Trajectory',
            eval_plot_image.transpose(2, 0, 1),
            epoch
        )
        del eval_plot_image

        print(f"\nValidation - Total: {valid_metrics['total_loss']:.4f}")

        # 검증 후 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'depth_net_state_dict': self.depth_net.state_dict(),
            'pose_net_state_dict': self.pose_net.state_dict(),
            'mono_optimizer_state_dict': self.optimizer.state_dict(),
            'mono_scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        # Save AMP state if using
        if self.use_amp:
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