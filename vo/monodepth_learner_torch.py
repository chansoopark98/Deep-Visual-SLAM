import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from model.layers import (
    disp_to_depth, 
    BackprojectDepth, 
    Project3D, 
    transformation_from_parameters,
    get_smooth_loss,
    SSIM
)


class MonodepthLearner:
    """PyTorch implementation of monodepth2 learner"""
    
    def __init__(self, 
                 depth_net: nn.Module,
                 pose_net: nn.Module,
                 config: dict,
                 device: str = 'cuda'):
        
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.depth_net.to(self.device)
        self.pose_net.to(self.device)
        
        # Hyperparameters
        self.num_scales = 4
        self.num_source = config['Train']['num_source']  # 2
        
        self.image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        self.smoothness_ratio = config['Train']['smoothness_ratio']  # 0.001
        self.auto_mask = config['Train']['auto_mask']  # True
        self.ssim_ratio = config['Train']['ssim_ratio']  # 0.85
        self.min_depth = config['Train']['min_depth']  # 0.1
        self.max_depth = config['Train']['max_depth']  # 10.0
        
        # Initialize SSIM
        self.ssim = SSIM()
        self.ssim.to(self.device)
        
        # Initialize backprojection and projection layers
        self.backproject_depth = {}
        self.project_3d = {}
        
        for scale in range(self.num_scales):
            h = self.image_shape[0] // (2 ** scale)
            w = self.image_shape[1] // (2 ** scale)
            
            self.backproject_depth[scale] = BackprojectDepth(
                config['Train']['batch_size'], h, w
            ).to(self.device)
            
            self.project_3d[scale] = Project3D(
                config['Train']['batch_size'], h, w
            ).to(self.device)
    
    def compute_reprojection_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = self.ssim_ratio * ssim_loss + (1 - self.ssim_ratio) * l1_loss
        
        return reprojection_loss
    
    def rescale_intrinsics(self, intrinsics: torch.Tensor, 
                          original_height: int, original_width: int,
                          target_height: int, target_width: int) -> torch.Tensor:
        """Rescale camera intrinsics for different image sizes"""
        h_scale = target_height / original_height
        w_scale = target_width / original_width
        
        batch_size = intrinsics.shape[0]
        
        # Create scaling matrix
        scale_matrix = torch.tensor([
            [w_scale, 0, 0],
            [0, h_scale, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=intrinsics.device)
        
        scale_matrix = scale_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply scaling
        scaled_intrinsics = torch.matmul(scale_matrix, intrinsics)
        
        return scaled_intrinsics
    
    def forward_mono(self, sample: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, ...]:
        """Forward pass for monocular training"""
        
        # Move data to device
        for key in sample:
            sample[key] = sample[key].to(self.device)
        
        # Extract inputs
        left_image = sample['source_left']  # [B, 3, H, W]
        right_image = sample['source_right']  # [B, 3, H, W]
        tgt_image = sample['target_image']  # [B, 3, H, W]
        intrinsic = sample['intrinsic']  # [B, 3, 3]
        
        outputs = {}
        pixel_losses = 0.
        smooth_losses = 0.
        
        H, W = self.image_shape
        
        # Depth prediction
        if training:
            self.depth_net.train()
        else:
            self.depth_net.eval()
            
        depth_outputs = self.depth_net(tgt_image)
        
        # Store disparity outputs
        for scale in range(self.num_scales):
            outputs[("disp", scale)] = depth_outputs[("disp", scale)]
        
        # Pose prediction
        if training:
            self.pose_net.train()
        else:
            self.pose_net.eval()
        
        # Concatenate images for pose network
        # Left -> Target
        concat_left_tgt = torch.cat([left_image, tgt_image], dim=1)
        axisangle_left, translation_left = self.pose_net(concat_left_tgt)
        
        # Target -> Right  
        concat_tgt_right = torch.cat([tgt_image, right_image], dim=1)
        axisangle_right, translation_right = self.pose_net(concat_tgt_right)
        
        # Create transformation matrices (4x4)
        # Left to Target (invert=True because we want Target to Left)
        T_left_to_tgt_4x4 = transformation_from_parameters(
            axisangle_left[:, 0], translation_left[:, 0], invert=True
        )
        
        # Target to Right (invert=False)
        T_tgt_to_right_4x4 = transformation_from_parameters(
            axisangle_right[:, 0], translation_right[:, 0], invert=False
        )
        
        # Extract 3x4 matrices for projection
        T_left_to_tgt = T_left_to_tgt_4x4[:, :3, :]
        T_tgt_to_right = T_tgt_to_right_4x4[:, :3, :]
        
        # Compute losses for each scale
        for scale in range(self.num_scales):
            h_s = H // (2**scale)
            w_s = W // (2**scale)
            
            # Get disparity and depth
            disp = outputs[("disp", scale)]
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            
            # Target image at current scale
            if scale != 0:
                target = F.interpolate(
                    tgt_image, 
                    [h_s, w_s],
                    mode="bilinear",
                    align_corners=False
                )
                K_s = self.rescale_intrinsics(intrinsic, H, W, h_s, w_s)
            else:
                target = tgt_image
                K_s = intrinsic
            
            inv_K_s = torch.inverse(K_s)
            
            # Compute reprojection for each source
            reprojection_losses = []
            
            # Left image reprojection
            if scale != 0:
                left_s = F.interpolate(left_image, [h_s, w_s], mode="bilinear", align_corners=False)
            else:
                left_s = left_image
                
            # Backproject depth and project to left view
            cam_points = self.backproject_depth[scale](depth, inv_K_s)
            pix_coords = self.project_3d[scale](cam_points, K_s, T_left_to_tgt)
            
            left_warped = F.grid_sample(
                left_s, pix_coords,
                padding_mode="border",
                align_corners=False
            )
            
            left_reproj_loss = self.compute_reprojection_loss(left_warped, target)
            reprojection_losses.append(left_reproj_loss)
            
            # Right image reprojection
            if scale != 0:
                right_s = F.interpolate(right_image, [h_s, w_s], mode="bilinear", align_corners=False)
            else:
                right_s = right_image
                
            # Backproject depth and project to right view
            cam_points = self.backproject_depth[scale](depth, inv_K_s)
            pix_coords = self.project_3d[scale](cam_points, K_s, T_tgt_to_right)
            
            right_warped = F.grid_sample(
                right_s, pix_coords,
                padding_mode="border",
                align_corners=False
            )
            
            right_reproj_loss = self.compute_reprojection_loss(right_warped, target)
            reprojection_losses.append(right_reproj_loss)
            
            reprojection_losses = torch.cat(reprojection_losses, 1)
            
            # Auto-masking
            if self.auto_mask:
                identity_reprojection_losses = []
                
                # Left identity
                identity_left_loss = self.compute_reprojection_loss(left_s, target)
                identity_reprojection_losses.append(identity_left_loss)
                
                # Right identity
                identity_right_loss = self.compute_reprojection_loss(right_s, target)
                identity_reprojection_losses.append(identity_right_loss)
                
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                
                # Add random noise to break ties
                identity_reprojection_losses += torch.randn(
                    identity_reprojection_losses.shape,
                    device=self.device
                ) * 1e-5
                
                # Combine losses
                combined = torch.cat([identity_reprojection_losses, reprojection_losses], dim=1)
            else:
                combined = reprojection_losses
            
            # Take minimum loss
            to_optimise, idxs = torch.min(combined, dim=1)
            pixel_loss = to_optimise.mean()
            pixel_losses += pixel_loss
            
            # Smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            
            smooth_loss = get_smooth_loss(norm_disp, target)
            smooth_loss = smooth_loss / (2 ** scale)
            smooth_losses += smooth_loss * self.smoothness_ratio
        
        # Average losses
        pixel_losses = pixel_losses / self.num_scales
        smooth_losses = smooth_losses / self.num_scales
        total_loss = pixel_losses + smooth_losses
        
        # Extract depth predictions
        pred_depths = []
        for scale in range(self.num_scales):
            _, depth = disp_to_depth(outputs[("disp", scale)], self.min_depth, self.max_depth)
            pred_depths.append(depth)
        
        return total_loss, pixel_losses, smooth_losses, pred_depths
    
    def forward_stereo(self, sample: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, ...]:
        """Forward pass for stereo training"""
        
        # Move data to device
        for key in sample:
            sample[key] = sample[key].to(self.device)
        
        # Extract inputs
        src_image = sample['source_image']  # [B, 3, H, W]
        tgt_image = sample['target_image']  # [B, 3, H, W]
        intrinsic = sample['intrinsic']  # [B, 3, 3]
        stereo_pose = sample['pose']  # [B, 6]
        
        outputs = {}
        pixel_losses = 0.
        smooth_losses = 0.
        
        H, W = self.image_shape
        
        # Depth prediction
        if training:
            self.depth_net.train()
        else:
            self.depth_net.eval()
            
        depth_outputs = self.depth_net(tgt_image)
        
        # Process each scale
        for scale in range(self.num_scales):
            h_s = H // (2**scale)
            w_s = W // (2**scale)
            
            # Get disparity and depth
            disp = depth_outputs[("disp", scale)]
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            
            # Resize images if needed
            if scale != 0:
                src_s = F.interpolate(src_image, [h_s, w_s], mode="bilinear", align_corners=False)
                tgt_s = F.interpolate(tgt_image, [h_s, w_s], mode="bilinear", align_corners=False)
                K_s = self.rescale_intrinsics(intrinsic, H, W, h_s, w_s)
            else:
                src_s = src_image
                tgt_s = tgt_image
                K_s = intrinsic
            
            # Convert pose to transformation matrix (4x4)
            T_4x4 = transformation_from_parameters(
                stereo_pose[:, :3].unsqueeze(1),  # axis-angle
                stereo_pose[:, 3:].unsqueeze(1),   # translation
                invert=False
            )
            
            # Extract 3x4 matrix for projection
            T = T_4x4[:, :3, :]
            
            # Warp source to target
            inv_K_s = torch.inverse(K_s)
            cam_points = self.backproject_depth[scale](depth, inv_K_s)
            pix_coords = self.project_3d[scale](cam_points, K_s, T)
            
            warped = F.grid_sample(
                src_s, pix_coords,
                padding_mode="border",
                align_corners=False
            )
            
            # Compute losses
            reproj_loss = self.compute_reprojection_loss(warped, tgt_s)
            
            if self.auto_mask:
                identity_loss = self.compute_reprojection_loss(src_s, tgt_s)
                reproj_loss = torch.min(reproj_loss, identity_loss)
            
            pixel_loss = reproj_loss.mean()
            pixel_losses += pixel_loss
            
            # Smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, tgt_s)
            smooth_loss = smooth_loss / (2 ** scale)
            smooth_losses += smooth_loss * self.smoothness_ratio
        
        # Average losses
        pixel_losses = pixel_losses / self.num_scales
        smooth_losses = smooth_losses / self.num_scales
        total_loss = pixel_losses + smooth_losses
        
        # Extract depth predictions
        pred_depths = []
        for scale in range(self.num_scales):
            _, depth = disp_to_depth(depth_outputs[("disp", scale)], self.min_depth, self.max_depth)
            pred_depths.append(depth)
        
        return total_loss, pixel_losses, smooth_losses, pred_depths