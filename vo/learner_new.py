from __future__ import absolute_import, division, print_function
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from learner_func import (
    disp_to_depth, 
    BackprojectDepth, 
    Project3D, 
    transformation_from_parameters,
    get_smooth_loss,
    SSIM
)

class MonodepthTrainer:
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
        self.batch_size = config['Train']['batch_size']  # 8

        self.image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        self.smoothness_ratio = config['Train']['smoothness_ratio']  # 0.001
        self.auto_mask = config['Train']['auto_mask']  # True
        self.ssim_ratio = config['Train']['ssim_ratio']  # 0.85
        self.min_depth = config['Train']['min_depth']  # 0.1
        self.max_depth = config['Train']['max_depth']  # 10.0
        self.use_multiscale = config['Train']['use_multiscale']  # Default is False
        
        # Initialize SSIM
        self.ssim = SSIM()
        self.ssim.to(self.device)

        # Initialize backprojection and projection layers
        self.backproject_depth: Dict[int, BackprojectDepth] = {}
        self.project_3d: Dict[int, Project3D] = {}

        for scale in range(self.num_scales):
            h = self.image_shape[0] // (2 ** scale)
            w = self.image_shape[1] // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

    @torch.compile
    def _compute_reprojection_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Computes reprojection loss between a batch of predicted and target images
        pred: torch.Tensor (0 ~ 1)
            Predicted images of shape [B, C, H, W]
        target: torch.Tensor (0 ~ 1)
            Target images of shape [B, C, H, W]
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (self.ssim_ratio * ssim_loss) + ((1 - self.ssim_ratio) * l1_loss)

        return reprojection_loss

    def process_batch(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        sample: Dict[str, torch.Tensor]
        {
            ('K', 0):        # torch.FloatTensor [4, 4] (intrinsic, scale 0)
            ('inv_K', 0):    # torch.FloatTensor [4, 4] (inverse intrinsic, scale 0)
            ('K', 1):        # torch.FloatTensor [4, 4] (intrinsic, scale 1)
            ('inv_K', 1):    # torch.FloatTensor [4, 4] (inverse intrinsic, scale 1)
            ('K', 2):        # ...
            ('inv_K', 2):    # ...
            ('K', 3):        # ...
            ('inv_K', 3):    # ...
            ('source_left', 0):   # torch.FloatTensor [3, H0, W0] (scale 0)
            ('target_image', 0):  # torch.FloatTensor [3, H0, W0] (scale 0)
            ('source_right', 0):  # torch.FloatTensor [3, H0, W0] (scale 0)
            ('source_left', 1):   # torch.FloatTensor [3, H1, W1] (scale 1)
            ('target_image', 1):  # torch.FloatTensor [3, H1, W1] (scale 1)
            ('source_right', 1):  # torch.FloatTensor [3, H1, W1] (scale 1)
            ('source_left', 2):   # ...
            ('target_image', 2):  # ...
            ('source_right', 2):  # ...
            ('source_left', 3):   # ...
            ('target_image', 3):  # ...
            ('source_right', 3):  # ...
        }
        """
        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].to(self.device)


        outputs: dict = self.depth_net(sample[('target_image', 0)])

        outputs.update(self._predict_poses(sample))

        self._generate_images_pred(sample, outputs)
        losses = self._compute_losses(sample, outputs)

        return outputs, losses

    def _predict_poses(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        # Left -> Target (invert=True)
        concat_left_tgt = torch.cat([sample[('source_left', 0)], sample[('target_image', 0)]], dim=1)
        axisangle_left, translation_left = self.pose_net(concat_left_tgt)

        # Target -> Right (invert=False)
        concat_tgt_right = torch.cat([sample[('target_image', 0)], sample[('source_right', 0)]], dim=1)
        axisangle_right, translation_right = self.pose_net(concat_tgt_right)  # [B, 1, 3]

        outputs[("axisangle", 0, -1)] = axisangle_left
        outputs[("translation", 0, -1)] = translation_left
        outputs[("axisangle", 0, 1)] = axisangle_right
        outputs[("translation", 0, 1)] = translation_right

        # Invert the matrix if the frame id is negative
        outputs[("cam_T_cam", 0, -1)] = transformation_from_parameters(
            axisangle_left[:, 0], translation_left[:, 0], invert=True)
        outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
            axisangle_right[:, 0], translation_right[:, 0], invert=False)

        return outputs

    @torch.compile
    def _generate_images_pred(self, sample: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> None:
        for scale in range(self.num_scales):
            disp_raw = outputs[("disp", scale)]

            if self.use_multiscale: # v1 multiscale
                disp_up = disp_raw
                source_scale = scale
            else: # v2 full-res supervision
                disp_up = F.interpolate(
                           disp_raw,
                           [self.image_shape[0], self.image_shape[1]],
                           mode="bilinear",
                           align_corners=False)
                source_scale = 0
                
            outputs[("disp_up", scale)] = disp_up

            _, depth = disp_to_depth(disp_up, self.min_depth, self.max_depth)

            outputs[("depth", scale)] = depth

            for frame_id in [-1, 1]:
                """
                frame_id: -1 >> left to target,
                frame_id: 1 >> target to right
                """
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, sample[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, sample[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if frame_id == -1:
                    source_image = sample[('source_left', source_scale)]
                else:
                    source_image = sample[('source_right', source_scale)]
                    
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_image,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True
                    )
    
                outputs[("color_identity", frame_id, scale)] = source_image

    @torch.compile
    def _compute_losses(self, inputs, outputs):
        """
        Compute the reprojection and smoothness losses for a minibatch
        
        sample: Dict
            source_left: [B, 3, H, W] tensor
            target_image: [B, 3, H, W] tensor
            source_right: [B, 3, H, W] tensor
            (K, scale=4): [B, 4, 4] tensor
            (inv_K, scale=4): [B, 4, 4] tensor
        
        outputs: Dict[str, torch.Tensor]
            (disp, scale): [B, 1, H, W] tensor
            (depth, scale): [B, 1, H, W] tensor
            (color, frame_id, scale): [B, 3, H, W] tensor
            (color_identity, frame_id, scale): [B, 3, H, W] tensor
            (sample, frame_id, scale): [B, H, W, 2]
            (axisangle, scale, frame_id): [B, 1, 3] tensor
            (translation, scale, frame_id): [B, 1, 3] tensor
            (cam_T_cam, scale, frame_id): [B, 4, 4] tensor
        """
        losses = {}
        total_loss = 0

        for scale in range(self.num_scales):
            loss = 0
            reprojection_losses = []

            if self.use_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp_up", scale)]
            target = inputs[('target_image', source_scale)]

            for frame_id in [-1, 1]:
                pred = outputs[("color", frame_id, source_scale)]
                reprojection_losses.append(self._compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if self.auto_mask:
                identity_reprojection_losses = []
                for frame_id in [-1, 1]:
                    pred = outputs[("color_identity", frame_id, source_scale)]

                    identity_reprojection_losses.append(
                        self._compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            if self.auto_mask:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_losses

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.auto_mask:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            mean_disp = torch.clamp(mean_disp, min=0.001)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, 
                                          inputs[('target_image', 0)])

            loss += self.smoothness_ratio * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses