import numpy as np
from sympy import im
import yaml
import sys
import os
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.depthnet import DepthNet
# from model.posenet import PoseNet
from model.posenet_single import PoseNet
from utils.visualization import Visualizer
from vo.dataset.vo_loader import VoDataLoader
from eval_traj import EvalTrajectory
from utils.utils import remove_prefix_from_state_dict
from learner_func import (
    disp_to_depth, 
    BackprojectDepth, 
    Project3D, 
    transformation_from_parameters,
    get_smooth_loss,
    SSIM
)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pose_net = PoseNet(
            num_layers=18,
            pretrained=True,
            num_input_images=2)
    pose_net.to(device)
    pose_state_dict = torch.load('./weights/vo/pose_net_epoch_30.pth')
    pose_state_dict = remove_prefix_from_state_dict(pose_state_dict, prefix="_orig_mod.")
    pose_net.load_state_dict(pose_state_dict)
    pose_net.eval()

    depth_net = DepthNet(
            num_layers=18,
            pretrained=True,
            num_input_images=1,
            scales=range(4),
            num_output_channels=1,
            use_skips=True)
    depth_net.to(device)
    depth_state_dict = torch.load('./weights/vo/depth_net_epoch_30.pth')
    depth_state_dict = remove_prefix_from_state_dict(depth_state_dict, prefix="_orig_mod.")
    depth_net.load_state_dict(depth_state_dict)
    depth_net.eval()

    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    with torch.no_grad():
        config['Train']['batch_size'] = 1
        num_source = config['Train']['num_source']
        image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        batch_size = config['Train']['batch_size']

        eval_tool = EvalTrajectory(depth_model=depth_net, pose_model=pose_net, config=config)

        data_loader = VoDataLoader(config=config)
        mono_iter = iter(data_loader.test_mono_loader)
        min_batches = data_loader.num_test_mono
        predict_pbar = tqdm(range(min_batches), desc=f'Predicting Trajectory', unit='batch')

        visualizer = Visualizer(window_size=(1920, 1080),
                                draw_plane=False, is_record=True, video_fps=30, video_name="0414_traj.mp4")

        for batch_idx in predict_pbar:
            data_sample = next(mono_iter)
            for key in data_sample:
                if isinstance(data_sample[key], torch.Tensor):
                    data_sample[key] = data_sample[key].to(device)

            intrinsic = data_sample[('K', 0)][0].detach().cpu().numpy()  # [3, 3]
            target_image = data_sample[('target_image', 0)]
            source_right = data_sample[('source_right', 0)]
            concat_tgt_right = torch.cat([target_image, source_right], dim=1)
            axisangle_right, translation_right = pose_net(concat_tgt_right)  # [B, 1, 3]

            pred_transform = transformation_from_parameters(
                axisangle_right[:, 0], translation_right[:, 0], invert=False)
            pred_transform = pred_transform.detach().cpu().numpy()[0]

            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

            outputs = depth_net(target_image)
            disp = outputs[("disp", 0)]
            _, depth_map = disp_to_depth(disp=disp,
                                      min_depth=config['Train']['min_depth'],
                                      max_depth=config['Train']['max_depth']) # [B, 1, H, W]
            depth_map = depth_map.detach().cpu().numpy()[0, 0]  # [H, W]
            

            updated_pose = visualizer.world_pose @ pred_transform
            visualizer.world_pose = updated_pose

            # Draw point cloud
            denornalized_target = data_loader.denormalize_image(target_image)
            denornalized_target = denornalized_target[0].detach().cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            visualizer.draw_pointcloud(denornalized_target, depth_map, intrinsic, visualizer.world_pose)

            # Draw trajectory
            visualizer.draw_trajectory(visualizer.world_pose, color="red", line_width=2)
            
            # draw camera model
            visualizer.draw_camera_model(visualizer.world_pose, scale=0.5, name_prefix="camera")

            # animation
            # visualizer.set_camera_poisition(visualizer.world_pose)
            visualizer.render()
        visualizer.close()