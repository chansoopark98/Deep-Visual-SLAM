import tensorflow as tf
import pyvista as pv
import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.monodepth2 import DispNet, PoseNet
from utils.visualization import Visualizer
from eval import EvalTrajectory, pose_axis_angle_vec2mat
from kalman_filter import ESEKF, ImuParameters


def imu_pose_to_transform(pose):
    """
    Convert IMU pose [p_x, p_y, p_z, q_w, q_x, q_y, q_z] to a 4x4 transformation matrix.
    :param pose: Array of shape [7], where the first 3 elements are position and the next 4 are quaternion.
    :return: Array of shape [4, 4], 4x4 transformation matrix.
    """
    # Extract position and quaternion
    position = pose[:3]  # [3]
    q_w, q_x, q_y, q_z = pose[3:]  # Quaternion components

    # Compute the rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)]
    ])

    # Create the 4x4 transformation matrix
    transform = np.eye(4)  # Initialize as identity matrix
    transform[:3, :3] = R  # Set rotation
    transform[:3, 3] = position  # Set translation

    return transform

if __name__ == '__main__':
    from dataset.data_loader import DataLoader
    from tqdm import tqdm

    with open('./vio/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    with tf.device('/GPU:0'):
        config['Train']['batch_size'] = 1
        num_source = config['Train']['num_source']
        image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        batch_size = config['Train']['batch_size']

        depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='disp_resnet')
        depth_net(tf.random.normal((1, *image_shape, 3)))
        depth_net.load_weights('./weights/vo_dataAug_ep101_lr1e-4_norm1.0_decay1e-4/depth_net_epoch_20_model.h5')

        pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
        posenet_input_shape = [(batch_size, *image_shape, 6)]
        pose_net.build(posenet_input_shape)
        pose_net.load_weights('./weights/vo_dataAug_ep101_lr1e-4_norm1.0_decay1e-4/pose_net_epoch_20_model.h5')

        eval_tool = EvalTrajectory(depth_model=depth_net, pose_model=pose_net, config=config)

        data_loader = DataLoader(config=config)
        test_tqdm = tqdm(data_loader.test_dataset, total=data_loader.num_test_samples)

        visualizer = Visualizer(draw_plane=True, is_record=True, video_fps=24, video_name="visualization.mp4")

        imu_freq = 100

        imu_parameters = ImuParameters()
        imu_parameters.frequency = imu_freq
        imu_parameters.sigma_a_n = 0.019
        imu_parameters.sigma_w_n = 0.015
        imu_parameters.sigma_a_b = 0.0001
        imu_parameters.sigma_w_b =  2.0e-5 

        init_nominal_state = np.zeros((19,))
        init_nominal_state[16:19] = np.array([-9.81, 0, 0])
        imu_estimator = ESEKF(init_nominal_state, imu_parameters)

        for idx, (ref_images, target_image, imus, intrinsics) in enumerate(test_tqdm):
            left_images = ref_images[:, :num_source] # [B, num_source, H, W, 3]
            left_imus = imus[:, :num_source] # [B, num_source, seq_len, 6]
            left_image = left_images[:, 0] # [B, H, W, 3]
            left_imu = left_imus[:, 0] # [B,  seq_len, 6]

            intrinsic = intrinsics[0]
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

            disp_raw = depth_net(target_image, training=False)

            depth_map = eval_tool.disp_to_depth(disp=disp_raw[0],
                                            min_depth=config['Train']['min_depth'],
                                            max_depth=config['Train']['max_depth']) # [B, H, W]
            depth_map = depth_map[0].numpy() # [H, W]

            input_images = tf.concat([left_image, target_image], axis=3)
            pose = pose_net(input_images, training=False) # [B, 6]
            pred_transform = pose_axis_angle_vec2mat(pose, invert=True)[0] # [4, 4] transformation matrix

            updated_pose = visualizer.world_pose @ pred_transform.numpy()
            visualizer.world_pose = updated_pose

            # Draw point cloud
            denornalized_target = data_loader.denormalize_image(target_image[0]).numpy()  # [H, W, 3]
            visualizer.draw_pointcloud(denornalized_target, depth_map, intrinsic, visualizer.world_pose)
 
            # Draw trajectory
            visualizer.draw_trajectory(visualizer.world_pose, color="red", line_width=2)
            
            # draw camera model
            visualizer.draw_camera_model(visualizer.world_pose, scale=0.5, name_prefix="camera")

            # animation
            visualizer.set_camera_poisition(visualizer.world_pose)
            visualizer.render()
        visualizer.close()