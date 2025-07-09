import tensorflow as tf
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.depth_net import DispNet
from model.pose_net import PoseNet
from utils.visualization import Visualizer
from eval import EvalTrajectory
from utils.projection_utils import pose_axis_angle_vec2mat

if __name__ == '__main__':
    from vo.dataset.mono_loader import MonoLoader
    from tqdm import tqdm

    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    with tf.device('/GPU:0'):
        config['Train']['batch_size'] = 1
        num_source = config['Train']['num_source']
        image_shape = (config['Train']['img_h'], config['Train']['img_w'])
        batch_size = config['Train']['batch_size']

        depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='disp_resnet')
        dispnet_input_shape = (config['Train']['batch_size'],
                               config['Train']['img_h'], config['Train']['img_w'], 3)
        # depth_net(tf.random.normal((1, *image_shape, 3)))
        depth_net.build(dispnet_input_shape)
        _ = depth_net(tf.random.normal(dispnet_input_shape))
        exp_name = 'mode=axisAngle_res=(480, 640)_ep=31_bs=8_initLR=0.0001_endLR=1e-05_prefix=Monodepth2-resnet18-irs-withoutCustom_withoutScaling'
        depth_net.load_weights(f'./weights/vo/{exp_name}/depth_net_epoch_30_model.weights.h5')

        pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
        posenet_input_shape = [(batch_size, *image_shape, 6)]
        pose_net.build(posenet_input_shape)
        pose_net.load_weights(f'./weights/vo/{exp_name}/pose_net_epoch_30_model.weights.h5')

        eval_tool = EvalTrajectory(depth_model=depth_net, pose_model=pose_net, config=config)

        data_loader = MonoLoader(config=config)
        test_tqdm = tqdm(data_loader.test_mono_datasets, total=data_loader.num_mono_test)

        visualizer = Visualizer(draw_plane=True, is_record=True, video_fps=24, video_name="visualization.mp4")

        for idx, (sample) in enumerate(test_tqdm):
            left_image = sample['source_left']  # [B, H, W, 3]
            right_image = sample['source_right']  # [B, H, W, 3]
            tgt_image = sample['target_image']  # [B, H, W, 3]
            intrinsic = sample['intrinsic']  # [B, 3, 3] - target camera intrinsic


            intrinsic = intrinsic[0]
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

            disp_raw = depth_net(tgt_image, training=False)

            depth_map = eval_tool.disp_to_depth(disp=disp_raw[0],
                                            min_depth=config['Train']['min_depth'],
                                            max_depth=config['Train']['max_depth']) # [B, H, W]
            depth_map = depth_map[0].numpy() # [H, W]

            input_images = tf.concat([tgt_image, right_image], axis=3)
            pose = pose_net(input_images, training=False) # [B, 6]

            pred_transform = pose_axis_angle_vec2mat(pose, depth=depth_map, invert=False, is_stereo=False)[0] # [4, 4] transformation matrix
            

            updated_pose = visualizer.world_pose @ pred_transform.numpy()
            visualizer.world_pose = updated_pose

            # Draw point cloud
            denornalized_target = data_loader.denormalize_image(tgt_image[0]).numpy()  # [H, W, 3]
            visualizer.draw_pointcloud(denornalized_target, depth_map, intrinsic, visualizer.world_pose)
 
            # Draw trajectory
            visualizer.draw_trajectory(visualizer.world_pose, color="red", line_width=2)
            
            # draw camera model
            visualizer.draw_camera_model(visualizer.world_pose, scale=0.5, name_prefix="camera")

            # animation
            # visualizer.set_camera_poisition(visualizer.world_pose)
            visualizer.render()
        visualizer.close()