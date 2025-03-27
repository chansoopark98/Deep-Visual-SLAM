import tensorflow as tf
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.depth_net import DispNet
from model.pose_net import PoseNet, PoseNetExtra

if __name__ == '__main__':


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
        exp_name = 'mode=axisAngle_res=(480, 640)_ep=31_bs=16_initLR=0.0001_endLR=1e-05'
        depth_net.load_weights(f'./weights/vo/{exp_name}/depth_net_epoch_24_model.weights.h5')

        pose_net = PoseNetExtra(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
        posenet_input_shape = (batch_size, *image_shape, 6)
        pose_net.build(posenet_input_shape)
        _ = pose_net(tf.random.normal(posenet_input_shape))
        pose_net.load_weights(f'./weights/vo/{exp_name}/pose_net_epoch_24_model.weights.h5')

        # export model
        export_dir = './weights/vo/export'
        os.makedirs(export_dir, exist_ok=True)
        
        tf.saved_model.save(depth_net, os.path.join(export_dir, 'depth_net'))
        tf.saved_model.save(pose_net, os.path.join(export_dir, 'pose_net'))
        print('Model exported to', export_dir) 