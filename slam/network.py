import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import tensorflow as tf, tf_keras
from model.depth_net import DispNet
from model.pose_net import PoseNet
import matplotlib.pyplot as plt
from vo.utils.d3vo_projection_utils import pose_axis_angle_vec2mat
import cv2

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast(max_disp - min_disp, tf.float32) * disp
    depth = tf.cast(1., tf.float32) / scaled_disp
    return depth
    
class Networks:
    def __init__(self,
                 depth_weight_path: str = None,
                 pose_weight_path: str = None,
                 image_shape: tuple = (480, 640)):
        self.image_shape = image_shape
        self.batch_size = 1

        self.depth_net = DispNet(image_shape=self.image_shape, batch_size=self.batch_size, prefix='disp_resnet')
        dispnet_input_shape = (self.batch_size, *self.image_shape, 3)
        self.depth_net.build(dispnet_input_shape)
        _ = self.depth_net(tf.random.normal(dispnet_input_shape))
        self.depth_net.load_weights(depth_weight_path)
        self.depth_net.trainable = False

        self.pose_net = PoseNet(image_shape=image_shape, batch_size=self.batch_size, prefix='mono_posenet')
        posenet_input_shape = (self.batch_size, *self.image_shape, 6)
        self.pose_net.build(posenet_input_shape)
        _ = self.pose_net(tf.random.normal(posenet_input_shape))
        self.pose_net.load_weights(pose_weight_path)
        self.pose_net.trainable = False
    
    def _preprocess(self, image, is_bgr=True):
        # bgr to rgb
        if is_bgr:
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        tensor = tf.cast(image, tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        tensor /= 255.0
        return tensor

    def depth(self, image):
        tensor = self._preprocess(image, is_bgr=True)
        disps = self.depth_net(tensor, training=False)
        depth = disp_to_depth(disps[0], 0.1, 10.0)
        depth = tf.squeeze(depth, axis=0)
        depth = tf.squeeze(depth, axis=-1)
        depth = tf.clip_by_value(depth, 0.1, 10.0)
 
        return depth.numpy()
    
    def pose(self, img1, img2, depth, translation_scale=5.6):
        # bgr to rgb
        left = self._preprocess(img1, is_bgr=True)
        right = self._preprocess(img2, is_bgr=True)
        depth = tf.expand_dims(depth, axis=0)
        
        pair_image = tf.concat([left, right], axis=-1)
        pair_image = tf.cast(pair_image, tf.float32)
        
        pose= self.pose_net(pair_image, training=False)
    
        transformation = pose_axis_angle_vec2mat(pose,  tf.expand_dims(depth, axis=-1), invert=True)
        transformation = tf.squeeze(transformation, axis=0)
        
        return transformation.numpy()