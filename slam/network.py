import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import tensorflow as tf

def getTransMatrix(trans_vec):
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    batch_size= trans_vec.get_shape().as_list()[0]
    # [B, 1, 1]
    one = tf.ones([batch_size,1,1], dtype=tf.float32)
    zero = tf.zeros([batch_size,1,1], dtype=tf.float32)

    T = tf.concat([
        one, zero, zero, trans_vec[:, :, :1],
        zero, one, zero, trans_vec[:, :, 1:2],
        zero, zero, one, trans_vec[:, :, 2:3],
        zero, zero, zero, one

    ], axis=2)

    T = tf.reshape(T,[batch_size, 4, 4])


    # T = tf.zeros([trans_vec.get_shape().as_list()[0],4,4],dtype=tf.float32)
    # for i in range(4):
    #     T[:,i,i] = 1
    # trans_vec = tf.reshape(trans_vec, [-1,3,1])
    # T[:,:3,3] = trans_vec
    return T

def rotFromAxisAngle(vec):
    """
    Convert axis angle into rotation matrix
    not euler angle but Axis
    :param vec: [B, 1, 3]
    :return:
    """
    angle = tf.norm(vec,ord=2,axis=2,keepdims=True)
    axis = vec / (angle + 1e-7)

    ca = tf.cos(angle)
    sa = tf.sin(angle)

    C = 1 - ca

    x = axis[:,:,:1]
    y = axis[:,:,1:2]
    z = axis[:,:,2:3]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # [B, 1, 1]
    one = tf.ones_like(zxC, dtype=tf.float32)
    zero = tf.zeros_like(zxC, dtype=tf.float32)

    rot_matrix = tf.concat([
        x * xC + ca, xyC - zs, zxC + ys, zero,
        xyC + zs, y * yC + ca, yzC - xs, zero,
        zxC - ys, yzC + xs, z * zC + ca, zero,
        zero, zero, zero, one
    ],axis=2)

    rot_matrix = tf.reshape(rot_matrix, [-1,4,4])


    # rot_matrix = tf.zeros([vec.get_shape().as_list()[0],4,4], dtype= tf.float32)
    #
    # rot_matrix[:, 0, 0] = tf.squeeze()
    # rot_matrix[:, 0, 1] = tf.squeeze()
    # rot_matrix[:, 0, 2] = tf.squeeze()
    # rot_matrix[:, 1, 0] = tf.squeeze()
    # rot_matrix[:, 1, 1] = tf.squeeze()
    # rot_matrix[:, 1, 2] = tf.squeeze()
    # rot_matrix[:, 2, 0] = tf.squeeze(zxC - ys)
    # rot_matrix[:, 2, 1] = tf.squeeze(yzC + xs)
    # rot_matrix[:, 2, 2] = tf.squeeze(z * zC + ca)
    # rot_matrix[:, 3, 3] = 1

    return rot_matrix

def pose_axis_angle_vec2mat(vec, invert=False):
    """
    Convert axis angle and translation into 4x4 matrix
    :param vec: [B,1,6] with former 3 vec is axis angle
    :return:
    """
    # batch_size, _ = vec.get_shape().as_list()
    batch_size = vec.shape[0]

    axisvec = tf.slice(vec, [0, 0], [-1, 3])
    axisvec = tf.reshape(axisvec, [batch_size, 1, 3])

    translation = tf.slice(vec, [0, 3], [-1, 3])
    translation = tf.reshape(translation, [batch_size, 1, 3])


    R = rotFromAxisAngle(axisvec)

    if invert:
        R = tf.transpose(R, [0,2,1])
        translation *= -1
    t = getTransMatrix(translation)

    if invert:
        M = tf.matmul(R,t)
    else:
        M = tf.matmul(t,R)
    return M

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast(max_disp - min_disp, tf.float32) * disp
    depth = tf.cast(1., tf.float32) / scaled_disp
    return depth
    
class Networks:
    def __init__(self, image_shape: tuple):
        # load tf.saved_model
        self.depth_net = tf.saved_model.load('./weights/vo/export/depth_net')
        self.pose_net = tf.saved_model.load('./weights/vo/export/pose_net')

    def depth(self, image):
        tensor = tf.cast(image, tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        tensor /= 255.0
        disp = self.depth_net(tensor, training=False)[0]
        depth = disp_to_depth(disp, 0.1, 10.0)
        depth = tf.squeeze(depth, axis=0)
        depth = tf.squeeze(depth, axis=-1)
        return depth.numpy()
    
    def pose(self, img1, img2, depth, translation_scale=5.6):
        print(depth.shape)

        pair_image = tf.concat([img1, img2], axis=-1)
        pair_image = tf.cast(pair_image, tf.float32)
        pair_image /= 255.0
        pair_image = tf.expand_dims(pair_image, 0)
        pose = self.pose_net(pair_image, training=False)

        # scaling
        # pytorch original translation[:, 0] * (1 / depth).mean() * translation_scale)
        # pose = [axis_angle, translation]
        # split axis angle and translation
        axisAngle = pose[:, :3]
        translation = pose[:, 3:]
        translation *= tf.reduce_mean(1 / depth) * translation_scale
        refined_pose = tf.concat([axisAngle, translation], axis=1)
        transformation = pose_axis_angle_vec2mat(refined_pose, invert=True)
        transformation = tf.squeeze(transformation, axis=0)
        return transformation.numpy()