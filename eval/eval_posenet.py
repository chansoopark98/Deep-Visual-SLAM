import tensorflow as tf, tf_keras
import numpy as np
import os
import glob
import yaml
import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_loader import RedwoodDataLoader
from model.pose_net import PoseNet

if __name__ == '__main__':
    with open('./vo/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

    dataset = RedwoodDataLoader(config)
    total_samples = dataset.get_dataset_size(dataset.test_dir)

    # load model
    batch_size = config['Train']['batch_size']
    image_shape = (config['Train']['img_h'], config['Train']['img_w'])

    pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
    posenet_input_shape = (batch_size, *image_shape, 6)
    pose_net.build(posenet_input_shape)
    pose_net.load_weights('./assets/weights/vo/0303_mars/pose_net_epoch_30_model.weights.h5')
    pose_net.trainable = False

    for sample in tqdm.tqdm(dataset.generate_datasets(dataset.test_dir), 
                           desc="Processing samples", 
                           unit="sample",
                           total=total_samples):
        target_image = sample['target_image']
        right_image = sample['right_image']
        rotation = sample['rotation']
        translation = sample['translation']

        target_image = tf.convert_to_tensor(target_image, dtype=tf.float32)
        right_image = tf.convert_to_tensor(right_image, dtype=tf.float32)
        concat_image = tf.concat([target_image, right_image], axis=-1)
        concat_image = tf.expand_dims(concat_image, axis=0)

        pred_pose = pose_net(concat_image, training=False)
        pred_rotation = pred_pose[:, :3]
        pred_translation = pred_pose[:, 3:]

        print('pred rotation shape : ', pred_rotation.shape)
        print('pred translation shape : ', pred_translation.shape)

        print('target image shape : ', target_image.shape)
        print('right image shape : ', right_image.shape)
        print('rotation shape : ', rotation.shape)
        print('translation shape : ', translation.shape)
