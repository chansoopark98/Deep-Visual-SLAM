import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from model.monodepth2 import build_disp_net, build_posenet

def build_vio(config) -> keras.models.Model:
    batch_size = config['Train']['batch_size']
    image_size = (config['Train']['img_h'],
                  config['Train']['img_w'])
    
    dispnet = build_disp_net(image_shape=image_size,
                             batch_size=batch_size)
    dispnet.load_weights('./depth/weights/test_monodepth/epoch_45_model.weights.h5')

    posenet = build_posenet(image_shape=image_size,
                            batch_size=batch_size)
    
    source_image = keras.layers.Input(shape=(*image_size, 6), batch_size=batch_size)
    target_image = keras.layers.Input(shape=(*image_size, 3), batch_size=batch_size)

    src_tgt1 = keras.layers.concatenate([source_image[:, :, :, :3], target_image], axis=-1)
    src_tgt2 = keras.layers.concatenate([target_image, source_image[:, :, :, 3:]], axis=-1)

    pose_ctp = posenet(src_tgt1)
    pose_ctn = posenet(src_tgt2)

    poses = keras.layers.concatenate([pose_ctp, pose_ctn], axis=1)
    # poses = keras.layers.Concatenate(axis=1)([pose_ctp, pose_ctn])

    pred_disps = dispnet(target_image)

    model = keras.models.Model(inputs=[source_image, target_image],
                               outputs=[pred_disps, poses])
    
    return model


class TSPVIO(keras.Model):
    def __init__(self, config, *args, **kwargs):
        # super(TSPVIO, self).__init__()
        super().__init__(*args, **kwargs)
        self.config = config
        self.seq_len = 10
        self.batch_size = self.config['Train']['batch_size']
        self.image_size = (self.config['Train']['img_h'],
                           self.config['Train']['img_w'])
    
        self.dispnet = build_disp_net(image_shape=self.image_size,
                                      batch_size=self.batch_size)
        self.posenet = build_posenet(image_shape=self.image_size,
                                                  batch_size=self.batch_size)
        
        # self.dispnet.load_weights('./vio/model/epoch_50_model.h5')

    
    def call(self, inputs, training):
        source_image, target_image = inputs
        print()
        # Source to Target concatenation for pose prediction
        # src_tgt1 = tf.concat([source_image[:, :, :, :3], target_image], axis=-1)
        src_tgt1 = keras.layers.concatenate([source_image[:, :, :, :3], target_image], axis=-1)
        # src_tgt2 = tf.concat([target_image, source_image[:, :, :, 3:]], axis=-1)
        src_tgt2 = keras.layers.concatenate([target_image, source_image[:, :, :, 3:]], axis=-1)

        pose_ctp = self.posenet(src_tgt1, training=training)
        pose_ctn = self.posenet(src_tgt2, training=training)

        # Concatenate poses as in the original code
        # poses = tf.concat([pose_ctp, pose_ctn], axis=1)
        poses = keras.layers.concatenate([pose_ctp, pose_ctn], axis=1)

        pred_disps = self.dispnet(target_image, training=training)

        return pred_disps, poses

if __name__ == '__main__':
    import yaml
    with tf.device('/device:GPU:1'):
        with open('../config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model = TSPVIO(config)
        model.built=True
        model.build_model(8)
        model.summary()
        source_left = tf.ones((8, 432, 768, 3))
        source_right = tf.ones((8, 432, 768, 3))
        target_image = tf.ones((8, 432, 768, 3))
        

        pred_disp, pose = model.call([source_left, source_right, target_image])

        for disp in pred_disp:
            print(disp.shape) # 1/1, 1/2, 1/4, 1/8
        print(pose.shape)
