import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from model.monodepth2 import build_disp_net, build_feature_net

batch_norm_decay = 0.95
batch_norm_epsilon = 1e-5

class ImuNet(keras.Model):
    def __init__(self):
        super(ImuNet, self).__init__()
        self.imu_dropout = 0.2
        self.imu_len = 256

        self.encoder_conv = keras.Sequential([
            # Layer1
            keras.layers.Conv1D(64, kernel_size=3, padding='same',
                                use_bias=False,
                                name='IMU_ENCODER_conv1'),
            keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                            epsilon=batch_norm_epsilon,
                                            name='IMU_ENCODER_bn1',),
            keras.layers.LeakyReLU(0.2,
                                      name='IMU_ENCODER_leaky_relu1'),
            keras.layers.Dropout(self.imu_dropout,
                                    name='IMU_ENCODER_dropout1'),

            # Layer 2
            keras.layers.Conv1D(128, kernel_size=3, padding='same',
                                   use_bias=False,
                                   name='IMU_ENCODER_conv2'),
            keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                            epsilon=batch_norm_epsilon,
                                            name='IMU_ENCODER_bn2'),
            keras.layers.LeakyReLU(0.2, name='IMU_ENCODER_leaky_relu2'),

            keras.layers.Dropout(self.imu_dropout,
                                    name='IMU_ENCODER_dropout2'),

            # Layer 3
            keras.layers.Conv1D(256, kernel_size=3, padding='same',
                                   use_bias=False,              
                                   name='IMU_ENCODER_conv3'),
            keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                            epsilon=batch_norm_epsilon,
                                            name='IMU_ENCODER_bn3'),
            keras.layers.LeakyReLU(0.2, name='IMU_ENCODER_leaky_relu3'),

            keras.layers.Dropout(self.imu_dropout,
                                    name='IMU_ENCODER_dropout3'),
        ])

        self.proj = keras.layers.Dense(self.imu_len, 
                                       name='IMU_ENCODER_dense')
        
    def call(self, x, training=True):
        # x: (batch, imu_seq_len, 6)
        x_shape = tf.shape(x)
        x = self.encoder_conv(x, training=training) # batch, imu_seq_len, 256         
        out = self.proj(tf.reshape(x, (x_shape[0], -1))) # batch, 256
        out = tf.expand_dims(out, 1)
        return out


def build_posenet(image_shape,
                  batch_size):
    # Input (Batch, 1, 512)
    # Output (Batch, 6)
    features_in = keras.layers.Input(shape=image_shape, batch_size=batch_size)


    cells = [keras.layers.LSTMCell(512,
                                   dropout=0.5,
                                   name='Pose_lstm_1'),
            keras.layers.LSTMCell(512,
                                  dropout=0.5,
                                  name='Pose_lstm_2')
                                  ]
    stacked_lstm_cells = keras.layers.StackedRNNCells(cells, name='Pose_stacked_lstm_cells')
        
    lstms = keras.layers.RNN(stacked_lstm_cells, return_sequences=True, return_state=True, name='Pose_rnn')

    features, _, _ = lstms(features_in)


    features = keras.layers.Dense(256)(features)
    features = keras.layers.LeakyReLU(0.2)(features) 
    features = keras.layers.Dense(6)(features)

    model = keras.models.Model(inputs=features_in,
                               outputs=features)
    return model

def build_vio_imu(config) -> keras.models.Model:
    batch_size = config['Train']['batch_size']
    image_size = (config['Train']['img_h'],
                  config['Train']['img_w'])
    
    dispnet = build_disp_net(image_shape=image_size,
                             batch_size=batch_size)
    dispnet.load_weights('./depth/weights/test_monodepth/epoch_45_model.weights.h5')
    
    feature_net = build_feature_net(image_shape=image_size,
                            batch_size=batch_size)
    
    posenet = build_posenet(image_shape=(1, 512), batch_size=batch_size)
    
    imu_net = ImuNet()
    
    source_image = keras.layers.Input(shape=(*image_size, 3), batch_size=batch_size)
    target_image = keras.layers.Input(shape=(*image_size, 3), batch_size=batch_size)
    imu = keras.layers.Input(shape=(11, 6), batch_size=batch_size)

    images = keras.layers.concatenate([source_image, target_image], axis=-1)

    feat_img = feature_net(images)
    feat_imu = imu_net(imu)
    
    concat_feat = keras.layers.concatenate([feat_img, feat_imu], axis=-1)    

    poses = posenet(concat_feat)

    pred_disps = dispnet(target_image)

    model = keras.models.Model(inputs=[source_image, target_image, imu],
                               outputs=[pred_disps, poses])
    
    return model
    
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
        with open('./vio/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model = build_vio_imu(config)
        model.build([[config['Train']['batch_size'], config['Train']['img_h'], config['Train']['img_w'], 6],
                     [config['Train']['batch_size'], config['Train']['img_h'], config['Train']['img_w'], 3],
                     [config['Train']['batch_size'], 11, 6]])
        
        model.summary()

        # source_left = tf.ones((8, 432, 768, 3))
        # source_right = tf.ones((8, 432, 768, 3))
        # target_image = tf.ones((8, 432, 768, 3))
        

        # pred_disp, pose = model.call([source_left, source_right, target_image])

        # for disp in pred_disp:
        #     print(disp.shape) # 1/1, 1/2, 1/4, 1/8
        # print(pose.shape)
