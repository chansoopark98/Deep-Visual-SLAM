import tensorflow as tf, tf_keras
try:
    from .flownet import CustomFlow
except:
    from flownet import CustomFlow


def std_conv(filter_size, out_channel, stride, use_bias=True, pad='same', name='conv'):
    conv_layer = tf_keras.layers.Conv2D(out_channel,
                                        (filter_size, filter_size),
                                         strides=(stride, stride), 
                                         use_bias=use_bias,
                                         padding=pad,
                                         name=name+'_'+'conv')
    return conv_layer

class PoseNetExtra(tf_keras.Model):
    """
    - 입력: (B, H, W, 6)  (ex: 소스+타겟 concat)
    - 내부: ResNet-18 인코더 -> Conv/ReduceMean -> Reshape -> scale
    - 출력: (B, 1, 6)  (Monodepth2식 pose)
    """
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='pose_resnet',
                 **kwargs):
        super(PoseNetExtra, self).__init__(**kwargs)
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size
        self.pose_scale = 0.1

        self.encoder = CustomFlow(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True).build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True
        
        # 공통 특징 추출층
        self.shared_features = tf_keras.Sequential([
            std_conv(1, 256, 1, use_bias=False, name='shared_conv1'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.LeakyReLU(),
            std_conv(3, 256, 1, use_bias=False, name='shared_conv2'),
            tf_keras.layers.BatchNormalization(),
            tf_keras.layers.LeakyReLU(),
        ])

        self.global_pool = tf_keras.layers.GlobalAveragePooling2D()

        
        # 회전 브랜치
        self.rotation_branch = tf_keras.Sequential([
            tf_keras.layers.Dense(256),
            tf_keras.layers.LeakyReLU(),
            tf_keras.layers.Dense(3)  # 축-각도 또는 오일러 각도
        ])
        
        # 이동 브랜치
        self.translation_branch = tf_keras.Sequential([
            tf_keras.layers.Dense(256),
            tf_keras.layers.LeakyReLU(),
            tf_keras.layers.Dense(3)  # XYZ 이동
        ])

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        features = self.shared_features(x)

        features = self.global_pool(features)

        # 분리된 예측
        rotation = self.rotation_branch(features) * 0.01  # 회전에 적합한 스케일링
        translation = self.translation_branch(features) * 0.1  # 이동에 적합한 스케일링
        
        # 결합된 포즈 벡터
        return tf.concat([rotation, translation], axis=-1)
    
class PoseNet(tf_keras.Model):
    """
    - 입력: (B, H, W, 6)  (ex: 소스+타겟 concat)
    - 내부: ResNet-18 인코더 -> Conv/ReduceMean -> Reshape -> scale
    - 출력: (B, 1, 6)  (Monodepth2식 pose)
    """
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='pose_resnet',
                 **kwargs):
        super(PoseNet, self).__init__(**kwargs)
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        self.encoder = CustomFlow(image_shape=(self.image_height, self.image_width, 6), batch_size=1, pretrained=True).build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True
        
        # filter_size, out_channel, stride, pad='same', name='conv'
        self.pose_conv0 = std_conv(1, 256, 1, name='pose_conv0')  # kernel=1
        self.pose_act0 = tf_keras.layers.ReLU(name='pose_relu')

        self.pose_conv1 = std_conv(3, 256, 1, name='pose_conv1')  # kernel=3
        self.pose_act1 = tf_keras.layers.ReLU(name='pose_relu1')

        self.pose_conv2 = std_conv(3, 256, 1, name='pose_conv2')  # kernel=3
        self.pose_act2 = tf_keras.layers.ReLU(name='pose_relu2')
    
        self.pose_conv3 = tf_keras.layers.Conv2D(
            filters=6, kernel_size=(1, 1), strides=(1, 1),
            activation=None, name='pose_conv3'
        )

        self.reshape_layer = tf_keras.layers.Reshape((6,), name='pose_reshape')

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training) 

        x = self.pose_conv0(x)
        x = self.pose_act0(x)

        x = self.pose_conv1(x)
        x = self.pose_act1(x)

        x = self.pose_conv2(x)
        x = self.pose_act2(x)
            
        x = self.pose_conv3(x)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.reshape_layer(x)# [B, 6]
        x = x * 0.01 # scale
        return x
    
if __name__ == '__main__':
    # Test PoseNet
    image_shape = (480, 640)
    batch_size = 2
    posenet = PoseNetExtra(image_shape=image_shape, batch_size=batch_size)
    posenet.build((batch_size, image_shape[0], image_shape[1], 6))
    posenet.summary()
    
    # Test forward
    inputs = tf.random.normal((batch_size, image_shape[0], image_shape[1], 6))
    outputs = posenet(inputs)
    print(outputs.shape)  # [B, 6]