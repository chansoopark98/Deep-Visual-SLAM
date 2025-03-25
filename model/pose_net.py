import tensorflow as tf, tf_keras
try:
    from .flownet import CustomFlow
except:
    from flownet import CustomFlow


def std_conv(filter_size, out_channel, stride, pad='same', name='conv'):
    conv_layer = tf_keras.layers.Conv2D(out_channel,
                                        (filter_size, filter_size),
                                         strides=(stride, stride), 
                                         padding=pad,
                                         name=name+'_'+'conv')
    return conv_layer

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
    image_shape = (128, 416)
    batch_size = 2
    posenet = PoseNet(image_shape=image_shape, batch_size=batch_size)
    posenet.build((batch_size, image_shape[0], image_shape[1], 6))
    posenet.summary()
    
    # Test forward
    inputs = tf.random.normal((batch_size, image_shape[0], image_shape[1], 6))
    outputs = posenet(inputs)
    print(outputs.shape)  # [B, 6]