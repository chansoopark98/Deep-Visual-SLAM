import tensorflow as tf, tf_keras
try:
    from .model_utils import *
    from .flownet import CustomFlow
except:
    from model_utils import *
    from flownet import CustomFlow
    
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
        self.pose_conv1 = std_conv(3, 256, 1, name='pose_conv1')  # kernel=3
        self.pose_conv2 = std_conv(3, 256, 1, name='pose_conv2')  # kernel=3
        self.pose_conv3 = tf_keras.layers.Conv2D(
            filters=6, kernel_size=(1, 1), strides=(1, 1),
            activation=None, name='pose_conv3'
        )

        self.conv_a = tf_keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            activation=None, name='conv_a'
        )
        self.conv_b = tf_keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            activation=None, name='conv_b'
        )

        self.relu = tf.nn.relu
        self.tanh = tf.nn.tanh
        self.softplus = tf.nn.softplus

    def call(self, inputs, training=False):
        """
        inputs: [B, H, W, 6] (source + target image concat)
        return: [B, 1, 6]
        """
        # PoseNet Forward
        x = self.encoder(inputs, training=training) # [B, H/32, W/32, 512]
        
        x = self.pose_conv0(x)
        x = self.relu(x)

        x = self.pose_conv1(x)
        x = self.relu(x)

        x = self.pose_conv2(x)
        x = self.relu(x)
            
        x = self.pose_conv3(x)  # [B, H/32, W/32, 6]

        out_pose = self.relu(x)
        out_pose = tf.reduce_mean(out_pose, axis=[1, 2], keepdims=False)

        out_a = self.conv_a(x)
        out_a = self.softplus(out_a)
        out_a = tf.reduce_mean(out_a, axis=[1, 2], keepdims=False)

        out_b = self.conv_b(x)
        out_b = self.tanh(out_b)
        out_b = tf.reduce_mean(out_b, axis=[1, 2], keepdims=False)

        # scaling
        out_pose = out_pose * 0.01
        out_a = out_a * 0.01
        out_b = out_b * 0.01

        a = tf.reshape(out_a, (-1, 1, 1, 1))
        b = tf.reshape(out_b, (-1, 1, 1, 1))
        return out_pose, a, b
    
if __name__ == '__main__':
    # Test PoseNet
    image_shape = (128, 416)
    batch_size = 2
    posenet = PoseNet(image_shape=image_shape, batch_size=batch_size)
    posenet.build((batch_size, image_shape[0], image_shape[1], 6))
    posenet.summary()
    
    # Test forward
    inputs = tf.random.normal((batch_size, image_shape[0], image_shape[1], 6))
    pose, a, b = posenet(inputs)
    print(pose)
    print('a:', a.shape)
    print('b:', b.shape)