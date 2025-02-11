import tensorflow as tf
try:
    from .model_utils import *
    from .efficientnetv2 import EfficientNetV2Encoder
    from .mobilenetv3 import MobilenetV3Large
    from .resnet import resnet_18
    from .resnet_tf import Resnet
    from .raft.raft_backbone import CustomRAFT
    from .flownet import CustomFlow
except:
    from model_utils import *
    from efficientnetv2 import EfficientNetV2Encoder
    from mobilenetv3 import MobilenetV3Large
    from resnet import resnet_18
    from resnet_tf import Resnet
    from raft.raft_backbone import CustomRAFT
    from flownet import CustomFlow

class ResNet18Encoder(tf.keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix: str = 'resnet18',
                 **kwargs):
        super(ResNet18Encoder, self).__init__(**kwargs)

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.prefix_str = prefix

        # 여기서 filters/kernels/strides는 고정
        self.filters = [64, 64, 128, 256, 512]
        self.kernels = [7, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 2]

        # ============ Define Layers ============

        # conv1
        # std_conv(kernels[0], filters[0], strides[0], name=prefix+'_conv1')
        self.conv1 = std_conv(self.kernels[0], self.filters[0], self.strides[0], name=self.prefix_str+'_conv1')
        self.bn_conv1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                           epsilon=batch_norm_epsilon, name=self.prefix_str+'_'+'BatchNorm')
        self.act_conv1 = tf.keras.layers.ReLU(name=self.prefix_str+'_relu1')
        
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name=self.prefix_str+'_pool')

        # conv2_x
        self.conv2_1 = ResidualBlock(num_channel=self.filters[1], prefix=self.prefix_str+'_conv2_1')
        self.conv2_2 = ResidualBlock(num_channel=self.filters[1], prefix=self.prefix_str+'_conv2_2')

        # conv3_x
        self.conv3_1 = ResidualBlockFirst(out_channel=self.filters[2], stride=self.strides[2], prefix=self.prefix_str+'_conv3_1')
        self.conv3_2 = ResidualBlock(num_channel=self.filters[2], prefix=self.prefix_str+'_conv3_2')

        # conv4_x
        self.conv4_1 = ResidualBlockFirst(out_channel=self.filters[3], stride=self.strides[3], prefix=self.prefix_str+'_conv4_1')
        self.conv4_2 = ResidualBlock(num_channel=self.filters[3], prefix=self.prefix_str+'_conv4_2')

        # conv5_x
        self.conv5_1 = ResidualBlockFirst(out_channel=self.filters[4], stride=self.strides[4], prefix=self.prefix_str+'_conv5_1')
        self.conv5_2 = ResidualBlock(num_channel=self.filters[4], prefix=self.prefix_str+'_conv5_2')

    def call(self, inputs, training=False):
        """
        inputs: (B, H, W, 3)
        returns: (x, [skip1, skip2, skip3, skip4])
        """
        # conv1
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = self.act_conv1(x)
        skip4 = x  # (H/2, W/2, 64)

        x = self.pool(x)  # (H/4, W/4, 64)

        # conv2_x
        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        skip3 = x  # (H/4, W/4, 64)

        # conv3_x
        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        skip2 = x  # (H/8, W/8, 128)

        # conv4_x
        x = self.conv4_1(x, training=training)
        x = self.conv4_2(x, training=training)
        skip1 = x  # (H/16, W/16, 256)

        # conv5_x
        x = self.conv5_1(x, training=training)
        x = self.conv5_2(x, training=training)
        # x: (H/32, W/32, 512)

        return x, [skip1, skip2, skip3, skip4]

class DispNet(tf.keras.Model):
    """
    Encoder + Disp Decoder
    """
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix: str = 'Dispnet',
                 **kwargs):
        super(DispNet, self).__init__(**kwargs)

        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size
        self.prefix_str = prefix

        self.encoder = Resnet(image_shape=(*image_shape, 3),
                              batch_size=batch_size,
                              pretrained=True,
                              prefix=prefix + '_resnet18').build_model()
        
        # Depth Decoder
        print('Building Depth Decoder Model')
        filters = [16, 32, 64, 128, 256]

        # disp 5
        self.iconv5 = reflect_conv(3, filters[4], 1, 'iconv5')
        self.iconv5_resize = tf.keras.layers.Resizing(
            height=self.image_height // 16,
            width=self.image_width // 16,
            interpolation='nearest',
            name='iconv5_resize'
        )
        self.upconv5 = reflect_conv(3, filters[4], 1, 'upconv5')

        # disp 4
        self.iconv4 = reflect_conv(3, filters[3], 1, 'iconv4')
        self.iconv4_resize = tf.keras.layers.Resizing(
            height=self.image_height // 8,
            width=self.image_width // 8,
            interpolation='nearest',
            name='iconv4_resize'
        )
        self.upconv4 = reflect_conv(3, filters[3], 1, 'upconv4')
        self.disp4 = reflect_conv(3, 1, 1, 'disp4', activation_fn=tf.nn.sigmoid)

        # disp 3
        self.iconv3 = reflect_conv(3, filters[2], 1, 'iconv3')
        self.iconv3_resize = tf.keras.layers.Resizing(
            height=self.image_height // 4,
            width=self.image_width // 4,
            interpolation='nearest',
            name='iconv3_resize'
        )
        self.upconv3 = reflect_conv(3, filters[2], 1, 'upconv3')
        self.disp3 = reflect_conv(3, 1, 1, 'disp3', activation_fn=tf.nn.sigmoid)

        # disp 2
        self.iconv2 = reflect_conv(3, filters[1], 1, 'iconv2')
        self.iconv2_resize = tf.keras.layers.Resizing(
            height=self.image_height // 2,
            width=self.image_width // 2,
            interpolation='nearest',
            name='iconv2_resize'
        )
        self.upconv2 = reflect_conv(3, filters[1], 1, 'upconv2')
        self.disp2 = reflect_conv(3, 1, 1, 'disp2', activation_fn=tf.nn.sigmoid)

        # disp 1
        self.iconv1 = reflect_conv(3, filters[0], 1, 'iconv1')
        self.iconv1_resize = tf.keras.layers.Resizing(
            height=self.image_height,
            width=self.image_width,
            interpolation='nearest',
            name='iconv1_resize'
        )
        self.upconv1 = reflect_conv(3, filters[0], 1, 'upconv1')
        self.disp1 = reflect_conv(3, 1, 1, 'disp1', activation_fn=tf.nn.sigmoid)

    def call(self, inputs, training=False):
        """
        inputs: [B, image_height, image_width, 3]
        returns: disp1, disp2, disp3, disp4
        """
        # 1) 인코더
        x, skips = self.encoder(inputs, training=training)

        # disp5
        iconv5 = self.iconv5(x, training=training)  # [B,H/32, W/32, 256]
        iconv5_upsample = self.iconv5_resize(iconv5)
        iconv5_concat = tf.concat([iconv5_upsample, skips[0]], axis=3)
        upconv5 = self.upconv5(iconv5_concat, training=training)

        # disp4
        iconv4 = self.iconv4(upconv5, training=training)
        iconv4_upsample = self.iconv4_resize(iconv4)
        iconv4_concat = tf.concat([iconv4_upsample, skips[1]], axis=3)
        upconv4 = self.upconv4(iconv4_concat, training=training)
        disp4 = self.disp4(upconv4, training=training)

        # disp3
        iconv3 = self.iconv3(upconv4, training=training)
        iconv3_upsample = self.iconv3_resize(iconv3)
        iconv3_concat = tf.concat([iconv3_upsample, skips[2]], axis=3)
        upconv3 = self.upconv3(iconv3_concat, training=training)
        disp3 = self.disp3(upconv3, training=training)

        # disp2
        iconv2 = self.iconv2(upconv3, training=training)
        iconv2_upsample = self.iconv2_resize(iconv2)
        iconv2_concat = tf.concat([iconv2_upsample, skips[3]], axis=3)
        upconv2 = self.upconv2(iconv2_concat, training=training)
        disp2 = self.disp2(upconv2, training=training)

        # disp1
        iconv1 = self.iconv1(upconv2, training=training)
        iconv1_upsample = self.iconv1_resize(iconv1)
        upconv1 = self.upconv1(iconv1_upsample, training=training)
        disp1 = self.disp1(upconv1, training=training)

        return disp1, disp2, disp3, disp4

class PoseNet(tf.keras.Model):
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

        # self.encoder = resnet_18()
        self.encoder = CustomFlow(image_shape=(*image_shape, 6),
                                  batch_size=batch_size,
                                  prefix='custom_flow').build_model()

        # filter_size, out_channel, stride, pad='same', name='conv'
        self.pose_conv0 = std_conv(1, 256, 1, name='pose_conv0')  # kernel=1
        self.pose_conv1 = std_conv(3, 256, 1, name='pose_conv1')  # kernel=3
        self.pose_conv2 = std_conv(3, 256, 1, name='pose_conv2')  # kernel=3
        self.pose_conv3 = tf.keras.layers.Conv2D(
            filters=6, kernel_size=(1,1), strides=(1,1),
            activation=None, name='pose_conv3'
        )

        # 3) ReduceMeanLayer, Reshape
        self.reduce_mean_layer = ReduceMeanLayer(prefix='pose_reduce_mean')
        self.reshape_layer = tf.keras.layers.Reshape((6,), name='pose_reshape')

    def call(self, inputs, training=False):
        """
        inputs: [B, H, W, 6]
        return: [B, 1, 6]
        """
        # 1) ResNet 인코더
        x = self.encoder(inputs, training=training) 
        # x: 최종 conv5_x 특징맵, shape [B, H/32, W/32, 512]

        # 2) pose_conv0 -> pose_conv1 -> pose_conv2 -> pose_conv3
        x = self.pose_conv0(x)
        x = self.pose_conv1(x)
        x = self.pose_conv2(x)
        x = self.pose_conv3(x)  # [B, H/32, W/32, 6]

        # 3) reduce_mean -> reshape -> scale
        x = self.reduce_mean_layer(x)  # [B, 1, 1, 6] => keepdims=True
        x = self.reshape_layer(x)      # [B, 6]
        x = x * 0.01 # scale
        return x

class MonoDepth2Model(tf.keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 **kwargs):
        super(MonoDepth2Model, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.num_scales = 4
        self.num_source = 2
        self.ssim_ratio = 0.85
        self.smooth_ratio = 1e-3
        self.auto_mask = True
        self.min_depth = 0.1
        self.max_depth = 10.

        self.depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='disp_resnet')
        self.depth_net(tf.random.normal((1, *image_shape, 3)))
        self.depth_net.load_weights('./assets/weights/depth/nyu_diode_diml_metricDepth_ep30.h5')

        self.pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')

    def call(self, inputs, training=False):
        src_left = inputs[..., :3]
        tgt_image = inputs[..., 3:6] 
        src_right = inputs[..., 6:]

        disp_raw = self.depth_net(tgt_image, training=training) # disp [1,2,3,4]
        
        pred_disp_list = []
        for s in range(self.num_scales):
            scale_h = self.image_shape[0] // (2 ** s)
            scale_w = self.image_shape[1] // (2 ** s)
            scaled_disp = tf.image.resize(disp_raw[s], [scale_h, scale_w], method=tf.image.ResizeMethod.BILINEAR)
            pred_disp_list.append(scaled_disp)

        # Pose
        # src_left, src_right
        cat_left = tf.concat([src_left, tgt_image], axis=3)   # [B,H,W,6]
        cat_right = tf.concat([tgt_image, src_right], axis=3) # [B,H,W,6]
        pose_left = self.pose_net(cat_left, training=training)    # [B,6]
        pose_right = self.pose_net(cat_right, training=training)  # [B,6]
        pred_poses = tf.stack([pose_left, pose_right], axis=1)    # [B,2,6]

        return pred_disp_list, pred_poses


if __name__ == '__main__':
    dispnet = DispNet(image_shape=(256, 256), batch_size=1, prefix='disp_resnet')
    monodepth = MonoDepth2Model(image_shape=(256, 256), batch_size=1)
    dummy = tf.random.normal((1, 256, 256, 3))
    dummy_src = tf.random.normal((1, 256, 256, 6))
    imu_src = tf.random.normal((1, 10, 6))

    # test
    disp1, disp2, disp3, disp4 = dispnet(dummy, True)
    print(disp1.shape, disp2.shape, disp3.shape, disp4.shape)