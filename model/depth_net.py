import tensorflow as tf, tf_keras
try:
    from .model_utils import *
    from .resnet_tf import Resnet, Resnet34
    from .cam_conv import AddCAMCoords

except:
    from model_utils import *
    from resnet_tf import Resnet, Resnet34
    from cam_conv import AddCAMCoords

class DispNet(tf_keras.Model):
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

        self.add_coord = AddCAMCoords(coord_maps=False,
                                      centered_coord=True,
                                      norm_coord_maps=True,
                                      with_r=False,
                                      bord_dist=False,
                                      scale_centered_coord=(self.image_height, self.image_width),
                                      fov_maps=True,
                                      data_format='channels_last')
        
        self.channels = self.add_coord.additional_channels() + 3
        self.encoder = Resnet(image_shape=(*image_shape, self.channels),
                              batch_size=batch_size,
                              pretrained=True,
                              prefix=prefix + '_resnet18').build_model()
        
        # Depth Decoder
        print('Building Depth Decoder Model')
        filters = [16, 32, 64, 128, 256]

        # disp 5
        self.iconv5 = reflect_conv(3, filters[4], 1, 'iconv5')
        self.iconv5_resize = tf_keras.layers.Resizing(
            height=self.image_height // 16,
            width=self.image_width // 16,
            interpolation='bilinear',
            name='iconv5_resize'
        )
        self.upconv5 = reflect_conv(3, filters[4], 1, 'upconv5')

        # disp 4
        self.iconv4 = reflect_conv(3, filters[3], 1, 'iconv4')
        self.iconv4_resize = tf_keras.layers.Resizing(
            height=self.image_height // 8,
            width=self.image_width // 8,
            interpolation='bilinear',
            name='iconv4_resize'
        )
        self.upconv4 = reflect_conv(3, filters[3], 1, 'upconv4')
        self.disp4 = reflect_conv(3, 1, 1, 'disp4', activation_fn=tf.nn.sigmoid)

        # disp 3
        self.iconv3 = reflect_conv(3, filters[2], 1, 'iconv3')
        self.iconv3_resize = tf_keras.layers.Resizing(
            height=self.image_height // 4,
            width=self.image_width // 4,
            interpolation='bilinear',
            name='iconv3_resize'
        )
        self.upconv3 = reflect_conv(3, filters[2], 1, 'upconv3')
        self.disp3 = reflect_conv(3, 1, 1, 'disp3', activation_fn=tf.nn.sigmoid)

        # disp 2
        self.iconv2 = reflect_conv(3, filters[1], 1, 'iconv2')
        self.iconv2_resize = tf_keras.layers.Resizing(
            height=self.image_height // 2,
            width=self.image_width // 2,
            interpolation='bilinear',
            name='iconv2_resize'
        )
        self.upconv2 = reflect_conv(3, filters[1], 1, 'upconv2')
        self.disp2 = reflect_conv(3, 1, 1, 'disp2', activation_fn=tf.nn.sigmoid)

        # disp 1
        self.iconv1 = reflect_conv(3, filters[0], 1, 'iconv1')
        self.iconv1_resize = tf_keras.layers.Resizing(
            height=self.image_height,
            width=self.image_width,
            interpolation='bilinear',
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
        rgb, intrinsic = inputs
        input_tensors = self.add_coord(rgb, intrinsic, training=training)
        x, skips = self.encoder(input_tensors, training=training)

        x = tf.cast(x, tf.float32)
        skips = [tf.cast(skip, tf.float32) for skip in skips]

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


if __name__ == '__main__':
    dispnet = DispNet(image_shape=(256, 256), batch_size=1, prefix='disp_resnet')
    monodepth = MonoDepth2Model(image_shape=(256, 256), batch_size=1)
    dummy = tf.random.normal((1, 256, 256, 3))
    dummy_src = tf.random.normal((1, 256, 256, 6))
    imu_src = tf.random.normal((1, 10, 6))

    # test
    disp1, disp2, disp3, disp4 = dispnet(dummy, True)
    print(disp1.shape, disp2.shape, disp3.shape, disp4.shape)