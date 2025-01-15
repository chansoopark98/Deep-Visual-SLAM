import tensorflow as tf
try:
    from .model_utils import *
    from .efficientnetv2 import EfficientNetV2Encoder
    from .mobilenetv3 import MobilenetV3Large
    from .resnet import resnet_18
    from .resnet_tf import Resnet
    from .raft.raft_backbone import CustomRAFT
except:
    from model_utils import *
    from efficientnetv2 import EfficientNetV2Encoder
    from mobilenetv3 import MobilenetV3Large
    from resnet import resnet_18
    from resnet_tf import Resnet
    from raft.raft_backbone import CustomRAFT

class PredictFlow(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(PredictFlow, self).__init__()
        self.conv_out = tf.keras.layers.Conv2D(filters=2,
                                      kernel_size=3, 
                                      strides=1,
                                      name=name,
                                      padding='same')

    def call(self, inputs):
        return self.conv_out(inputs)
    
class Flownet(tf.keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='flownet',
                 **kwargs):
        super(Flownet, self).__init__(**kwargs)

        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        self.encoder = resnet_18()

        filters = [16, 32, 64, 128, 256]

        # base
        self.common_resize = tf.keras.layers.Resizing(
            height=self.image_height,
            width=self.image_width,
            interpolation='bilinear',
            name=prefix+'_common_resize',
        )

        self.conv5 = std_conv(3, 256, 1, name='conv5')
        self.iconv5_resize = tf.keras.layers.Resizing(
            height=self.image_height // 16,
            width=self.image_width // 16,
            interpolation='bilinear',
            name=prefix+'_iconv5_resize'
        )
        self.upconv5 = reflect_conv(3, filters[4], 1, 'upconv5')
        self.upflow5 = PredictFlow(name='upflow5')

        self.conv4 = reflect_conv(3, filters[3], 1, 'conv4')
        self.iconv4_resize = tf.keras.layers.Resizing(
            height=self.image_height // 8,
            width=self.image_width // 8,
            interpolation='bilinear',
            name=prefix+'_iconv4_resize'
        )
        self.upconv4 = reflect_conv(3, filters[3], 1, 'upconv4')
        self.upflow4 = PredictFlow(name='upflow4')

        self.conv3 = reflect_conv(3, filters[2], 1, 'conv3')
        self.iconv3_resize = tf.keras.layers.Resizing(
            height=self.image_height // 4,
            width=self.image_width // 4,
            interpolation='bilinear',
            name=prefix+'_iconv3_resize'
        )
        self.upconv3 = reflect_conv(3, filters[2], 1, 'upconv3')
        self.upflow3 = PredictFlow(name='upflow3')

        self.conv2 = reflect_conv(3, filters[1], 1, 'conv2')
        self.iconv2_resize = tf.keras.layers.Resizing(
            height=self.image_height // 2,
            width=self.image_width // 2,
            interpolation='bilinear',
            name=prefix+'_iconv2_resize'
        )
        self.upconv2 = reflect_conv(3, filters[1], 1, 'upconv2')
        self.upflow2 = PredictFlow(name='upflow2')

        self.conv1 = reflect_conv(3, filters[0], 1, 'conv1')
        self.iconv1_resize = tf.keras.layers.Resizing(
            height=self.image_height,
            width=self.image_width,
            interpolation='bilinear',
            name=prefix+'_iconv1_resize'
        )
        self.upconv1 = reflect_conv(3, filters[0], 1, 'upconv1')
        self.upflow1 = PredictFlow(name='upflow1')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        left, right = inputs
        model_input = tf.concat([left, right], axis=-1)

        x, skips = self.encoder(model_input, training=training)

        x = self.conv5(x, training=training)  # [B,H/32, W/32, 256]
        x = self.iconv5_resize(x)
        x = tf.concat([x, skips[0]], axis=3)
        x = self.upconv5(x, training=training)
        flow5 = self.common_resize(self.upflow5(x))

        # disp4
        x = self.conv4(x, training=training)
        x = self.iconv4_resize(x)
        x = tf.concat([x, skips[1]], axis=3)
        x = self.upconv4(x, training=training)
        flow4 = self.common_resize(self.upflow4(x))
        
        # disp3
        x = self.conv3(x, training=training)
        x = self.iconv3_resize(x)
        x = tf.concat([x, skips[2]], axis=3)
        x = self.upconv3(x, training=training)
        flow3 = self.common_resize(self.upflow3(x))
        
        # disp2
        x = self.conv2(x, training=training)
        x = self.iconv2_resize(x)
        x = tf.concat([x, skips[3]], axis=3)
        x = self.upconv2(x, training=training)
        flow2 = self.common_resize(self.upflow2(x))
        
        # disp1
        x = self.conv1(x, training=training)
        x = self.iconv1_resize(x)
        x = self.upconv1(x, training=training)
        x = self.common_resize(self.upflow1(x))
        return [flow5, flow4, flow3, flow2, x]

class CustomFlow:
    def __init__(self, image_shape, batch_size, pretrained=True, prefix='custom_flow'):
        """
        Initializes the Flownet class.

        Args:
            image_shape (tuple): Input image shape (height, width, channels).
            batch_size (int): Batch size (not used directly in the model, but for reference).
            prefix (str): Prefix for the model name.
        """
        if len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple of (height, width, channels)")

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.prefix = prefix

    def build_model(self) -> tf.keras.Model:
        base_model = Flownet(image_shape=self.image_shape, batch_size=self.batch_size, prefix='flownet')
        model_input_shape = [(self.batch_size, self.image_shape[0], self.image_shape[1], 3),
                             (self.batch_size, self.image_shape[0], self.image_shape[1], 3)]
        base_model.build(input_shape=model_input_shape)
        dummy_input = [tf.random.normal((self.batch_size, self.image_shape[0], self.image_shape[1], 3)),
                       tf.random.normal((self.batch_size, self.image_shape[0], self.image_shape[1], 3))]
        _ = base_model(dummy_input, training=False)
        
        if self.pretrained:
            pretrained_weights = './assets/weights/flow/flownet/epoch_230_model.h5'
            base_model.load_weights(pretrained_weights)
        
        base_model.summary()

        new_input = tf.keras.layers.Input(shape=(self.image_shape[0], self.image_shape[1], 6),
                                          batch_size=self.batch_size)
        outputs = base_model.get_layer('res_net_type_i')(new_input)

        partial_model = tf.keras.Model(
            inputs=new_input,
            outputs=outputs,
            name=f"{self.prefix}_partial"
        )
        return partial_model

if __name__ == '__main__':
    custom_flow = CustomFlow(image_shape=(384, 512, 3), batch_size=1, pretrained=True).build_model()