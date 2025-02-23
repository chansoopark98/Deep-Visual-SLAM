import tensorflow as tf, tf_keras
try:
    from .model_utils import *
    from .efficientnetv2 import EfficientNetV2Encoder
    from .mobilenetv3 import MobilenetV3Large
    from .resnet import resnet_18
    from .mobilenetv4 import MobilenetV4
    from .resnet_tf import Resnet
    # from .raft.raft_backbone import CustomRAFT
except:
    from model_utils import *
    from efficientnetv2 import EfficientNetV2Encoder
    from mobilenetv3 import MobilenetV3Large
    from mobilenetv4 import MobilenetV4
    from resnet import resnet_18
    from resnet_tf import Resnet
    # from raft.raft_backbone import CustomRAFT

class PredictFlow(tf_keras.layers.Layer):
    def __init__(self, input_filters, name=None):
        super(PredictFlow, self).__init__()
        self.conv_3x3 = tf_keras.layers.Conv2D(filters=input_filters,
                                      kernel_size=3, 
                                      strides=1,
                                      name=name+'_3x3',
                                      padding='same')
        self.activation = tf_keras.layers.LeakyReLU(name=name+'_leaky_relu', alpha=0.2)
        self.conv_1x1 = tf_keras.layers.Conv2D(filters=2,
                                        kernel_size=1, 
                                        strides=1,
                                        name=name+'_1x1',
                                        padding='same')
        

    def call(self, inputs, training=True):
        x = self.conv_3x3(inputs)
        x = self.activation(x)
        x = self.conv_1x1(x)
        return x
    
class Flownet(tf_keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='flownet',
                 **kwargs):
        super(Flownet, self).__init__(**kwargs)

        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        # self.encoder = resnet_18()
        # self.encoder = Resnet(image_shape=(self.image_height, self.image_width, 6),
        #                       batch_size=self.batch_size,
        #                       pretrained=True,
        #                       prefix=prefix+'_resnet').build_model()
        self.encoder = MobilenetV4(image_shape=(self.image_height, self.image_width, 6),
                                   batch_size=self.batch_size,
                                   prefix=prefix+'_mobilenetv4').build_model()
        self.encoder.summary()
        self.encoder.trainable = True

        filters = [16, 32, 64, 128, 256]

        self.conv5 = std_conv(3, 256, 1, name='conv5')
        self.iconv5_resize = tf.keras.layers.Resizing(
            height=self.image_height // 16,
            width=self.image_width // 16,
            interpolation='bilinear',
            name=prefix+'_iconv5_resize'
        )
        self.upconv5 = reflect_conv(3, filters[4], 1, 'upconv5', activation_fn=tf.nn.leaky_relu)
        self.upflow5 = PredictFlow(filters[4], name='upflow5')

        self.conv4 = reflect_conv(3, filters[3], 1, 'conv4', activation_fn=tf.nn.leaky_relu)
        self.iconv4_resize = tf.keras.layers.Resizing(
            height=self.image_height // 8,
            width=self.image_width // 8,
            interpolation='bilinear',
            name=prefix+'_iconv4_resize'
        )
        self.upconv4 = reflect_conv(3, filters[3], 1, 'upconv4', activation_fn=tf.nn.leaky_relu)
        self.upflow4 = PredictFlow(filters[3], name='upflow4')

        self.conv3 = reflect_conv(3, filters[2], 1, 'conv3', activation_fn=tf.nn.leaky_relu)
        self.iconv3_resize = tf.keras.layers.Resizing(
            height=self.image_height // 4,
            width=self.image_width // 4,
            interpolation='bilinear',
            name=prefix+'_iconv3_resize'
        )
        self.upconv3 = reflect_conv(3, filters[2], 1, 'upconv3', activation_fn=tf.nn.leaky_relu)
        self.upflow3 = PredictFlow(filters[2], name='upflow3')

        self.conv2 = reflect_conv(3, filters[1], 1, 'conv2', activation_fn=tf.nn.leaky_relu)
        self.iconv2_resize = tf.keras.layers.Resizing(
            height=self.image_height // 2,
            width=self.image_width // 2,
            interpolation='bilinear',
            name=prefix+'_iconv2_resize'
        )
        self.upconv2 = reflect_conv(3, filters[1], 1, 'upconv2', activation_fn=tf.nn.leaky_relu)
        self.upflow2 = PredictFlow(filters[1], name='upflow2')

        self.conv1 = reflect_conv(3, filters[0], 1, 'conv1', activation_fn=tf.nn.leaky_relu)
        self.iconv1_resize = tf.keras.layers.Resizing(
            height=self.image_height,
            width=self.image_width,
            interpolation='bilinear',
            name=prefix+'_iconv1_resize'
        )
        self.upconv1 = reflect_conv(3, filters[0], 1, 'upconv1', activation_fn=tf.nn.leaky_relu)
        self.upflow1 = PredictFlow(filters[0], name='upflow1')

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        x, skips = self.encoder(inputs, training=training)

        x = tf.cast(x, tf.float32)
        skips = [tf.cast(skip, tf.float32) for skip in skips]

        x = self.conv5(x, training=training)  # [B,H/32, W/32, 256]
        x = self.iconv5_resize(x)
        x = tf.concat([x, skips[0]], axis=3)
        x = self.upconv5(x, training=training)
        flow5 = self.upflow5(x)
        flow5 = tf.image.resize(flow5, (self.image_height, self.image_width), method='bilinear')

        # disp4
        x = self.conv4(x, training=training)
        x = self.iconv4_resize(x)
        x = tf.concat([x, skips[1]], axis=3)
        x = self.upconv4(x, training=training)
        flow4 = self.upflow4(x)
        flow4 = tf.image.resize(flow4, (self.image_height, self.image_width), method='bilinear')
        
        # disp3
        x = self.conv3(x, training=training)
        x = self.iconv3_resize(x)
        x = tf.concat([x, skips[2]], axis=3)
        x = self.upconv3(x, training=training)
        flow3 = self.upflow3(x)
        flow3 = tf.image.resize(flow3, (self.image_height, self.image_width), method='bilinear')
        
        # disp2
        x = self.conv2(x, training=training)
        x = self.iconv2_resize(x)
        x = tf.concat([x, skips[3]], axis=3)
        x = self.upconv2(x, training=training)
        flow2 = self.upflow2(x)
        flow2 = tf.image.resize(flow2, (self.image_height, self.image_width), method='bilinear')
        
        # disp1
        x = self.conv1(x, training=training)
        x = self.iconv1_resize(x)
        x = self.upconv1(x, training=training)
        x = self.upflow1(x)

        return [x, flow2, flow3, flow4, flow5]

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
        base_model = Flownet(image_shape=self.image_shape, batch_size=self.batch_size, prefix=self.prefix)
        # base_model = MobilenetV4(image_shape=self.image_shape,
        #                            batch_size=self.batch_size,
        #                            prefix=self.prefix+'_mobilenetv4').build_model()
        base_model.trainable = True
        model_input_shape = (self.batch_size, *self.image_shape) # (batch_size, height, width, 6)
        
        dummy_input = tf.random.normal(model_input_shape)
        _ = base_model(dummy_input, training=False)
        
        if self.pretrained:
            pretrained_weights = './assets/weights/flow/flow/MobileNetV4_Medium/epoch_100_model.weights.h5'
            base_model.load_weights(pretrained_weights, skip_mismatch=True)
        

        new_input = tf_keras.layers.Input(shape=(self.image_shape[0], self.image_shape[1], 6),
                                          batch_size=self.batch_size)
        
        outputs, _ = base_model.get_layer('flownet_mobilenetv4_model')(new_input)

        partial_model = tf_keras.Model(
            inputs=new_input,
            outputs=outputs,
            name=f"{self.prefix}_partial"
        )
        return partial_model

if __name__ == '__main__':
    custom_flow = CustomFlow(image_shape=(384, 512, 6), batch_size=1, prefix='flownet', pretrained=True).build_model()