import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras

batch_norm_decay = 0.95
batch_norm_epsilon = 1e-5
pose_scale = 0.01

class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def build_disp_net(image_shape: tuple, batch_size: int) -> keras.models.Model:
    resnet_model = build_resnet18(image_shape=(*image_shape, 3),
                                                batch_size=batch_size,
                                                prefix='disp_resnet')
    
    x, skips = resnet_model.output
    
    image_height = image_shape[0]
    image_width = image_shape[1]

    # Disp Decoder
    print('Building Depth Decoder Model')
    filters = [16, 32, 64, 128, 256]

    # disp 5
    iconv5 = _conv_reflect(3, filters[4], 1, 'iconv5')(x)
    iconv5_upsample = keras.layers.Resizing(height=image_height // 16, width=image_width // 16, interpolation='bilinear')(iconv5)
    iconv5_concat = keras.layers.Concatenate(axis=3)([iconv5_upsample, skips[0]])
    upconv5 = _conv_reflect(3, filters[4], 1, 'upconv5')(iconv5_concat)

    # disp 4
    iconv4 = _conv_reflect(3, filters[3], 1, 'iconv4')(upconv5)
    iconv4_upsample = keras.layers.Resizing(height=image_height // 8, width=image_width // 8, interpolation='bilinear')(iconv4)
    iconv4_concat = keras.layers.Concatenate(axis=3)([iconv4_upsample, skips[1]])
    upconv4 = _conv_reflect(3, filters[3], 1, 'upconv4')(iconv4_concat)
    disp4 = _conv_reflect(3, 1, 1, 'disp4', activation_fn=tf.nn.sigmoid)(upconv4)

    # disp 3
    iconv3 = _conv_reflect(3, filters[2], 1, 'iconv3')(upconv4)
    iconv3_upsample = keras.layers.Resizing(height=image_height // 4, width=image_width // 4, interpolation='bilinear')(iconv3)
    iconv3_concat = keras.layers.Concatenate(axis=3)([iconv3_upsample, skips[2]])
    upconv3 = _conv_reflect(3, filters[2], 1, 'upconv3')(iconv3_concat)
    disp3 = _conv_reflect(3, 1, 1, 'disp3', activation_fn=tf.nn.sigmoid)(upconv3)

    # disp 2
    iconv2 = _conv_reflect(3, filters[1], 1, 'iconv2')(upconv3)
    iconv2_upsample = keras.layers.Resizing(height=image_height // 2, width=image_width // 2, interpolation='bilinear')(iconv2)
    iconv2_concat = keras.layers.Concatenate(axis=3)([iconv2_upsample, skips[3]])
    upconv2 = _conv_reflect(3, filters[1], 1, 'upconv2')(iconv2_concat)
    disp2 = _conv_reflect(3, 1, 1, 'disp2', activation_fn=tf.nn.sigmoid)(upconv2)

    # disp 1
    iconv1 = _conv_reflect(3, filters[0], 1, 'iconv1')(upconv2)
    iconv1_upsample = keras.layers.Resizing(height=image_height, width=image_width, interpolation='bilinear')(iconv1)
    upconv1 = _conv_reflect(3, filters[0], 1, 'upconv1')(iconv1_upsample)
    disp1 = _conv_reflect(3, 1, 1, 'disp1', activation_fn=tf.nn.sigmoid)(upconv1)

    model = keras.models.Model(inputs=[resnet_model.input], outputs=[disp1, disp2, disp3, disp4])
    return model


class ReduceMeanLayer(keras.layers.Layer):
    def __init__(self, name='unit', **kwargs):
        super(ReduceMeanLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

# class ReduceMeanLayer(keras.layers.Layer):
#     def call(self, inputs):
#         return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
    
def build_posenet(image_shape: tuple, batch_size: int) -> keras.models.Model:
    # ResNet-18
    resnet_model = build_resnet18(image_shape=(*image_shape, 6),
                                                batch_size=batch_size,
                                                prefix='pose_resnet')
    res18, _ = resnet_model.output  # [x, [skip1, skip2, skip3, skip4]]
    
    res18_concat = _conv(1, 256, 1, name='pose_conv0')(res18)

    pose_conv1 = _conv(3, 256, 1, name='pose_conv1')(res18_concat)
    pose_conv2 = _conv(3, 256, 1, name='pose_conv2')(pose_conv1)
    pose_conv3 = keras.layers.Conv2D(6, (1, 1), strides=(1, 1), activation=None, name='pose_conv3')(pose_conv2)

    pose_final = ReduceMeanLayer()(pose_conv3)
    pose_final = keras.layers.Reshape((1, 6))(pose_final)
    pose_final = keras.layers.Lambda(lambda x: x * tf.cast(pose_scale, tf.float32))(pose_final)

    model = keras.models.Model(inputs=[resnet_model.input], outputs=pose_final)
    return model


def build_resnet18(image_shape: tuple, batch_size: int, prefix: str) -> keras.models.Model:
    model_input = keras.layers.Input(shape=image_shape, batch_size=batch_size)

    print('Building ResNet-18 Model')
    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 1, 2, 2, 2]

    # conv1
    print('\tBuilding unit: conv1')
    x = _conv(kernels[0], filters[0], strides[0], name=prefix+'_conv1')(model_input)
    x = _bn(name=prefix+'_conv1')(x)
    x = _activate(type='relu', name=prefix+'_relu1')(x)
    skip4 = x

    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool')(x)

    # conv2_x
    x = ResidualBlock(num_channel=filters[1], name=prefix+'_conv2_1')(x)
    x = ResidualBlock(num_channel=filters[1], name=prefix+'_conv2_2')(x)
    skip3 = x

    # conv3_x
    x = ResidualBlockFirst(out_channel=filters[2], stride=strides[2], name=prefix+'_conv3_1')(x)
    x = ResidualBlock(num_channel=filters[2], name=prefix+'_conv3_2')(x)
    skip2 = x

    # conv4_x
    x = ResidualBlockFirst(out_channel=filters[3], stride=strides[3], name=prefix+'_conv4_1')(x)
    x = ResidualBlock(num_channel=filters[3], name=prefix+'_conv4_2')(x)
    skip1 = x

    # conv5_x
    x = ResidualBlockFirst(out_channel=filters[4], stride=strides[4], name=prefix+'_conv5_1')(x)
    x = ResidualBlock(num_channel=filters[4], name=prefix+'_conv5_2')(x)
    
    model = keras.models.Model(inputs=model_input, outputs=[x, [skip1, skip2, skip3, skip4]])
    return model

class ResidualBlockFirst(keras.layers.Layer):
    def __init__(self, out_channel, stride, name='unit', **kwargs):
        super(ResidualBlockFirst, self).__init__(name=name, **kwargs)
        self.name = name
        self.out_channel = out_channel
        self.stride = stride
        
        self.conv1 = keras.layers.Conv2D(out_channel, (3, 3), strides=(stride, stride), padding='same', name=name+'_'+'conv1')
        self.bn1 = keras.layers.BatchNormalization(name=name+'_'+'BatchNorm')
        self.relu1 = keras.layers.ReLU(name=name+'_'+'relu1')
        self.conv2 = keras.layers.Conv2D(out_channel, (3, 3), strides=(1, 1), padding='same', name=name+'_'+'conv2')
        self.bn2 = keras.layers.BatchNormalization(name=name+'_'+'BatchNorm_1')
        self.relu2 = keras.layers.ReLU(name=name+'_'+'relu2')
        
        if stride != 1 or out_channel != self.out_channel:
            self.shortcut_conv = keras.layers.Conv2D(out_channel, (1, 1), strides=(stride, stride), padding='same', name=name+'_'+'shortcut')
        else:
            self.shortcut_conv = None

    def call(self, inputs, training=False):
        in_channel = inputs.shape[-1]
        
        print(f'\tBuilding residual unit: {self.name}')
        if in_channel == self.out_channel:
            if self.stride == 1:
                short_cut = tf.identity(inputs)
            else:
                short_cut = self.shortcut_conv(inputs)
        else:
            short_cut = self.shortcut_conv(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = keras.layers.add([x, short_cut], name=self.name+'_'+'add')
        x = self.relu2(x)
        return x

class ResidualBlock(keras.layers.Layer):
    def __init__(self, num_channel, name='unit', **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self.name = name
        self.num_channel = num_channel
        self.conv1 = keras.layers.Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', name=name+'_conv1')
        self.bn1 = keras.layers.BatchNormalization(name=name+'_BatchNorm')
        self.relu1 = keras.layers.ReLU(name=name+'_relu1')
        self.conv2 = keras.layers.Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', name=name+'_conv2')
        self.bn2 = keras.layers.BatchNormalization(name=name+'_BatchNorm_1')
        self.relu2 = keras.layers.ReLU(name=name+'_relu2')

    def call(self, inputs, training=False):
        print(f'\tBuilding residual unit: {self.name}')
        short_cut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = keras.layers.add([x, short_cut], name=self.name+'_add')
        x = self.relu2(x)
        return x

def _conv(filter_size, out_channel, stride, pad='same', name='conv'):
    return keras.layers.Conv2D(out_channel, (filter_size, filter_size), strides=(stride, stride), padding=pad, name=name+'_'+'conv')

def _conv_reflect(filter_size, out_channel, stride, name='conv', activation_fn=tf.nn.elu):
    pad_size = filter_size // 2

    conv_reflect = keras.Sequential([
        ReflectionPadding2D(padding=(pad_size, pad_size)),
        keras.layers.Conv2D(out_channel, 
                            (filter_size, filter_size), 
                            strides=(stride, stride), 
                            padding='valid', 
                            activation=activation_fn, 
                            name=name + '_' + 'conv_reflect')
    ])
    
    return conv_reflect

def _bn(name='BatchNorm'):
    return keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                         epsilon=batch_norm_epsilon, name=name+'_'+'BatchNorm')

def _activate(type='relu', name='relu'):
    if type == 'elu':
        activation = keras.layers.ELU(name=name)
    else:
        activation = keras.layers.ReLU(name=name)
    return activation

if __name__ == '__main__':
    print('Test code')
    disp = build_disp_net((256, 512), 1)
    pose = build_posenet((256, 512), 1)
    print(disp)
    print(pose)