import tensorflow as tf

batch_norm_decay = 0.95
batch_norm_epsilon = 1e-5
pose_scale = 0.01

def std_conv(filter_size, out_channel, stride, pad='same', name='conv'):
    return tf.keras.layers.Conv2D(out_channel, (filter_size, filter_size), strides=(stride, stride), padding=pad, name=name+'_'+'conv')

def hard_sigmoid(x):
    return tf.keras.layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)

def hard_swish(x):
    return tf.keras.layers.Multiply()([x, hard_sigmoid(x)])

def reflect_conv(filter_size, out_channel, stride, name='conv', activation_fn=tf.nn.elu):
    """
    returns a `tf.keras.Sequential` with ReflectionPadding2D + Conv2D(...).
    """
    pad_size = filter_size // 2
    conv_reflect = tf.keras.Sequential([
        ReflectionPadding2D(padding=(pad_size, pad_size), name=name + '_reflect_pad'),
        tf.keras.layers.Conv2D(out_channel,
                               kernel_size=(filter_size, filter_size),
                               strides=(stride, stride),
                               padding='valid',
                               activation=activation_fn,
                               name=name + '_conv_reflect'),
    ], name=name + '_reflect_conv_seq')

    return conv_reflect

def batch_norm(name='BatchNorm'):
    return tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                              epsilon=batch_norm_epsilon, name=name+'_'+'BatchNorm')

def activation(type='relu', name='relu'):
    if type == 'elu':
        activation = tf.keras.layers.ELU(name=name)
    else:
        activation = tf.keras.layers.ReLU(name=name)
    return activation

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0],
                s[1] + 2 * self.padding[0],
                s[2] + 2 * self.padding[1],
                s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]], 'REFLECT')

class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, prefix='unit', **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

class ResidualBlockFirst(tf.keras.layers.Layer):
    def __init__(self, out_channel, stride, prefix='unit', **kwargs):
        super(ResidualBlockFirst, self).__init__(**kwargs)
        self.prefix = prefix
        self.out_channel = out_channel
        self.stride = stride
        
        self.conv1 = tf.keras.layers.Conv2D(out_channel, (3, 3), strides=(stride, stride), padding='same', name=prefix+'_'+'conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon,
                                                      name=prefix+'_'+'BatchNorm')
        self.relu1 = tf.keras.layers.ReLU(name=prefix+'_'+'relu1')
        self.conv2 = tf.keras.layers.Conv2D(out_channel, (3, 3), strides=(1, 1), padding='same', name=prefix+'_'+'conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon,
                                                      name=prefix+'_'+'BatchNorm_1')
        self.relu2 = tf.keras.layers.ReLU(name=prefix+'_'+'relu2')
        
        if stride != 1 or out_channel != self.out_channel:
            self.shortcut_conv = tf.keras.layers.Conv2D(out_channel, (1, 1), strides=(stride, stride), padding='same', name=prefix+'_'+'shortcut')
        else:
            self.shortcut_conv = None

    def call(self, inputs, training=False):
        in_channel = inputs.shape[-1]
    
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
        
        x = tf.keras.layers.add([x, short_cut], name=self.name+'_'+'add')
        x = self.relu2(x)
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_channel, prefix='unit', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.num_channel = num_channel
        self.conv1 = tf.keras.layers.Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', name=prefix+'_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon,
                                                      name=prefix+'_BatchNorm')
        self.relu1 = tf.keras.layers.ReLU(name=prefix+'_relu1')
        self.conv2 = tf.keras.layers.Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', name=prefix+'_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon,
                                                      name=prefix+'_BatchNorm_1')
        self.relu2 = tf.keras.layers.ReLU(name=prefix+'_relu2')

    def call(self, inputs, training=False):
        short_cut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.keras.layers.add([x, short_cut], name=self.name+'_add')
        x = self.relu2(x)
        return x