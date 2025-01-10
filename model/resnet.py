import tensorflow as tf

batch_norm_decay = 0.95
batch_norm_epsilon = 1e-5

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            use_bias=False,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            use_bias=False,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride,
                                                       use_bias=False))
            self.downsample.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                                   epsilon=batch_norm_epsilon))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            use_bias=False,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            use_bias=False,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            use_bias=False,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride,
                                                   use_bias=False))
        self.downsample.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                               epsilon=batch_norm_epsilon))

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            use_bias=False,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        skip4 = x # H/2

        x = self.pool1(x) # H/4

        x = self.layer1(x, training=training)
        skip3 = x # H/4

        x = self.layer2(x, training=training)
        skip2 = x # H/8

        x = self.layer3(x, training=training)
        skip1 = x # H/16
        x = self.layer4(x, training=training)
        return x, [skip1, skip2, skip3, skip4]


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            use_bias=False,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay,
                                                      epsilon=batch_norm_epsilon)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        skip4 = x

        x = self.pool1(x)

        x = self.layer1(x, training=training)
        skip3 = x

        x = self.layer2(x, training=training)
        skip2 = x

        x = self.layer3(x, training=training)
        skip1 = x
        x = self.layer4(x, training=training)
        return x, [skip1, skip2, skip3, skip4]

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block

def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])


def resnet_34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])


def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])


def resnet_101():
    return ResNetTypeII(layer_params=[3, 4, 23, 3])


def resnet_152():
    return ResNetTypeII(layer_params=[3, 8, 36, 3])

if __name__ == '__main__':
    model = resnet_18()
    model.build(input_shape=(1, 224, 224, 3))

    model.summary()

    # inference
    input = tf.random.normal((1, 224, 224, 3))
    x, skips = model(input)

    print(x.shape)
    for skip in skips:
        print(skip.shape)