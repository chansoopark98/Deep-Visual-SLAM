import tensorflow as tf
from tensorflow import keras

BATCH_NORM_DECAY = 0.95

def basic_block(inputs, num_channels, kernel_size, num_blocks, skip_blocks, name):
    """Basic residual block"""
    x = inputs

    for i in range(num_blocks):
        if i not in skip_blocks:
            x1 = conv_bn_relu(x, num_channels, kernel_size, strides=[1,1], name=name + '.'+str(i))
            x = keras.layers.Add()([x, x1])
            x = keras.layers.Activation('relu')(x)
    return x

def basic_block_down(inputs, num_channels, kernel_size, name):
    """Residual block with strided downsampling"""
    x = inputs
    x1 = conv_bn_relu(x, num_channels, kernel_size, strides=[2,1], name=name+'.0')
    x = keras.layers.Conv2D(num_channels, kernel_size=1, strides=2, padding='same', activation='linear', use_bias=False, name=name+'.0.downsample.0')(x)
    x = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=1e-5, name=name+'.0.downsample.1')(x)
    x = keras.layers.Add()([x, x1])
    x = keras.layers.Activation('relu')(x)
    return x
         
def resnet_18(inputs, build_partial=True):
    x = keras.layers.ZeroPadding2D(padding=(3,3), name='pad')(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='linear', use_bias=False, name='conv1')(x)
    x = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=1e-5, name='bn1')(x)
    x = keras.layers.Activation('relu', name='relu')(x)

    skip1 = x # H/2

    x = keras.layers.ZeroPadding2D(padding=(1,1), name='pad1')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    x = basic_block(x, num_channels=64, kernel_size=3, num_blocks=2, skip_blocks=[], name='layer1')

    skip2 = x # H/4

    x = basic_block_down(x, num_channels=128, kernel_size=3, name='layer2')
    x = basic_block(x, num_channels=128, kernel_size=3, num_blocks=2, skip_blocks=[0], name='layer2')

    skip3 = x # H/8

    x = basic_block_down(x, num_channels=256, kernel_size=3, name='layer3')
    x = basic_block(x, num_channels=256, kernel_size=3, num_blocks=2, skip_blocks=[0], name='layer3')

    skip4 = x # H/16

    x = basic_block_down(x, num_channels=512, kernel_size=3, name='layer4')
    x = basic_block(x, num_channels=512, kernel_size=3, num_blocks=2, skip_blocks=[0], name='layer4')

    if build_partial:
        return x, [skip4, skip3, skip2, skip1]
    else:
        x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
        x = keras.layers.Dense(units=1000, use_bias=True, activation='linear', name='fc')(x)
        return x
    

def resnet_34(inputs, build_partial=True):
    x = keras.layers.ZeroPadding2D(padding=(3,3), name='pad')(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='linear', use_bias=False, name='conv1')(x)
    x = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=1e-5, name='bn1')(x)
    x = keras.layers.Activation('relu', name='relu')(x)

    skip1 = x # H/2
    
    x = keras.layers.ZeroPadding2D(padding=(1,1), name='pad1')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    x = basic_block(x, num_channels=64, kernel_size=3, num_blocks=3, skip_blocks=[], name='layer1')

    skip2 = x # H/4

    x = basic_block_down(x, num_channels=128, kernel_size=3, name='layer2')       
    x = basic_block(x, num_channels=128, kernel_size=3, num_blocks=4, skip_blocks=[0], name='layer2')       

    skip3 = x # H/8

    x = basic_block_down(x, num_channels=256, kernel_size=3, name='layer3')       
    x = basic_block(x, num_channels=256, kernel_size=3, num_blocks=6, skip_blocks=[0], name='layer3')  

    skip4 = x # H/16

    x = basic_block_down(x, num_channels=512, kernel_size=3, name='layer4')       
    x = basic_block(x, num_channels=512, kernel_size=3, num_blocks=3, skip_blocks=[0], name='layer4')  

    if build_partial:
        return x, [skip4, skip3, skip2, skip1]
    else:
        x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
        x = keras.layers.Dense(units=1000, use_bias=True, activation='linear', name='fc')(x)
    return x

def conv_bn_relu(x, num_channels, kernel_size, strides, name):
    """Layer consisting of 2 consecutive batch normalizations with 1 first relu"""
    if strides[0] == 2:
        x = keras.layers.ZeroPadding2D(padding=(1,1), name=name+'.pad')(x)
        x = keras.layers.Conv2D(num_channels, kernel_size, strides[0], padding='valid', activation='linear', use_bias=False, name=name+'.conv1')(x)
    else:
        x = keras.layers.Conv2D(num_channels, kernel_size, strides[0], padding='same', activation='linear',  use_bias=False, name=name+'.conv1')(x)      
    x = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=1e-5, name=name+'.bn1')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(num_channels, kernel_size, strides[1], padding='same', activation='linear', use_bias=False, name=name+'.conv2')(x)
    x = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=1e-5, name=name+'.bn2')(x)
    return x