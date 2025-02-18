import tensorflow as tf
from model.monodepth2 import DispNet, PoseNet

# test_posenet
image_shape = (480, 640)
model = PoseNet(image_shape=(480, 640), batch_size=1)
model.build(input_shape=(1, *image_shape, 6))
model.load_weights('test_posenet.h5')
model(tf.random.normal((1, *image_shape, 6)))
model.save('./assets/export_model', save_format='tf')
