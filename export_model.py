import tensorflow as tf, tf_keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from model.pose_net import PoseNet
import os
import tensorflowjs as tfjs

# tfjs.converters.convert_tf_saved_model()

class ExportWrapper(tf_keras.Model):
    def __init__(self, model: tf_keras.Model, image_shape):
        super(ExportWrapper, self).__init__()
        self.model = model
        self.model.build(input_shape=(1, *image_shape, 6))
        self.model.load_weights('test_posenet.weights.h5')

        self.image_shape = image_shape
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def preprocess(self, image):
        """
        image (tf.Tensor): 입력 이미지 [1, H, W, 6].
        """
        # 입력값을 0-1 사이로 정규화
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image_left = (image[..., :3] - self.mean) / self.std
        image_right = (image[..., 3:] - self.mean) / self.std
        image = tf.concat([image_left, image_right], axis=-1)
        return image

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        inputs = self.preprocess(inputs)
        outputs = self.model(inputs, training=training)
        outputs = tf.squeeze(outputs, axis=0)
        return outputs
    
    def get_config(self):
        config = super(ExportWrapper, self).get_config()
        config.update({
            'image_shape': self.image_shape,
            # 'model'은 비직렬화 가능하므로 여기서는 제외합니다.
        })
        return config

image_shape = (480, 640)
base_model = PoseNet(image_shape=image_shape, batch_size=1)

wrapped_model = ExportWrapper(model=base_model, image_shape=image_shape)
wrapped_model.build(input_shape=(1, *image_shape, 6))
outputs = wrapped_model(tf.random.normal((1, *image_shape, 6)))
print(outputs.shape)

wrapped_model.save('./assets/export_model_with_preprocess', save_format='tf')
#path of the directory where you want to save your model



#  --control_flow_v2 True 

# tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model --quantize_float16 --control_flow_v2 True ./assets/export_model_with_preprocess ./assets/export_model_with_preprocess_js_fp16

#   --input_format {tf_hub,keras_saved_model,tfjs_layers_model,keras_keras,tf_saved_model,keras,tf_frozen_model}
#                         Input format. For "keras", the input path can be one of the two following formats: - A topology+weights combined HDF5 (e.g., generated with
#                         `tf_keras.model.save_model()` method). - A weights-only HDF5 (e.g., generated with Keras Model's `save_weights()` method). For "keras_saved_model", the input_path must
#                         point to a subfolder under the saved model folder that is passed as the argument to tf.contrib.save_model.save_keras_model(). The subfolder is generated automatically
#                         by tensorflow when saving keras model in the SavedModel format. It is usually named as a Unix epoch time (e.g., 1542212752). For "tf" formats, a SavedModel, frozen
#                         model, or TF-Hub module is expected.