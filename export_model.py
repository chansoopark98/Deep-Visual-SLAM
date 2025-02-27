import tensorflow as tf, tf_keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from model.pose_net import PoseNet
import os
import tensorflowjs as tfjs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_h', type=str, default=240, help='Image height')
parser.add_argument('--img_w', type=str, default=320, help='Image width')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the model')
parser.add_argument('--pretrained_dir', type=str, default='./weights/vo/tf2.16_vo_bs8_ep21_lr1e-4_1e-5_240x320_/pose_net_epoch_18_model.weights.h5')
parser.add_argument('--saved_model_dir', type=str, default='./assets/saved_models/', help='Output directory for the model')
parser.add_argument('--tfjs_dir', type=str, default='./assets/tfjs_models/', help='Output directory for the tfjs model')
args = parser.parse_args()

class ExportWrapper(tf_keras.Model):
    def __init__(self, model: tf_keras.Model, image_shape):
        super(ExportWrapper, self).__init__()
        self.model = model
        # rename model 
        self.model._name = 'export_model'

        # rename model layers
        for layer in self.model.layers:
            layer._name = f"export_{layer.name}"

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

if __name__ == '__main__':
    os.makedirs(args.saved_model_dir, exist_ok=True)
    os.makedirs(args.tfjs_dir, exist_ok=True)
    image_shape = (args.img_h, args.img_w)
    base_model = PoseNet(image_shape=image_shape, batch_size=1)
    base_model.build(input_shape=(1, *image_shape, 6))
    base_model.load_weights(args.pretrained_dir)

    wrapped_model = ExportWrapper(model=base_model, image_shape=image_shape)
    wrapped_model.build(input_shape=(1, *image_shape, 6))
    outputs = wrapped_model(tf.random.normal((1, *image_shape, 6)))
    print(outputs.shape)

    # rename all layers
    for layer in wrapped_model.layers:
        layer._name = f"a_export_{layer.name}"

    # wrapped_model.save(args.saved_model_dir, save_format='tf')

    tf.saved_model.save(wrapped_model, args.saved_model_dir)


    #path of the directory where you want to save your model

    tfjs.converters.convert_tf_saved_model(args.saved_model_dir,
                                        args.tfjs_dir,
                                        quantization_dtype_map=tfjs.quantization.QUANTIZATION_DTYPE_FLOAT16,
                                            control_flow_v2=True, 
                                            )
    

#  --control_flow_v2 True 

# tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model --quantize_float16 --control_flow_v2 True ./assets/saved_models ./assets/export_model_with_preprocess_js_fp16

#   --input_format {tf_hub,keras_saved_model,tfjs_layers_model,keras_keras,tf_saved_model,keras,tf_frozen_model}
#                         Input format. For "keras", the input path can be one of the two following formats: - A topology+weights combined HDF5 (e.g., generated with
#                         `tf_keras.model.save_model()` method). - A weights-only HDF5 (e.g., generated with Keras Model's `save_weights()` method). For "keras_saved_model", the input_path must
#                         point to a subfolder under the saved model folder that is passed as the argument to tf.contrib.save_model.save_keras_model(). The subfolder is generated automatically
#                         by tensorflow when saving keras model in the SavedModel format. It is usually named as a Unix epoch time (e.g., 1542212752). For "tf" formats, a SavedModel, frozen
#                         model, or TF-Hub module is expected.