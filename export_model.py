import tensorflow as tf
from model.monodepth2 import  PoseNet
# import tensorflow_model_optimization as tfmot

# clustering_params = {
#     "number_of_clusters": 8,
#     "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
# }
# clustered_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)
# clustered_model.compile()
# clustered_model.fit()
# final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

class ExportWrapper(tf.keras.Model):
    def __init__(self, model, image_shape):
        super(ExportWrapper, self).__init__()
        self.model = model
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
    
    def call(self, inputs, training=False):
        inputs = self.preprocess(inputs)
        outputs = self.model(inputs, training=training)
        outputs = tf.squeeze(outputs, axis=0)
        return outputs

image_shape = (480, 640)
base_model = PoseNet(image_shape=image_shape, batch_size=1)
base_model.build(input_shape=(1, *image_shape, 6))
base_model.load_weights('test_posenet.h5')

wrapped_model = ExportWrapper(model=base_model, image_shape=image_shape)
wrapped_model.build(input_shape=(1, *image_shape, 6))
outputs = wrapped_model(tf.random.normal((1, *image_shape, 6)))
print(outputs.shape)
wrapped_model.save('./assets/export_model_with_preprocess', save_format='tf')