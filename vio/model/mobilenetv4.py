import tensorflow as tf, tf_keras
import tensorflow_models as tfm
import keras

small = [
    "block_group_convbn_0", # (H/2, W/2, 32)
    "block_group_convbn_2", # (H/4, W/4, 32)
    "block_group_convbn_4", # (H/8, W/8, 64)
    "block_group_uib_10", # (H/16, W/16, 96)
    "block_group_uib_16", # (H/32, W/32, 128)
]

medium = [
    "block_group_convbn_0", # (H/2, W/2, 32)
    "block_group_fused_ib_1", # (H/4, W/4, 48)
    "block_group_uib_3", # (H/8, W/8, 80)
    "block_group_uib_11", # (H/16, W/16, 160)
    "block_group_uib_22"
]

class MobilenetV4:
    def __init__(self, image_shape, batch_size, prefix='mobilenetv4'):
        """
        Initializes the MobilenetV4 class.

        Args:
            image_shape (tuple): Input image shape (height, width, channels).
            batch_size (int): Batch size (not used directly in the model, but for reference).
            prefix (str): Prefix for the model name.
                'MobileNetV4ConvSmall'
                'MobileNetV4ConvMedium'
                'MobileNetV4ConvLarge'
                'MobileNetV4HybridMedium'
                'MobileNetV4HybridLarge'
                'MobileNetV4ConvMediumSeg'
        """
        if len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple of (height, width, channels)")

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.prefix = prefix

    def build_model(self) -> tf_keras.Model:
        """
        Builds a MobileNetV4-based functional model with skip connections.

        Returns:
            tf.keras.Model: Functional model.
        """
        if self.image_shape[2] == 3:
            pretrained_weights = 'imagenet'
        else:
            pretrained_weights = None
        
        input_specs = tf_keras.layers.InputSpec(shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        base_model = tfm.vision.backbones.MobileNet(model_id='MobileNetV4ConvMedium',
                                               input_specs=input_specs,)

        layer_names = medium

        outputs = [base_model.get_layer(name).output for name in layer_names]

        partial_model = tf_keras.Model(
            inputs=base_model.input,
            outputs=outputs,
            name=f"{self.prefix}_partial"
        )

        inputs = tf_keras.Input(shape=self.image_shape, name="input_image")
        features = partial_model(inputs)

        x = features[-1]  # block6o_add (H/32)

        skips = [
            features[3],  # block5h_add (H/16)
            features[2],  # block3d_add (H/8)
            features[1],  # block2d_add (H/4)
            features[0]   # block1b_add (H/2)
        ]

        return tf_keras.Model(inputs=inputs, outputs=[x, skips], name=f"{self.prefix}_model")


if __name__ == '__main__':
    image_shape = (480, 640, 3)
    batch_size = 4
    model_builder = MobilenetV4(image_shape=image_shape, batch_size=batch_size)
    model = model_builder.build_model()
    model.summary()
