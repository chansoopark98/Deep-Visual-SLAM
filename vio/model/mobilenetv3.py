import tensorflow as tf

class MobilenetV3Large:
    def __init__(self, image_shape, batch_size, prefix='mobilenetv3'):
        """
        Initializes the MobilenetV3Large class.

        Args:
            image_shape (tuple): Input image shape (height, width, channels).
            batch_size (int): Batch size (not used directly in the model, but for reference).
            prefix (str): Prefix for the model name.
        """
        if len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple of (height, width, channels)")

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.prefix = prefix

    def build_model(self) -> tf.keras.Model:
        """
        Builds a MobileNetV3Large-based functional model with skip connections.

        Returns:
            tf.keras.Model: Functional model.
        """
        if self.image_shape[2] == 3:
            pretrained_weights = 'imagenet'
        else:
            pretrained_weights = None

        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(self.image_shape[0], self.image_shape[1], 3),
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights=pretrained_weights,
            input_tensor=None,
            classes=0,
            pooling=None,
            dropout_rate=0.2,
            classifier_activation=None,
            include_preprocessing=False
        )

        layer_names = [
            "expanded_conv/Add",
            "expanded_conv_2/Add",
            "expanded_conv_5/Add",
            "expanded_conv_11/Add",
            "expanded_conv_14/Add",
        ]

        outputs = [base_model.get_layer(name).output for name in layer_names]

        partial_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=outputs,
            name=f"{self.prefix}_partial"
        )

        inputs = tf.keras.Input(shape=self.image_shape, name="input_image")
        features = partial_model(inputs)

        x = features[-1]  # block6o_add (H/32)

        skips = [
            features[3],  # block5h_add (H/16)
            features[2],  # block3d_add (H/8)
            features[1],  # block2d_add (H/4)
            features[0]   # block1b_add (H/2)
        ]

        return tf.keras.Model(inputs=inputs, outputs=[x, skips], name=f"{self.prefix}_model")

if __name__ == '__main__':
    image_shape = (480, 640, 3)
    batch_size = 4
    model_builder = MobilenetV3Large(image_shape=image_shape, batch_size=batch_size)
    model = model_builder.build_model()
