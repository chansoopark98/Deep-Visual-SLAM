import tensorflow as tf

class MobilenetV3Large:
    def __init__(self, image_shape, batch_size, prefix='mobilenetv3', pretrained=True, return_skips=False):
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
        self.pretrained = pretrained
        self.return_skips = return_skips

    def build_model(self) -> tf.keras.Model:
        """
        Builds a MobileNetV3Large-based functional model with skip connections.

        Returns:
            tf.keras.Model: Functional model.
        """
        if self.image_shape[2] == 3 and self.pretrained:
            pretrained_weights = 'imagenet'
        else:
            pretrained_weights = None
            
        base_model: tf.keras.Model = tf.keras.applications.MobileNetV3Large(
            input_shape=(self.image_shape[0], self.image_shape[1], self.image_shape[2]),
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
        base_model.summary()

        layer_names = [
            "expanded_conv/Add", # 240 320 16
            "expanded_conv_2/Add", # 120 160 24
            "expanded_conv_5/Add", # 60 80 40
            "expanded_conv_11/Add", # 30 40 112
            "expanded_conv_14/Add", # 15 20 160
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
            features[3],
            features[2],
            features[1],
            features[0],
        ]
        if self.return_skips:
            return tf.keras.Model(inputs=inputs, outputs=[x, skips], name=f"{self.prefix}_model")
        else:
            return tf.keras.Model(inputs=inputs, outputs=x, name=f"{self.prefix}_model")

if __name__ == '__main__':
    image_shape = (480, 640, 3)
    batch_size = 4
    model_builder = MobilenetV3Large(image_shape=image_shape, batch_size=batch_size)
    partial_model = model_builder.build_model()
    partial_model.build(input_shape=(batch_size, *image_shape))
    x, skips = partial_model(tf.random.normal((batch_size, *image_shape)))
    print(f'outputs {x.shape}')
    for skip in skips:
        print(skip.shape)
