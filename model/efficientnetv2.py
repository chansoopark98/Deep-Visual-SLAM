import tensorflow as tf

def get_efficientv2b0(image_shape):
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=image_shape,
        classes=0,
        classifier_activation=None,
        include_preprocessing=False
    )
    base_model.summary()
    layer_names = [
        "block6h_add",
        "block5e_add",
        "block3b_add",
        "block2b_add",
        "block1a_project_activation",
    ]

    outputs = [base_model.get_layer(name).output for name in layer_names]

    partial_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name=f"efficientnetv2b0_partial"
    )
    return partial_model

def get_efficientv2s(image_shape):
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=image_shape,
        classes=0,
        classifier_activation=None,
        include_preprocessing=False
    )
    layer_names = [
        "block6o_add",
        "block5h_add",
        "block3d_add",
        "block2d_add",
        "block1b_add",
    ]

    outputs = [base_model.get_layer(name).output for name in layer_names]

    partial_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name=f"efficientnetv2s_partial"
    )
    return partial_model

class EfficientNetV2Encoder:
    def __init__(self, image_shape, model_type, batch_size, prefix='efficientnetv2'):
        """
        Initializes the EfficientNetV2Encoder class.

        Args:
            image_shape (tuple): Input image shape (height, width, channels).
            batch_size (int): Batch size (not used directly in the model, but for reference).
            prefix (str): Prefix for the model name.
        """
        if len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple of (height, width, channels)")

        self.image_shape = image_shape
        self.model_type = model_type
        self.batch_size = batch_size
        self.prefix = prefix

    def build_model(self) -> tf.keras.Model:

        if self.model_type == 's':
            effnet = get_efficientv2s(self.image_shape)

        elif self.model_type == 'b0':
            effnet = get_efficientv2b0(self.image_shape)

        inputs = tf.keras.Input(shape=self.image_shape, name="input_image")
        features = effnet(inputs)

        x = features[0]  # (H/32)

        skips = [
            features[1],  # (H/16)
            features[2],  # (H/8)
            features[3],  # (H/4)
            features[4]   # (H/2)
        ]
        return tf.keras.Model(inputs=inputs, outputs=[x, skips], name=f"{self.prefix}_model")
    
if __name__ == '__main__':
    image_shape = (224, 224, 3)
    batch_size = 32
    model = EfficientNetV2Encoder(image_shape, 'b0', batch_size)
    model.build_model().summary()