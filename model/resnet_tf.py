import tensorflow as tf
try:
    from .resnet_original import resnet_18
except:
    from resnet_original import resnet_18

class Resnet:
    def __init__(self, image_shape, batch_size, pretrained=True, prefix='base'):
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
        self.pretrained = pretrained
        self.prefix = prefix

    def build_model(self) -> tf.keras.Model:
        """
        Builds a MobileNetV3Large-based functional model with skip connections.

        Returns:
            tf.keras.Model: Functional model.
        """
        inputs = tf.keras.Input(shape=self.image_shape)
        outputs = resnet_18(inputs=inputs)
        base_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self.pretrained:
            pretrained_weights = './assets/weights/resnet18.h5'
            base_model.load_weights(pretrained_weights)
        
        base_model.summary()

        layer_names = [
            "relu",
            "activation_3",
            "activation_7",
            "activation_11",
            "activation_15",
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
    model_builder = Resnet(image_shape=image_shape, batch_size=batch_size)
    model = model_builder.build_model()
    model.summary()
    # model.save(f'./assets/weights/backbone_resnet18.h5')
    tf.keras.models.save_model(model, 'assets/weigths/backbone_resnet18.h5')
    encoder = tf.keras.models.load_model('assets/weigths/backbone_resnet18.h5')
    encoder.summary()
