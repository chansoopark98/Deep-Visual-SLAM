import tensorflow as tf
try:
    from .resnet_original import resnet_18
except:
    from resnet_original import resnet_18

class Resnet:
    def __init__(self, image_shape, batch_size, pretrained=True, prefix='base'):
        """
        Initializes the Resnet18 class.

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
        Builds a Resnet-based functional model with skip connections.

        Returns:
            tf.keras.Model: Functional model.
        """
        inputs = tf.keras.Input(shape=self.image_shape)
        outputs = resnet_18(inputs=inputs, build_partial=True) # x, [skip4, skip3, skip2, skip1]
        base_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{self.prefix}_resnet18")

        if self.pretrained:
            pretrained_weights = './assets/weights/resnet18.h5'
            base_model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
        
        for layer in base_model.layers:
            layer._name = f"{self.prefix}_{layer.name}"        

        return base_model

if __name__ == '__main__':
    image_shape = (480, 640, 6)
    batch_size = 4
    model_builder = Resnet(image_shape=image_shape, batch_size=batch_size)
    model = model_builder.build_model()
    # model.summary()
    # model.save(f'./assets/weights/backbone_resnet18.h5')
    # tf.keras.models.save_model(model, 'assets/weigths/backbone_resnet18.h5')
    # encoder = tf.keras.models.load_model('assets/weigths/backbone_resnet18.h5')
    # encoder.summary()
    # model.summary()
    test = model(tf.random.normal((batch_size, *image_shape)))
    print(f'outputs {len(test)}')
