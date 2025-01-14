import tensorflow as tf 
try:
    from raft import RAFT
except:
    from .raft import RAFT

class CustomRAFT:
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
        base_model = RAFT()
        model_input_shape = [(self.batch_size, self.image_shape[0], self.image_shape[1], 3),
                             (self.batch_size, self.image_shape[0], self.image_shape[1], 3)]
        base_model.build(model_input_shape)
        base_model.summary()
        if self.pretrained:
            base_model.load_weights('./assets/weights/raft/model')

        inputs = tf.keras.Input(shape=(self.image_shape[0], self.image_shape[1], 6),
                                batch_size=self.batch_size)
        left = inputs[:, :, :, :3]
        right = inputs[:, :, :, 3:]

        outputs = base_model.get_layer('basic_encoder')([left, right])  # Connect the layer
        new_model = tf.keras.Model(inputs, outputs, name=f'{self.prefix}_partial')
        return new_model
    
if __name__ == '__main__':
    test_model = CustomRAFT(image_shape=(256, 256, 3), batch_size=4, pretrained=True, prefix='raft')
    model = test_model.build_model()
    model.summary()