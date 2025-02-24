import tensorflow as tf

class ConvNext:
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
        input_tensor = tf.keras.Input(shape=self.image_shape, batch_size=self.batch_size)
        base_model = tf.keras.applications.ConvNeXtTiny(
                                           include_top=False,
                                           weights='imagenet', 
                                           
                                           input_tensor=input_tensor,
                                             classes=0, include_preprocessing=False)
        base_model.summary()
        
        """
        tf.__operators__.add_2 # 1/4 96
        tf.__operators__.add_5 # 1/8 192
        tf.__operators__.add_14 # 1/16 384
        layer_normalization # 1/32 768
        """
        for layer in base_model.layers:
            layer._name = f"{self.prefix}_{layer.name}"        

        return base_model
    
if __name__ == '__main__':
    image_shape = (480, 640, 3)
    batch_size = 4
    model_builder = ConvNext(image_shape=image_shape, batch_size=batch_size, pretrained='base')
    model = model_builder.build_model()