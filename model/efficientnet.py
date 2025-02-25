import tensorflow as tf, tf_keras
from .efficientnet_lite import efficientnet_lite_b0

eff_lite_b0_layers = [
        "block1a_project_bn", # 1/2@ 16
        "block2b_add", # 1/4 @ 24
        "block3b_add", # 1/8 @ 40
        "block5c_add", # 1/16@ 112
        "block7a_project_bn" # 1/32@ 320
]

class EfficientNet:
    def __init__(self, image_shape, batch_size, pretrained=True, return_skips=True, prefix='base'):
        if len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple of (height, width, channels)")

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.return_skips = return_skips
        self.prefix = prefix

    def build_model(self) -> tf_keras.Model:
        inputs = tf_keras.Input(shape=self.image_shape)

        if self.pretrained:        
            weights = 'imagenet'
        else:
            weights = None
        base_model = efficientnet_lite_b0(input_shape=self.image_shape,
                                      include_top=False, 
                                      weights=weights, 
                                      classes=0)

        base_model.summary() 
        outputs = [base_model.get_layer(name).output for name in eff_lite_b0_layers]
        partial_model = tf_keras.Model(inputs=base_model.inputs, outputs=outputs, name=f"{self.prefix}_partial")

        for layer in partial_model.layers:
            layer._name = f"{self.prefix}_{layer.name}"        

        inputs = tf_keras.Input(shape=self.image_shape, name="input_image")
        features = partial_model(inputs)

        x = features[-1]  # block6o_add (H/32)

        skips = [
            features[3],
            features[2],
            features[1],
            features[0],
        ]
        if self.return_skips:
            return tf_keras.Model(inputs=inputs, outputs=[x, skips], name=f"{self.prefix}_model")
        else:
            return tf_keras.Model(inputs=inputs, outputs=x, name=f"{self.prefix}_model")
    
if __name__ == '__main__':
    image_shape = (480, 640, 6)
    batch_size = 4
    model_builder = EfficientNet(image_shape=image_shape, batch_size=batch_size,
                                 pretrained=True, return_skips=True, prefix='efficientnet')
    model = model_builder.build_model()
    test = model(tf.random.normal((batch_size, *image_shape)))
    print(f'outputs {len(test)}')
