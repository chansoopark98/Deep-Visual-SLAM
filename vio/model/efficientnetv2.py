import tensorflow as tf

class EfficientNetV2Encoder(tf.keras.Model):
    def __init__(self, 
                 image_shape: tuple,
                 batch_size: int,
                 prefix: str = 'efficientnetv2',
                 **kwargs):
        super(EfficientNetV2Encoder, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.prefix = prefix

        self.base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=(image_shape[0], image_shape[1], 3),
            classes=0,
            classifier_activation=None,
            include_preprocessing=False
        )
        self.layer_names = [
            "block1b_add",
            "block2d_add",
            "block3d_add",
            "block5h_add",
            "block6o_add",
        ]

        outputs = [self.base_model.get_layer(name).output for name in self.layer_names]
        
        self.partial_model = tf.keras.Model(inputs=self.base_model.input, 
                                            outputs=outputs,
                                            name=f"{prefix}_partial")

    def call(self, inputs, training=False):
        """
        inputs: [B, H, W, 3]
        returns:
            x    : 가장 깊은 feature (H/32 크기)
            skips: [skip4, skip3, skip2, skip1] 형태로 반환
                   (기존 DispNet 구조가 skip[0], skip[1], ... 접근)
        """
        features = self.partial_model(inputs, training=training)
        
        x = features[-1]  # block6o_add (H/32)
        
        skips = [
            features[3],  # block5h_add (H/16)
            features[2],  # block3d_add (H/8)
            features[1],  # block2d_add (H/4)
            features[0]   # block1b_add (H/2)
        ]
        return x, skips