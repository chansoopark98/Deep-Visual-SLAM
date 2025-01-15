import tensorflow as tf

class DiodeHandler:
    def __init__(self, target_size: tuple) -> None:
        """
        Initializes the DiodeHandler class for resizing DIODE dataset images.

        Args:
            target_size (tuple): Target resolution (height, width) for resized images and depth maps.
        """
        self.target_size = target_size

    @tf.function(jit_compile=True)
    def preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Resizes and preprocesses DIODE dataset RGB and depth images.

        Args:
            rgb (tf.Tensor): Input RGB tensor of shape [H, W, 3].
            depth (tf.Tensor): Input depth tensor of shape [H, W, 1].

        Returns:
            tuple: Resized RGB and depth tensors with target size.
        """
        # Resize
        resized_rgb = tf.image.resize(rgb, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        resized_depth = tf.image.resize(depth, self.target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Type casting
        resized_rgb = tf.cast(resized_rgb, tf.uint8)
        resized_depth = tf.cast(resized_depth, tf.float32)

        # Ensure shape for stability
        resized_rgb = tf.ensure_shape(resized_rgb, [*self.target_size, 3])
        resized_depth = tf.ensure_shape(resized_depth, [*self.target_size, 1])

        return resized_rgb, resized_depth

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dummy = tf.ones((480, 640, 3))
    dummy_depth = tf.ones((480, 640, 1))
    handler = DiodeHandler(target_size=(792, 1408))
    a, b, c = handler.preprocess(dummy, dummy_depth)
    print(a.shape, b.shape)
    plt.imshow(a)
    plt.show()
    plt.imshow(b[:, :, 0], cmap='plasma')
    plt.show()