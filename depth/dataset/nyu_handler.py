import tensorflow as tf

class NyuHandler:
    def __init__(self) -> None:
        """
        Initializes the NyuHandler class for cropping NYU dataset images.
        """
        # Cropping boundaries for NYU dataset
        self.bound_left = 43
        self.bound_right = 608
        self.bound_top = 45
        self.bound_bottom = 472

    @tf.function(jit_compile=True)
    def nyu_crop_resize(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Crops the valid region of the NYU dataset without resizing.

        Args:
            rgb (tf.Tensor): Input RGB tensor of shape [480, 640, 3].
            depth (tf.Tensor): Input depth tensor of shape [480, 640, 1].

        Returns:
            tuple: Cropped RGB and depth tensors.
        """
        # Crop the valid region
        cropped_image = rgb[self.bound_top:self.bound_bottom, self.bound_left:self.bound_right, :]
        cropped_depth = depth[self.bound_top:self.bound_bottom, self.bound_left:self.bound_right, :]

        # Type casting
        cropped_image = tf.cast(cropped_image, tf.uint8)
        cropped_depth = tf.cast(cropped_depth, tf.float32)

        # Ensure shapes for compatibility
        cropped_image = tf.ensure_shape(cropped_image, [self.bound_bottom - self.bound_top, 
                                                        self.bound_right - self.bound_left, 3])
        cropped_depth = tf.ensure_shape(cropped_depth, [self.bound_bottom - self.bound_top, 
                                                        self.bound_right - self.bound_left, 1])

        return cropped_image, cropped_depth

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dummy = tf.ones((480, 640, 3))
    dummy_depth = tf.ones((480, 640, 1))
    handler = NyuHandler(target_size=(792, 1408))
    a, b = handler.nyu_crop_resize(dummy, dummy_depth)
    print(a.shape, b.shape)
    plt.imshow(a)
    plt.show()
    plt.imshow(b[:, :, 0], cmap='plasma')
    plt.show()