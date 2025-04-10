import tensorflow as tf

class DimlHandler:
    def __init__(self, image_size: tuple) -> None:
        """
        Initializes the DimlHandler class for resizing DIML dataset images.

        Args:
            image_size (tuple): Target resolution (height, width) for resized images and depth maps.
        """
        # Define Kinect v2 camera intrinsic parameters (resolution: 480x640)
        self.kinect_intrinsic = {
            'fx': 366.0,
            'fy': 366.0,
            'cx': 320.0,
            'cy': 240.0,
            'width': 640,
            'height': 480
        }
        
        # Create intrinsic matrix as tensor
        self.K = tf.constant([
            [self.kinect_intrinsic['fx'], 0, self.kinect_intrinsic['cx']],
            [0, self.kinect_intrinsic['fy'], self.kinect_intrinsic['cy']],
            [0, 0, 1]
        ], dtype=tf.float32)
        self.image_size = image_size

    @tf.function(jit_compile=True)
    def preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        Resizes and preprocesses DIML dataset RGB and depth images.

        Args:
            rgb (tf.Tensor): Input RGB tensor of shape [H, W, 3].
            depth (tf.Tensor): Input depth tensor of shape [H, W, 1].

        Returns:
            tuple: Resized RGB and depth tensors with target size.
        """
        # Resize RGB and depth
        resized_rgb = self._resize_image(rgb, method=tf.image.ResizeMethod.BILINEAR)
        resized_depth = self._resize_image(depth, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Ensure correct data types
        resized_rgb = tf.cast(resized_rgb, tf.uint8)
        resized_depth = tf.cast(resized_depth, tf.float32)

        # Ensure output shape
        resized_rgb = tf.ensure_shape(resized_rgb, [*self.image_size, 3])
        resized_depth = tf.ensure_shape(resized_depth, [*self.image_size, 1])

        return resized_rgb, resized_depth, self.K

    def _resize_image(self, image: tf.Tensor, method: tf.image.ResizeMethod) -> tf.Tensor:
        """
        Resizes the given image to the target size using the specified method.

        Args:
            image (tf.Tensor): Input tensor to resize.
            method (tf.image.ResizeMethod): Method for resizing (e.g., bilinear, nearest).

        Returns:
            tf.Tensor: Resized image tensor.
        """
        return tf.image.resize(image, self.image_size, method=method)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dummy = tf.ones((480, 640, 3))
    dummy_depth = tf.ones((480, 640, 1))
    handler = DimlHandler(image_size=(792, 1408))
    a, b, c = handler.preprocess(dummy, dummy_depth)
    print(a.shape, b.shape)
    plt.imshow(a)
    plt.show()
    plt.imshow(b[:, :, 0], cmap='plasma')
    plt.show()