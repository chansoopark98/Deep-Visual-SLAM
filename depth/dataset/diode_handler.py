import tensorflow as tf

class DiodeHandler(object):
    def __init__(self, target_size: tuple) -> None:
        self.target_size = target_size
        
        self.original_size = (768, 1024)
        self.original_intrinsic_matrix = tf.constant([[886.81, 0., 512.],
                                             [0., 927.06, 384.],
                                             [0., 0., 1.]], dtype=tf.float32)

    @tf.function(jit_compile=True)
    def preprocess(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        rgb = tf.image.resize(rgb, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, self.target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Compute scaling factors
        # TODO
        scale_x = tf.cast(self.target_size[1] / self.original_size[1], tf.float32)  # target_width / original_width
        scale_y = tf.cast(self.target_size[0] / self.original_size[0], tf.float32)  # target_height / original_height
        new_fx = tf.cast(self.original_intrinsic_matrix[0, 0] * scale_x, tf.float32)
        new_fy = tf.cast(self.original_intrinsic_matrix[1, 1] * scale_y, tf.float32)
        new_cx = tf.cast(self.original_intrinsic_matrix[0, 2] * scale_x, tf.float32)
        new_cy = tf.cast(self.original_intrinsic_matrix[1, 2] * scale_y, tf.float32)
        adjusted_intrinsic_matrix = tf.constant([[new_fx, 0, new_cx],
												 [0, new_fy, new_cy],
												 [0, 0, 1]], dtype=tf.float32)
        
        
        # Adjust intrinsic matrix
        # adjusted_intrinsic_matrix = tf.constant([[self.original_intrinsic_matrix[0, 0] * scale_x, 0, self.original_intrinsic_matrix[0, 2] * scale_x],
        #                                          [0, self.original_intrinsic_matrix[1, 1] * scale_y, self.original_intrinsic_matrix[1, 2] * scale_y],
        #                                          [0, 0, 1]], dtype=tf.float32)

        return rgb, depth, adjusted_intrinsic_matrix

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