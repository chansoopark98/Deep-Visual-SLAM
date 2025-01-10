import tensorflow as tf
from dataset.dataset_utils import rescale_camera_intrinsic

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
        intrinsic = rescale_camera_intrinsic(self.original_intrinsic_matrix,
                                             self.original_size,
												self.target_size)
        rgb = tf.cast(rgb, tf.uint8)
        depth = tf.cast(depth, tf.float32)
        
        rgb = tf.ensure_shape(rgb, [*self.target_size, 3])
        depth = tf.ensure_shape(depth, [*self.target_size, 1])

        return rgb, depth, intrinsic

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