import tensorflow as tf
from dataset.dataset_utils import rescale_camera_intrinsic

class NyuHandler(object):
    def __init__(self, target_size: tuple) -> None:
        self.target_size = target_size
          # Original intrinsic matrix
        original_cx = 3.2558244941119034e+02
        original_cy = 2.5373616633400465e+02

        bound_left = 43
        bound_top = 45

        new_cx = original_cx - bound_left
        new_cy = original_cy - bound_top

        self.original_size = (480, 640)
        self.original_intrinsic_matrix = tf.constant([[5.1885790117450188e+02, 0., new_cx],
                                             [0., 5.1946961112127485e+02, new_cy],
                                             [0., 0., 1.]], dtype=tf.float32)
        

    @tf.function(jit_compile=True)
    def nyu_crop_resize(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
        """
        1) 유효 영역을 crop
        2) 16:9 해상도(self.image_size)에 맞춰 letterbox (padding) 적용
        """
        bound_left = 43
        bound_right = 608
        bound_top = 45
        bound_bottom = 472

        # Crop
        cropped_image = rgb[bound_top:bound_bottom, bound_left:bound_right, :]
        cropped_depth = depth[bound_top:bound_bottom, bound_left:bound_right, :]

        # Resize
        cropped_image = tf.image.resize(cropped_image, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        cropped_depth = tf.image.resize(cropped_depth, self.target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        intrinsic = rescale_camera_intrinsic(self.original_intrinsic_matrix,
                                             self.original_size,
                                             self.target_size)
        # 타입 변환
        cropped_image = tf.cast(cropped_image, tf.uint8)
        cropped_depth = tf.cast(cropped_depth, tf.float32)

        # shape 보장
        cropped_image = tf.ensure_shape(cropped_image, [None, None, 3])
        cropped_depth = tf.ensure_shape(cropped_depth, [None, None, 1])
        
        return cropped_image, cropped_depth, intrinsic

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