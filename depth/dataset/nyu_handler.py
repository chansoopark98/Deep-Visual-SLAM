import tensorflow as tf

class NyuHandler(object):
    def __init__(self, image_size: tuple) -> None:
        self.image_size = image_size

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

        # 1. Crop
        cropped_image = rgb[bound_top:bound_bottom, bound_left:bound_right, :]
        cropped_depth = depth[bound_top:bound_bottom, bound_left:bound_right, :]

        # 2. resize_with_pad을 이용해 16:9 사이즈 만들기
        target_h, target_w = self.image_size  # 예: (360, 640)
        padded_image = tf.image.resize_with_pad(
            cropped_image,
            target_height=target_h,
            target_width=target_w,
            method=tf.image.ResizeMethod.BILINEAR,
        )
        padded_depth = tf.image.resize_with_pad(
            cropped_depth,
            target_height=target_h,
            target_width=target_w,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

        # 타입 변환
        padded_image = tf.cast(padded_image, tf.uint8)
        padded_depth = tf.cast(padded_depth, tf.float32)

        # shape 보장
        padded_image = tf.ensure_shape(padded_image, [None, None, 3])
        padded_depth = tf.ensure_shape(padded_depth, [None, None, 1])
        
        return padded_image, padded_depth

    # @tf.function
    # def manual_letterbox(self, image, target_h, target_w, resize_method=tf.image.ResizeMethod.BILINEAR):
    #     """원본 aspect ratio 유지, 남는 공간은 0으로 패딩"""
    #     # 현재 이미지 크기
    #     orig_h = tf.shape(image)[0]
    #     orig_w = tf.shape(image)[1]
    #     orig_aspect = tf.cast(orig_w, tf.float32) / tf.cast(orig_h, tf.float32)
    #     target_aspect = tf.cast(target_w, tf.float32) / tf.cast(target_h, tf.float32)

    #     # 원본과 목표의 가로세로 비율 비교
    #     def scale_by_width():
    #         # 가로를 target에 맞추고, 세로는 비율에 따라
    #         new_w = target_w
    #         new_h = tf.cast(tf.math.round(tf.cast(new_w, tf.float32)/orig_aspect), tf.int32)
    #         return new_h, new_w

    #     def scale_by_height():
    #         # 세로를 target에 맞추고, 가로는 비율에 따라
    #         new_h = target_h
    #         new_w = tf.cast(tf.math.round(tf.cast(new_h, tf.float32)*orig_aspect), tf.int32)
    #         return new_h, new_w

    #     new_h, new_w = tf.cond(orig_aspect > target_aspect, scale_by_width, scale_by_height)
        
    #     # 먼저 resize
    #     resized = tf.image.resize(image, (new_h, new_w), method=resize_method)

    #     # pad 값 계산
    #     pad_top = (target_h - new_h) // 2
    #     pad_bottom = target_h - new_h - pad_top
    #     pad_left = (target_w - new_w) // 2
    #     pad_right = target_w - new_w - pad_left

    #     # [ [top, bottom], [left, right], [channel, channel] ]
    #     padded = tf.pad(resized, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT", constant_values=0)

    #     return padded
        
    # @tf.function(jit_compile=True)
    # def nyu_crop_resize(self, rgb: tf.Tensor, depth: tf.Tensor) -> tuple:
    #     bound_left = 43
    #     bound_right = 608
    #     bound_top = 45
    #     bound_bottom = 472
    #     target_size = (480, 640)

    #     # crop
    #     cropped_image = rgb[bound_top:bound_bottom, bound_left:bound_right, :]
    #     cropped_depth = depth[bound_top:bound_bottom, bound_left:bound_right, :]

    #     # letterbox (16:9)
    #     cropped_image = self.manual_letterbox(cropped_image, target_size[0], target_size[1], resize_method=tf.image.ResizeMethod.BILINEAR)
    #     cropped_depth = self.manual_letterbox(cropped_depth, target_size[0], target_size[1], resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #     return tf.cast(cropped_image, tf.uint8), tf.cast(cropped_depth, tf.float32)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dummy = tf.ones((480, 640, 3))
    dummy_depth = tf.ones((480, 640, 1))
    handler = NyuHandler(image_size=(792, 1408))
    a, b = handler.nyu_crop_resize(dummy, dummy_depth)
    print(a.shape, b.shape)
    plt.imshow(a)
    plt.show()
    plt.imshow(b[:, :, 0], cmap='plasma')
    plt.show()