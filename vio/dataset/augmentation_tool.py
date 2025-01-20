import tensorflow as tf

class Augmentations:
    def __init__(self, image_size: tuple):
        if not isinstance(image_size, tuple):
            raise TypeError("image_size must be a tuple")
        if len(image_size) != 2:
            raise ValueError("image_size must have two elements")
        self.image_size = image_size

    @tf.function(jit_compile=True)
    def left_right_flip(self, image, imu):
        """
        Args:
            image: (H, W, 3) shape, Tensor
            imu: (Seq, 6) shape, Tensor
                 imu[:, 0] = gx
                 imu[:, 1] = gy
                 imu[:, 2] = gz
                 imu[:, 3] = ax
                 imu[:, 4] = ay
                 imu[:, 5] = az

        Returns:
            image_flipped: 좌우 반전된 이미지 (H, W, 3)
            imu_flipped: IMU 데이터 (Seq, 6)
                         (gx, -gy, -gz, -ax, ay, az)
        """
        # 1) 이미지 좌우 반전
        image_flipped = tf.image.flip_left_right(image)

        # 2) IMU 데이터 변환
        # (gx, gy, gz, ax, ay, az) -> (gx, -gy, -gz, -ax, ay, az)
        gx, gy, gz, ax, ay, az = tf.unstack(imu, axis=-1)  # shape: (Seq,)

        gy = -gy
        gz = -gz
        ax = -ax

        imu_flipped = tf.stack([gx, gy, gz, ax, ay, az], axis=-1)

        return image_flipped, imu_flipped
        