import tensorflow as tf

class Augmentations:
    def __init__(self, image_size: tuple):
        if not isinstance(image_size, tuple):
            raise TypeError("image_size must be a tuple")
        if len(image_size) != 2:
            raise ValueError("image_size must have two elements")
        self.image_size = image_size

    @tf.function(jit_compile=True)
    def imu_left_right_flip(self, imu):
        """
        Args:
            image: (N, H, W, 3) shape, Tensor
            imu: (N, Seq, 6) shape, Tensor
                 imu[:, :, 0] = gx
                 imu[:, :, 1] = gy
                 imu[:, :, 2] = gz
                 imu[:, :, 3] = ax
                 imu[:, :, 4] = ay
                 imu[:, :, 5] = az

        Returns:
            image_flipped: 좌우 반전된 이미지 (N, H, W, 3)
            imu_flipped: IMU 데이터 (N, Seq, 6)
                         (gx, -gy, -gz, -ax, ay, az)
        """
        # 2) IMU 데이터 변환
        # (gx, gy, gz, ax, ay, az) -> (gx, -gy, -gz, -ax, ay, az)
        gx, gy, gz, ax, ay, az = tf.unstack(imu, axis=-1)  # shape: (Seq,)

        gy = -gy
        gz = -gz
        ax = -ax

        imu_flipped = tf.stack([gx, gy, gz, ax, ay, az], axis=-1)

        return imu_flipped
    
    @tf.function(jit_compile=True)
    def image_left_right_flip(self, image):
        """
        Args:
            image: (N, H, W, 3) shape, Tensor

        Returns:
            image_flipped: 좌우 반전된 이미지 (N, H, W, 3)
        """
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped
    
    @tf.function(jit_compile=True)
    def brightness(self, image, delta):
        """
        Args:
            image: (N, H, W, 3) shape, Tensor
            delta: 밝기 조절 값
        
        Returns:
            image_brightened: 밝기 조절된 이미지 (N, H, W, 3)
        """
        image_brightened = tf.image.adjust_brightness(image, delta)
        return image_brightened