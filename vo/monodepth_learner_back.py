import tensorflow as tf, tf_keras
from utils.projection_utils import projective_inverse_warp

class Learner(object):
    def __init__(self,
                 depth_model: tf_keras.Model,
                 pose_model: tf_keras.Model,
                 config: dict):
        self.depth_net = depth_model
        self.pose_net = pose_model
        self.config = config

        # 예시 하이퍼파라미터
        self.num_scales = 4
        self.num_source = self.config['Train']['num_source'] # 2
        
        self.image_shape = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.smoothness_ratio = self.config['Train']['smoothness_ratio'] # 0.001
        self.auto_mask = self.config['Train']['auto_mask'] # True
        self.ssim_ratio = self.config['Train']['ssim_ratio'] # 0.85
        self.min_depth = self.config['Train']['min_depth'] # 0.1
        self.max_depth = self.config['Train']['max_depth'] # 10.0

        if self.config['Train']['mode'] in ['axisAngle', 'euler']:
            self.pose_mode = self.config['Train']['mode']
            if self.pose_mode == 'axisAngle':
                self.is_euler = False
            else:
                self.is_euler = True
        else:
            raise ValueError('Invalid pose mode')

    @tf.function() # ok
    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1. / max_depth
        max_disp = 1. / min_depth
        scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast(max_disp - min_disp, tf.float32) * disp
        depth = tf.cast(1., tf.float32) / scaled_disp
        return depth
    
    def compute_reprojection_loss(self, reproj_image, tgt_image):
        """
        L1 + SSIM photometric loss
        """
        l1_loss = tf.reduce_mean(tf.abs(reproj_image - tgt_image), axis=3, keepdims=True)
        ssim_loss = tf.reduce_mean(self.ssim(reproj_image, tgt_image), axis=3, keepdims=True)

        loss = self.ssim_ratio * ssim_loss + (1. - self.ssim_ratio) * l1_loss
        return loss

    def ssim(self, x, y):
        # 현재 코드 동일
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode='REFLECT')
        y = tf.pad(y, [[0,0],[1,1],[1,1],[0,0]], mode='REFLECT')

        mu_x = tf.nn.avg_pool2d(x, ksize=3, strides=1, padding='VALID')
        mu_y = tf.nn.avg_pool2d(y, ksize=3, strides=1, padding='VALID')

        sigma_x  = tf.nn.avg_pool2d(x**2, ksize=3, strides=1, padding='VALID') - mu_x**2
        sigma_y  = tf.nn.avg_pool2d(y**2, ksize=3, strides=1, padding='VALID') - mu_y**2
        sigma_xy = tf.nn.avg_pool2d(x*y,  ksize=3, strides=1, padding='VALID') - mu_x*mu_y

        C1 = 0.01**2
        C2 = 0.03**2

        SSIM_n = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
        SSIM_d = (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
        SSIM_raw = SSIM_n / (SSIM_d + 1e-10)  # +1e-10 to avoid /0

        SSIM_loss = tf.clip_by_value((1.0 - SSIM_raw)*0.5, 0.0, 1.0)
        return SSIM_loss

    def get_smooth_loss(self, disp, img):
        disp = tf.cast(disp, tf.float32)
        img = tf.cast(img, tf.float32)
        """
        Edge-aware smoothness: disp gradients * exp(-|img grads|)
        """
        disp_mean = tf.reduce_mean(disp, axis=[1,2], keepdims=True) + 1e-7
        norm_disp = disp / disp_mean

        disp_dx = tf.abs(norm_disp[:, 1:, :, :] - norm_disp[:, :-1, :, :])
        disp_dy = tf.abs(norm_disp[:, :, 1:, :] - norm_disp[:, :, :-1, :])

        img_dx = tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :])
        img_dy = tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

        weight_x = tf.exp(-tf.reduce_mean(img_dx, axis=3, keepdims=True))
        weight_y = tf.exp(-tf.reduce_mean(img_dy, axis=3, keepdims=True))

        smoothness_x = disp_dx * weight_x
        smoothness_y = disp_dy * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
    
    @tf.function(jit_compile=True)
    def rescale_intrinsics(self, intrinsics, original_height, original_width, target_height, target_width):
        """
        intrinsics(tf.Tensor): [B, 3, 3]
        """
        # float 변환
        orig_h = tf.cast(original_height, tf.float32)
        orig_w = tf.cast(original_width,  tf.float32)
        tgt_h  = tf.cast(target_height,   tf.float32)
        tgt_w  = tf.cast(target_width,    tf.float32)

        # 세로/가로 스케일 비율
        h_scale = tgt_h / orig_h
        w_scale = tgt_w / orig_w

        # 기존 fx, fy, cx, cy 추출
        fx = intrinsics[:, 0, 0]  # (B,)
        fy = intrinsics[:, 1, 1]  # (B,)
        cx = intrinsics[:, 0, 2]  # (B,)
        cy = intrinsics[:, 1, 2]  # (B,)

        # 새로운 스케일 반영
        fx_new = fx * w_scale
        fy_new = fy * h_scale
        cx_new = cx * w_scale
        cy_new = cy * h_scale

        # skew는 제외(혹은 0으로 둔다고 가정)
        # 보통 pinhole 모델에서는 skew가 0이므로, 여기서는 0으로 세팅
        skew_0 = tf.zeros_like(fx_new)  # (B,)
        skew_1 = tf.zeros_like(fy_new)  # (B,)

        # batch 차원만큼 3x3을 다시 구성
        row0 = tf.stack([fx_new, skew_0, cx_new], axis=1)  # (B, 3)
        row1 = tf.stack([skew_1, fy_new, cy_new], axis=1)  # (B, 3)
        # 마지막 행은 [0, 0, 1]
        # -> batch 크기에 맞춰 tile 또는 stack
        # shape: (B, 3)
        row2 = tf.tile(tf.constant([[0., 0., 1.]], tf.float32), [tf.shape(intrinsics)[0], 1])

        # 최종 (B, 3, 3)으로 스택
        intrinsics_rescaled = tf.stack([row0, row1, row2], axis=1)
        return intrinsics_rescaled
        
    def forward_step(self, ref_images, tgt_image, intrinsic, training=True) -> tf.Tensor:
   
        left_images = ref_images[:, :self.num_source] # [B, num_source, H, W, 3]
        right_images = ref_images[:, self.num_source:] # [B, num_source, H, W, 3]

        pixel_losses = 0.
        smooth_losses = 0.
        
        H = tf.shape(tgt_image)[1]
        W = tf.shape(tgt_image)[2]

        # Generate Depth and Pose Results
        pred_depths = []
        pred_poses = []
        scaled_tgts = []

        disp_raw = self.depth_net(tgt_image, training=training) # disp raw result (H, W, 1)

        for scale_idx in range(self.num_scales):
            resized_disp = tf.image.resize(disp_raw[scale_idx], [H, W], method=tf.image.ResizeMethod.BILINEAR)
            resized_depth = self.disp_to_depth(resized_disp, self.min_depth, self.max_depth)
            pred_depths.append(resized_depth)

            tgt_scaled = tf.image.resize(tgt_image,
                                         [H // (2**scale_idx), W //
                                          (2**scale_idx)],
                                         method=tf.image.ResizeMethod.BILINEAR)
            scaled_tgts.append(tgt_scaled)
        
        for src_idx in range(self.num_source):
            left_image = left_images[:, src_idx]  # [B, H, W, 3]
            right_image = right_images[:, src_idx]  # [B, H, W, 3]

            cat_left = tf.concat([left_image, tgt_image], axis=3)   # [B,H,W,6]
            cat_right = tf.concat([tgt_image, right_image], axis=3) # [B,H,W,6]

            # no use imu
            pose_left = self.pose_net(cat_left, training=training)    # [B,6]
            pose_right = self.pose_net(cat_right, training=training)  # [B,6]

            pose_left = tf.cast(pose_left, tf.float32)
            pose_right = tf.cast(pose_right, tf.float32)
            
            pred_poses.append([pose_left, pose_right])

        for s in range(self.num_scales):
            # reprojection loss
            reprojection_list = []

            for i in range(self.num_source):
                left_image = left_images[:, i]  # [B, H, W, 3]
                right_image = right_images[:, i]  # [B, H, W, 3]
                src_image_stack = [left_image, right_image]

                current_poses = pred_poses[i]  # shape [B, 2, 6]

                # left-right and right-left
                for j in range(2): # j=0: left, j=1: right
                    curr_depth = pred_depths[s]
                    curr_src = src_image_stack[j]
                    curr_pose = current_poses[j] # shape [B,6]

                    curr_proj_image = projective_inverse_warp(
                        curr_src,
                        tf.squeeze(curr_depth, axis=3),
                        curr_pose,
                        intrinsics=intrinsic,
                        invert=(j == 0),
                        euler=self.is_euler
                    )
                    # photometric
                    curr_reproj_loss = self.compute_reprojection_loss(curr_proj_image, tgt_image)

                    # for loss
                    reprojection_list.append(curr_reproj_loss)

            # shape => [B, H/(2^s), W/(2^s), num_source]
            reprojection_losses = tf.concat(reprojection_list, axis=3)

            # 3.3) auto_mask
            combined = reprojection_losses
            if self.auto_mask:
                identity_list = []
                for i in range(self.num_source):
                    left_image = left_images[:, i]  # [B, H, W, 3]
                    right_image = right_images[:, i]  # [B, H, W, 3]
                    src_image_stack = [left_image, right_image]

                    for j in range(2):
                        # identity reprojection => src==tgt scaled
                        identity_loss = self.compute_reprojection_loss(
                            src_image_stack[j], tgt_image)
                        identity_list.append(identity_loss)
                identity_losses = tf.concat(identity_list, axis=3)
                # random noise
                identity_losses += tf.random.normal(tf.shape(identity_losses), stddev=1e-5)

                combined = tf.concat([identity_losses, reprojection_losses], axis=3)

            # min across channel => pick best
            reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))

            # smoothness loss
            smooth_loss = self.get_smooth_loss(disp_raw[s], scaled_tgts[s])
            smooth_loss = smooth_loss / (2.0**s)

            pixel_losses += reprojection_loss
            smooth_losses += smooth_loss * self.smoothness_ratio
            
        # total loss
        num_scales_f = tf.cast(self.num_scales, tf.float32)
        pixel_losses = pixel_losses / num_scales_f
        smooth_losses = smooth_losses / num_scales_f
        total_loss = pixel_losses + smooth_losses

        return total_loss, pixel_losses, smooth_losses, pred_depths