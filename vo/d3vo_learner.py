import tensorflow as tf
from tensorflow import keras
from utils.d3vo_projection_utils import projective_inverse_warp

class Learner(object):
    def __init__(self,
                 depth_model: keras.Model,
                 pose_model: keras.Model,
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
        self.ab_ratio = self.config['Train']['ab_ratio'] # 0.01
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
    
    def compute_reprojection_loss(self, reproj_image, tgt_image, sigma=None):
        """
        L1 + SSIM photometric loss with uncertainty weighting
        """
        l1_loss = tf.reduce_mean(tf.abs(reproj_image - tgt_image), axis=3, keepdims=True)
        ssim_loss = tf.reduce_mean(self.ssim(reproj_image, tgt_image), axis=3, keepdims=True)

        loss = self.ssim_ratio * ssim_loss + (1. - self.ssim_ratio) * l1_loss
        
        if sigma is not None:
            # Match PyTorch implementation - multiply by sigma
            loss = loss * sigma
        
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
        left_image = ref_images[0] # [B, H, W, 3]
        right_image = ref_images[1] # [B, H, W, 3]

        pixel_losses = 0.
        smooth_losses = 0.
        uncertainty_losses = 0.
        
        H = tf.shape(tgt_image)[1]
        W = tf.shape(tgt_image)[2]

        # Generate Depth and Pose Results
        pred_depths = []
        pred_sigmas = []
        pred_poses = []
        scaled_tgts = []

        disp_raw, sigma_raw = self.depth_net(tgt_image, training=training) # disp and sigma results

        for scale_idx in range(self.num_scales):
            scaled_disp = disp_raw[scale_idx]
            scaled_sigma = sigma_raw[scale_idx]
            scaled_depth = self.disp_to_depth(scaled_disp, self.min_depth, self.max_depth)
            
            pred_depths.append(scaled_depth)
            pred_sigmas.append(scaled_sigma)

            tgt_scaled = tf.image.resize(tgt_image,
                                       [H // (2**scale_idx), W // (2**scale_idx)],
                                       method=tf.image.ResizeMethod.BILINEAR)
            scaled_tgts.append(tgt_scaled)
        
        # Predict Poses with brightness correction parameters
        concat_left_tgt = tf.concat([left_image, tgt_image], axis=3)   # [B,H,W,6]
        concat_tgt_right = tf.concat([tgt_image, right_image], axis=3) # [B,H,W,6]

        pose_left, left_a, left_b = self.pose_net(concat_left_tgt, training=training)    # [B,6], [B,1], [B,1]
        pose_right, right_a, right_b = self.pose_net(concat_tgt_right, training=training)  # [B,6], [B,1], [B,1]

        pose_left = tf.cast(pose_left, tf.float32)
        pose_right = tf.cast(pose_right, tf.float32)
        
        pred_poses = [pose_left, pose_right]
        pred_as = [left_a, right_a]
        pred_bs = [left_b, right_b]

        for s in range(self.num_scales):
            reprojection_list = []
            identity_reprojection_list = []
            ab_losses = []

            for i in range(2): # left, right
                curr_depth = pred_depths[s]
                curr_sigma = pred_sigmas[s] 
                curr_src = ref_images[i]
                curr_pose = pred_poses[i]
                curr_a = pred_as[i]
                curr_b = pred_bs[i]

                # Scale adjustment for different scales
                h_s = H // (2**s)
                w_s = W // (2**s)
                
                if s != 0:
                    curr_depth = tf.image.resize(curr_depth, [h_s, w_s],
                                                 method=tf.image.ResizeMethod.BILINEAR)
                    curr_sigma = tf.image.resize(curr_sigma, [h_s, w_s],
                                                 method=tf.image.ResizeMethod.BILINEAR)
                    curr_src = tf.image.resize(curr_src, [h_s, w_s],
                                               method=tf.image.ResizeMethod.BILINEAR)
                    scaled_intrinsic = self.rescale_intrinsics(intrinsic, H, W, h_s, w_s)
                else:
                    scaled_intrinsic = intrinsic

                curr_proj_image = projective_inverse_warp(
                    curr_src,
                    tf.squeeze(curr_depth, axis=3),
                    curr_pose,
                    intrinsics=scaled_intrinsic,
                    invert=(i == 0),
                    euler=self.is_euler
                )

                # Apply brightness correction
                curr_a_expanded = tf.expand_dims(tf.expand_dims(curr_a, 1), 1)  # [B, 1, 1, 1]
                curr_b_expanded = tf.expand_dims(tf.expand_dims(curr_b, 1), 1)  # [B, 1, 1, 1]

                if i == 0:
                    a_inv = 1.0 / curr_a_expanded
                    b_inv = -curr_b_expanded / curr_a_expanded
                    curr_a_expanded = a_inv
                    curr_b_expanded = b_inv
          
                curr_proj_image = curr_proj_image * curr_a_expanded + curr_b_expanded
                
                
                # Photometric loss with uncertainty
                curr_reproj_loss = self.compute_reprojection_loss(curr_proj_image, scaled_tgts[s], curr_sigma)
                reprojection_list.append(curr_reproj_loss)

                # ab loss
                ab_losses.append((curr_a_expanded - 1) ** 2 + curr_b_expanded ** 2)
        
                if self.auto_mask:
                    scaled_src = tf.image.resize(curr_src, [h_s, w_s], 
                                                 method=tf.image.ResizeMethod.BILINEAR)
                    # 여기도 동일하게 차원 확장 적용
                    scaled_src = scaled_src * curr_a_expanded + curr_b_expanded
                    identity_reproj_loss = self.compute_reprojection_loss(scaled_src, scaled_tgts[s], curr_sigma)
                    identity_reprojection_list.append(identity_reproj_loss)

            reprojection_losses = tf.concat(reprojection_list, axis=3)
            
            if self.auto_mask:
                identity_reprojection_losses = tf.concat(identity_reprojection_list, axis=3)
                min_reproj_loss = tf.reduce_min(reprojection_losses, axis=3, keepdims=True)
                min_identity_reproj_loss = tf.reduce_min(identity_reprojection_losses, axis=3, keepdims=True)
                min_identity_reproj_loss += tf.random.normal(tf.shape(min_identity_reproj_loss), stddev=1e-5)
                combined = tf.minimum(min_reproj_loss, min_identity_reproj_loss)
            else:
                combined = tf.reduce_min(reprojection_losses, axis=3, keepdims=True)
            
            reprojection_loss = tf.reduce_mean(combined)

            # Smoothness and uncertainty losses
            mean_disp = tf.reduce_mean(disp_raw[s], [1, 2], keepdims=True)
            norm_disp = disp_raw[s] / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, scaled_tgts[s])

            ab_losses = tf.concat(ab_losses, axis=3)
            ab_loss = tf.reduce_sum(ab_losses, axis=3, keepdims=True)

            reg_loss = smooth_loss + self.ab_ratio * tf.reduce_mean(ab_loss)
            
            uncertainty_loss = tf.reduce_mean((pred_sigmas[s] - 1.0) ** 2)

            pixel_losses += reprojection_loss
            smooth_losses += self.smoothness_ratio * reg_loss / (2 ** s)
            uncertainty_losses += uncertainty_loss

        num_scales_f = tf.cast(self.num_scales, tf.float32)
        pixel_losses = pixel_losses / num_scales_f
        smooth_losses = smooth_losses / num_scales_f
        uncertainty_losses = uncertainty_losses / num_scales_f
        
        total_loss = pixel_losses + smooth_losses + uncertainty_losses

        return total_loss, pixel_losses, smooth_losses, uncertainty_losses, pred_depths