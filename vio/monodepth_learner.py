import tensorflow as tf
from utils.projection_utils import projective_inverse_warp

class Learner(object):
    def __init__(self,
                 depth_model: tf.keras.Model,
                 pose_model: tf.keras.Model,
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

        # Monodepth2: return (1-SSIM)/2 => [0..1] 범위
        # clip to avoid negative
        SSIM_loss = tf.clip_by_value((1.0 - SSIM_raw)*0.5, 0.0, 1.0)
        return SSIM_loss

    def get_smooth_loss(self, disp, img):
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
        
    def forward_step(self, ref_images, tgt_image, imu, intrinsic, training=True) -> tf.Tensor:
        """
        ref_images: (batch_size, num_source * 2, img_h, img_w, 3)
            left_images: ref_images[:, :num_source]
            right_images: ref_images[:, num_source:]

        tgt_image: (batch_size, img_h, img_w, 3)

        # TODO: IMU 추가
        Expected shape :
        imus: (batch_size, imu_seq_len, 12)
           left_imus: imug[:, :num_source]
           right_imus: imug[:, :, num_source:]

        intrinsic: (batch_size, 3, 3)
        """
        left_images = ref_images[:, :self.num_source] # [B, num_source, H, W, 3]
        right_images = ref_images[:, self.num_source:] # [B, num_source, H, W, 3]

        left_imus = imu[:, :self.num_source] # [B, num_source, imu_seq_len, 6]
        right_imus = imu[:, self.num_source:] # [B, num_source, imu_seq_len, 6]

        # TODO
        # left_imus = imus[:, :self.num_source] # [B, num_source, imu_seq_len, 12]
        # right_imus = imus[:, self.num_source:] # [B, num_source, imu_seq_len, 12]

        pixel_losses = 0.
        smooth_losses = 0.
        total_loss = 0.

        pred_depths = []
        pred_auto_masks = []
        
        left_warped = []
        right_warped = []

        left_warped_losses = []
        right_warped_losses = []
        
        for src_idx in range(self.num_source):
            left_image = left_images[:, src_idx]  # [B, H, W, 3]
            right_image = right_images[:, src_idx]  # [B, H, W, 3]

            left_imu = left_imus[:, src_idx]  # [B, imu_seq_len, 6]
            right_imu = right_imus[:, src_idx]  # [B, imu_seq_len, 6]

            # input_images = tf.concat([left_image, tgt_image, right_image], axis=3)  # [B, H, W, 9]

            # pred_disps, pred_poses = self.model(input_images, training=training)
            disp_raw = self.depth_net(tgt_image, training=training)

            pred_disps = []
            for s in range(self.num_scales):
                scale_h = self.image_shape[0] // (2 ** s)
                scale_w = self.image_shape[1] // (2 ** s)
                scaled_disp = tf.image.resize(disp_raw[s], [scale_h, scale_w], method=tf.image.ResizeMethod.BILINEAR)
                pred_disps.append(scaled_disp)

            cat_left = tf.concat([left_image, tgt_image], axis=3)   # [B,H,W,6]
            cat_right = tf.concat([tgt_image, right_image], axis=3) # [B,H,W,6]
            pose_left = self.pose_net([cat_left, left_imu], training=training)    # [B,6]
            pose_right = self.pose_net([cat_right, right_imu], training=training)  # [B,6]
            pred_poses = tf.stack([pose_left, pose_right], axis=1)    # [B,2,6]
            pred_poses = tf.cast(pred_poses, tf.float32)

            # Parse input images
            src_image_stack = tf.concat([left_image, right_image], axis=3)

            # Multi-scale photometric + auto-mask + smoothness
            H = tf.shape(tgt_image)[1]
            W = tf.shape(tgt_image)[2]

            for s in range(self.num_scales):
                # disp_s shape => [B, H/(2^s), W/(2^s), 1]
                disp_s = pred_disps[s]
                disp_s = tf.cast(disp_s, tf.float32)
                # target scaled => nearest or bilinear, here nearest
                tgt_scaled = tf.image.resize(tgt_image,
                                            [H // (2**s), W // (2**s)],
                                            method=tf.image.ResizeMethod.BILINEAR)
                # src scaled
                src_scaled = tf.image.resize(src_image_stack,
                                            [H // (2**s), W // (2**s)],
                                            method=tf.image.ResizeMethod.BILINEAR)

                # 3.1) depth = disp->depth
                depth_s = self.disp_to_depth(disp_s, self.min_depth, self.max_depth)
                pred_depths.append(depth_s)

                # 3.2) reprojection loss
                reprojection_list = []

                for i in range(self.num_source):  # 0=left,1=right
                    # src_i
                    curr_src = src_scaled[..., i*3:(i+1)*3]
                    # pose => [B,num_source,6]
                    curr_pose = pred_poses[:, i, :]  # shape [B,6]
                    # warp
                    # intrinsics_simplified = ...
                    resized_intrinsics = self.rescale_intrinsics(intrinsic, H, W, H // (2**s), W // (2**s))

                    curr_proj_image = projective_inverse_warp(
                        curr_src,
                        tf.squeeze(depth_s, axis=3),
                        curr_pose,
                        intrinsics=resized_intrinsics,
                        invert=(i==0),
                        
                    )
                    # photometric
                    curr_reproj_loss = self.compute_reprojection_loss(curr_proj_image, tgt_scaled)

                    # for visualization
                    if s == 0:
                        if i == 0:
                            left_warped.append(curr_proj_image)
                            left_warped_losses.append(curr_reproj_loss)
                        else:
                            right_warped.append(curr_proj_image)
                            right_warped_losses.append(curr_reproj_loss)

                    # for loss
                    reprojection_list.append(curr_reproj_loss)

                # shape => [B, H/(2^s), W/(2^s), num_source]
                reprojection_losses = tf.concat(reprojection_list, axis=3)

                # 3.3) auto_mask
                combined = reprojection_losses
                if self.auto_mask:
                    identity_list = []
                    for i in range(self.num_source):
                        # identity reprojection => src==tgt scaled
                        identity_loss = self.compute_reprojection_loss(
                            src_scaled[..., i*3:(i+1)*3], tgt_scaled
                        )
                        identity_list.append(identity_loss)
                    identity_losses = tf.concat(identity_list, axis=3)
                    # random noise
                    identity_losses += tf.random.normal(tf.shape(identity_losses), stddev=1e-5)

                    combined = tf.concat([identity_losses, reprojection_losses], axis=3)

                    # For visualization
                    if s == 0:
                        pred_auto_masks.append(tf.expand_dims(tf.cast(tf.argmin(combined, axis=3) > 1,tf.float32) * 255, -1))

                # min across channel => pick best
                reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))

                # 3.4) smoothness
                smooth_loss = self.get_smooth_loss(disp_s, tgt_scaled)
                # scale 보정 => smooth_loss /= (2^s)
                smooth_loss = smooth_loss / (2.0**s)

                scale_total_loss = reprojection_loss + self.smoothness_ratio * smooth_loss
                total_loss += scale_total_loss

                pixel_losses += reprojection_loss
                smooth_losses += smooth_loss

        left_warped = tf.stack(left_warped, axis=1) # [B, num_source, H, W, 3]
        right_warped = tf.stack(right_warped, axis=1) # [B, num_source, H, W, 3]
        left_warped_losses = tf.stack(left_warped_losses, axis=1) # [B, num_source, H, W, 1]
        right_warped_losses = tf.stack(right_warped_losses, axis=1) # [B, num_source, H, W, 1]

        pred_auto_masks = tf.stack(pred_auto_masks, axis=1) # [B, num_source, H, W, 1]    

        vis_outputs = {
            'left_warped': left_warped, # [B, num_source, H, W, 3]
            'right_warped': right_warped, # [B, num_source, H, W, 3]
            'left_warped_losses': left_warped_losses, # [B, num_source, H, W, 1]
            'right_warped_losses': right_warped_losses, # [B, num_source, H, W, 1]
            'target': tgt_image, # [B, H, W, 3]
            'left_images': left_images, # [B, num_source, H, W, 3]
            'right_images': right_images, # [B, num_source, H, W, 3]
            'masks': pred_auto_masks, # [B, num_source, H, W, 1]
        }
            
        # 평균 내기
        num_scales_f = tf.cast(self.num_scales, tf.float32)
        total_loss = total_loss / num_scales_f
        pixel_losses = pixel_losses / num_scales_f
        smooth_losses = smooth_losses / num_scales_f

        return total_loss, pixel_losses, smooth_losses, pred_depths, vis_outputs

"""
# # 1. Forward pass
        # pred_disps, pred_poses = self.model(images, training=training)
        
        # pred_poses = tf.cast(pred_poses, tf.float32)
        
        # # 2. Parse input images
        # left_image = images[..., :3]
        # tgt_image = images[..., 3:6]
        # right_image = images[..., 6:9]
        # src_image_stack = tf.concat([left_image, right_image], axis=3)

        # # 3. Multi-scale photometric + auto-mask + smoothness
        # H = tf.shape(tgt_image)[1]
        # W = tf.shape(tgt_image)[2]

        # pixel_losses = 0.
        # smooth_losses = 0.
        # total_loss = 0.

        # pred_depths = []
        # pred_auto_masks = []
        # warped_images = [] # [[B, H, W, 3], [B, H, W, 3]] # left to target, right to target
        # warped_losses = [] # [[B, H, W, 1], [B, H, W, 1]] # left to target, right to target

        # for s in range(self.num_scales):
        #     # disp_s shape => [B, H/(2^s), W/(2^s), 1]
        #     disp_s = pred_disps[s]
        #     disp_s = tf.cast(disp_s, tf.float32)
        #     # target scaled => nearest or bilinear, here nearest
        #     tgt_scaled = tf.image.resize(tgt_image,
        #                                  [H // (2**s), W // (2**s)],
        #                                   method=tf.image.ResizeMethod.BILINEAR)
        #     # src scaled
        #     src_scaled = tf.image.resize(src_image_stack,
        #                                  [H // (2**s), W // (2**s)],
        #                                   method=tf.image.ResizeMethod.BILINEAR)

        #     # 3.1) depth = disp->depth
        #     depth_s = self.disp_to_depth(disp_s, self.min_depth, self.max_depth)
        #     pred_depths.append(depth_s)

        #     # 3.2) reprojection loss
        #     reprojection_list = []
        #     for i in range(self.num_source):  # 0=left,1=right
        #         # src_i
        #         curr_src = src_scaled[..., i*3:(i+1)*3]
        #         # pose => [B,num_source,6]
        #         curr_pose = pred_poses[:, i, :]  # shape [B,6]
        #         # warp
        #         # intrinsics_simplified = ...
        #         resized_intrinsics = self.rescale_intrinsics(intrinsic, H, W, H // (2**s), W // (2**s))

        #         curr_proj_image = projective_inverse_warp(
        #             curr_src,
        #             tf.squeeze(depth_s, axis=3),
        #             curr_pose,
        #             intrinsics=resized_intrinsics,
        #             invert=(i==0),
                    
        #         )
        #         # photometric
        #         curr_reproj_loss = self.compute_reprojection_loss(curr_proj_image, tgt_scaled)

        #         # for visualization
        #         if s == 0:
        #             warped_images.append(curr_proj_image)
        #             warped_losses.append(curr_reproj_loss)

        #         # for loss
        #         reprojection_list.append(curr_reproj_loss)

        #     # shape => [B, H/(2^s), W/(2^s), num_source]
        #     reprojection_losses = tf.concat(reprojection_list, axis=3)

        #     # 3.3) auto_mask
        #     combined = reprojection_losses
        #     if self.auto_mask:
        #         identity_list = []
        #         for i in range(self.num_source):
        #             # identity reprojection => src==tgt scaled
        #             identity_loss = self.compute_reprojection_loss(
        #                 src_scaled[..., i*3:(i+1)*3], tgt_scaled
        #             )
        #             identity_list.append(identity_loss)
        #         identity_losses = tf.concat(identity_list, axis=3)
        #         # random noise
        #         identity_losses += tf.random.normal(tf.shape(identity_losses), stddev=1e-5)

        #         combined = tf.concat([identity_losses, reprojection_losses], axis=3)

        #         # For visualization
        #         if s == 0:
        #             pred_auto_masks.append(tf.expand_dims(tf.cast(tf.argmin(combined, axis=3) > 1,tf.float32) * 255, -1))

        #     # min across channel => pick best
        #     reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))

        #     # 3.4) smoothness
        #     smooth_loss = self.get_smooth_loss(disp_s, tgt_scaled)
        #     # scale 보정 => smooth_loss /= (2^s)
        #     smooth_loss = smooth_loss / (2.0**s)

        #     scale_total_loss = reprojection_loss + self.smoothness_ratio * smooth_loss
        #     total_loss += scale_total_loss

        #     pixel_losses += reprojection_loss
        #     smooth_losses += smooth_loss

        # # visualizations output
        # vis_outputs = {
        #     'warped_image': warped_images,
        #     'warped_loss': warped_losses,
        #     'target': tgt_image,
        #     'left_image': left_image,
        #     'right_image': right_image,
        #     'mask': pred_auto_masks,
        # }
            
        # # 평균 내기
        # num_scales_f = tf.cast(self.num_scales, tf.float32)
        # total_loss = total_loss / num_scales_f
        # pixel_losses = pixel_losses / num_scales_f
        # smooth_losses = smooth_losses / num_scales_f

        # return total_loss, pixel_losses, smooth_losses, pred_depths, vis_outputs
"""