import tensorflow as tf
from tensorflow import keras
from utils.projection_utils import projective_inverse_warp

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

        loss = (self.ssim_ratio * ssim_loss) + ((1. - self.ssim_ratio) * l1_loss)
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

        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = tf.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_disp_y = tf.abs(disp[:, :-1, :, :] - disp[:, 1:, :, :])

        grad_img_x = tf.reduce_mean(tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 3, keepdims=True)
        grad_img_y = tf.reduce_mean(tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :]), 3, keepdims=True)

        grad_disp_x *= tf.exp(-grad_img_x)
        grad_disp_y *= tf.exp(-grad_img_y)

        return tf.reduce_mean(grad_disp_x) + tf.reduce_mean(grad_disp_y)
        
    @tf.function(jit_compile=True)
    def rescale_intrinsics(self, intrinsics, original_height, original_width, target_height, target_width):
        """
        intrinsics(tf.Tensor): [B, 3, 3]
        """
        # 스케일 비율 계산
        h_scale = tf.cast(target_height, tf.float32) / tf.cast(original_height, tf.float32)
        w_scale = tf.cast(target_width, tf.float32) / tf.cast(original_width, tf.float32)

        # 배치 크기
        batch_size = tf.shape(intrinsics)[0]
        
        # 스케일링 행렬 생성
        scale_matrix = tf.stack([
            tf.stack([w_scale, 0.0, 0.0], axis=0),
            tf.stack([0.0, h_scale, 0.0], axis=0), 
            tf.stack([0.0, 0.0, 1.0], axis=0)
        ], axis=0)
        
        # 배치 차원으로 확장
        scale_matrix = tf.tile(tf.expand_dims(scale_matrix, 0), [batch_size, 1, 1])
        
        # 행렬 곱셈으로 스케일링 적용
        scaled_intrinsics = tf.matmul(scale_matrix, intrinsics)
        
        return scaled_intrinsics
    
    def forward_mono(self, sample, training=True) -> tf.Tensor:
        left_image = sample['source_left']  # [B, H, W, 3]
        right_image = sample['source_right']  # [B, H, W, 3]
        tgt_image = sample['target_image']  # [B, H, W, 3]
        intrinsic = sample['intrinsic']  # [B, 3, 3] - target camera intrinsic
        ref_images = [left_image, right_image]  # [B, H, W, 3] * 2

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
            scaled_disp = disp_raw[scale_idx]
            scaled_depth = self.disp_to_depth(scaled_disp, self.min_depth, self.max_depth)
            pred_depths.append(scaled_depth)

            tgt_scaled = tf.image.resize(tgt_image,
                                        [H // (2**scale_idx), W // (2**scale_idx)],
                                        method=tf.image.ResizeMethod.BILINEAR)
            scaled_tgts.append(tgt_scaled)
        
        # Predict Poses
        concat_left_tgt = tf.concat([left_image, tgt_image], axis=3)   # [B,H,W,6]
        concat_tgt_right = tf.concat([tgt_image, right_image], axis=3) # [B,H,W,6]

        # no use imu
        pose_left = self.pose_net(concat_left_tgt, training=training)    # [B,6]
        pose_right = self.pose_net(concat_tgt_right, training=training)  # [B,6]

        pose_left = tf.cast(pose_left, tf.float32)
        pose_right = tf.cast(pose_right, tf.float32)
        
        pred_poses = [pose_left, pose_right]

        for s in range(self.num_scales):
            # reprojection loss
            reprojection_list = []
            identity_reprojection_list = []  # 원본 구현처럼 분리

            curr_depth = pred_depths[s] # [B,H,W,1]
            h_s = H // (2**s)
            w_s = W // (2**s)

            for i in range(2): # left, right
                curr_src = ref_images[i] # [B,H,W,3]
                curr_pose = pred_poses[i] # [B,6]

                if s != 0:
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
                    is_stereo=False,
                    euler=False,
                )
                
                # Photometric loss
                curr_reproj_loss = self.compute_reprojection_loss(curr_proj_image, scaled_tgts[s])
                reprojection_list.append(curr_reproj_loss)
                

            reprojection_losses = tf.concat(reprojection_list, axis=3)  # [B, H_s, W_s, 2]
            
            if self.auto_mask:
                identity_reprojection_losses = []
                
                for i in range(2):  # left, right
                    # 원본과 동일: source_scale에 따른 처리
                    if s != 0:
                        # v1_multiscale이 False인 경우 source_scale = 0
                        pred = tf.image.resize(ref_images[i], [h_s, w_s], 
                                            method=tf.image.ResizeMethod.BILINEAR)
                    else:
                        pred = ref_images[i]
                    
                    identity_loss = self.compute_reprojection_loss(pred, scaled_tgts[s])
                    identity_reprojection_losses.append(identity_loss)
                
                identity_reprojection_losses = tf.concat(identity_reprojection_losses, 3)
                
                # identity_reprojection_loss = tf.reduce_mean(identity_reprojection_losses, 3, keepdims=True)
                identity_reprojection_loss = identity_reprojection_losses
                
                # 원본과 동일: 랜덤 노이즈 추가
                identity_reprojection_loss += tf.random.normal(
                    tf.shape(identity_reprojection_loss), 
                    mean=0.0, 
                    stddev=1e-5
                )
                
                # 원본과 동일: identity와 reprojection loss 결합
                combined = tf.concat([identity_reprojection_loss, reprojection_losses], axis=3)
            else:
                combined = reprojection_losses

            combined = tf.reduce_min(combined, axis=3, keepdims=True)  # [B, H_s, W_s, 1]
            
            # 최종 reprojection loss
            reprojection_loss = tf.reduce_mean(combined)

            # smoothness loss
            mean_disp = tf.reduce_mean(disp_raw[s], [1, 2], keepdims=True)
            norm_disp = disp_raw[s] / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, scaled_tgts[s])
            smooth_loss = smooth_loss / (2.0**s)

            pixel_losses += reprojection_loss
            smooth_losses += smooth_loss * self.smoothness_ratio
            
        # total loss
        num_scales_f = tf.cast(self.num_scales, tf.float32)
        pixel_losses = pixel_losses / num_scales_f
        smooth_losses = smooth_losses / num_scales_f
        
        total_loss = pixel_losses + smooth_losses

        return total_loss, pixel_losses, smooth_losses, pred_depths
    
    def forward_stereo(self, sample, training=True) -> tf.Tensor:
        src_image = sample['source_image']  # [B, H, W, 3]
        tgt_image = sample['target_image']  # [B, H, W, 3]
        intrinsic = sample['intrinsic']  # [B, 3, 3] - target camera intrinsic
        stereo_pose = sample['pose'] # src to tgt pose [B, 6] (axis-angle + translation)

        pixel_losses = 0.
        smooth_losses = 0.
        
        H, W = self.image_shape

        # Generate Depth
        disp_raw = self.depth_net(tgt_image, training=training)  # [[B, H, W, 1], [B, H/2, W/2, 1], ...]

        for s in range(self.num_scales):
            h_s = H // (2**s)
            w_s = W // (2**s)
            
            # Current scale depth
            disp_s = disp_raw[s]
            depth_s = self.disp_to_depth(disp_s, self.min_depth, self.max_depth)
            
            # Resize images to current scale
            if s != 0:
                src_s = tf.image.resize(src_image, [h_s, w_s], 
                                    method=tf.image.ResizeMethod.BILINEAR)
                tgt_s = tf.image.resize(tgt_image, [h_s, w_s], 
                                    method=tf.image.ResizeMethod.BILINEAR)
                scaled_intrinsic = self.rescale_intrinsics(intrinsic, H, W, h_s, w_s)
            else:
                src_s = src_image
                tgt_s = tgt_image
                scaled_intrinsic = intrinsic
            
            # Stereo reprojection (L->R or R->L)
            proj_image = projective_inverse_warp(
                src_s,
                tf.squeeze(depth_s, axis=3),
                stereo_pose,
                intrinsics=scaled_intrinsic,
                invert=False,
                is_stereo=True,
                euler=False,
            )
            
            # Photometric loss
            reproj_loss = self.compute_reprojection_loss(proj_image, tgt_s)

            if self.auto_mask:
                # Identity reprojection loss (no warping)
                identity_loss = self.compute_reprojection_loss(src_s, tgt_s)
                
                reproj_loss = tf.minimum(reproj_loss, identity_loss)
            
            pixel_loss = tf.reduce_mean(reproj_loss)
            pixel_losses += pixel_loss
            
            # Smoothness loss  
            mean_disp = tf.reduce_mean(disp_raw[s], axis=[1, 2], keepdims=True)
            norm_disp = disp_raw[s] / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, tgt_s)
            smooth_loss = smooth_loss / (2.0**s)
            smooth_losses += smooth_loss * self.smoothness_ratio
        
        # Average over scales
        pixel_losses = pixel_losses / float(self.num_scales)
        smooth_losses = smooth_losses / float(self.num_scales)
        
        total_loss = pixel_losses + smooth_losses
        
        # Return predictions
        pred_depths = []
        for s in range(self.num_scales):
            depth = self.disp_to_depth(disp_raw[s], self.min_depth, self.max_depth)
            pred_depths.append(depth)

        return total_loss, pixel_losses, smooth_losses, pred_depths

    # @tf.function
    def matrix_to_axis_angle_vectorized(self, matrix, depth=None):
        """
        4x4 변환 행렬을 6DoF axis-angle 벡터로 변환
        pose_axis_angle_vec2mat의 역변환
        
        Args:
            matrix: [B, 4, 4] 변환 행렬
            depth: [B, H, W, 1] depth map (선택사항, 스케일 정규화용)
        
        Returns:
            [B, 6] axis-angle + translation 벡터
        """
        batch_size = tf.shape(matrix)[0]
        
        # Translation 추출
        translation = matrix[:, :3, 3]  # [B, 3]
        
        # Rotation matrix 추출
        R = matrix[:, :3, :3]  # [B, 3, 3]
        
        # Depth 스케일링 역적용 (있는 경우)
        if depth is not None:
            inv_depth = 1.0 / (depth + 1e-6)
            mean_inv_depth = tf.reduce_mean(inv_depth, axis=[1, 2, 3])  # [B]
            mean_inv_depth = tf.reshape(mean_inv_depth, [batch_size, 1])
            # 역스케일링 적용
            translation = translation / (mean_inv_depth + 1e-8)
        
        # Rotation matrix to axis-angle
        # 1. Trace를 이용한 angle 계산
        trace = tf.linalg.trace(R)  # [B]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = tf.clip_by_value(cos_angle, -1.0, 1.0)
        angle = tf.acos(cos_angle)  # [B]
        
        # 2. Axis 계산
        # 작은 각도 처리
        eps = 1e-7
        is_small_angle = tf.less(tf.abs(angle), eps)
        
        # Skew-symmetric 부분에서 axis 추출
        # axis = (R - R^T) / (2 * sin(angle))
        R_transpose = tf.transpose(R, [0, 2, 1])
        R_skew = R - R_transpose
        
        sin_angle = tf.sin(angle)
        sin_angle_safe = tf.where(is_small_angle, tf.ones_like(sin_angle), sin_angle)
        
        axis_x = R_skew[:, 2, 1] / (2.0 * sin_angle_safe + eps)
        axis_y = R_skew[:, 0, 2] / (2.0 * sin_angle_safe + eps)  
        axis_z = R_skew[:, 1, 0] / (2.0 * sin_angle_safe + eps)
        
        # 배치 처리를 위한 조건부 axis 계산
        axis_default = tf.stack([axis_x, axis_y, axis_z], axis=1)  # [B, 3]
        axis_small = tf.zeros([batch_size, 3])  # 작은 각도일 때는 0 벡터
        
        # 최종 axis 선택
        axis = tf.where(tf.expand_dims(is_small_angle, 1), axis_small, axis_default)
        
        # Axis 정규화
        axis_norm = tf.norm(axis, axis=1, keepdims=True)
        axis_normalized = tf.where(
            tf.expand_dims(is_small_angle, 1),
            axis,  # 작은 각도일 때는 정규화 안 함
            axis / (axis_norm + eps)
        )
        
        # Axis-angle 벡터 생성
        angle_expanded = tf.expand_dims(angle, 1)  # [B, 1]
        axis_angle = axis_normalized * angle_expanded  # [B, 3]
        
        # 작은 각도의 경우 0 벡터로 설정
        axis_angle = tf.where(tf.expand_dims(is_small_angle, 1), 
                            tf.zeros_like(axis_angle), 
                            axis_angle)
        
        # Translation과 결합
        pose_vec = tf.concat([axis_angle, translation], axis=1)  # [B, 6]
        
        return pose_vec