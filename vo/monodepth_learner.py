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
        self.predictive_mask = self.config['Train']['predictive_mask'] # False
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
    

    def forward_step(self, sample: dict, training: bool = True) -> tf.Tensor:
        """
        최적화된 forward step - JIT 컴파일 친화적
        """
        # Unpack sample
        left_image = sample['source_left']
        right_image = sample['source_right']
        tgt_image = sample['target_image']
        intrinsic = sample['intrinsic']
        gt_poses = sample['poses']
        data_type = sample['data_type']  # 0: temporal, 1: stereo
        use_pose_net = sample['use_pose_net']

        pixel_losses = 0.
        smooth_losses = 0.
        
        batch_size = tf.shape(tgt_image)[0]
        H = tf.shape(tgt_image)[1]
        W = tf.shape(tgt_image)[2]

        # Depth prediction
        disp_outputs = self.depth_net(tgt_image, training=training)
        
        # Pose prediction - 조건부 계산 제거, 모두 계산 후 선택
        concat_left_tgt = tf.concat([left_image, tgt_image], axis=3)
        concat_tgt_right = tf.concat([tgt_image, right_image], axis=3)
        
        temporal_pose_left = self.pose_net(concat_left_tgt, training=training)
        temporal_pose_right = self.pose_net(concat_tgt_right, training=training)
        
        # Stereo poses
        stereo_pose_left = self.matrix_to_axis_angle_vectorized(gt_poses[:, 0, :, :])
        stereo_pose_right = self.matrix_to_axis_angle_vectorized(gt_poses[:, 1, :, :])
        
        # Select poses based on use_pose_net
        use_pose_net_f = tf.cast(tf.expand_dims(use_pose_net, 1), tf.float32)
        one_minus_use = 1.0 - use_pose_net_f
        
        pose_left = temporal_pose_left * use_pose_net_f + stereo_pose_left * one_minus_use
        pose_right = temporal_pose_right * use_pose_net_f + stereo_pose_right * one_minus_use

        # 데이터 타입별 invert 설정
        is_temporal = tf.cast(tf.equal(data_type, 0), tf.float32)
        is_stereo = 1.0 - is_temporal

        # Multi-scale processing
        for s in range(self.num_scales):
            h_s = H // (2 ** s)
            w_s = W // (2 ** s)
            
            # Current scale depth
            disp_s = disp_outputs[s]
            depth_s = self.disp_to_depth(disp_s, self.min_depth, self.max_depth)
            
            # Resize target
            if s == 0:
                tgt_s = tgt_image
                scaled_K = intrinsic
            else:
                tgt_s = tf.image.resize(tgt_image, [h_s, w_s])
                scaled_K = self.rescale_intrinsics(intrinsic, H, W, h_s, w_s)
            
            # Process both source images
            reprojection_losses = []
            
            for i in range(2):
                if i == 0:
                    src = left_image
                    pose = pose_left
                else:
                    src = right_image  
                    pose = pose_right
                
                if s > 0:
                    src = tf.image.resize(src, [h_s, w_s])
                
                # Adjust pose for temporal left
                should_invert = is_temporal * tf.cast(tf.equal(i, 0), tf.float32)
                should_invert = tf.expand_dims(should_invert, 1)
                
                adjusted_pose = pose * (1.0 - 2.0 * should_invert)
                
                # Warp
                proj_img = projective_inverse_warp(
                    src,
                    tf.squeeze(depth_s, axis=3),
                    adjusted_pose,
                    intrinsics=scaled_K,
                    invert=False,
                    euler=False
                )
                
                # Compute loss
                reproj_loss = self.compute_reprojection_loss(proj_img, tgt_s)
                reprojection_losses.append(reproj_loss)
                
                # Identity loss for auto-masking
                if self.auto_mask:
                    identity_loss = self.compute_reprojection_loss(src, tgt_s)
                    reprojection_losses.append(identity_loss)
            
            # Combine losses
            if self.auto_mask:
                # Stack losses: [reproj_left, identity_left, reproj_right, identity_right]
                all_losses = tf.concat(reprojection_losses, axis=3)
                
                # For stereo: use only reprojection losses (indices 0, 2)
                # For temporal: use all losses
                stereo_losses = tf.concat([reprojection_losses[0], reprojection_losses[2]], axis=3)
                
                # Select based on data type
                is_stereo_4d = tf.reshape(is_stereo, [batch_size, 1, 1, 1])
                selected_losses = all_losses * (1.0 - is_stereo_4d) + \
                                  tf.concat([stereo_losses, stereo_losses], axis=3) * is_stereo_4d
                
                # Take minimum
                min_loss = tf.reduce_min(selected_losses[:, :, :, :4], axis=3, keepdims=True)
            else:
                # No auto-masking
                combined = tf.concat(reprojection_losses, axis=3)
                min_loss = tf.reduce_min(combined, axis=3, keepdims=True)
            
            # Accumulate losses
            pixel_loss = tf.reduce_mean(min_loss)
            smooth_loss = self.get_smooth_loss(disp_s, tgt_s) / (2 ** s)
            
            pixel_losses += pixel_loss
            smooth_losses += smooth_loss * self.smoothness_ratio
        
        # Average over scales
        pixel_losses = pixel_losses / float(self.num_scales)
        smooth_losses = smooth_losses / float(self.num_scales)
        
        total_loss = pixel_losses + smooth_losses
        
        # Return depth predictions for visualization
        pred_depths = []
        for s in range(self.num_scales):
            depth = self.disp_to_depth(disp_outputs[s], self.min_depth, self.max_depth)
            pred_depths.append(depth)

        return total_loss, pixel_losses, smooth_losses, pred_depths

    @tf.function
    def matrix_to_axis_angle_vectorized(self, matrix):
        """
        완전히 벡터화된 4x4 to 6DoF 변환
        """
        # Translation
        translation = matrix[:, :3, 3]
        
        # Rotation matrix
        R = matrix[:, :3, :3]
        
        # Compute angle from trace
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        angle = tf.acos(tf.clip_by_value((trace - 1.0) / 2.0, -1.0, 1.0))
        
        # Compute axis
        # When angle is small, use [0, 0, 1] as default
        # Otherwise use (R - R^T) / (2 * sin(angle))
        sin_angle = tf.sin(angle)
        use_default = tf.less(tf.abs(sin_angle), 1e-7)
        
        axis_x = tf.where(use_default, 0.0, (R[:, 2, 1] - R[:, 1, 2]) / (2.0 * sin_angle + 1e-8))
        axis_y = tf.where(use_default, 0.0, (R[:, 0, 2] - R[:, 2, 0]) / (2.0 * sin_angle + 1e-8))
        axis_z = tf.where(use_default, 1.0, (R[:, 1, 0] - R[:, 0, 1]) / (2.0 * sin_angle + 1e-8))
        
        axis = tf.stack([axis_x, axis_y, axis_z], axis=1)
        
        # Normalize
        axis_norm = tf.maximum(tf.norm(axis, axis=1, keepdims=True), 1e-8)
        axis = axis / axis_norm
        
        # Axis-angle
        angle_expanded = tf.expand_dims(angle, 1)
        axis_angle = axis * tf.where(tf.expand_dims(use_default, 1), 0.0, angle_expanded)
        
        return tf.concat([translation, axis_angle], axis=1)