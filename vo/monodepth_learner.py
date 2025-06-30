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
        sample dict 구조:
        - source_left: 왼쪽 소스 이미지
        - source_right: 오른쪽 소스 이미지 (스테레오의 경우)
        - target_image: 타겟 이미지 
        - intrinsic: 카메라 내부 파라미터
        - poses: GT poses (stereo의 경우) 또는 None (temporal의 경우)
        - baseline: 스테레오 베이스라인
        - data_type: 0 = 'stereo', 1 = 'temporal'
        - use_pose_net: pose network 사용 여부
        """
        # Unpack sample
        left_image = sample['source_left']
        right_image = sample['source_right']
        tgt_image = sample['target_image']
        intrinsic = sample['intrinsic']
        gt_poses = sample['poses']  # stereo의 경우 GT poses [B, 2, 4, 4]
        # baseline = sample['baseline']
        data_type = sample['data_type']  # integer로 변경
        use_pose_net = sample['use_pose_net']
        
        ref_images = [left_image, right_image]

        pixel_losses = 0.
        smooth_losses = 0.
        
        H = tf.shape(tgt_image)[1]
        W = tf.shape(tgt_image)[2]

        # Generate Depth Results
        pred_depths = []
        scaled_tgts = []

        disp_raw = self.depth_net(tgt_image, training=training)

        for scale_idx in range(self.num_scales):
            scaled_disp = disp_raw[scale_idx]
            scaled_depth = self.disp_to_depth(scaled_disp, self.min_depth, self.max_depth)
            pred_depths.append(scaled_depth)

            tgt_scaled = tf.image.resize(tgt_image,
                                        [H // (2**scale_idx), W // (2**scale_idx)],
                                        method=tf.image.ResizeMethod.BILINEAR)
            scaled_tgts.append(tgt_scaled)
        
        # Pose 처리 - inner function 제거
        # 1. Temporal poses (모든 샘플에 대해)
        concat_left_tgt = tf.concat([left_image, tgt_image], axis=3)
        concat_tgt_right = tf.concat([tgt_image, right_image], axis=3)
        
        temporal_pose_left = self.pose_net(concat_left_tgt, training=training)
        temporal_pose_right = self.pose_net(concat_tgt_right, training=training)
        
        # 2. Stereo poses (모든 샘플에 대해)
        stereo_pose_left = self.matrix_to_axis_angle_simple(gt_poses[:, 0, :, :])
        stereo_pose_right = self.matrix_to_axis_angle_simple(gt_poses[:, 1, :, :])
        
        # 3. use_pose_net에 따라 선택
        use_pose_net_expanded = tf.expand_dims(use_pose_net, axis=1)  # [B, 1]
        
        pose_left = tf.where(
            use_pose_net_expanded,
            temporal_pose_left,
            stereo_pose_left
        )
        
        pose_right = tf.where(
            use_pose_net_expanded,
            temporal_pose_right,
            stereo_pose_right
        )
        
        pred_poses = [pose_left, pose_right]

        # Multi-scale loss computation
        for s in range(self.num_scales):
            reprojection_list = []
            identity_reprojection_list = []

            curr_depth = pred_depths[s]  # [B,H,W,1]
            h_s = H // (2**s)
            w_s = W // (2**s)

            for i in range(2):  # left, right
                curr_src = ref_images[i]  # [B,H,W,3]
                curr_pose = pred_poses[i]  # [B,6]

                if s != 0:
                    curr_src = tf.image.resize(curr_src, [h_s, w_s], 
                                            method=tf.image.ResizeMethod.BILINEAR)
                    scaled_intrinsic = self.rescale_intrinsics(intrinsic, H, W, h_s, w_s)
                else:
                    scaled_intrinsic = intrinsic

                # 효율적인 warping
                is_stereo = tf.equal(data_type, 1)  # 1 = stereo
                should_invert = tf.logical_and(
                    tf.logical_not(is_stereo),  # temporal이고
                    tf.equal(i, 0)              # left일 때
                )  # [B]
                
                adjusted_pose = self.adjust_pose_for_invert(curr_pose, should_invert)
                
                curr_proj_image = projective_inverse_warp(
                    curr_src,
                    tf.squeeze(curr_depth, axis=3),
                    adjusted_pose,
                    intrinsics=scaled_intrinsic,
                    invert=False,
                    euler=False
                )
                
                # Photometric loss
                curr_reproj_loss = self.compute_reprojection_loss(curr_proj_image, scaled_tgts[s])
                reprojection_list.append(curr_reproj_loss)
                
                # Identity reprojection loss
                if self.auto_mask:
                    identity_reproj_loss = self.compute_reprojection_loss(curr_src, scaled_tgts[s])
                    identity_reprojection_list.append(identity_reproj_loss)

            reprojection_losses = tf.concat(reprojection_list, axis=3)  # [B, H_s, W_s, 2]
            
            # Auto-masking
            if self.auto_mask:
                identity_reprojection_losses = tf.concat(identity_reprojection_list, axis=3)
                
                # Add small noise for numerical stability
                identity_reprojection_losses += tf.random.normal(
                    tf.shape(identity_reprojection_losses), 
                    mean=0.0, 
                    stddev=1e-5
                )

                # 각 샘플별로 다른 masking 적용
                is_stereo = tf.equal(data_type, 1)  # 1 = stereo
                is_stereo_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(is_stereo, -1), -1), -1)
                
                # Stereo masking: occlusion만 고려
                stereo_masked = tf.reduce_min(reprojection_losses, axis=3, keepdims=True)
                
                # Temporal masking: auto-masking 적용
                temporal_combined = tf.concat([identity_reprojection_losses, reprojection_losses], axis=3)
                temporal_masked = tf.reduce_min(temporal_combined, axis=3, keepdims=True)
                
                # 선택
                combined = tf.where(
                    is_stereo_expanded,
                    stereo_masked,
                    temporal_masked
                )
            else:
                combined = tf.reduce_min(reprojection_losses, axis=3, keepdims=True)

            # 최종 reprojection loss
            reprojection_loss = tf.reduce_mean(combined)

            # Smoothness loss
            mean_disp = tf.reduce_mean(disp_raw[s], [1, 2], keepdims=True)
            norm_disp = disp_raw[s] / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, scaled_tgts[s])
            smooth_loss = smooth_loss / (2.0**s)

            pixel_losses += reprojection_loss
            smooth_losses += smooth_loss * self.smoothness_ratio
        
        # Average over scales
        num_scales_f = tf.cast(self.num_scales, tf.float32)
        pixel_losses = pixel_losses / num_scales_f
        smooth_losses = smooth_losses / num_scales_f
        
        total_loss = pixel_losses + smooth_losses

        return total_loss, pixel_losses, smooth_losses, pred_depths


    @tf.function(jit_compile=True)
    def adjust_pose_for_invert(self, pose, should_invert):
        """
        invert가 필요한 경우 pose를 역변환
        pose: [B, 6] - (tx, ty, tz, rx, ry, rz)
        should_invert: [B] - boolean
        """
        # Translation 부분 반전
        translation = pose[:, :3]  # [B, 3]
        rotation = pose[:, 3:]     # [B, 3]
        
        # invert가 True인 경우 translation과 rotation을 반전
        inverted_translation = -translation
        inverted_rotation = -rotation
        
        # should_invert를 [B, 1]로 확장
        should_invert_expanded = tf.expand_dims(should_invert, axis=1)
        
        # 조건에 따라 선택
        adjusted_translation = tf.where(
            should_invert_expanded,
            inverted_translation,
            translation
        )
        
        adjusted_rotation = tf.where(
            should_invert_expanded,
            inverted_rotation,
            rotation
        )
        
        adjusted_pose = tf.concat([adjusted_translation, adjusted_rotation], axis=1)
        
        return adjusted_pose

    @tf.function(jit_compile=True)
    def matrix_to_axis_angle_simple(self, matrix):
        """
        간단하고 안정적인 4x4 to 6DoF 변환
        scipy.spatial.transform.Rotation.as_rotvec() 참조
        """
        batch_size = tf.shape(matrix)[0]
        
        # Translation 추출
        translation = matrix[:, :3, 3]  # [B, 3]
        
        # Rotation matrix
        R = matrix[:, :3, :3]  # [B, 3, 3]
        
        # Compute rotation vector using Rodrigues' formula
        # 1. Compute trace
        trace = tf.linalg.trace(R)
        
        # 2. Compute angle
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)
        theta = tf.acos(cos_theta)
        
        # 3. Compute axis (when theta != 0)
        # Use the fact that R - R^T gives us the axis direction
        R_minus_RT = R - tf.transpose(R, [0, 2, 1])
        
        # Extract axis components
        axis_x = R_minus_RT[:, 2, 1]
        axis_y = R_minus_RT[:, 0, 2]  
        axis_z = R_minus_RT[:, 1, 0]
        
        axis = tf.stack([axis_x, axis_y, axis_z], axis=1)
        
        # Normalize axis
        axis_norm = tf.norm(axis, axis=1, keepdims=True) + 1e-8
        axis = axis / axis_norm
        
        # Handle special case when theta ≈ 0
        is_small_angle = tf.less(theta, 1e-5)
        theta = tf.where(is_small_angle, tf.zeros_like(theta), theta)
        
        # Axis-angle representation
        axis_angle = axis * tf.expand_dims(theta, 1)
        
        # Combine translation and rotation
        pose_6dof = tf.concat([translation, axis_angle], axis=1)
        
        return pose_6dof