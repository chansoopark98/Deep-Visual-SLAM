import tensorflow as tf

class DepthLearner(object):
    def __init__(self, model, optimizer, **config):
        """
        model: 이미 build된 tf.keras.Model
        optimizer: tf.keras.optimizers.Optimizer
        config: hyperparams, etc.
        """
        self.model = model
        self.optimizer = optimizer

        self.num_scales = 4
        self.min_depth = 0.1
        self.max_depth = 10.
    
    def depth_to_disparity(self, depth, eps=1e-4):
        # Compute disparity
        disparity = tf.where(depth >= eps, 1.0 / depth, tf.zeros_like(depth))
        depth = tf.cast(depth, tf.float32)
        return disparity

    def disparity_to_depth(self, disparity, eps=1e-4):
        # Compute absolute disparity
        abs_disp = tf.abs(disparity)

        # Compute depth
        depth = tf.where(abs_disp >= eps, 1.0 / abs_disp, tf.zeros_like(abs_disp))
        depth = tf.cast(depth, tf.float32)
        return depth
    
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
    
    def l1_loss(self, pred, gt, valid_mask):
        abs_diff = tf.abs(pred - gt)
        masked_abs_diff = tf.boolean_mask(abs_diff, valid_mask)
    
        return tf.reduce_mean(masked_abs_diff)

    def scale_invariant_log_loss(self, pred, gt):
        if len(pred.shape) == 4 and pred.shape[-1] == 1:
            pred = tf.squeeze(pred, axis=-1)
        if len(gt.shape) == 4 and gt.shape[-1] == 1:
            gt = tf.squeeze(gt, axis=-1)

        # (1) 유효 깊이 마스크(gt>0) 생성
        valid_mask = tf.cast(gt > 0, tf.float32)
        valid_count = tf.reduce_sum(valid_mask) + 1e-8  # 분모가 0이 되지 않도록
        
        # pred = tf.clip_by_value(pred, 0.1, 10.)
        # gt = tf.clip_by_value(gt, 0.1, 10.)

        # (3) 로그 차이 계산 + 유효 픽셀에만 적용
        log_diff = (tf.math.log(pred) - tf.math.log(gt)) * valid_mask

        diff_sq  = tf.reduce_sum(tf.square(log_diff)) / valid_count
        diff_sum = tf.square(tf.reduce_sum(log_diff) / valid_count)

        # loss = diff_sq - 0.5 * diff_sum
        loss = diff_sq * diff_sum
        return loss
    
    def silog_loss(self,
                prediction: tf.Tensor,
                target: tf.Tensor,
                valid_mask: tf.Tensor,
                variance_focus: float = 0.5) -> tf.Tensor:
        """
        Compute SILog loss (scale-invariant log loss) in TensorFlow,
        WITH valid_mask. (즉, target > 0인 픽셀에 대해서만 손실 계산)

        Args:
            prediction (tf.Tensor): 예측 값 (B,H,W) 혹은 (B,H,W,1) 형태 가정
            target (tf.Tensor): 정답 값 (B,H,W) 혹은 (B,H,W,1) 형태 가정
            variance_focus (float): SILog 분산 항에 곱해질 스칼라 (기본값 0.85)

        Returns:
            tf.Tensor: 스칼라 형태의 SILog loss.
        """

        # 0 이하인 prediction은 log()에서 문제가 되므로 eps로 치환(또는 max 사용)
        eps = 1e-6
        prediction = tf.maximum(prediction, eps)

        # valid_mask가 True인 위치만 골라서 계산 (boolean_mask)
        valid_prediction = tf.boolean_mask(prediction, valid_mask)
        valid_target = tf.boolean_mask(target, valid_mask)

        # 로그 차이 계산: log(pred) - log(gt)
        d = tf.math.log(valid_prediction) - tf.math.log(valid_target)

        # SILog 식: E[d^2] - variance_focus * (E[d])^2
        d2_mean = tf.reduce_mean(tf.square(d))  # E[d^2]
        d_mean = tf.reduce_mean(d)              # E[d]
        silog_expr = d2_mean - variance_focus * tf.square(d_mean)

        # 최종 스케일링
        loss_val = tf.sqrt(silog_expr)

        return loss_val
    
    def multi_scale_loss(self, pred_depths, gt_depth, rgb, valid_mask) -> dict:
        alpha = [1/2, 1/4, 1/8, 1/16] 
        smooth_losses = 0.0
        log_losses = 0.0
        l1_losses = 0.0

        smooth_loss_factor = 1.0
        log_loss_factor = 1.0
        l1_loss_factor = 1.0

        original_shape = gt_depth.shape[1:3]  # (H, W)

        for i in range(self.num_scales):
            # i-th 스케일 depth
            pred_depth = pred_depths[i]

            pred_depth_resized = tf.image.resize(
                pred_depth,
                original_shape,
                method=tf.image.ResizeMethod.BILINEAR
            )
            # smoothness loss
            smooth_losses += self.get_smooth_loss(pred_depth_resized, rgb) * alpha[i]
            
            # scale-invariant log loss
            log_losses += self.silog_loss(pred_depth_resized, gt_depth, valid_mask) * alpha[i]

            # l1 loss
            l1_losses += self.l1_loss(pred_depth_resized, gt_depth, valid_mask) * alpha[i]
        
        loss_dict = {
            'smooth_loss': smooth_losses * smooth_loss_factor,
            'log_loss': log_losses * log_loss_factor,
            'l1_loss': l1_losses * l1_loss_factor
        }
        return loss_dict

    def forward_step(self, rgb, depth, training=True) -> list[dict, list]:
        """
        rgb: [B, H, W, 3]
        depth: [B, H, W] or [B, H, W, 1] (GT depth; 0은 invalid)
        """
        # 1. Forward pass (Disparity 예측)
        pred_disps = self.model(rgb, training=training)
        # pred_disps: list [disp1, disp2, disp3, disp4]
        
        # 2. Disps -> Depths
        depth = tf.where(depth >= self.max_depth, 0., depth)
        valid_mask = depth > 0

        pred_depths = [
            self.disparity_to_depth(disp)
            for disp in pred_disps
        ]

        # 3. multi-scale loss 계산
        loss_dict = self.multi_scale_loss(pred_depths=pred_depths,
                                          gt_depth=depth,
                                          rgb=rgb,
                                          valid_mask=valid_mask)

        return loss_dict, pred_depths