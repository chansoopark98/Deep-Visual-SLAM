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

    def disp_to_depth(self, disp, min_depth, max_depth):
        """
        Disparity -> Depth 변환
        disp: [B, H, W, 1] (sigmoid 출력을 가정)
        min_depth, max_depth: 예측할 수 있는 깊이 범위
        """
        # min_disp, max_disp를 이용해 disp를 [min_disp, max_disp] 범위로 스케일링
        min_disp = 1.0 / max_depth
        max_disp = 1.0 / min_depth
        scaled_disp = tf.cast(min_disp, tf.float32) + \
                      tf.cast(max_disp - min_disp, tf.float32) * disp
        depth = 1.0 / scaled_disp
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
    
    def l1_loss(self, pred, gt):
        valid_mask = tf.cast(gt > 0., tf.float32)
        valid_count = tf.reduce_sum(valid_mask) + 1e-8

        abs_diff = tf.abs(pred - gt)
        masked_abs_diff = abs_diff * valid_mask

        return tf.reduce_sum(masked_abs_diff) / valid_count

    def scale_invariant_log_loss(self, pred, gt):
        """
        Eigen et al. (2014)의 Scale-invariant log loss 에
        'gt_depth == 0' 픽셀 마스킹 로직 추가
        pred, gt: shape [B, H, W] 또는 [B, H, W, 1]
        """
        # 만약 채널이 4차원 (B,H,W,1)이라면 squeeze해서 [B,H,W]로 맞춰줌
        if len(pred.shape) == 4 and pred.shape[-1] == 1:
            pred = tf.squeeze(pred, axis=-1)
        if len(gt.shape) == 4 and gt.shape[-1] == 1:
            gt = tf.squeeze(gt, axis=-1)

        # (1) 유효 깊이 마스크(gt>0) 생성
        valid_mask = tf.cast(gt > 0, tf.float32)
        valid_count = tf.reduce_sum(valid_mask) + 1e-8  # 분모가 0이 되지 않도록
        
        pred = tf.clip_by_value(pred, 0.1, 10.)
        gt = tf.clip_by_value(gt, 0.1, 10.)

        # (3) 로그 차이 계산 + 유효 픽셀에만 적용
        log_diff = (tf.math.log(pred) - tf.math.log(gt)) * valid_mask

        diff_sq  = tf.reduce_sum(tf.square(log_diff)) / valid_count
        diff_sum = tf.square(tf.reduce_sum(log_diff) / valid_count)

        loss = diff_sq - 0.5 * diff_sum
        return loss

    def multi_scale_loss(self, pred_depths, gt_depth, rgb) -> dict:
        alpha = [1/2, 1/4, 1/8, 1/16] 
        smooth_losses = 0.0
        log_losses = 0.0
        l1_losses = 0.0

        original_shape = gt_depth.shape[1:3]  # (H, W)

        for i in range(self.num_scales):
            # i-th 스케일 depth
            pred_depth = pred_depths[i]

            # GT와 크기가 다를 수 있으므로, GT 크기로 resize
            # (batch, origH, origW, 1) 형태로 맞추기
            pred_depth_resized = tf.image.resize(
                pred_depth,
                original_shape,
                method=tf.image.ResizeMethod.BILINEAR
            )
            # smoothness loss
            smooth_losses += self.get_smooth_loss(pred_depth_resized, rgb) * alpha[i]

            # scale-invariant log loss
            log_losses += self.scale_invariant_log_loss(pred_depth_resized, gt_depth) * alpha[i]

            # l1 loss
            l1_losses += self.l1_loss(pred_depth_resized, gt_depth) * alpha[i]
        
        loss_dict = {
            'smooth_loss': smooth_losses,
            'log_loss': log_losses,
            'l1_loss': l1_losses
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
        pred_depths = [
            self.disp_to_depth(disp, self.min_depth, self.max_depth) 
            for disp in pred_disps
        ]

        # 3. multi-scale loss 계산
        loss_dict = self.multi_scale_loss(pred_depths=pred_depths, gt_depth=depth, rgb=rgb)

        return loss_dict, pred_depths
