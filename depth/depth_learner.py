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

        # (2) clip으로 매우 작은 값이나 너무 큰 값 방어
        pred = tf.clip_by_value(pred, 1e-3, 1000.)
        # gt가 0인 부분은 무시하기 때문에, clip 시에도 valid_mask가 1인 부분만 실제로 사용됨
        gt = tf.clip_by_value(gt, 1e-3, 1000.)

        # (3) 로그 차이 계산 + 유효 픽셀에만 적용
        log_diff = (tf.math.log(pred) - tf.math.log(gt)) * valid_mask

        # (4) scale-invariant loss 계산
        #     1) diff_sq = mean( (log_pred - log_gt)^2 )
        #     2) diff_sum = ( mean( (log_pred - log_gt) ) )^2
        #        단, 여기서는 "mean" 대신 "sum / valid_count"로 구현
        diff_sq  = tf.reduce_sum(tf.square(log_diff)) / valid_count
        diff_sum = tf.square(tf.reduce_sum(log_diff) / valid_count)

        loss = diff_sq - 0.5 * diff_sum
        return loss

    def multi_scale_loss(self, pred_depths, gt_depth, alpha=[0.5, 0.2, 0.2, 0.1]):
        """
        여러 스케일에서 예측된 depth와 원본 GT depth 간의 손실을 가중합
        pred_depths: list of depth tensors [disp1->depth, disp2->depth, disp3->depth, disp4->depth]
        gt_depth: [B, H, W] 혹은 [B, H, W, 1]
        alpha: 스케일별 가중치
        """
        total_loss = 0.0
        original_shape = gt_depth.shape[1:3]  # (H, W)

        for i in range(self.num_scales):
            # i-th 스케일 depth
            pred_depth = pred_depths[i]

            # GT와 크기가 다를 수 있으므로, GT 크기로 resize
            # (batch, origH, origW, 1) 형태로 맞추기
            pred_depth_resized = tf.image.resize(
                pred_depth,
                original_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )

            # scale-invariant log loss
            loss_i = self.scale_invariant_log_loss(pred_depth_resized, gt_depth)
            total_loss += alpha[i] * loss_i

        return total_loss

    def forward_step(self, rgb, depth, training=True) -> tf.Tensor:
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
        total_loss = self.multi_scale_loss(pred_depths=pred_depths, gt_depth=depth)

        return total_loss, pred_depths
