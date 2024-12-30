import tensorflow as tf

def scale_invariant_log_loss(pred, gt):
    """
    Eigen et al. (2014)의 scale-invariant log depth loss 예시
    pred, gt: [B, H, W] shape (이미 같은 스케일이라고 가정)
    """
    # 안전을 위해 클리핑
    pred = tf.clip_by_value(pred, clip_value_min=1e-3, clip_value_max=1000.)
    gt   = tf.clip_by_value(gt,   clip_value_min=1e-3, clip_value_max=1000.)

    log_diff = tf.math.log(pred) - tf.math.log(gt)
    diff_sq = tf.reduce_mean(tf.square(log_diff))  # 1st term
    diff_sum = tf.square(tf.reduce_mean(log_diff)) # 2nd term
    return diff_sq - 0.5 * diff_sum


def multi_scale_loss(disps, gt_depths, loss_fn=scale_invariant_log_loss, alpha=[0.5, 0.2, 0.2, 0.1]):
    """
    disps: list of [disp1, disp2, disp3, disp4]
    gt_depths: list of GT depth tensors (downsample된 것들)
    alpha: 각 스케일별 가중치
    loss_fn: 스케일 간 일치하는 (pred, gt)에 적용할 기본 loss 함수
    """
    total_loss = 0.0
    for i, (disp, gt) in enumerate(zip(disps, gt_depths)):
        # Disparity -> Depth 변환 (예: 1/(disp+epsilon))
        pred_depth = 1.0 / tf.maximum(disp, 1e-6)
        loss_i = loss_fn(pred_depth, gt)
        total_loss += alpha[i] * loss_i

    return total_loss
