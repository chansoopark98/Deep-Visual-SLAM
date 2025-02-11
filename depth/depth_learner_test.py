import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Dict, Tuple

EPS = 1e-6

class DepthLearner:
    def __init__(self, model: tf.keras.Model, config: Dict[str, any]) -> None:
        """
        DepthLearner 클래스 초기화.
        Args:
            model (tf.keras.Model): 학습 및 추론에 사용할 모델.
            config (Dict[str, any]): 학습 관련 설정 (예: 'relative' 또는 'metric', min_depth, max_depth 등).
        """
        self.model = model
        self.train_mode: str = config['Train']['mode']  # 'relative' 또는 'metric'
        self.min_depth: float = config['Train']['min_depth']  # 예: 0.1
        self.max_depth: float = config['Train']['max_depth']  # 예: 10.0
        self.num_scales: int = 4  # 멀티스케일 예측 수

    def disp_to_depth(self, disp: tf.Tensor) -> tf.Tensor:
        """
        disparity를 depth로 변환.
        """
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / scaled_disp
        return tf.cast(depth, tf.float32)

    def scaled_depth_to_disp(self, depth: tf.Tensor) -> tf.Tensor:
        """
        depth를 disparity로 변환.
        """
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = 1.0 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return tf.cast(disp, tf.float32)

    def compute_midas_mae_loss(self, pred: tf.Tensor, gt: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        MiDaS 스타일의 scale/shift 불변 MAE 손실을 계산.
        - 예측(depth)와 GT(depth)를 각각 유효 픽셀에 대해 중앙값으로 정규화한 후 L1 오차를 계산.
        
        Args:
            pred: 예측 depth [B, H, W, 1].
            gt: 정답 depth [B, H, W, 1].
            mask: 유효 픽셀 마스크 (gt > 0).
            
        Returns:
            scale/shift 불변 MAE 손실 값.
        """
        # 유효 픽셀만 선택
        pred_valid = tf.boolean_mask(pred, mask)
        gt_valid   = tf.boolean_mask(gt, mask)
        
        # 중앙값 계산 (0이 되지 않도록 EPS 추가)
        m_pred = tf.maximum(tfp.stats.percentile(pred_valid, 50.0, interpolation='nearest'), EPS)
        m_gt   = tf.maximum(tfp.stats.percentile(gt_valid,   50.0, interpolation='nearest'), EPS)
        
        # 중앙값으로 정규화 (scale/shift 보정)
        pred_norm = pred / m_pred
        gt_norm   = gt   / m_gt
        
        # L1 손실 계산 (유효 픽셀에 대해서만)
        loss = tf.abs(pred_norm - gt_norm)
        loss = tf.reduce_mean(tf.boolean_mask(loss, mask))
        return loss

    def compute_gradient_loss(self, pred: tf.Tensor, gt: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        MiDaS 스타일의 scale 불변 gradient 손실을 계산.
        - 예측 depth와 GT depth에 대해 수평, 수직 방향의 gradient를 구하고, 그 차이를 L1 손실로 계산.
        
        Args:
            pred: 정규화된 예측 depth [B, H, W, 1].
            gt: 정규화된 정답 depth [B, H, W, 1].
            mask: 유효 픽셀 마스크 (gt > 0).
        
        Returns:
            gradient 손실 값.
        """
        # 수평, 수직 방향의 gradient (간단한 차분 방식)
        grad_x_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_y_pred = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        grad_x_gt   = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        grad_y_gt   = gt[:, 1:, :, :] - gt[:, :-1, :, :]
        
        # 마찬가지로, gradient에 해당하는 유효 마스크 (인접 픽셀 모두 유효한 경우)
        mask_x = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_y = mask[:, 1:, :, :] * mask[:, :-1, :, :]
        
        # gradient 차이 L1 손실 계산
        loss_x = tf.abs(grad_x_pred - grad_x_gt)
        loss_y = tf.abs(grad_y_pred - grad_y_gt)
        
        loss_x = tf.reduce_mean(tf.boolean_mask(loss_x, mask_x))
        loss_y = tf.reduce_mean(tf.boolean_mask(loss_y, mask_y))
        
        return (loss_x + loss_y) / 2.0

    def multi_scale_midas_loss(self, pred_depths: List[tf.Tensor], gt_depth: tf.Tensor) -> tf.Tensor:
        """
        멀티스케일에 대해 MiDaS 스타일의 손실(MAE + Gradient 손실)을 계산.
        각 스케일마다 GT를 해당 크기로 리사이즈한 후 손실을 계산하고, 스케일별 가중치를 곱하여 합산.
        
        Args:
            pred_depths: 여러 스케일에서 예측한 depth 리스트 (각 텐서 크기는 다를 수 있음).
            gt_depth: 원본 정답 depth, [B, H, W, 1].
            
        Returns:
            총 손실 값.
        """
        total_loss = 0.0
        # 각 스케일에 부여할 가중치 (필요에 따라 조정)
        alpha = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
        
        mae_losses = 0.
        smooth_losses = 0.

        for i, pred in enumerate(pred_depths):
            # 예측 스케일에 맞게 GT 리사이즈 (nearest 방법으로 마스크에도 영향 없이)
            pred_shape = tf.shape(pred)[1:3]
            gt_resized = tf.image.resize(gt_depth, pred_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            # 유효 마스크: gt가 0보다 큰 영역
            mask = tf.cast(gt_resized > 0, tf.float32)
            
            # MAE 손실 계산
            mae_loss = self.compute_midas_mae_loss(pred, gt_resized, mask)
            
            # 중앙값 정규화: 위에서 사용했던 것과 동일한 방식으로 정규화
            pred_valid = tf.boolean_mask(pred, mask)
            gt_valid   = tf.boolean_mask(gt_resized, mask)
            m_pred = tf.maximum(tfp.stats.percentile(pred_valid, 50.0, interpolation='nearest'), EPS)
            m_gt   = tf.maximum(tfp.stats.percentile(gt_valid,   50.0, interpolation='nearest'), EPS)
            pred_norm = pred / m_pred
            gt_norm   = gt_resized / m_gt
            
            # Gradient 손실 계산
            grad_loss = self.compute_gradient_loss(pred_norm, gt_norm, mask)
            
            mae_losses += alpha[i] * mae_loss
            smooth_losses += alpha[i] * grad_loss
        
        return {
            'smooth_loss': smooth_losses,
            'log_loss': mae_losses,
            'l1_loss': 0.
        }
        # return total_loss

    def forward_step(self, rgb: tf.Tensor, depth: tf.Tensor, training: bool = True
                    ) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """
        한 번의 forward step을 수행하여, 예측 depth를 계산하고, 멀티스케일 MiDaS 손실을 반환.
        
        Args:
            rgb: 입력 RGB 이미지, [B, H, W, 3].
            depth: 정답 depth, [B, H, W, 1] (또는 [B, H, W]).
            training: 학습 모드 여부.
            
        Returns:
            - 손실 딕셔너리 (여기서는 'midas_loss' 항목만 사용).
            - 예측 depth 리스트 (멀티스케일).
        """
        # 모델을 통해 disparity 예측 (멀티스케일 출력)
        pred_disps = self.model(rgb, training=training)
        
        # 유효 마스크: depth 값이 0보다 크고, (train_mode가 'relative'면 1, 아니면 max_depth)보다 작은 영역
        # valid_mask = (depth > 0.) & (depth < (1. if self.train_mode == 'relative' else self.max_depth))
        
        # 예측 disparity를 depth로 변환
        pred_depths = [self.disp_to_depth(disp) for disp in pred_disps]
        
        # MiDaS 스타일의 멀티스케일 손실 계산
        loss_dict = self.multi_scale_midas_loss(pred_depths, depth)
        
        return loss_dict, pred_depths
