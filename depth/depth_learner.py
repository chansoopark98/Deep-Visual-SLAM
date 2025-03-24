from typing import Dict, Any, List, Tuple
import tensorflow as tf, tf_keras

class DepthLearner:
    def __init__(self, model: tf_keras.Model, config: Dict[str, Any]) -> None:
        """
        Initializes the DepthLearner class.

        Args:
            model (tf.keras.Model): The Keras model for training and inference.
            config (Dict[str, Any]): Configuration dictionary containing training parameters.
        """
        self.model = model
        self.train_mode: str = config['Train']['mode']  # 'relative' or 'metric'
        self.min_depth: float = config['Train']['min_depth']  # Minimum depth (e.g., 0.1)
        self.max_depth: float = config['Train']['max_depth']  # Maximum depth (e.g., 10.0)
        self.num_scales: int = 4  # Number of scales used

    @tf.function(jit_compile=True)
    def disp_to_depth(self, disp: tf.Tensor) -> tf.Tensor:
        """
        Converts disparity to depth.

        Args:
            disp (tf.Tensor): Input disparity map.

        Returns:
            tf.Tensor: Converted depth map.
        """
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / scaled_disp
        return tf.cast(depth, tf.float32)

    @tf.function(jit_compile=True)
    def scaled_depth_to_disp(self, depth: tf.Tensor) -> tf.Tensor:
        """
        Converts scaled depth to disparity.

        Args:
            depth (tf.Tensor): Input depth map.

        Returns:
            tf.Tensor: Converted disparity map.
        """
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = 1.0 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return tf.cast(disp, tf.float32)

    @tf.function(jit_compile=True)
    def get_smooth_loss(self, disp: tf.Tensor, img: tf.Tensor) -> tf.Tensor:
        """
        Computes the edge-aware smoothness loss.

        Args:
            disp (tf.Tensor): Disparity map.
            img (tf.Tensor): Reference image.

        Returns:
            tf.Tensor: Smoothness loss value.
        """
        disp_mean = tf.reduce_mean(disp, axis=[1, 2], keepdims=True) + 1e-7
        norm_disp = disp / disp_mean

        disp_dx, disp_dy = self.compute_gradients(norm_disp)
        img_dx, img_dy = self.compute_gradients(img)

        weight_x = tf.exp(-tf.reduce_mean(img_dx, axis=3, keepdims=True))
        weight_y = tf.exp(-tf.reduce_mean(img_dy, axis=3, keepdims=True))

        smoothness_x = disp_dx * weight_x
        smoothness_y = disp_dy * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    @tf.function(jit_compile=True)
    def compute_gradients(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Computes gradients in the x and y directions for a tensor.

        Args:
            tensor (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Gradients in the x and y directions.
        """
        tensor_dx = tf.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
        tensor_dy = tf.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        return tensor_dx, tensor_dy

    @tf.function(jit_compile=True)
    def l1_loss(self, pred: tf.Tensor, gt: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
        """
        Computes the L1 loss for valid pixels only.

        Args:
            pred (tf.Tensor): Predicted depth map.
            gt (tf.Tensor): Ground truth depth map.
            valid_mask (tf.Tensor): Boolean mask indicating valid pixels.

        Returns:
            tf.Tensor: Scalar L1 loss value.
        """
        abs_diff = tf.abs(pred - gt)
        masked_abs_diff = tf.boolean_mask(abs_diff, valid_mask)
        return tf.reduce_mean(masked_abs_diff)

    # @tf.function(jit_compile=True)
    def silog_loss(self, prediction: tf.Tensor, target: tf.Tensor, valid_mask: tf.Tensor,
                   variance_focus: float = 0.5) -> tf.Tensor:
        """
        Computes the scale-invariant logarithmic (SILog) loss.

        Args:
            prediction (tf.Tensor): Predicted depth map.
            target (tf.Tensor): Ground truth depth map.
            valid_mask (tf.Tensor): Boolean mask indicating valid pixels.
            variance_focus (float): Weight for the variance term in SILog loss (default: 0.5).

        Returns:
            tf.Tensor: Scalar SILog loss value.
        """
        eps = 1e-6
        prediction = tf.maximum(prediction, eps)

        valid_prediction = tf.boolean_mask(prediction, valid_mask)
        valid_target = tf.boolean_mask(target, valid_mask)

        d = tf.math.log(valid_prediction) - tf.math.log(valid_target)
        d2_mean = tf.reduce_mean(tf.square(d))
        d_mean = tf.reduce_mean(d)
        silog_expr = d2_mean - variance_focus * tf.square(d_mean)

        return tf.sqrt(silog_expr)

    # @tf.function(jit_compile=True)
    def multi_scale_loss(self, pred_depths: List[tf.Tensor], gt_depth: tf.Tensor,
                         rgb: tf.Tensor, valid_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        alpha = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
        smooth_losses, log_losses, l1_losses = 0.0, 0.0, 0.0

        smooth_loss_factor = 1.0
        log_loss_factor = 1.0
        l1_loss_factor = 0.1

        original_shape = gt_depth.shape[1:3]

        for i in range(self.num_scales):
            pred_depth = pred_depths[i]

            pred_depth_resized = tf.image.resize(
                pred_depth, original_shape, method=tf.image.ResizeMethod.BILINEAR
            )

            resized_disp = self.scaled_depth_to_disp(pred_depth_resized)

            smooth_losses += self.get_smooth_loss(resized_disp, rgb) * alpha[i]
            log_losses += self.silog_loss(pred_depth_resized, gt_depth, valid_mask) * alpha[i]
            # l1_losses += self.l1_loss(pred_depth_resized, gt_depth, valid_mask) * alpha[i]

        return {
            'smooth_loss': smooth_losses * smooth_loss_factor,
            'log_loss': log_losses * log_loss_factor,
            'l1_loss': l1_losses * l1_loss_factor
        }
    
    def forward_step(self, rgb: tf.Tensor, depth: tf.Tensor, intrinsic, training: bool = True
                    ) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """
        Performs a forward step, predicting depth and calculating losses.

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [B, H, W, 3].
            depth (tf.Tensor): Ground truth depth tensor of shape [B, H, W] or [B, H, W, 1].
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
                - Loss dictionary containing smoothness, SILog, and L1 losses.
                - List of predicted depth maps at different scales.
        """

        pred_disps = self.model(rgb, training=training)

        valid_mask = (depth > 0.) & (depth < (1. if self.train_mode == 'relative' else self.max_depth))

        pred_depths = [self.disp_to_depth(disp) for disp in pred_disps]

        loss_dict = self.multi_scale_loss(
            pred_depths=pred_depths,
            gt_depth=depth,
            rgb=rgb,
            valid_mask=valid_mask
        )

        return loss_dict, pred_depths