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
        """
        Computes multi-scale loss including smoothness, SILog, and L1 losses.

        Args:
            pred_depths (List[tf.Tensor]): List of predicted depth maps at different scales.
            pred_disps (List[tf.Tensor]): List of predicted disparity maps at different scales.
            gt_depth (tf.Tensor): Ground truth depth map.
            rgb (tf.Tensor): RGB image for edge-aware smoothness.
            valid_mask (tf.Tensor): Boolean mask indicating valid pixels.

        Returns:
            Dict[str, tf.Tensor]: Dictionary containing the computed loss values.
        """
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
    
    def create_normalized_coords(self, rgb_tensor, K_tensor):
        """
        rgb_tensor: (B, H, W, 3)
        K_tensor:   (B, 3, 3)
        return:     (B, H, W, 2) normalized xy
        """
        # 현재 텐서플로 그래프 상에서 동적으로 shape을 추출
        shape = tf.shape(rgb_tensor)
        b = shape[0]
        h = shape[1]
        w = shape[2]

        # 픽셀 그리드 생성 (y, x)
        # meshgrid() -> shape=(H, W)
        # tf.range를 사용해 0..H-1, 0..W-1
        y_coords = tf.linspace(0.0, tf.cast(h - 1, tf.float32), h)
        x_coords = tf.linspace(0.0, tf.cast(w - 1, tf.float32), w)
        # shape=(H, W)
        xx, yy = tf.meshgrid(x_coords, y_coords, indexing='xy')

        # (H, W, 1) -> (1, H, W, 1) batch broadcast를 위해
        yy = yy[tf.newaxis, ..., tf.newaxis]
        xx = xx[tf.newaxis, ..., tf.newaxis]

        # Intrinsics로부터 focal, principal point 추출
        # tf.split 등으로 fx, fy, cx, cy를 구해서 shape=(B,1,1,1)로 만듦
        fx = K_tensor[:, 0, 0][:, tf.newaxis, tf.newaxis, tf.newaxis]  # (B,1,1,1)
        fy = K_tensor[:, 1, 1][:, tf.newaxis, tf.newaxis, tf.newaxis]
        cx = K_tensor[:, 0, 2][:, tf.newaxis, tf.newaxis, tf.newaxis]
        cy = K_tensor[:, 1, 2][:, tf.newaxis, tf.newaxis, tf.newaxis]

        # 모든 batch에 대해 동일한 (xx, yy)를 사용해야 하므로 tile
        xx_tiled = tf.tile(xx, [b, 1, 1, 1])  # (B,H,W,1)
        yy_tiled = tf.tile(yy, [b, 1, 1, 1])  # (B,H,W,1)

        # 정규화 (x - cx) / fx, (y - cy) / fy
        x_normalized = (xx_tiled - cx) / fx
        y_normalized = (yy_tiled - cy) / fy

        # concat하여 (B,H,W,2)
        coords = tf.concat([x_normalized, y_normalized], axis=-1)
        return coords
    
    def build_coord_channels(self, img, K):
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        # 픽셀 그리드 생성 (0 ~ W-1, 0 ~ H-1)
        x_coords = tf.linspace(0.0, tf.cast(W-1, tf.float32), W)
        y_coords = tf.linspace(0.0, tf.cast(H-1, tf.float32), H)
        X, Y = tf.meshgrid(x_coords, y_coords)
        X = tf.expand_dims(X, axis=0)  # (1,H,W)
        Y = tf.expand_dims(Y, axis=0)
        # K에서 focal length와 principal point 추출
        fx = K[0,0]; fy = K[1,1]
        cx = K[0,2]; cy = K[1,2]
        # Centered Coordinate: (u-cx), (v-cy)
        cc_x = (X - cx) / fx   # fx로 나눠서 정규화 (라디안당 픽셀)
        cc_y = (Y - cy) / fy
        # 화각 기반 각도 맵 (atan 사용)
        fov_x = tf.math.atan(cc_x)
        fov_y = tf.math.atan(cc_y)
        # -1~1 정규화 좌표 (CoordConv 용)
        X_norm = (X / tf.cast(W-1, tf.float32)) * 2.0 - 1.0
        Y_norm = (Y / tf.cast(H-1, tf.float32)) * 2.0 - 1.0
        # 채널 스택: [cc_x, cc_y, fov_x, fov_y, X_norm, Y_norm]
        return tf.stack([cc_x, cc_y, fov_x, fov_y, X_norm, Y_norm], axis=-1)

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
        # coord_map = self.create_normalized_coords(rgb, intrinsic)
        # test_coord_map = self.build_coord_channels(rgb, intrinsic)
        # model_input = tf.concat([rgb, coord_map], axis=-1)
        pred_disps = self.model([rgb, intrinsic], training=training)

        valid_mask = (depth > 0.) & (depth < (1. if self.train_mode == 'relative' else self.max_depth))

        pred_depths = [self.disp_to_depth(disp) for disp in pred_disps]

        loss_dict = self.multi_scale_loss(
            pred_depths=pred_depths,
            gt_depth=depth,
            rgb=rgb,
            valid_mask=valid_mask
        )

        return loss_dict, pred_depths