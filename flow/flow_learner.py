import tensorflow as tf

class FlowLearner(object):
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.gamma = 0.8
        self.lamda_l1 = 1.0
        self.lamda_epe = 0.1
        self.max_flow = config['Train']['max_flow']

    def sequence_loss(self, y_true, y_pred) -> dict:
        """
        Combined L1 loss and EPE loss for optical flow sequence predictions.
        Args:
            y_true: Tensor of ground truth optical flow, shape (bs, h, w, 2)
            y_pred: List of predicted optical flows, each with shape (bs, h, w, 2)
        Returns:
            Combined loss (L1 + EPE) over all predictions.
        """
        y_pred = tf.cast(y_pred, tf.float32)
        flow_gt = y_true

        # Create validity mask
        valid_cond_1 = tf.abs(flow_gt[:, :, :, 0]) < 1000
        valid_cond_2 = tf.abs(flow_gt[:, :, :, 1]) < 1000
        valid = valid_cond_1 & valid_cond_2

        # Magnitude-based validity
        mag = tf.sqrt(tf.reduce_sum(flow_gt**2, axis=-1))
        valid = valid & (mag < self.max_flow)
        valid = tf.expand_dims(tf.cast(valid, tf.float32), axis=-1)

        # Initialize losses
        n_predictions = len(y_pred)
        flow_loss_l1 = 0.0
        flow_loss_epe = 0.0

        # Weight for each prediction
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            
            # L1 Loss
            i_loss_l1 = tf.abs(y_pred[i] - flow_gt)
            flow_loss_l1 += i_weight * tf.reduce_mean(valid * i_loss_l1)
            
            # EPE Loss
            i_loss_epe = tf.sqrt(tf.reduce_sum((y_pred[i] - flow_gt)**2, axis=-1, keepdims=True))
            flow_loss_epe += i_weight * tf.reduce_mean(valid * i_loss_epe)

        # Combine losses
        loss_dict = {
            'l1_loss': self.lamda_l1 * flow_loss_l1,
            'epe_loss': self.lamda_epe * flow_loss_epe
        }
        return loss_dict

    def forward_step(self, left, right, flow, training=True) -> list[dict, list]:
        inputs = tf.concat([left, right], axis=-1)
        pred_flows = self.model(inputs, training=training)
        loss_dict = self.sequence_loss(flow, pred_flows)
        return loss_dict, pred_flows