import tensorflow as tf

class FlowLearner(object):
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.gamma = 0.8
        self.lamda_l1 = 1.0
        self.lamda_epe = 0.1
        self.max_flow = config['Train']['max_flow']

    @tf.function(jit_compile=True)
    def sequence_loss(self, y_true, y_pred, valid) -> dict:
        y_pred = tf.cast(y_pred, tf.float32)
        flow_gt = y_true

        # Initialize losses
        n_predictions = len(y_pred)
        flow_loss_l1 = 0.0

        # Weight for each prediction
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            
            # L1 Loss
            i_loss_l1 = tf.abs(y_pred[i] - flow_gt)
            flow_loss_l1 += i_weight * tf.reduce_mean(valid * i_loss_l1)

        # Combine losses
        total_loss = self.lamda_l1 * flow_loss_l1
        return total_loss

    @tf.function()
    def forward_step(self, left, right, flow, mask, training=True) -> list[dict, list]:
        inputs = tf.concat([left, right], axis=-1)
        pred_flows = self.model(inputs, training=training)
        total_loss = self.sequence_loss(flow, pred_flows, mask)
        return total_loss, pred_flows