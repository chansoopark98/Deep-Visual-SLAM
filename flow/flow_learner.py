import tensorflow as tf

class FlowLearner(object):
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.max_flow = config['Train']['max_flow']

    @tf.function(jit_compile=True)
    def sequence_loss(self, y_true, y_pred, gamma=0.8):
        y_pred = tf.cast(y_pred, tf.float32)
        flow_gt = y_true
        
        valid_cond_1 = tf.abs(flow_gt[:, :, :, 0]) < 1000
        valid_cond_2 = tf.abs(flow_gt[:, :, :, 1]) < 1000
        valid = valid_cond_1 & valid_cond_2

        n_predictions = len(y_pred)
        flow_loss = 0.0

        # exclude invalid pixels and extremely large displacements
        mag = tf.sqrt(tf.reduce_sum(flow_gt**2, axis=-1))
        valid = valid & (mag < self.max_flow)
        # as float and expand channel axis
        valid = tf.expand_dims(tf.cast(valid, tf.float32), axis=-1)

        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            i_loss = tf.abs(y_pred[i] - flow_gt)
            flow_loss += i_weight * tf.reduce_mean(valid * i_loss)

        return flow_loss

    @tf.function(jit_compile=True)
    def forward_step(self, left, right, flow, training=True) -> list[dict, list]:
        pred_flows = self.model([left, right], training=training)
        loss_dict = {
            'flow_loss': self.sequence_loss(flow, pred_flows)
        }
        
        return loss_dict, pred_flows