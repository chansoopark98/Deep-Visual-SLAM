import tensorflow as tf, tf_keras

class DepthMetrics(tf_keras.metrics.Metric):
    def __init__(self, mode, min_depth, max_depth, name='depth_metrics', **kwargs):
        """
        Initializes DepthMetrics to evaluate depth estimation models.

        Args:
            mode (str): Evaluation mode ('relative' or 'metric').
            min_depth (float): Minimum valid depth value.
            max_depth (float): Maximum valid depth value.
            name (str): Name of the metric (default: 'depth_metrics').
            **kwargs: Additional arguments for tf.keras.metrics.Metric.
        """
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Variables for accumulation (in float32 format)
        self.sum_abs_diff = self.add_weight(name='sum_abs_diff', initializer='zeros', dtype=tf.float32)
        self.sum_abs_rel = self.add_weight(name='sum_abs_rel', initializer='zeros', dtype=tf.float32)
        self.sum_sq_rel  = self.add_weight(name='sum_sq_rel',  initializer='zeros', dtype=tf.float32)
        
        self.sum_mse = self.add_weight(name='sum_mse', initializer='zeros', dtype=tf.float32)
        self.sum_mse_log = self.add_weight(name='sum_mse_log', initializer='zeros', dtype=tf.float32)
        self.sum_abs_log = self.add_weight(name='sum_abs_log', initializer='zeros', dtype=tf.float32)

        self.sum_a1 = self.add_weight(name='sum_a1', initializer='zeros', dtype=tf.float32)
        self.sum_a2 = self.add_weight(name='sum_a2', initializer='zeros', dtype=tf.float32)
        self.sum_a3 = self.add_weight(name='sum_a3', initializer='zeros', dtype=tf.float32)

        self.total_count = self.add_weight(name='total_count', initializer='zeros', dtype=tf.float32)

    # @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        """
        Updates the metric states based on ground truth and predictions.

        Args:
            y_true (tf.Tensor): Ground truth depth tensor.
            y_pred (tf.Tensor): Predicted depth tensor.
            sample_weight (optional): Weights for the samples (default: None).
        """

        # Relative depth (min-max norm) to metric depth
        if self.mode == 'relative':
            y_true = y_true * (self.max_depth - self.min_depth) + self.min_depth
            y_pred = y_pred * (self.max_depth - self.min_depth) + self.min_depth
            
            y_true = tf.clip_by_value(y_true, self.min_depth, self.max_depth)
            y_pred = tf.clip_by_value(y_pred, self.min_depth, self.max_depth)

        # Flatten for calculation (entire batch)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # clip by value (to avoid NaN)
        epsilon = 1e-6
        y_true = tf.clip_by_value(y_true, epsilon, 100.)
        y_pred = tf.clip_by_value(y_pred, epsilon, 100.)

        # thresh = max(gt/pred, pred/gt)
        thresh = tf.maximum(y_true / y_pred, y_pred / y_true)

        # (thresh < 1.25), 1.25^2, 1.25^3
        a1 = tf.reduce_mean(tf.cast(thresh < 1.25, tf.float32))
        a2 = tf.reduce_mean(tf.cast(thresh < (1.25**2), tf.float32))
        a3 = tf.reduce_mean(tf.cast(thresh < (1.25**3), tf.float32))

        # abs_diff
        abs_diff = tf.reduce_mean(tf.abs(y_true - y_pred))
        # abs_rel
        abs_rel = tf.reduce_mean(tf.abs(y_true - y_pred) / y_true)
        # sq_rel
        sq_rel = tf.reduce_mean(((y_true - y_pred)**2) / y_true)

        # mse (RMSE)
        mse = tf.reduce_mean((y_true - y_pred)**2)
        
        # log(gt) - log(pred)
        mse_log = tf.reduce_mean((tf.math.log(y_true) - tf.math.log(y_pred))**2)
        # abs_log
        abs_log = tf.reduce_mean(tf.abs(tf.math.log(y_true) - tf.math.log(y_pred)))

        batch_size = tf.cast(tf.size(y_true), tf.float32)

        # 누적
        self.sum_abs_diff.assign_add(abs_diff * batch_size)
        self.sum_abs_rel.assign_add(abs_rel * batch_size)
        self.sum_sq_rel.assign_add(sq_rel * batch_size)
        self.sum_mse.assign_add(mse * batch_size)
        self.sum_mse_log.assign_add(mse_log * batch_size)
        self.sum_abs_log.assign_add(abs_log * batch_size)
        self.sum_a1.assign_add(a1 * batch_size)
        self.sum_a2.assign_add(a2 * batch_size)
        self.sum_a3.assign_add(a3 * batch_size)
        self.total_count.assign_add(batch_size)

    def result(self) -> tf.Tensor:
        """
        Computes and returns the root mean squared error (RMSE) based on accumulated states.

        Returns:
            tf.Tensor: RMSE value.
        """
        count = tf.maximum(self.total_count, 1.0)
        # RMSE
        rmse = tf.sqrt(self.sum_mse / count)
        return rmse

    def get_all_metrics(self) -> dict:
        """
        Returns all computed metrics as a dictionary.

        Returns:
            dict: Dictionary containing all metrics ('abs_diff', 'rmse', 'a1', etc.).
        """

        count = tf.maximum(self.total_count, 1.0)

        abs_diff = self.sum_abs_diff / count
        abs_rel  = self.sum_abs_rel  / count
        sq_rel   = self.sum_sq_rel   / count
        
        rmse = tf.sqrt(self.sum_mse / count)
        rmse_log = tf.sqrt(self.sum_mse_log / count)
        abs_log = self.sum_abs_log / count

        a1 = self.sum_a1 / count
        a2 = self.sum_a2 / count
        a3 = self.sum_a3 / count

        return {
            'abs_diff': abs_diff,
            'abs_rel': abs_rel,
            'sq_rel':  sq_rel,
            'rmse':    rmse,
            'rmse_log': rmse_log,
            'abs_log': abs_log,
            'a1':      a1,
            'a2':      a2,
            'a3':      a3
        }

    def reset_states(self) -> None:
        """
        Resets all accumulated states for the metrics. Typically called at the end of each epoch.

        Returns:
            None
        """
        for var in self.variables:
            var.assign(tf.zeros_like(var))

# test
if __name__ == '__main__':
    depth_metrics = DepthMetrics(mode='metric', min_depth=0.1, max_depth=100.0)
    y_true = tf.random.uniform((4, 256, 256, 1), 0.1, 100.0)
    y_pred = tf.random.uniform((4, 256, 256, 1), 0.1, 100.0)
    depth_metrics.update_state(y_true, y_pred)
    print(depth_metrics.result())
    print(depth_metrics.get_all_metrics())
    depth_metrics.reset_states()
    print(depth_metrics.result())
    print(depth_metrics.get_all_metrics())
    print