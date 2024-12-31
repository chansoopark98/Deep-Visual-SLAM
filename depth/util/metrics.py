import tensorflow as tf

class DepthMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='depth_metrics', **kwargs):
        super().__init__(name=name, **kwargs)

        # 누적을 위한 변수들 (float32 형태)
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

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true: GT depth, shape (B, ...)
        y_pred: Pred depth, shape (B, ...)
        둘 다 tf.float32라고 가정
        """
        # Flatten해서 계산 (배치 전체)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # 혹시나 0 값이 있을 수 있으므로, clip
        # (필요 시, 또는 Masking이 필요한 경우에는 별도 처리)
        epsilon = 1e-6
        y_true = tf.clip_by_value(y_true, epsilon, 1e6)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1e6)

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

        # mse (RMSE 위해)
        mse = tf.reduce_mean((y_true - y_pred)**2)
        # mse_log (RMSE log 위해)
        # log(gt) - log(pred)
        mse_log = tf.reduce_mean((tf.math.log(y_true) - tf.math.log(y_pred))**2)
        # abs_log
        abs_log = tf.reduce_mean(tf.abs(tf.math.log(y_true) - tf.math.log(y_pred)))

        batch_size = tf.cast(tf.size(y_true), tf.float32)  # 이 배치 내 픽셀(혹은 유효 depth)의 개수

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

    def result(self):
        """
        기본적으로 'rmse' (sqrt of mse)만 반환.
        나머지 지표는 get_all_metrics()로 별도 접근.
        """
        count = tf.maximum(self.total_count, 1.0)
        # RMSE
        rmse = tf.sqrt(self.sum_mse / count)
        return rmse

    def get_all_metrics(self):
        """
        누적된 값으로 최종 지표들을 dictionary 형태로 반환
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

    def reset_states(self):
        """Epoch 별로 metric 초기화"""
        for var in self.variables:
            var.assign(tf.zeros_like(var))
