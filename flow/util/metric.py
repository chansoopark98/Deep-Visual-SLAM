import tensorflow as tf

class EndPointError(tf.keras.metrics.Metric):
    ''' Calculates end-point-error and relating metrics '''
    def __init__(self, max_flow=400, **kwargs):
        super().__init__(**kwargs)
        self.max_flow = max_flow

        self.epe = self.add_weight(name='epe', initializer='zeros')
        self.u1 = self.add_weight(name='u1', initializer='zeros')
        self.u3 = self.add_weight(name='u3', initializer='zeros')
        self.u5 = self.add_weight(name='u5', initializer='zeros')

        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        flow_gt, valid = y_true
        
        # exclude invalid pixels and extremely large displacements
        mag = tf.sqrt(tf.reduce_sum(flow_gt**2, axis=-1))
        valid = valid & (mag < self.max_flow)

        epe = tf.sqrt(tf.reduce_sum((y_pred[-1] - flow_gt)**2, axis=-1))
        epe = epe[valid]
        rate_under1 = tf.cast(epe < 1, dtype=tf.float32)
        rate_under3 = tf.cast(epe < 3, dtype=tf.float32)
        rate_under5 = tf.cast(epe < 5, dtype=tf.float32)

        self.epe.assign_add(tf.reduce_mean(epe))
        self.u1.assign_add(tf.reduce_mean(rate_under1))
        self.u3.assign_add(tf.reduce_mean(rate_under3))
        self.u5.assign_add(tf.reduce_mean(rate_under5))
        self.count.assign_add(1)

    def result(self):
        count = tf.cast(self.count, dtype=self.epe.dtype)
        result = {
            'epe': self.epe / count,
            'u1': self.u1 / count,
            'u3': self.u3 / count,
            'u5': self.u5 / count
        }
        return result