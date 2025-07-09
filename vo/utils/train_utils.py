import keras
import tensorflow as tf

class StepLR(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, step_size, gamma):
        super(StepLR, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.step_size = step_size
        self.gamma = gamma
    
    def __call__(self, step):
        # Calculate the decay factor based on the step (epoch)
        decay_factor = self.gamma ** tf.math.floor(step / self.step_size)
        return self.initial_learning_rate * decay_factor
    
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'step_size': self.step_size,
            'gamma': self.gamma
        }
    