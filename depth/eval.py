import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import Dict, List, Tuple
from depth_learner import DepthLearner
from model.monodepth2 import DispNet
from util.plot import plot_images
from util.metrics import DepthMetrics
from dataset.data_loader import DataLoader
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import yaml

np.set_printoptions(suppress=True)

class Evaluate(object):
    def __init__(self, config: dict) -> None:
        """
        Initializes the Evaluate class.

        Args:
            config (dict): Configuration dictionary containing model, dataset, and evaluation settings.
        """
        self.config = config
        self.configure_default_ops()
        print('initialize')

    def configure_default_ops(self) -> None:
        """
        Configures default operations including model setup, dataset loading, and metric initialization.
        """
        # 1. Model
        self.batch_size = self.config['Train']['batch_size']
        self.model = DispNet(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                             batch_size=self.batch_size)

        model_input_shape = (self.config['Train']['batch_size'],
                             self.config['Train']['img_h'], self.config['Train']['img_w'], 3)
        self.model.build(model_input_shape)
        _ = self.model(tf.random.normal(model_input_shape))
        self.model.load_weights('./assets/weights/depth/nyu_diode_diml_metricDepth_ep30.h5')
        self.model.summary()

        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
    
        self.valid_dataset = self.data_loader.valid_dataset
        self.valid_samples = self.data_loader.num_valid_samples
        
        # 3. Learner
        self.learner = DepthLearner(model=self.model, config=self.config)

        # 4. Metrics
        self.valid_depth_metrics = DepthMetrics(mode=self.config['Train']['mode'],
                                                min_depth=self.config['Train']['min_depth'],
                                                max_depth=self.config['Train']['max_depth'],
                                                name='valid_depth_metrics')
    
    @tf.function()
    def eval_step(self, rgb: tf.Tensor, depth: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        loss_dict, pred_depths = self.learner.forward_step(rgb, depth, training=False)
        return loss_dict, pred_depths
    
    @tf.function()
    def update_valid_metric(self, pred_depth: tf.Tensor, gt_depth: tf.Tensor) -> None:
        self.valid_depth_metrics.update_state(gt_depth, pred_depth)

    def eval(self) -> None:
        # Evaluation
        valid_tqdm = tqdm(self.valid_dataset, total=self.valid_samples)
        valid_tqdm.set_description('Evaluation || ')
        for _, (rgb, depth) in enumerate(valid_tqdm):
            _, pred_depths = self.eval_step(rgb, depth)

            self.update_valid_metric(pred_depths[0], depth)

        metrics_dict = self.valid_depth_metrics.get_all_metrics()
        for metric_name, metric_value in metrics_dict.items():
            print(f"{metric_name}: {metric_value}")
        
        self.valid_depth_metrics.reset_states()

if __name__ == '__main__':
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    with tf.device('/device:GPU:0'):
        trainer = Evaluate(config=config)
        trainer.eval()