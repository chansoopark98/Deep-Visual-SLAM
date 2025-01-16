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
from datetime import datetime
from tqdm import tqdm
import yaml
import csv

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
        self.config['Train']['batch_size'] = 1 # For evaluation, batch size should be 1
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

        # 5. Evaluation results
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join('./depth/results', current_time)
        self.rgb_dir = os.path.join(self.output_dir, 'rgb')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
    
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
        for idx, (rgb, depth) in enumerate(valid_tqdm):
            loss_dict, pred_depths = self.eval_step(rgb, depth)
 
            self.update_valid_metric(pred_depths[0], depth)

            denorm_rgb = self.data_loader.denormalize_image(rgb)
            batch_plot = plot_images(denorm_rgb, pred_depths, depth,
                                     mode='metric', depth_max=self.config['Train']['max_depth']).numpy()
            
            
            rounded_loss = round(float((sum(loss_dict.values()))), 3)
            
            batch_plot_path = os.path.join(self.rgb_dir, f'batch_{idx}_{rounded_loss}.png')
            tf.io.write_file(batch_plot_path, tf.image.encode_png(batch_plot[0]))
        
        # Save metrics to CSV
        metrics_dict = self.valid_depth_metrics.get_all_metrics()

        csv_path = os.path.join(self.output_dir, 'evaluation_metrics.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])
            for metric_name, metric_value in metrics_dict.items():
                print(f"{metric_name}: {metric_value}")
                writer.writerow([metric_name, metric_value])
       
        self.valid_depth_metrics.reset_states()

if __name__ == '__main__':
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    with tf.device('/device:GPU:0'):
        trainer = Evaluate(config=config)
        trainer.eval()