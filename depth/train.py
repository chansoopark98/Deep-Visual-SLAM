import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import Dict, Any, List, Tuple
from depth_learner import DepthLearner
# from depth_learner_test import DepthLearner
from model.depth_net import DispNet
from util.plot import plot_images
from util.metrics import DepthMetrics
from dataset.data_loader import DataLoader
from datetime import datetime
import tensorflow as tf, tf_keras
import keras
from tqdm import tqdm
import numpy as np
import yaml

np.set_printoptions(suppress=True)

class Trainer(object):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.configure_train_ops()
        print('initialize')

    def configure_train_ops(self) -> None:
        """
        Configures training operations including model, dataset, optimizer, metrics, and logger.

        - Sets mixed precision policy.
        - Builds and initializes the model with predefined shapes.
        - Loads training and validation datasets.
        - Configures optimizer with learning rate schedule and loss scaling.
        - Initializes metrics for training and validation.

        Returns:
            None
        """
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

        # 1. Model
        self.batch_size = self.config['Train']['batch_size']
        self.model = DispNet(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                             batch_size=self.batch_size)

        model_input_shape = (self.config['Train']['batch_size'],
                             self.config['Train']['img_h'], self.config['Train']['img_w'], 5)
        self.model.build(model_input_shape)
        _ = self.model(tf.random.normal(model_input_shape))

        if self.config['Train']['mode'] == 'metric':
            self.model.load_weights('./assets/weights/depth/metric_epoch_45_model.weights.h5', skip_mismatch=True) # Pretrained relative depth weights
        self.model.summary()

        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
        self.train_dataset = self.data_loader.train_dataset
        self.train_samples = self.data_loader.num_train_samples

        self.valid_dataset = self.data_loader.valid_dataset
        self.valid_samples = self.data_loader.num_valid_samples
        
        # 3. Optimizer
        self.scheduler = keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['final_lr'],
                                                                              power=self.config['Train']['power'])
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                                  beta_1=self.config['Train']['beta1'],
                                                  weight_decay=self.config['Train']['weight_decay'] if self.config['Train']['weight_decay'] > 0 else None
                                                  )
        
        self.optimizer = keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # 4. Learner
        self.learner = DepthLearner(model=self.model, config=self.config)

        # 5. Metrics
        self.train_total_loss = tf_keras.metrics.Mean(name='train_total_loss')
        self.train_smooth_loss = tf_keras.metrics.Mean(name='train_smooth_loss')
        self.train_log_loss = tf_keras.metrics.Mean(name='train_log_loss')
        self.train_l1_loss = tf_keras.metrics.Mean(name='train_l1_loss')

        self.valid_total_loss = tf_keras.metrics.Mean(name='valid_total_loss')
        self.valid_smooth_loss = tf_keras.metrics.Mean(name='valid_smooth_loss')
        self.valid_log_loss = tf_keras.metrics.Mean(name='valid_log_loss')
        self.valid_l1_loss = tf_keras.metrics.Mean(name='valid_l1_loss')

        self.valid_depth_metrics = DepthMetrics(mode=self.config['Train']['mode'],
                                                min_depth=self.config['Train']['min_depth'],
                                                max_depth=self.config['Train']['max_depth'],
                                                name='valid_depth_metrics')

        # 7. Logger
        depth_train_type = self.config['Train']['mode']
        if depth_train_type not in ['relative', 'metric']:
            raise ValueError("Invalid depth training type. Choose 'relative' or 'metric'.")
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join('depth', self.config['Directory']['log_dir'], 
                        depth_train_type, current_time + '_')
        self.train_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        self.save_path = '{0}/depth/{1}'.format(self.config['Directory']['weights'],
                                     self.config['Directory']['exp_name'])
        os.makedirs(self.save_path, exist_ok=True)

    @tf.function(jit_compile=True)
    def train_step(self, rgb: tf.Tensor, depth: tf.Tensor, intrinsic) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """
        Executes a single training step with backpropagation.

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [B, H, W, 3].
            depth (tf.Tensor): Ground truth depth tensor of shape [B, H, W] or [B, H, W, 1].

        Returns:
            Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
                - Dictionary of loss values (e.g., smooth, log, and L1 losses).
                - List of predicted depth tensors at different scales.
        """
        with tf.GradientTape() as tape:
            loss_dict, pred_depths = self.learner.forward_step(rgb, depth, intrinsic, training=True)
            total_loss = sum(loss_dict.values())
            scaled_loss = self.optimizer.scale_loss(total_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(scaled_gradients, self.model.trainable_variables))
        return loss_dict, pred_depths
    
    @tf.function(jit_compile=True)
    def validation_step(self, rgb: tf.Tensor, depth: tf.Tensor, intrinsic) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """
        Computes loss and predicted depths for validation data.

        Args:
            rgb (tf.Tensor): Input RGB tensor of shape [B, H, W, 3].
            depth (tf.Tensor): Ground truth depth tensor of shape [B, H, W] or [B, H, W, 1].

        Returns:
            Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
                - Loss dictionary containing smooth, log, and L1 losses.
                - List of predicted depth tensors at different scales.
        """
        loss_dict, pred_depths = self.learner.forward_step(rgb, depth, intrinsic, training=False)
        return loss_dict, pred_depths
    
    @tf.function()
    def update_train_metric(self, loss_dict: Dict[str, tf.Tensor]) -> None:
        """
        Updates training metrics with the loss values.

        Args:
            loss_dict (Dict[str, tf.Tensor]): Dictionary containing smooth, log, and L1 losses.

        Returns:
            None
        """
        total_loss = tf.reduce_sum(list(loss_dict.values()))
        self.train_total_loss.update_state(total_loss)
        self.train_smooth_loss.update_state(loss_dict['smooth_loss'])
        self.train_log_loss.update_state(loss_dict['log_loss'])
        self.train_l1_loss.update_state(loss_dict['l1_loss'])

    @tf.function()
    def update_valid_metric(self, loss_dict: Dict[str, tf.Tensor], pred_depth: tf.Tensor, gt_depth: tf.Tensor) -> None:
        """
        Updates validation metrics with the loss values and depth predictions.

        Args:
            loss_dict (Dict[str, tf.Tensor]): Dictionary containing smooth, log, and L1 losses.
            pred_depth (tf.Tensor): Predicted depth tensor.
            gt_depth (tf.Tensor): Ground truth depth tensor.

        Returns:
            None
        """
        total_loss = tf.reduce_sum(list(loss_dict.values()))
        self.valid_total_loss.update_state(total_loss)
        self.valid_smooth_loss.update_state(loss_dict['smooth_loss'])
        self.valid_log_loss.update_state(loss_dict['log_loss'])
        self.valid_l1_loss.update_state(loss_dict['l1_loss'])
        self.valid_depth_metrics.update_state(gt_depth, pred_depth)

    def train(self) -> None:
        """
        Trains the model over multiple epochs with distributed data and logging.

        - Executes training and validation for each epoch.
        - Logs metrics, images, and depth predictions to TensorBoard.
        - Saves model weights at regular intervals.

        Args:
            None

        Returns:
            None
        """
        for epoch in range(self.config['Train']['epoch']):
            lr = self.scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr

            train_tqdm = tqdm(self.train_dataset,
                              total=self.train_samples)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for idx, (rgb, depth, intrinsic) in enumerate(train_tqdm):
                train_loss_result, pred_train_depths = self.train_step(rgb, depth, intrinsic)

                # Update train metrics
                self.update_train_metric(train_loss_result)

                current_step = self.train_samples * epoch + idx

                if current_step % self.config['Train']['train_plot_interval'] == 0:
                    # Draw depth plot
                    target_image = self.data_loader.denormalize_image(rgb)

                    train_depth_plot = plot_images(image=target_image,
                                                   pred_depths=pred_train_depths,
                                                   gt_depth=depth,
                                                   mode=self.config['Train']['mode'],
                                                   depth_max=self.config['Train']['max_depth'])

                    with self.train_summary_writer.as_default():
                        # Logging train images
                        tf.summary.image('Train/Depth Result',
                                         train_depth_plot, step=current_step)
                        del train_depth_plot

                train_tqdm.set_postfix(
                    total_loss=self.train_total_loss.result().numpy(),
                    smooth_loss=self.train_smooth_loss.result().numpy()
                    )
                
            # End train session
            with self.train_summary_writer.as_default():
                # Logging train total, pixel, smooth loss
                tf.summary.scalar(f'Train/{self.train_total_loss.name}' ,
                                    self.train_total_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_smooth_loss.name}',
                                    self.train_smooth_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_log_loss.name}',
                                    self.train_log_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_l1_loss.name}',
                                    self.train_l1_loss.result(), step=epoch)

            # Validation
            valid_tqdm = tqdm(self.valid_dataset,
                              total=self.valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (rgb, depth, intrinsic) in enumerate(valid_tqdm):
                valid_loss_result, pred_valid_depths = self.validation_step(rgb, depth, intrinsic)

                # Update valid metrics
                self.update_valid_metric(valid_loss_result, pred_valid_depths[0], depth)

                current_step = self.valid_samples * epoch + idx
                    
                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    # Draw depth plot
                    target_image = self.data_loader.denormalize_image(rgb)
                    valid_depth_plot = plot_images(image=target_image,
                                                   pred_depths=pred_valid_depths,
                                                   gt_depth=depth,
                                                   mode=self.config['Train']['mode'],
                                                   depth_max=self.config['Train']['max_depth'])

                    with self.valid_summary_writer.as_default():
                        # Logging valid images
                        tf.summary.image('Valid/Depth Result',
                                         valid_depth_plot, step=current_step)
                        del valid_depth_plot

                valid_tqdm.set_postfix(
                    total_loss=self.valid_total_loss.result().numpy(),
                )
            
            # End valid session
            with self.valid_summary_writer.as_default():
                        # Logging valid total, pixel, smooth loss
                        tf.summary.scalar(f'Valid/{self.valid_total_loss.name}',
                                          self.valid_total_loss.result(), step=epoch)
                        tf.summary.scalar(f'Valid/{self.valid_smooth_loss.name}',
                                            self.valid_smooth_loss.result(), step=epoch)
                        tf.summary.scalar(f'Valid/{self.valid_log_loss.name}',
                                            self.valid_log_loss.result(), step=epoch)
                        tf.summary.scalar(f'Valid/{self.valid_l1_loss.name}',
                                            self.valid_l1_loss.result(), step=epoch)            
                        
            with self.valid_summary_writer.as_default():
                metrics_dict = self.valid_depth_metrics.get_all_metrics()
                for metric_name, metric_value in metrics_dict.items():
                    tf.summary.scalar(f"Eval/{metric_name}", metric_value, step=epoch)

            if epoch % self.config['Train']['save_freq'] == 0:
                self.model.save_weights(self.save_path + '/{0}_epoch_{1}_model.weights.h5'.format(self.config['Train']['mode'],
                                                                                          epoch))
                
            # Reset metrics
            self.train_total_loss.reset_states()
            self.train_smooth_loss.reset_states()
            self.train_log_loss.reset_states()
            self.train_l1_loss.reset_states()

            self.valid_total_loss.reset_states()
            self.valid_smooth_loss.reset_states()
            self.valid_log_loss.reset_states()
            self.valid_l1_loss.reset_states()
            self.valid_depth_metrics.reset_states()

if __name__ == '__main__':
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get GPU configuration and set visible GPUs
    gpu_config = config.get('Experiment', {})
    visible_gpus = gpu_config.get('gpus', [])
    gpu_vram = gpu_config.get('gpu_vram', None)
    gpu_vram_factor = gpu_config.get('gpu_vram_factor', None)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            if visible_gpus:
                selected_gpus = [gpus[i] for i in visible_gpus]
                tf.config.set_visible_devices(selected_gpus, 'GPU')
            else:
                print("No GPUs specified in config. Using all available GPUs.")
                selected_gpus = gpus
            
            if gpu_vram and gpu_vram_factor:
                for gpu in selected_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_vram * gpu_vram_factor)]
                    )
            
            print(f"Using GPUs: {selected_gpus}")
        except RuntimeError as e:
            print(f"Error during GPU configuration: {e}")
    else:
        print('No GPU devices found')
        raise SystemExit

    trainer = Trainer(config=config)
    trainer.train()