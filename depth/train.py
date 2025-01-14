import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from depth_learner import DepthLearner
from model.monodepth2 import DispNet
from util.plot import plot_images
from util.metrics import DepthMetrics
from dataset.data_loader import DataLoader
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import yaml

np.set_printoptions(suppress=True)

class Trainer(object):
    def __init__(self, config, strategy: tf.distribute.Strategy) -> None:
        self.config = config
        self.strategy = strategy
        self.configure_train_ops()
        print('initialize')

    def configure_train_ops(self) -> None:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        # 1. Model
        self.batch_size = self.config['Train']['batch_size']
        self.model = DispNet(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                             batch_size=self.batch_size)

        model_input_shape = (self.config['Train']['batch_size'],
                             self.config['Train']['img_h'], self.config['Train']['img_w'], 3)
        self.model.build(model_input_shape)
        _ = self.model(tf.random.normal(model_input_shape))
        self.model.load_weights('./assets/weights/depth/epoch_10_model.h5')
        self.model.summary()

        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
        self.train_dataset = self.data_loader.train_dataset
        self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
        self.train_samples = self.data_loader.num_train_samples

        self.valid_dataset = self.data_loader.valid_dataset
        self.valid_dataset = self.strategy.experimental_distribute_dataset(self.valid_dataset)
        self.valid_samples = self.data_loader.num_valid_samples
        
        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['final_lr'],
                                                                              power=self.config['Train']['power'])
        
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.config['Train']['init_lr'],
                                                  beta_1=self.config['Train']['beta1'],
                                                  weight_decay=self.config['Train']['weight_decay'] if self.config['Train']['weight_decay'] > 0 else None
                                                  ) # 
        
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # 4. Learner
        self.learner = DepthLearner(model=self.model, config=self.config)

        # 5. Metrics
        self.train_total_loss = tf.keras.metrics.Mean(name='train_total_loss')
        self.train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
        self.train_log_loss = tf.keras.metrics.Mean(name='train_log_loss')
        self.train_l1_loss = tf.keras.metrics.Mean(name='train_l1_loss')

        self.valid_total_loss = tf.keras.metrics.Mean(name='valid_total_loss')
        self.valid_smooth_loss = tf.keras.metrics.Mean(name='valid_smooth_loss')
        self.valid_log_loss = tf.keras.metrics.Mean(name='valid_log_loss')
        self.valid_l1_loss = tf.keras.metrics.Mean(name='valid_l1_loss')

        self.valid_depth_metrics = DepthMetrics(mode=self.config['Train']['mode'],
                                                min_depth=self.config['Train']['min_depth'],
                                                max_depth=self.config['Train']['max_depth'],
                                                name='valid_depth_metrics')

        # 7. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join('depth', self.config['Directory']['log_dir'] + \
            '/' + current_time + '_')
        self.train_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        os.makedirs('{0}/{1}'.format(self.config['Directory']['weights'],
                                     self.config['Directory']['exp_name']),
                    exist_ok=True)

    @tf.function()
    def train_step(self, rgb, depth):
        with tf.GradientTape() as tape:
            loss_dict, pred_depths = self.learner.forward_step(rgb, depth, training=True)
            total_loss = sum(loss_dict.values())
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_dict, pred_depths
    
    @tf.function()
    def validation_step(self, rgb, depth):
        loss_dict, pred_depths = self.learner.forward_step(rgb, depth, training=False)
        return loss_dict, pred_depths
    
    @tf.function()
    def distributed_train_step(self, rgb, depth):
        loss_dict, pred_depths = self.strategy.run(self.train_step, args=(rgb, depth,))
        
        reduced_loss_dict = {
            key: self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            for key, value in loss_dict.items()
            }
        return reduced_loss_dict, pred_depths
    
    @tf.function()
    def distributed_valid_step(self, rgb, depth):
        loss_dict, pred_depths = self.strategy.run(self.validation_step, args=(rgb, depth,))
        
        reduced_loss_dict = {
            key: self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            for key, value in loss_dict.items()
            }
        return reduced_loss_dict, pred_depths
    
    @tf.function()
    def update_train_metric(self, loss_dict):
        total_loss = tf.reduce_sum(list(loss_dict.values()))
        self.train_total_loss.update_state(total_loss)
        self.train_smooth_loss.update_state(loss_dict['smooth_loss'])
        self.train_log_loss.update_state(loss_dict['log_loss'])
        self.train_l1_loss.update_state(loss_dict['l1_loss'])

    @tf.function()
    def update_valid_metric(self, loss_dict, pred_depth, gt_depth):
        total_loss = tf.reduce_sum(list(loss_dict.values()))
        self.valid_total_loss.update_state(total_loss)
        self.valid_smooth_loss.update_state(loss_dict['smooth_loss'])
        self.valid_log_loss.update_state(loss_dict['log_loss'])
        self.valid_l1_loss.update_state(loss_dict['l1_loss'])
        self.valid_depth_metrics.update_state(gt_depth, pred_depth)

    def train(self) -> None:
        for epoch in range(self.config['Train']['epoch']):
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr

            train_tqdm = tqdm(self.train_dataset,
                              total=self.train_samples)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for idx, (rgb, depth) in enumerate(train_tqdm):
                train_loss_result, pred_train_depths = self.distributed_train_step(rgb, depth)

                # Update train metrics
                self.strategy.run(self.update_train_metric, args=(train_loss_result,))

                current_step = self.train_samples * epoch + idx

                if current_step % self.config['Train']['train_plot_interval'] == 0:
                    # Draw depth plot
                    local_rgb = self.strategy.experimental_local_results(rgb)[0]
                    local_depth = self.strategy.experimental_local_results(depth)[0]
                    local_pred_depth = self.strategy.experimental_local_results(pred_train_depths)[0]

                    target_image = self.data_loader.denormalize_image(local_rgb)

                    train_depth_plot = plot_images(image=target_image,
                                                   pred_depths=local_pred_depth,
                                                   gt_depth=local_depth,
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
                                    self.train_total_loss.result(), step=current_step)
                tf.summary.scalar(f'Train/{self.train_smooth_loss.name}',
                                    self.train_smooth_loss.result(), step=current_step)
                tf.summary.scalar(f'Train/{self.train_log_loss.name}',
                                    self.train_log_loss.result(), step=current_step)
                tf.summary.scalar(f'Train/{self.train_l1_loss.name}',
                                    self.train_l1_loss.result(), step=current_step)

            # Validation
            valid_tqdm = tqdm(self.valid_dataset,
                              total=self.valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (rgb, depth) in enumerate(valid_tqdm):
                valid_loss_result, pred_valid_depths = self.distributed_valid_step(rgb, depth)

                # Update valid metrics
                self.strategy.run(self.update_valid_metric, args=(valid_loss_result, pred_valid_depths[0], depth))

                current_step = self.valid_samples * epoch + idx
                    
                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    # Draw depth plot
                    local_rgb = self.strategy.experimental_local_results(rgb)[0]
                    local_depth = self.strategy.experimental_local_results(depth)[0]
                    local_pred_depth = self.strategy.experimental_local_results(pred_valid_depths)[0]

                    target_image = self.data_loader.denormalize_image(local_rgb)
                    valid_depth_plot = plot_images(image=target_image,
                                                   pred_depths=local_pred_depth,
                                                   gt_depth=local_depth,
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
                                          self.valid_total_loss.result(), step=current_step)
                        tf.summary.scalar(f'Valid/{self.valid_smooth_loss.name}',
                                            self.valid_smooth_loss.result(), step=current_step)
                        tf.summary.scalar(f'Valid/{self.valid_log_loss.name}',
                                            self.valid_log_loss.result(), step=current_step)
                        tf.summary.scalar(f'Valid/{self.valid_l1_loss.name}',
                                            self.valid_l1_loss.result(), step=current_step)            
                        
            with self.valid_summary_writer.as_default():
                metrics_dict = self.valid_depth_metrics.get_all_metrics()
                for metric_name, metric_value in metrics_dict.items():
                    tf.summary.scalar(f"Eval/{metric_name}", metric_value, step=current_step)

            if epoch % 5 == 0:
                self.model.save_weights('{0}/{1}/epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
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

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        trainer = Trainer(config=config, strategy=strategy)
        trainer.train()