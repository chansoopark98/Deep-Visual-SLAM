import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf, tf_keras
import keras
from model.flownet import Flownet
from flow_learner import FlowLearner
from util.metric import EndPointError
from util.plot import plot_images
from dataset.data_loader import DataLoader
from datetime import datetime
from tqdm import tqdm
import numpy as np
import yaml

np.set_printoptions(suppress=True)

class Trainer(object):
    def __init__(self, config) -> None:
        self.config = config
        self.configure_train_ops()
        print('initialize')

    def configure_train_ops(self) -> None:
        policy = tf_keras.mixed_precision.Policy('float32')
        tf_keras.mixed_precision.set_global_policy(policy)

        # 1. Model
        self.batch_size = self.config['Train']['batch_size']
        
        # self.model = RAFT()
        self.model = Flownet(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                             batch_size=self.config['Train']['batch_size'],
                             prefix='flownet'
                             )
        self.model.trainable = True

        model_input_shape = (self.config['Train']['batch_size'], self.config['Train']['img_h'], self.config['Train']['img_w'], 6)
        self.model.build(model_input_shape)
        _ = self.model(tf.random.normal(model_input_shape))
        self.model.summary(line_length=100)

        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
        self.train_dataset = self.data_loader.train_dataset
        self.train_samples = self.data_loader.num_train_samples

        self.valid_dataset = self.data_loader.valid_dataset
        self.valid_samples = self.data_loader.num_valid_samples
        
        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['final_lr'],
                                                                              power=self.config['Train']['power'])
        
        self.optimizer = keras.optimizers.AdamW(learning_rate=self.config['Train']['init_lr'],
                                                  beta_1=self.config['Train']['beta1'],
                                                  weight_decay=self.config['Train']['weight_decay'] if self.config['Train']['weight_decay'] > 0 else None)
        self.optimizer = keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # 4. Learner
        self.learner = FlowLearner(model=self.model, config=self.config)

        # 5. Metrics
        self.train_total_loss = tf_keras.metrics.Mean(name='train_total_loss')

        self.valid_total_loss = tf_keras.metrics.Mean(name='valid_total_loss')

        self.valid_flow_metrics = EndPointError(max_flow=self.config['Train']['max_flow'])
        
        # 7. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join('flow', self.config['Directory']['log_dir'] + \
            '/' + current_time + '_')
        self.train_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        self.save_path = '{0}/flow/{1}'.format(self.config['Directory']['weights'],
                                     self.config['Directory']['exp_name'])
        os.makedirs(self.save_path, exist_ok=True)

    @tf.function()
    def train_step(self, left, right, flow, mask):
        with tf.GradientTape() as tape:
            total_loss, pred_flows = self.learner.forward_step(left, right, flow, mask, training=True)
            scaled_loss = self.optimizer.scale_loss(total_loss)

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config['Train']['clip_norm'])
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss, pred_flows[-1]
    
    @tf.function()
    def validation_step(self, left, right, flow, mask):
        loss_dict, pred_flows = self.learner.forward_step(left, right, flow, mask, training=False)
        return loss_dict, pred_flows[-1]
    
    @tf.function()
    def update_train_metric(self, loss):
        self.train_total_loss.update_state(loss)

    @tf.function()
    def update_valid_metric(self, loss, pred_flow, gt_flow):
        self.valid_total_loss.update_state(loss)
        self.valid_flow_metrics.update_state(gt_flow, pred_flow)

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
            for idx, (left, right, flow, mask) in enumerate(train_tqdm):
                train_loss_result, pred_train_flow = self.train_step(left, right, flow, mask)
                
                # Update train metrics
                self.update_train_metric(train_loss_result)

                current_step = self.train_samples * epoch + idx

                if current_step % self.config['Train']['train_plot_interval'] == 0:
                    # Draw flow plot
                    local_left = left
                    local_right = right
                    local_flow = flow
                    local_pred_flow = pred_train_flow

                    train_flow_plot = plot_images(left=local_left,
                                                  right=local_right,
                                                  gt_flow=local_flow,
                                                  pred_flow=local_pred_flow,
                                                  denorm_func=self.data_loader.denormalize_image)
    
                    with self.train_summary_writer.as_default():
                        # Logging train images
                        tf.summary.image('Train/Flow Result',
                                         train_flow_plot, step=current_step)
                        del train_flow_plot

                del train_loss_result, pred_train_flow

                train_tqdm.set_postfix(
                    total_loss=self.train_total_loss.result().numpy()
                    )
                
            # End train session
            with self.train_summary_writer.as_default():
                # Logging train total, pixel, smooth loss
                tf.summary.scalar(f'Train/{self.train_total_loss.name}' ,
                                    self.train_total_loss.result(), step=current_step)

            # Validation
            valid_tqdm = tqdm(self.valid_dataset,
                              total=self.valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (left, right, flow, mask) in enumerate(valid_tqdm):
                valid_loss_result, pred_valid_flow = self.validation_step(left, right, flow, mask)
                
                # Update valid metrics
                self.update_valid_metric(valid_loss_result, pred_valid_flow, flow)

                current_step = self.valid_samples * epoch + idx
                    
                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    local_left = left
                    local_right = right
                    local_flow = flow
                    local_pred_flow = pred_valid_flow

                    valid_flow_plot = plot_images(left=local_left,
                                right=local_right,
                                gt_flow=local_flow,
                                pred_flow=local_pred_flow,
                                denorm_func=self.data_loader.denormalize_image)

                    with self.valid_summary_writer.as_default():
                        # Logging valid images
                        tf.summary.image('Valid/Flow Result',
                                         valid_flow_plot, step=current_step)
                        del valid_flow_plot

                valid_tqdm.set_postfix(
                    total_loss=self.valid_total_loss.result().numpy(),
                )
            
            # End valid session
            with self.valid_summary_writer.as_default():
                # Logging valid total, pixel, smooth loss
                tf.summary.scalar(f'Valid/{self.valid_total_loss.name}',
                                    self.valid_total_loss.result(), step=current_step)
                        
            with self.valid_summary_writer.as_default():
                metrics_dict = self.valid_flow_metrics.get_all_metrics()
                for metric_name, metric_value in metrics_dict.items():
                    tf.summary.scalar(f"Eval/{metric_name}", metric_value, step=epoch)

            if epoch % 5 == 0:
                self.model.save_weights(self.save_path + '/epoch_{0}_model.weights.h5'.format(epoch))
                # self.model.save_weights('{0}/{1}/epoch_{2}_model.weights.h5'.format(self.config['Directory']['weights'],
                #                                                             self.config['Directory']['exp_name'],
                #                                                             epoch))
                
            # Reset metrics
            self.train_total_loss.reset_states()
            self.valid_total_loss.reset_states()
            self.valid_flow_metrics.reset_states()

if __name__ == '__main__':
    with open('./flow/config.yaml', 'r') as file:
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