import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
from dataset.data_loader import DataLoader
from utils.plot_utils import plot_images, plot_warp_images
from model.monodepth2 import MonoDepth2Model
from monodepth_learner import MonoDepth2Learner
from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml

np.set_printoptions(suppress=True)

class Trainer(object):
    def __init__(self, config, ) -> None:
        self.config = config
        
        self.configure_train_ops()
        print('initialize')
   
    def configure_train_ops(self) -> None:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        # 1. Model
        self.batch_size = self.config['Train']['batch_size']
        self.model = MonoDepth2Model(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                                     batch_size=self.config['Train']['batch_size'])
        model_input_shape = (self.config['Train']['batch_size'], self.config['Train']['img_h'], self.config['Train']['img_w'], 9)
        self.model.build(model_input_shape)
        self.model.summary()
        
        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
        
        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['init_lr'] * 0.1,
                                                                              power=0.9)
        
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.config['Train']['init_lr'],
                                                  beta_1=self.config['Train']['beta1'],
                                                  weight_decay=self.config['Train']['weight_decay'])
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # 4. Train Method
        self.learner = MonoDepth2Learner(model=self.model, optimizer=self.optimizer)

        # 5. Metrics
        self.train_total_loss = tf.keras.metrics.Mean(name='train_total_loss')
        self.train_pixel_loss = tf.keras.metrics.Mean(name='train_pixel_loss')
        self.train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
        self.valid_total_loss = tf.keras.metrics.Mean(name='valid_total_loss')
        self.valid_pixel_loss = tf.keras.metrics.Mean(name='valid_pixel_loss')
        self.valid_smooth_loss = tf.keras.metrics.Mean(name='valid_smooth_loss')

        # 6. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join('vio', self.config['Directory']['log_dir'] + \
            '/' + current_time + '_')
        self.train_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/valid')
        self.test_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/test')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        os.makedirs('{0}/{1}'.format(self.config['Directory']['weights'],
                                     self.config['Directory']['exp_name']),
                    exist_ok=True)
    
    @tf.function(jit_compile=True)
    def train_step(self, images, imus, intrinsic) -> tf.Tensor:
        with tf.GradientTape() as tape:
            total_loss, pixel_loss, smooth_loss, pred_depths, vis_outputs = self.learner.forward_step(images, intrinsic, training=True)
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        # loss update
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss, pixel_loss, smooth_loss, pred_depths, vis_outputs
    
    @tf.function(jit_compile=True)
    def validation_step(self, images, imus, intrinsic) -> tf.Tensor:
        total_loss, pixel_loss, smooth_loss, pred_depths, vis_outputs = self.learner.forward_step(images, intrinsic, training=False)
        return total_loss, pixel_loss, smooth_loss, pred_depths, vis_outputs

    def train(self) -> None:        
        for epoch in range(self.config['Train']['epoch']):    
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.data_loader.train_dataset, total=self.data_loader.num_train_samples)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for idx, (images, imus, intrinsic) in enumerate(train_tqdm):
                train_t_loss, train_p_loss, train_s_loss, pred_train_depths, train_vis_outputs = self.train_step(images, imus, intrinsic)

                # Update train metrics
                self.train_total_loss(train_t_loss)
                self.train_pixel_loss(train_p_loss)
                self.train_smooth_loss(train_s_loss)

                if idx % self.config['Train']['train_log_interval'] == 0:
                    current_step = self.data_loader.num_train_samples * epoch + idx

                    with self.train_summary_writer.as_default():
                        # Logging train total, pixel, smooth loss
                        tf.summary.scalar(f'Train/{self.train_total_loss.name}',
                                            self.train_total_loss.result(), step=current_step)
                        tf.summary.scalar(f'Train/{self.train_pixel_loss.name}',
                                            self.train_pixel_loss.result(), step=current_step)
                        tf.summary.scalar(f'Train/{self.train_smooth_loss.name}',
                                            self.train_smooth_loss.result(), step=current_step)

                if idx % self.config['Train']['train_plot_interval'] == 0:
                    # Draw depth plot
                    target_image = self.data_loader.denormalize_image(images[:, :, :, 3:6])
                    train_depth_plot = plot_images(image=target_image, pred_depths=pred_train_depths)
                    train_warp_plot = plot_warp_images(target_image=train_vis_outputs['target'],
                                                       left_image=train_vis_outputs['left_image'],
                                                       right_image=train_vis_outputs['right_image'],
                                                       warped_images=train_vis_outputs['warped_image'],
                                                       warped_losses=train_vis_outputs['warped_loss'],
                                                       masks=train_vis_outputs['mask'],
                                                       denrom_func=self.data_loader.denormalize_image)


                    with self.train_summary_writer.as_default():
                        # Logging train images
                        tf.summary.image('Train/Depth Result', train_depth_plot, step=current_step)
                        tf.summary.image('Train/Warp Result', train_warp_plot, step=current_step)
                        

                train_tqdm.set_postfix(
                    total_loss=self.train_total_loss.result().numpy(),
                    pixel_loss=self.train_pixel_loss.result().numpy(),
                    smooth_loss=self.train_smooth_loss.result().numpy())
            
            # Validation
            valid_tqdm = tqdm(self.data_loader.valid_dataset, total=self.data_loader.num_valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (images, imus, intrinsic) in enumerate(valid_tqdm):
                valid_t_loss, valid_p_loss, valid_s_loss, pred_valid_depths, valid_vis_outputs = self.validation_step(images, imus, intrinsic)

                # Update valid metrics
                self.valid_total_loss(valid_t_loss)
                self.valid_pixel_loss(valid_p_loss)
                self.valid_smooth_loss(valid_s_loss)

                if idx % self.config['Train']['valid_log_interval'] == 0:
                    self.data_loader.num_valid_samples * epoch + idx

                    with self.valid_summary_writer.as_default():
                        # Logging valid total, pixel, smooth loss
                        tf.summary.scalar(f'Valid/{self.valid_total_loss.name}',
                                          self.valid_total_loss.result(), step=current_step)
                        tf.summary.scalar(f'Valid/{self.valid_pixel_loss.name}',
                                          self.valid_pixel_loss.result(), step=current_step)
                        tf.summary.scalar(f'Valid/{self.valid_smooth_loss.name}',
                                          self.valid_smooth_loss.result(), step=current_step)
                
                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    # Draw depth plot
                    target_image = self.data_loader.denormalize_image(images[:, :, :, 3:6])
                    valid_depth_plot = plot_images(image=target_image, pred_depths=pred_valid_depths)
                    valid_warp_plot = plot_warp_images(target_image=valid_vis_outputs['target'],
                                                       left_image=valid_vis_outputs['left_image'],
                                                       right_image=valid_vis_outputs['right_image'],
                                                       warped_images=valid_vis_outputs['warped_image'],
                                                       warped_losses=valid_vis_outputs['warped_loss'],
                                                       masks=valid_vis_outputs['mask'],
                                                       denrom_func=self.data_loader.denormalize_image)


                    with self.valid_summary_writer.as_default():
                        # Logging valid images
                        tf.summary.image('Valid/Depth Result', valid_depth_plot, step=current_step)
                        tf.summary.image('Valid/Warp Result', valid_warp_plot, step=current_step)

                valid_tqdm.set_postfix(
                    total_loss=self.valid_total_loss.result().numpy(),
                    pixel_loss=self.valid_pixel_loss.result().numpy(),
                    smooth_loss=self.valid_smooth_loss.result().numpy()
                )

            if epoch % 5 == 0:
                self.model.save_weights('{0}/{1}/epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
                                                                            epoch))
            self.train_total_loss.reset_states()
            self.train_pixel_loss.reset_states()
            self.train_smooth_loss.reset_states()
            self.valid_total_loss.reset_states()
            self.valid_pixel_loss.reset_states()
            self.valid_smooth_loss.reset_states()

if __name__ == '__main__':
    with open('./vio/config.yaml', 'r') as file:
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

    # strategy = tf.distribute.MirroredStrategy()

    # with strategy.scope():
    trainer = Trainer(config=config)
    trainer.train()