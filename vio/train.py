import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
from dataset.data_loader import DataLoader
from utils.plot_utils import PlotTool
from eval import EvalTrajectory
from model.monodepth2 import DispNet, PoseNet
# from monodepth_learner import Learner
from monodepth_learner_new import Learner
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
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)

        # 1. Model
        self.batch_size = self.config['Train']['batch_size']

        image_shape = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.depth_net = DispNet(image_shape=image_shape, batch_size=self.batch_size, prefix='disp_resnet')
        self.depth_net(tf.random.normal((1, *image_shape, 3)))
        self.depth_net.load_weights('./assets/weights/depth/vo_pretrain_depth.h5')

        self.pose_net = PoseNet(image_shape=image_shape, batch_size=self.batch_size, prefix='mono_posenet')
        posenet_input_shape = [(self.batch_size, *image_shape, 6)]
        self.pose_net.build(posenet_input_shape)
        
        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
        
        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['end_lr'],
                                                                              power=0.9)
        
        # self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.config['Train']['init_lr'],
        #                                           beta_1=self.config['Train']['beta1'],
        #                                           weight_decay=self.config['Train']['weight_decay'])
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                                  beta_1=self.config['Train']['beta1'])
        
        all_variables = self.depth_net.trainable_variables + self.pose_net.trainable_variables
        self.optimizer.build(all_variables)
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # 4. Train Method
        self.learner = Learner(depth_model=self.depth_net,
                               pose_model=self.pose_net,
                               config=self.config)

        self.eval_tool = EvalTrajectory(depth_model=self.depth_net,
                                        pose_model=self.pose_net, config=self.config)

        self.plot_tool = PlotTool(config=self.config)

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
    
    # @tf.function(jit_compile=True)
    # def train_step(self, ref_images, target_image, intrinsic) -> tf.Tensor:
    #     with tf.GradientTape(persistent=True) as tape:
    #         total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_step(ref_images, target_image, intrinsic, training=True)
    #         scaled_loss = self.optimizer.get_scaled_loss(total_loss)

    #     # loss update
    #     depth_scale_grad = tape.gradient(scaled_loss, self.depth_net.trainable_variables)
    #     pose_scale_grad = tape.gradient(scaled_loss, self.pose_net.trainable_variables)

    #     depth_unscale_grad = self.optimizer.get_unscaled_gradients(depth_scale_grad)
    #     pose_unscale_grad = self.optimizer.get_unscaled_gradients(pose_scale_grad)

    #     self.optimizer.apply_gradients(zip(depth_unscale_grad, self.depth_net.trainable_variables))
    #     self.optimizer.apply_gradients(zip(pose_unscale_grad, self.pose_net.trainable_variables))
        
    #     return total_loss, pixel_loss, smooth_loss, pred_depths
    
    @tf.function(jit_compile=True)
    def train_step(self, ref_images, target_image, intrinsic):
        with tf.GradientTape() as tape:
            total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_step(
                ref_images, target_image, intrinsic, training=True)
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)
        # 모든 변수에 대한 그래디언트 한 번에 계산
        all_vars = self.depth_net.trainable_variables + self.pose_net.trainable_variables
        scaled_grads = tape.gradient(scaled_loss, all_vars)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        # 한 번에 apply_gradients 적용
        self.optimizer.apply_gradients(zip(grads, all_vars))
        return total_loss, pixel_loss, smooth_loss, pred_depths

    
    @tf.function(jit_compile=True)
    def validation_step(self, ref_images, target_image, intrinsic) -> tf.Tensor:
        total_loss, pixel_loss, smooth_loss, pred_depths = self.learner.forward_step(ref_images, target_image, intrinsic, training=False)
        return total_loss, pixel_loss, smooth_loss, pred_depths

    def train(self) -> None:        
        for epoch in range(self.config['Train']['epoch']):    
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.data_loader.train_dataset, total=self.data_loader.num_train_samples)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for idx, (ref_images, target_image, intrinsic) in enumerate(train_tqdm):
                train_t_loss, train_p_loss, train_s_loss, pred_train_depths = self.train_step(ref_images, target_image, intrinsic)

                # Update train metrics
                self.train_total_loss(train_t_loss)
                self.train_pixel_loss(train_p_loss)
                self.train_smooth_loss(train_s_loss)

                if idx % self.config['Train']['train_plot_interval'] == 0:
                    current_step = self.data_loader.num_train_samples * epoch + idx

                    # Draw depth plot
                    train_depth_plot = self.plot_tool.plot_images(images=target_image, # target_image
                                                                  pred_depths=pred_train_depths,
                                                                  denorm_func=self.data_loader.denormalize_image)

                    with self.train_summary_writer.as_default():
                        # Logging train images
                        tf.summary.image('Train/Depth Result', train_depth_plot, step=current_step)
                        
                train_tqdm.set_postfix(
                    total_loss=self.train_total_loss.result().numpy(),
                    pixel_loss=self.train_pixel_loss.result().numpy(),
                    smooth_loss=self.train_smooth_loss.result().numpy())
            
            # Logging train metrics
            with self.train_summary_writer.as_default():
                # Logging train total, pixel, smooth loss
                tf.summary.scalar(f'Train/{self.train_total_loss.name}',
                                    self.train_total_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_pixel_loss.name}',
                                    self.train_pixel_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_smooth_loss.name}',
                                    self.train_smooth_loss.result(), step=epoch)
            
            # Validation
            valid_tqdm = tqdm(self.data_loader.valid_dataset, total=self.data_loader.num_valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (ref_images, target_image, intrinsic) in enumerate(valid_tqdm):
                valid_t_loss, valid_p_loss, valid_s_loss, pred_valid_depths = self.validation_step(ref_images, target_image, intrinsic)

                # Update valid metrics
                self.valid_total_loss(valid_t_loss)
                self.valid_pixel_loss(valid_p_loss)
                self.valid_smooth_loss(valid_s_loss)
                self.eval_tool.update_state(ref_images, target_image, intrinsic)

                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    current_step = self.data_loader.num_valid_samples * epoch + idx
                    # Draw target image - target depth plot
                    valid_depth_plot = self.plot_tool.plot_images(images=target_image,
                                                                  pred_depths=pred_valid_depths,
                                                                  denorm_func=self.data_loader.denormalize_image)

                    with self.valid_summary_writer.as_default():
                        # Logging valid images
                        tf.summary.image('Valid/Depth Result', valid_depth_plot, step=current_step)

                valid_tqdm.set_postfix(
                    total_loss=self.valid_total_loss.result().numpy(),
                    pixel_loss=self.valid_pixel_loss.result().numpy(),
                    smooth_loss=self.valid_smooth_loss.result().numpy()
                )

            # Logging valid metrics
            with self.valid_summary_writer.as_default():
                # Logging valid total, pixel, smooth loss
                tf.summary.scalar(f'Valid/{self.valid_total_loss.name}',
                                    self.valid_total_loss.result(), step=epoch)
                tf.summary.scalar(f'Valid/{self.valid_pixel_loss.name}',
                                    self.valid_pixel_loss.result(), step=epoch)
                tf.summary.scalar(f'Valid/{self.valid_smooth_loss.name}',
                                    self.valid_smooth_loss.result(), step=epoch)

            # Eval
            eval_plot = self.eval_tool.eval_plot()
            with self.test_summary_writer.as_default():
                # Logging eval images
                tf.summary.image('Eval/Trajectory', eval_plot, step=epoch)
            
            self.depth_net.save_weights('{0}/{1}/depth_net_epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
                                                                                          self.config['Directory']['exp_name'],
                                                                                          epoch))
            self.pose_net.save_weights('{0}/{1}/pose_net_epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
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