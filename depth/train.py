import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from depth_learner import DepthLearner
from vio.model.monodepth2 import DispNet
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
    def __init__(self, config) -> None:
        self.config = config
        self.configure_train_ops()
        print('initialize')

    def configure_train_ops(self) -> None:
        # 1. Model
        self.batch_size = self.config['Train']['batch_size']
        self.model = DispNet(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                             batch_size=self.batch_size)

        model_input_shape = (self.config['Train']['batch_size'],
                             self.config['Train']['img_h'], self.config['Train']['img_w'], 3)
        self.model.build(model_input_shape)
        self.model.summary()

        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)

        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['init_lr'] * 0.1,
                                                                              power=0.9)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                                  ) # weight_decay=self.config['Train']['weight_decay']

        # 4. Learner
        self.learner = DepthLearner(model=self.model, optimizer=self.optimizer)

        # 5. Metrics
        self.train_total_loss = tf.keras.metrics.Mean(name='train_total_loss')
        self.train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
        self.train_log_loss = tf.keras.metrics.Mean(name='train_log_loss')
        self.train_l1_loss = tf.keras.metrics.Mean(name='train_l1_loss')

        self.valid_total_loss = tf.keras.metrics.Mean(name='valid_total_loss')
        self.valid_smooth_loss = tf.keras.metrics.Mean(name='valid_smooth_loss')
        self.valid_log_loss = tf.keras.metrics.Mean(name='valid_log_loss')
        self.valid_l1_loss = tf.keras.metrics.Mean(name='valid_l1_loss')

        self.valid_depth_metrics = DepthMetrics('valid_depth_metrics')

        # 6. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = self.config['Directory']['log_dir'] + \
            '/' + current_time + '_'
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
            loss_dict, pred_depths = self.learner.forward_step(
                rgb, depth, training=True)
            total_loss = sum(loss_dict.values())

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss_dict, pred_depths

    @tf.function()
    def validation_step(self, rgb, depth):
        loss_dict, pred_depths = self.learner.forward_step(
            rgb, depth, training=False)
        return loss_dict, pred_depths

    def train(self) -> None:
        for epoch in range(self.config['Train']['epoch']):
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr

            train_tqdm = tqdm(self.data_loader.train_dataset,
                              total=self.data_loader.num_train_samples)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for idx, (rgb, depth) in enumerate(train_tqdm):
                train_loss_result, pred_train_depths = self.train_step(rgb, depth)

                # Update train metrics
                self.train_total_loss.update_state(sum(train_loss_result.values()))
                self.train_smooth_loss.update_state(train_loss_result['smooth_loss'])
                self.train_log_loss.update_state(train_loss_result['log_loss'])
                self.train_l1_loss.update_state(train_loss_result['l1_loss'])

                if idx % self.config['Train']['train_log_interval'] == 0:
                    current_step = self.data_loader.num_train_samples * epoch + idx

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

                if idx % self.config['Train']['train_plot_interval'] == 0:
                    # Draw depth plot
                    target_image = self.data_loader.denormalize_image(rgb)
                    train_depth_plot = plot_images(image=target_image,
                                                   pred_depths=pred_train_depths,
                                                   gt_depth=depth)

                    with self.train_summary_writer.as_default():
                        # Logging train images
                        tf.summary.image('Train/Depth Result',
                                         train_depth_plot, step=current_step)

                train_tqdm.set_postfix(
                    total_loss=self.train_total_loss.result().numpy())

            # Validation
            valid_tqdm = tqdm(self.data_loader.valid_dataset,
                              total=self.data_loader.num_valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (rgb, depth) in enumerate(valid_tqdm):
                valid_loss_result, pred_valid_depths = self.validation_step(rgb, depth)

                # Update valid metrics
                self.valid_total_loss.update_state(sum(valid_loss_result.values()))
                self.valid_smooth_loss.update_state(valid_loss_result['smooth_loss'])
                self.valid_log_loss.update_state(valid_loss_result['log_loss'])
                self.valid_l1_loss.update_state(valid_loss_result['l1_loss'])
                self.valid_depth_metrics.update_state(depth, pred_valid_depths[0])

                if idx % self.config['Train']['valid_log_interval'] == 0:
                    self.data_loader.num_valid_samples * epoch + idx

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
                    
                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    # Draw depth plot
                    target_image = self.data_loader.denormalize_image(rgb)
                    valid_depth_plot = plot_images(image=target_image,
                                                   pred_depths=pred_valid_depths,
                                                   gt_depth=depth)

                    with self.valid_summary_writer.as_default():
                        # Logging valid images
                        tf.summary.image('Valid/Depth Result',
                                         valid_depth_plot, step=current_step)

                valid_tqdm.set_postfix(
                    total_loss=self.valid_total_loss.result().numpy(),
                )
            
            with self.valid_summary_writer.as_default():
                metrics_dict = self.valid_depth_metrics.get_all_metrics()
                for metric_name, metric_value in metrics_dict.items():
                    tf.summary.scalar(f"Valid/{metric_name}", metric_value, step=current_step)

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

    with tf.device('/device:GPU:1'):
        # args = parser.parse_args()

        # Set random seed
        SEED = 42
        # os.environ['PYTHONHASHSEED'] = str(SEED)
        # os.environ['TF_DETERMINISTIC_OPS'] = '0'
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)
        # tf.keras.utils.set_random_seed(SEED)
        # tf.config.experimental.enable_op_determinism()
        # np.random.seed(SEED)

        trainer = Trainer(config=config)

        trainer.train()