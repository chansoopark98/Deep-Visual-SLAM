import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import tensorflow as tf
import keras
import numpy as np
import gc
import yaml

import io
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from tqdm import tqdm
from util.load_datasets import GenerateDatasets
from model.monodepth2 import build_disp_net

matplotlib.use('Agg')
np.set_printoptions(suppress=True)

class Trainer(object):
    def __init__(self, config) -> None:
        self.config = config
        self._clear_session()
        self.configure_train_ops()
        print('initialize')

    def _clear_session(self):
        """
            Tensorflow 계산 그래프의 이전 session 및 메모리를 초기화
        """
        keras.backend.clear_session()
        _ = gc.collect()
   
    def configure_train_ops(self) -> None:
        """
            학습 관련 설정
            1. Model
            2. Dataset
            3. Optimizer
            4. Loss
            5. Metric
            6. Logger
        """
        # 1. Model
        self.model = build_disp_net(image_shape=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                                    batch_size=self.config['Train']['batch_size'])
        self.model.build([self.config['Train']['batch_size'],
                          self.config['Train']['img_h'],
                          self.config['Train']['img_w'],
                          3])
        self.model.summary()

        # 2. Dataset
        self.dataset = GenerateDatasets(data_dir=self.config['Directory']['data_dir'],
                                        image_size=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                                        batch_size=self.config['Train']['batch_size'])
        self.train_dataset = self.dataset.get_trainData(self.dataset.train_data)
        self.test_dataset = self.dataset.get_testData(self.dataset.valid_data)
        
        # 3. Optimizer
        self.warmup_scheduler = keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                           self.config['Train']['epoch'],
                                                                           self.config['Train']['init_lr'] * 0.1,
                                                                           power=0.9)
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                               weight_decay=self.config['Train']['weight_decay']) 

        # 4. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = self.config['Directory']['log_dir'] + '/' + current_time + '_'
        self.train_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        os.makedirs('{0}/{1}'.format(self.config['Directory']['weights'],
                                self.config['Directory']['exp_name']),
                                exist_ok=True)
        
    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1. / max_depth
        max_disp = 1. / min_depth
        scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast(max_disp - min_disp, tf.float32) * disp
        depth = tf.cast(1., tf.float32) / scaled_disp
        depth = tf.clip_by_value(depth, min_depth, max_depth)
        return depth
    
    def ssim(self, y_true, y_pred):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        y_true = tf.pad(y_true, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        y_pred = tf.pad(y_pred, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        mu_x = tf.nn.avg_pool2d(y_true, 3, 1, 'VALID')
        mu_y = tf.nn.avg_pool2d(y_pred, 3, 1, 'VALID')

        sigma_x = tf.nn.avg_pool2d(y_true ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = tf.nn.avg_pool2d(y_pred ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = tf.nn.avg_pool2d(y_true * y_pred, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    
    def grad_l1_loss(self, y_true, y_pred):
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        grad_loss = tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true)
        return grad_loss
        
    @tf.function(jit_compile=True)
    def train_step(self, image, depth) -> tf.Tensor:
        mask = tf.where(depth > 0., True, False)

        with tf.GradientTape() as tape:
            # Forward pass
            pred_disps = self.model(image, training=True)
            
            l1_losses = 0
            grad_losses = 0
            ssim_losses = 0
            
            display_depths = []
            
            for _, pred_disp in enumerate(pred_disps):
                _, h, w, _ = image.shape
                resized_pred_disp = tf.image.resize(pred_disp,
                                                        (h, w), tf.image.ResizeMethod.BILINEAR)
                
                resized_pred_depth = self.disp_to_depth(disp=resized_pred_disp,
                                                        min_depth=0.1,
                                                        max_depth=10.)

                display_depths.append(resized_pred_depth)

                ssim_loss = self.ssim(depth, resized_pred_depth)
                grad_loss = self.grad_l1_loss(depth, resized_pred_depth)
                l1_loss = tf.abs(depth - resized_pred_depth)

                ssim_loss = tf.reduce_mean(ssim_loss[mask])
                grad_loss = tf.reduce_mean(grad_loss[mask])
                l1_loss = tf.reduce_mean(l1_loss[mask])
                
                ssim_losses += ssim_loss
                grad_losses += grad_loss
                l1_losses += l1_loss * 0.1

            l1_losses /= 4.
            grad_losses /= 4.
            ssim_losses /= 4.

            total_loss =  l1_losses + grad_losses + ssim_losses
            
        losses = {
            'total_loss': total_loss,
            'l1_loss': l1_losses,
            'grad_loss': grad_losses,
            'ssim_loss': ssim_losses,
        }

        # loss update
        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return image, depth, display_depths, losses

    @tf.function(jit_compile=True) 
    def validation_step(self, image, depth) -> tf.Tensor:
        mask = tf.where(depth > 0., True, False)

        # Forward pass
        pred_disps = self.model(image, training=False)
            
        l1_losses = 0
        grad_losses = 0
        ssim_losses = 0
        
        display_depths = []
        
        for _, pred_disp in enumerate(pred_disps):
            _, h, w, _ = image.shape
            resized_pred_disp = tf.image.resize(pred_disp,
                                                    (h, w), tf.image.ResizeMethod.BILINEAR)
            
            resized_pred_depth = self.disp_to_depth(disp=resized_pred_disp,
                                                    min_depth=0.1,
                                                    max_depth=10.)

            display_depths.append(resized_pred_depth)

            ssim_loss = self.ssim(depth, resized_pred_depth)
            grad_loss = self.grad_l1_loss(depth, resized_pred_depth)
            l1_loss = tf.abs(depth - resized_pred_depth)

            ssim_loss = tf.reduce_mean(ssim_loss[mask])
            grad_loss = tf.reduce_mean(grad_loss[mask])
            l1_loss = tf.reduce_mean(l1_loss[mask])
            
            ssim_losses += ssim_loss
            grad_losses += grad_loss
            l1_losses += l1_loss * 0.1

            l1_losses /= 4.
            grad_losses /= 4.
            ssim_losses /= 4.

            total_loss =  l1_losses + grad_losses + ssim_losses
            
        losses = {
            'total_loss': total_loss,
            'l1_loss': l1_losses,
            'grad_loss': grad_losses,
            'ssim_loss': ssim_losses,
        }

        return image, depth, display_depths, losses

    def plot_images(self, image, depth, pred_depths):
        """
        세 개의 이미지를 하나의 plot에 그리는 함수

        :param warped_img: 왜곡된 이미지 (Tensor 또는 Numpy 배열)
        :param target_img: 타겟 이미지 (Tensor 또는 Numpy 배열)
        :param depth: 깊이 이미지 (Tensor 또는 Numpy 배열)
        """
        # Plot 설정
        depth_len = len(pred_depths)

        fig, axes = plt.subplots(1, 2 + depth_len, figsize=(20, 5))

        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(depth, vmin=0., vmax=10., cmap='plasma')
        axes[1].set_title('Depth')
        axes[1].axis('off')

        for idx in range(depth_len):
            pred_depth = pred_depths[idx][0].numpy()
            axes[2 + idx].imshow(pred_depth, vmin=0., vmax=10., cmap='plasma')
            axes[2 + idx].set_title(f'Pred Depth Scale {idx}')
            axes[2 + idx].axis('off')

        fig.tight_layout()

        # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plt.close()

        return buf

    def decode_items(self, image, depth, display_depths):
        image_npy = self.dataset.decode_image(image[0]).numpy()
        depth_npy = depth[0].numpy()
  
        plot_buffer = self.plot_images(image_npy, depth_npy, display_depths)

        image = tf.image.decode_png(plot_buffer.getvalue(), channels=4)
        return tf.expand_dims(image, 0)

    def train(self) -> None:
        train_freq = 50
        valid_freq = 20
        train_idx = 0
        valid_idx = 0
        for epoch in range(self.config['Train']['epoch']):    
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.train_dataset, total=self.dataset.number_train)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for i, (image, depth) in enumerate(train_tqdm):
                image, depth, display_depths, total_loss = self.train_step(image, depth)
                train_idx += 1

                if i % train_freq == 0:
                    train_plot_img = self.decode_items(image, depth, display_depths)
                    with self.train_summary_writer.as_default():
                        tf.summary.image('Train_plot', train_plot_img, step=train_idx)
                        tf.summary.scalar('total_loss', tf.reduce_mean(total_loss['total_loss']).numpy(), step=train_idx)
                        tf.summary.scalar('l1_loss', tf.reduce_mean(total_loss['l1_loss']).numpy(), step=train_idx)
                        tf.summary.scalar('grad_loss', tf.reduce_mean(total_loss['grad_loss']).numpy(), step=train_idx)
                        tf.summary.scalar('ssim_loss', tf.reduce_mean(total_loss['ssim_loss']).numpy(), step=train_idx)

            # Validation
            valid_tqdm = tqdm(self.test_dataset, total=self.dataset.number_test)
            valid_tqdm.set_description('Validation || ')
            for i, (image, depth) in enumerate(valid_tqdm):
                image, depth, display_depths, total_loss = self.validation_step(image, depth)
                valid_idx += 1
            
                if i % valid_freq == 0:
                    valid_plot_img = self.decode_items(image, depth, display_depths)

                    with self.valid_summary_writer.as_default():
                        tf.summary.image('Validation_plot', valid_plot_img, step=valid_idx)
                        tf.summary.scalar('total_loss', tf.reduce_mean(total_loss['total_loss']).numpy(), step=valid_idx)
                        tf.summary.scalar('l1_loss', tf.reduce_mean(total_loss['l1_loss']).numpy(), step=valid_idx)
                        tf.summary.scalar('grad_loss', tf.reduce_mean(total_loss['grad_loss']).numpy(), step=valid_idx)
                        tf.summary.scalar('ssim_loss', tf.reduce_mean(total_loss['ssim_loss']).numpy(), step=valid_idx)

            if epoch % 5 == 0:
                self.model.save_weights('{0}/{1}/epoch_{2}_model.weights.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
                                                                            epoch))

            # self._clear_session()

if __name__ == '__main__':
    # LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9" python trainer.py
    debug = False

    if debug:
        tf.executing_eagerly()
        tf.config.run_functions_eagerly(not debug)
        tf.config.optimizer.set_jit(False)
    # else:
        # tf.config.optimizer.set_jit(True)
    
    with open('./depth/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    with tf.device('/device:GPU:1'):
        # args = parser.parse_args()

        # Set random seed
        # SEED = 42
        # os.environ['PYTHONHASHSEED'] = str(SEED)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)

        trainer = Trainer(config=config)

        trainer.train()