import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
import numpy as np
import gc
import yaml
from tqdm import tqdm
from datetime import datetime
from dataset.tspxr_loader_v2 import TspxrTFDSGenerator
from model.model_warp import build_vio_imu
from utils.utils import *
from utils.plot_utils import plot_depths, plot_total
from utils.projection_utils import projective_inverse_warp, projective_inverse_warp_legacy
from loss.warp_loss import compute_reprojection_loss, get_smooth_loss
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
matplotlib.use('Agg')
"""
 This usually means you are trying to call the optimizer to update different parts of the model separately.
 Please call `optimizer.build(variables)`
 with the full list of trainable variables before the training loop or use legacy optimizer `keras.optimizers.legacy.Adam.'
"""
# tensorflow.python.framework.errors_impl.ResourceExhaustedError: Out of memory while trying to allocate

np.set_printoptions(suppress=True)


def ssim(y_true, y_pred):
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

def grad_l1_loss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    grad_loss = tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true)
    return grad_loss

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
        # self.model = TSPVIO(config=self.config)
        self.model = build_vio_imu(config=self.config)
        self.model.build([[self.config['Train']['batch_size'], self.config['Train']['img_h'], self.config['Train']['img_w'], 6],
                          [self.config['Train']['batch_size'], self.config['Train']['img_h'], self.config['Train']['img_w'], 3],
                          [self.config['Train']['batch_size'], 11, 6]])
        self.model.summary()
        # source_image = keras.layers.Input((*self.image_size, 6), batch_size=batch_size)
        # target_image = keras.layers.Input((*self.image_size, 3), batch_size=batch_size)

        # self.model.built = True
        # self.model.build_model(self.config['Train']['batch_size'])
        # self.model.summary()
        # self.model.load_weights('./weights/epoch_100_model.h5')

        # 2. Dataset
        self.dataset = TspxrTFDSGenerator(data_dir=self.config['Directory']['data_dir'],
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
                                               #weight_decay=self.config['Train']['weight_decay'],
                                               beta_1=0.9)
        
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

    @tf.function(jit_compile=True) 
    def train_step(self, source_image, target_image, target_depth, imu, intrinsic) -> tf.Tensor:
        """
        target_img: t-1
        source_img: t
        imu: (batch, seq, 6)
        """
        num_scale = 4
        mask = tf.where(target_depth > 0., True, False)
        
        with tf.GradientTape() as tape:
            pred_disps, poses = self.model([source_image, target_image, imu], training=True)
            
            depth_losses = 0.

            pred_auto_masks = []

            reprojection_losses = []

            image_diffs = []
            pred_depths = []
            warp_images = []

            # Forward pass
            for scale in range(num_scale):
                pred_disp = pred_disps[scale]
                pred_depth = self.disp_to_depth(disp=pred_disp,
                                                min_depth=0.1,
                                                max_depth=10.)
                
                pred_depth = tf.image.resize(pred_depth,
                                                (self.config['Train']['img_h'], self.config['Train']['img_w']),
                                                tf.image.ResizeMethod.BILINEAR)
                
                curr_proj_image = projective_inverse_warp_legacy(source_image,
                                                                 tf.squeeze(pred_depth, axis=-1),
                                                                 tf.squeeze(poses, axis=1),
                                                                 intrinsics=intrinsic,)
                
                curr_proj_error = tf.abs(curr_proj_image - target_image)

                reprojection_losses.append(compute_reprojection_loss(curr_proj_image, target_image))

                ssim_loss = ssim(target_depth, pred_depth)
                grad_loss = grad_l1_loss(target_depth, pred_depth)
                l1_loss = tf.abs(target_depth - pred_depth)

                ssim_loss = tf.reduce_mean(ssim_loss[mask]) * 0.85
                grad_loss = tf.reduce_mean(grad_loss[mask]) 
                l1_loss = tf.reduce_mean(l1_loss[mask]) * 0.1
                
                depth_loss = ssim_loss + grad_loss + l1_loss
                depth_losses += depth_loss

                image_diffs.append(curr_proj_error)
                pred_depths.append(pred_depth)
                warp_images.append(curr_proj_image)

        
            reprojection_losses = tf.concat(reprojection_losses, axis=3)

            # Auto Mask
            identity_reprojection_losses = compute_reprojection_loss(source_image, target_image)
            identity_reprojection_losses += (tf.random.normal(identity_reprojection_losses.get_shape()) * 1e-5)
            combined = tf.concat([identity_reprojection_losses, reprojection_losses], axis=3)
            pred_auto_masks.append(tf.expand_dims(tf.cast(tf.argmin(combined, axis=3) > 1,tf.float32) * 255, -1))

            combinded_reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))

            total_loss = combinded_reprojection_loss + depth_losses

        
            total_loss /= num_scale
            combinded_reprojection_loss /= num_scale
            depth_losses /= num_scale

        losses = {
            'reprojection_loss': combinded_reprojection_loss,
            'depth_losses': depth_losses,
            'total_loss': total_loss
        }

        # loss update
        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        for grad, var in zip(gradients, self.model.trainable_variables):
            if grad is None:
                print(f"No gradient provided for {var.name}")
                raise Exception('NO Gradient error')

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return losses, image_diffs, pred_depths, warp_images, pred_auto_masks, poses

    @tf.function(jit_compile=True)
    def validation_step(self, source_image, target_image, target_depth, imu, intrinsic) -> tf.Tensor:
        """
        target_img: t-1
        source_img: t
        imu: (batch, seq, 6)
        """
        num_scale = 4
        mask = tf.where(target_depth > 0., True, False)
        
        pred_disps, poses = self.model([source_image, target_image, imu], training=True)
        
        depth_losses = 0.

        pred_auto_masks = []

        reprojection_losses = []

        image_diffs = []
        pred_depths = []
        warp_images = []

        # Forward pass
        for scale in range(num_scale):
            pred_disp = pred_disps[scale]
            pred_depth = self.disp_to_depth(disp=pred_disp,
                                            min_depth=0.1,
                                            max_depth=10.)
            
            pred_depth = tf.image.resize(pred_depth,
                                            (self.config['Train']['img_h'], self.config['Train']['img_w']),
                                            tf.image.ResizeMethod.BILINEAR)
            
            curr_proj_image = projective_inverse_warp_legacy(source_image,
                                                                tf.squeeze(pred_depth, axis=-1),
                                                                tf.squeeze(poses, axis=1),
                                                                intrinsics=intrinsic,)
            
            curr_proj_error = tf.abs(curr_proj_image - target_image)

            reprojection_losses.append(compute_reprojection_loss(curr_proj_image, target_image))

            ssim_loss = ssim(target_depth, pred_depth)
            grad_loss = grad_l1_loss(target_depth, pred_depth)
            l1_loss = tf.abs(target_depth - pred_depth)

            ssim_loss = tf.reduce_mean(ssim_loss[mask]) * 0.85
            grad_loss = tf.reduce_mean(grad_loss[mask]) 
            l1_loss = tf.reduce_mean(l1_loss[mask]) * 0.1
            
            depth_loss = ssim_loss + grad_loss + l1_loss
            depth_losses += depth_loss

            image_diffs.append(curr_proj_error)
            pred_depths.append(pred_depth)
            warp_images.append(curr_proj_image)

        reprojection_losses = tf.concat(reprojection_losses, axis=3)

        # Auto Mask
        identity_reprojection_losses = compute_reprojection_loss(source_image, target_image)
        identity_reprojection_losses += (tf.random.normal(identity_reprojection_losses.get_shape()) * 1e-5)
        combined = tf.concat([identity_reprojection_losses, reprojection_losses], axis=3)
        pred_auto_masks.append(tf.expand_dims(tf.cast(tf.argmin(combined, axis=3) > 1,tf.float32) * 255, -1))

        combinded_reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))

        total_loss = combinded_reprojection_loss + depth_losses
    
        total_loss /= num_scale
        combinded_reprojection_loss /= num_scale
        depth_losses /= num_scale

        losses = {
            'reprojection_loss': combinded_reprojection_loss,
            'depth_losses': depth_losses,
            'total_loss': total_loss
        }

        
        return losses, image_diffs, pred_depths, warp_images, pred_auto_masks, poses
    
    def decode_items(self, source_image, target_image,
                     image_diffs, pred_depths, warp_images, pred_auto_mask, poses):
        """
        source_left: (B, H, W, 3)
        source_right: (B, H, W, 3)
        target_image: (B, H, W, 3)
        proj_image_stack_all: List [(B, H, W, 6)] * 4
        proj_error_stack_all: List [(B, H, W, 6)] * 4
        pred_depth_stacks: List [(B, H, W, 1)] * 4
        pred_auto_masks: List [(B, H, W, 1)] * 4
        """

        source_image = self.dataset.decode_image(source_image[0]).numpy()
        
        target_image = self.dataset.decode_image(target_image[0]).numpy()

        decoded_diff_images = []
        decoded_warp_images = []
        decoded_depths = []

        for i in range(len(warp_images)):
            # Diff images
            diff_image = self.dataset.decode_image(image_diffs[i][0])
            decoded_diff_images.append(diff_image.numpy())

            # Pred depths
            decoded_depths.append(pred_depths[i][0].numpy())

            # Warp images
            warp_image = self.dataset.decode_image(warp_images[i][0])
            decoded_warp_images.append(warp_image.numpy())
            
        decoded_masks = pred_auto_mask[0][0].numpy()

        # Plot raw image
        plot_image_buffer = plot_total(source=source_image,
                   target=target_image,
                   warp_list=decoded_warp_images,
                   diff_list=decoded_diff_images,
                   mask=decoded_masks,
                   pose=poses)
        plot_image = tf.image.decode_png(plot_image_buffer.getvalue(), channels=4)
        plot_image = tf.expand_dims(plot_image, 0)

        # Plot depth
        depth_plot_buffer = plot_depths(decoded_depths)
        plot_depth = tf.image.decode_png(depth_plot_buffer.getvalue(), channels=4)
        plot_depth = tf.expand_dims(plot_depth, 0)

        return plot_image, plot_depth

    def train(self) -> None:
        train_freq = 100
        valid_freq = 2
        train_idx = 0
        valid_idx = 0
        for epoch in range(self.config['Train']['epoch']):    
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.train_dataset, total=self.dataset.number_train_iters)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for i, (source_image, target_image, target_depth, imu, intrinsic) in enumerate(train_tqdm):
                total_loss, image_diffs, pred_depths,\
                    warp_images, pred_auto_masks, poses = self.train_step(source_image,
                                                                          target_image,
                                                                          target_depth,
                                                                          imu,
                                                                          intrinsic)
                
                train_idx += 1
        
                if i % train_freq == 0:
                    train_plot_image, train_plot_depth = self.decode_items(source_image,
                                                                           target_image,
                                                                           image_diffs,
                                                                           pred_depths,
                                                                           warp_images,
                                                                           pred_auto_masks,
                                                                           poses)

                    with self.train_summary_writer.as_default():
                        tf.summary.image('Train Plot Image', train_plot_image, step=train_idx)
                        tf.summary.image('Train Plot Depth', train_plot_depth, step=train_idx)
                        tf.summary.scalar('Train reprojection_loss', tf.reduce_mean(total_loss['reprojection_loss']).numpy(), step=train_idx)
                        tf.summary.scalar('Train depth_losses', tf.reduce_mean(total_loss['depth_losses']).numpy(), step=train_idx)
                        tf.summary.scalar('Train total_loss', tf.reduce_mean(total_loss['total_loss']).numpy(), step=train_idx)
            
            # Validation
            valid_tqdm = tqdm(self.test_dataset, total=self.dataset.number_test_iters)
            valid_tqdm.set_description('Validation || ')
            for i, (source_image, target_image, target_depth, imu, intrinsic) in enumerate(valid_tqdm):
                total_loss, image_diffs, pred_depths, warp_images, pred_auto_masks, poses = self.validation_step(source_image,
                                                                                                                 target_image,
                                                                                                                 target_depth,
                                                                                                                 imu,
                                                                                                                 intrinsic)
                valid_idx += 1
            
                if i % valid_freq == 0:
                    valid_plot_image, valid_plot_depth = self.decode_items(source_image,
                                                                           target_image,
                                                                           image_diffs,
                                                                           pred_depths,
                                                                           warp_images,
                                                                           pred_auto_masks,
                                                                           poses)

                    with self.valid_summary_writer.as_default():
                        tf.summary.image('Valid Plot Image', valid_plot_image, step=valid_idx)
                        tf.summary.image('Valid Plot Depth', valid_plot_depth, step=valid_idx)
                        tf.summary.scalar('Valid pixel_losses', tf.reduce_mean(total_loss['reprojection_loss']).numpy(), step=valid_idx)
                        tf.summary.scalar('Valid smooth_losses', tf.reduce_mean(total_loss['depth_losses']).numpy(), step=valid_idx)
                        tf.summary.scalar('Valid total_loss', tf.reduce_mean(total_loss['total_loss']).numpy(), step=valid_idx)

            if epoch % 5 == 0:
                self.model.save_weights('{0}/{1}/epoch_{2}_model.weights.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
                                                                            epoch))
            self._clear_session()

if __name__ == '__main__':
    # LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9" python trainer.py
    debug = False
    
    if debug:
        tf.executing_eagerly()
        tf.config.run_functions_eagerly(not debug)
        # tf.config.optimizer.set_jit(False)
        tf.config.experimental.enable_op_determinism()
    # else:
        # tf.config.optimizer.set_jit(True)
        
    with open('./vio/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    with tf.device('/device:GPU:1'):
        # args = parser.parse_args()

        # Set random seed
        # SEED = 42
        # os.enviro
        # n['PYTHONHASHSEED'] = str(SEED)
        # os.environ['TF_DETERMINISTIC_OPS'] = '0'
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)

        trainer = Trainer(config=config)

        trainer.train()