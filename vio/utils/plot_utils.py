import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
from utils.utils import *
import tensorflow as tf

class PlotTool:
    def __init__(self, config: dict) -> None:
        self.batch_size = config['Train']['batch_size']
        self.vis_batch_size = config['Train']['vis_batch_size']
        if self.vis_batch_size > self.batch_size:
            self.vis_batch_size = self.batch_size
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_source = config['Train']['num_source'] # 2
        self.num_scales = config['Train']['num_scale'] # 4

    def plot_images(self, images: tf.Tensor, pred_depths: tf.Tensor, denorm_func: callable):
        # Plot 설정
        image = denorm_func(images[0])
        pred_depths = [depth[0] for depth in pred_depths]
        
        fig, axes = plt.subplots(1, 1 + self.num_scales, figsize=(10, 10))

        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
    
        for idx in range(self.num_scales):
            depth = pred_depths[idx]
            axes[idx + 1].imshow(depth, vmin=0., vmax=10., cmap='plasma')
            axes[idx + 1].set_title(f'Scale {idx}')
            axes[idx + 1].axis('off')

        fig.tight_layout()

        # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plt.close(fig)
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        return tf.expand_dims(image, 0)

    def plot_warp_images(self,
                         vis_outputs: dict,
                         denorm_func: callable):
        """
        vis_outputs = {
            'left_warped': left_warped, # [B, num_source, H, W, 3]
            'right_warped': right_warped, # [B, num_source, H, W, 3]
            'left_warped_losses': left_warped_losses, # [B, num_source, H, W, 1]
            'right_warped_losses': right_warped_losses, # [B, num_source, H, W, 1]
            'target': tgt_image, # [B, H, W, 3]
            'left_images': left_images, # [B, num_source, H, W, 3]
            'right_images': right_images, # [B, num_source, H, W, 3]
            'masks': pred_auto_masks, # [B, num_source, H, W, 1]
        }
        """
        left_warped_images = vis_outputs['left_warped']
        right_warped_images = vis_outputs['right_warped']
        
        target_images = vis_outputs['target']
        left_images = vis_outputs['left_images']
        right_images = vis_outputs['right_images']
        
        batch = 0
        fig, axes = plt.subplots(self.num_source, 5, figsize=(20, 10)) # 2, 5
        
        target_image = target_images[batch, :, :, :]
        target_image = denorm_func(target_image)

        for i in range(self.num_source):
            left_image = left_images[batch, i, :, :, :] # (H, W, 3)
            left_image = denorm_func(left_image)    

            right_image = right_images[batch, i, :, :, :] # (H, W, 3)
            right_image = denorm_func(right_image)

            left_warped = left_warped_images[batch, i, :, :, :] # (H, W, 3)
            left_warped = denorm_func(left_warped)

            right_warped = right_warped_images[batch, i, :, :, :] # (H, W, 3)
            right_warped = denorm_func(right_warped)

            axes[i, 0].imshow(left_image, vmin=0, vmax=255)
            axes[i, 0].set_title('Left Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(left_warped, vmin=0, vmax=255)
            axes[i, 1].set_title('Left to Target')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(target_image, vmin=0, vmax=255)
            axes[i, 2].set_title('Target Image')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(right_warped, vmin=0, vmax=255)
            axes[i, 3].set_title('Right to Target')
            axes[i, 3].axis('off')

            axes[i, 4].imshow(right_image, vmin=0, vmax=255)
            axes[i, 4].set_title('Right Image')
            axes[i, 4].axis('off')

        # non-used plot
        # axes[1, 0].axis('off')
        # axes[1, 4].axis('off')
            
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plt.close()
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        return tf.expand_dims(image, 0)

    def plot_warp_loss(self, vis_outputs: dict):
        """
        vis_outputs = {
            'left_warped': left_warped, # [B, num_source, H, W, 3]
            'right_warped': right_warped, # [B, num_source, H, W, 3]
            'left_warped_losses': left_warped_losses, # [B, num_source, H, W, 1]
            'right_warped_losses': right_warped_losses, # [B, num_source, H, W, 1]
            'target': tgt_image, # [B, H, W, 3]
            'left_images': left_images, # [B, num_source, H, W, 3]
            'right_images': right_images, # [B, num_source, H, W, 3]
            'masks': pred_auto_masks, # [B, num_source, H, W, 1]
        }
        """
        left_warped_losses = vis_outputs['left_warped_losses']
        right_warped_losses = vis_outputs['right_warped_losses']
        masks = vis_outputs['masks']

        batch = 0
        fig, axes = plt.subplots(self.num_source, 3, figsize=(10, 10)) # 2, 5
        
        for i in range(self.num_source):
            left_warped_loss = left_warped_losses[batch, i, :, :, :] # (H, W, 1)
            right_warped_loss = right_warped_losses[batch, i, :, :, :] # (H, W, 1)
            mask = masks[batch, 0, :, :, :] # (H, W, 1)
            
            axes[i, 0].imshow(left_warped_loss)
            axes[i, 0].set_title('Left to Target Loss')
            axes[i, 0].axis('off')

            
            axes[i, 1].imshow(right_warped_loss)
            axes[i, 1].set_title('Right to Target Loss')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(mask)
            axes[i, 2].set_title('Mask')
            axes[i, 2].axis('off')

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plt.close()
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        return tf.expand_dims(image, 0)


# def plot_line_tensorboard(imgs: np.ndarray, pred: np.ndarray, gt: np.ndarray, decision: np.ndarray):
#     # 그림 및 gridspec 생성
#     fig = plt.figure(figsize=(15, 15))  # 크기 조정
#     gs = gridspec.GridSpec(11, 5)  # 4개의 열로 변경

#     # 11장의 이미지를 좌측에 표시
#     for i in range(11):
#         ax = fig.add_subplot(gs[i, 0])
#         ax.imshow(imgs[i], cmap='gray')
#         ax.axis('off')

#     # 우측에 첫번째 2차원 플롯 생성
#     ax2 = fig.add_subplot(gs[:, 1:3])
    
#     poses_est_mat = path_accu(pred)
#     gt_pose = path_accu(gt)
    
#     x_gt = np.asarray([pose[0, 3] for pose in gt_pose])
#     y_gt = np.asarray([pose[1, 3] for pose in gt_pose])
#     z_gt = np.asarray([pose[2, 3] for pose in gt_pose])

#     x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
#     y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
#     z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

#     fontsize_ = 10
#     plot_keys = ["Ground Truth", "Estimate"]
#     start_point = [0, 0]
#     style_pred = 'b-'
#     style_gt = 'r-'
#     style_O = 'ko'

#     ax2.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
#     ax2.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
#     ax2.plot(start_point[0], start_point[1], style_O, label='Start Point')
#     ax2.legend(loc="upper right", prop={'size': fontsize_})

#     if decision is not None:  
#         ax3 = fig.add_subplot(gs[:, 3:])
#         cout = decision[:, 0] * 100

#         cax = ax3.scatter(x_pred[1:-1], z_pred[1:-1], marker='o', c=cout)

#         # max_usage = max(cout)
#         # min_usage = min(cout)
#         max_usage = 1.
#         min_usage = 0.

#         ticks = np.floor(np.linspace(min_usage, max_usage, num=5))
#         cbar = fig.colorbar(cax, ax=ax3, ticks=ticks)
#         cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])
#         ax3.set_title('decision heatmap with window size {}'.format(11))

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)

#     plt.close()

#     return buf

# def plot_3d_tensorboard(imgs: np.ndarray, pred: np.ndarray, gt: np.ndarray, decision: np.ndarray=None):
#     # 그림 및 gridspec 생성
#     fig = plt.figure(figsize=(15, 15))  # 크기 조정
#     gs = gridspec.GridSpec(11, 5)  # 4개의 열로 변경

#     # 11장의 이미지를 좌측에 표시
#     for i in range(11):
#         ax = fig.add_subplot(gs[i, 0])
#         ax.imshow(imgs[i], cmap='gray')
#         ax.axis('off')

#     # 우측에 첫번째 2차원 플롯 생성
#     ax2 = fig.add_subplot(gs[:, 1:3], projection='3d')
    
#     poses_est_mat = path_accu(pred)
#     gt_pose = path_accu(gt)
    
#     x_gt = np.asarray([pose[0, 3] for pose in gt_pose])
#     y_gt = np.asarray([pose[1, 3] for pose in gt_pose])
#     z_gt = np.asarray([pose[2, 3] for pose in gt_pose])

#     x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
#     y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
#     z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

#     fontsize_ = 10
#     plot_keys = ["Ground Truth", "Estimate"]
#     start_point = [0, 0]
#     style_pred = 'b-'
#     style_gt = 'r-'
#     style_O = 'ko'

#     ax2.plot(x_gt, z_gt, y_gt, c='b', linestyle='dashed', marker='o', label=plot_keys[0])
#     ax2.plot(x_pred, z_pred, y_pred, c='r', linestyle='dashed', marker='x', label=plot_keys[1])

#     ax2.plot(start_point[0], start_point[1], style_O, label='Start Point')
#     ax2.legend(loc="upper right", prop={'size': fontsize_})
#     # ax2.set_xlim(0,)
#     # ax2.set_ylim(0,)
#     # ax2.set_zlim(0,)

#     if decision is not None:  
#         ax3 = fig.add_subplot(gs[:, 3:])
#         cout = decision[:, 0] * 100

#         cax = ax3.scatter(x_pred[1:-1], z_pred[1:-1], marker='o', c=cout)

#         # max_usage = max(cout)
#         # min_usage = min(cout)
#         max_usage = 1.
#         min_usage = 0.

#         ticks = np.floor(np.linspace(min_usage, max_usage, num=5))
#         cbar = fig.colorbar(cax, ax=ax3, ticks=ticks)
#         cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])
#         ax3.set_title('decision heatmap with window size {}'.format(11))

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)

#     plt.close()

#     return buf


# def plot_images(source_left, source_right, target_image):
#     # Plot 설정
#     fig, axes = plt.subplots(1, 3, figsize=(20, 10))

#     # 첫 번째 이미지 (source_left)

#     axes[0].imshow(source_left)
#     axes[0].set_title('source_left')
#     axes[0].axis('off')

#     axes[1].imshow(target_image)
#     axes[1].set_title('target_image')
#     axes[1].axis('off')

#     axes[2].imshow(source_right)
#     axes[2].set_title('source_right')
#     axes[2].axis('off')

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close()

#     return buf

# def plot_warped_image(decoded_proj_images, decoded_proj_errors, poses):
#     # Plot 설정
#     num_scales = len(decoded_proj_images)
#     fig, axes = plt.subplots(num_scales, 4, figsize=(20, 10))

#     left_pose = poses[0, 0, :].numpy()
#     right_pose = poses[0, 1, :].numpy()
    
#     # 첫 번째 이미지 (source_left)
#     for i in range(num_scales):
#         warped_left = decoded_proj_images[i][:, :, :3]
#         warped_right = decoded_proj_images[i][:, :, 3:]

#         error_left = decoded_proj_errors[i][:, :, :3]
#         error_right = decoded_proj_errors[i][:, :, 3:]

#         axes[i, 0].imshow(warped_left)
#         axes[i, 0].set_title('warped_left')
#         axes[i, 0].axis('off')
        
#         axes[i, 1].imshow(error_left)
#         axes[i, 1].set_title('error_left')
#         axes[i, 1].axis('off')

#         axes[i, 2].imshow(warped_right)
#         axes[i, 2].set_title('warped_right')
#         axes[i, 2].axis('off')

#         axes[i, 3].imshow(error_right)
#         axes[i, 3].set_title('error_right')
#         axes[i, 3].axis('off')


#     fig.text(0.5, 0.95, f"Left Pose: {left_pose}", ha='center', fontsize=12)
#     fig.text(0.5, 0.9, f"Right Pose: {right_pose}", ha='center', fontsize=12)

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close()

#     return buf


# def plot_total(source, target, warp_list, diff_list, mask, pose):
#     # Plot 설정
#     num_scales = len(warp_list)
#     fig, axes = plt.subplots(num_scales, 5, figsize=(20, 10))

#     decoded_pose = pose[0].numpy()
    
#     # 첫 번째 이미지 (source_left)
#     for i in range(num_scales):
#         axes[i, 0].imshow(source)
#         axes[i, 0].set_title('Source Image')
#         axes[i, 0].axis('off')
        
#         axes[i, 1].imshow(warp_list[i])
#         axes[i, 1].set_title('Source Warped')
#         axes[i, 1].axis('off')

#         axes[i, 2].imshow(target)
#         axes[i, 2].set_title('Target Image')
#         axes[i, 2].axis('off')

#         axes[i, 3].imshow(diff_list[i])
#         axes[i, 3].set_title('Differences')
#         axes[i, 3].axis('off')

#         axes[i, 4].imshow(mask)
#         axes[i, 4].set_title('Masks')
#         axes[i, 4].axis('off')


#     fig.text(0.5, 0.95, f"Pose: {decoded_pose}", ha='center', fontsize=12)

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close()

#     return buf


# def plot_warped_image_list(l2t_warped, r2t_warped, source_left, source_right, target_image, poses):
#     # Plot 설정
#     fig, axes = plt.subplots(4, 5, figsize=(20, 10))

#     left_pose = poses[0, 0, :].numpy()
#     right_pose = poses[0, 1, :].numpy()
    
#     # 첫 번째 이미지 (source_left)
#     for i in range(4):
#         warped_left = l2t_warped[i][0].numpy()
#         warped_right = r2t_warped[i][0].numpy()
        
#         warped_left = (warped_left * 0.225) + 0.45
#         warped_left *= 255.0
#         warped_left = np.uint8(warped_left)
        
#         warped_right = (warped_right * 0.225) + 0.45
#         warped_right *= 255.0
#         warped_right = np.uint8(warped_right)

#         axes[i, 0].imshow(source_left)
#         axes[i, 0].set_title('source_left')
#         axes[i, 0].axis('off')
        
#         axes[i, 1].imshow(warped_left)
#         axes[i, 1].set_title('left_warped')
#         axes[i, 1].axis('off')

#         axes[i, 2].imshow(target_image)
#         axes[i, 2].set_title('target_image')
#         axes[i, 2].axis('off')

#         axes[i, 3].imshow(warped_right)
#         axes[i, 3].set_title('right_warped')
#         axes[i, 3].axis('off')

#         axes[i, 4].imshow(source_right)
#         axes[i, 4].set_title('source_right')
#         axes[i, 4].axis('off')

#     fig.text(0.5, 0.95, f"Left Pose: {left_pose}", ha='center', fontsize=12)
#     fig.text(0.5, 0.90, f"Right Pose: {right_pose}", ha='center', fontsize=12)

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close()

#     return buf

# def plot_depths(depth_lists):
    
#     # Plot 설정
#     fig, axes = plt.subplots(1, 4, figsize=(20, 10))

#     # # 세 번째 이미지 (depth)
#     for i in range(len(depth_lists)):
#         depth = depth_lists[i]
#         axes[i].imshow(depth[:, :, 0], vmin=0., vmax=10., cmap='plasma')
#         axes[i].set_title(f'Depth Image : {i}')
#         axes[i].axis('off')

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)

#     plt.close()
#     return buf

# def plot_masks(mask_lists):
    
#     # Plot 설정
#     fig, axes = plt.subplots(1, 4, figsize=(20, 10))

#     # # 세 번째 이미지 (depth)
#     for i in range(len(mask_lists)):
#         mask = mask_lists[i]
#         axes[i].imshow(mask[:, :, 0])
#         axes[i].set_title(f'Depth Image : {i}')
#         axes[i].axis('off')

#     fig.tight_layout()

#     # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)

#     plt.close()
#     return buf