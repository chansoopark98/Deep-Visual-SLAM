import io
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *
import matplotlib.gridspec as gridspec
import tensorflow as tf

def plot_images(image: tf.Tensor, pred_depths: tf.Tensor):
    image = image[0]
    # Plot 설정
    depth_len = len(pred_depths)

    fig, axes = plt.subplots(1, 1 + depth_len, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    for idx in range(depth_len):
        pred_depth = pred_depths[idx][0].numpy()
        axes[1 + idx].imshow(pred_depth, vmin=0., vmax=10., cmap='plasma')
        axes[1 + idx].set_title(f'Pred Depth Scale {idx}')
        axes[1 + idx].axis('off')

    fig.tight_layout()

    # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    plt.close()
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    return tf.expand_dims(image, 0)

def plot_warp_images(target_image: tf.Tensor,
                     left_image: tf.Tensor,
                     right_image: tf.Tensor,
                     warped_images: list,
                     warped_losses: list,
                     masks: list,
                     denrom_func):
    
    left_image = denrom_func(left_image[0])
    target_image = denrom_func(target_image[0])
    right_image = denrom_func(right_image[0])

    warped_images = [denrom_func(warped_image[0]) for warped_image in warped_images]
    
    # left_to_target / target / right_to_target
    # left_to_target loss / right_to_target loss / mask
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    
    axes[0, 0].imshow(left_image, vmin=0, vmax=255)
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(warped_images[0], vmin=0, vmax=255)
    axes[0, 1].set_title('Left to Target')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(target_image, vmin=0, vmax=255)
    axes[0, 2].set_title('Target Image')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(warped_images[1], vmin=0, vmax=255)
    axes[0, 3].set_title('Right to Target')
    axes[0, 3].axis('off')

    axes[0, 4].imshow(right_image, vmin=0, vmax=255)
    axes[0, 4].set_title('Right Image')
    axes[0, 4].axis('off')


    axes[1, 1].imshow(warped_losses[0][0])
    axes[1, 1].set_title('Left to Target Loss')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(warped_losses[1][0])
    axes[1, 2].set_title('Right to Target Loss')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(masks[0][0])
    axes[1, 3].set_title('Mask')
    axes[1, 3].axis('off')
    fig.tight_layout()

    # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
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