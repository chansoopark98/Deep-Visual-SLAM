import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_images(left: tf.Tensor,
                right: tf.Tensor,
                gt_flow: tf.Tensor,
                pred_flow: tf.Tensor,
                denorm_func: callable) -> tf.Tensor:
    
    left = denorm_func(left[0])
    right = denorm_func(right[0])
    gt_flow = gt_flow[0]
    pred_flow = pred_flow[0]

    x_vmin = tf.reduce_min(gt_flow[:, :, 0])
    x_vmax = tf.reduce_max(gt_flow[:, :, 0])
    y_vmin = tf.reduce_min(gt_flow[:, :, 1])
    y_vmax = tf.reduce_max(gt_flow[:, :, 1])

    fig, axes = plt.subplots(1, 6, figsize=(20, 5))

    axes[0].imshow(left)
    axes[0].set_title('Left Image')
    axes[0].axis('off')

    axes[1].imshow(right)
    axes[1].set_title('Right Image')
    axes[1].axis('off')

    axes[2].imshow(gt_flow[:, :, 0], vmin=x_vmin, vmax=x_vmax)
    axes[2].set_title('GT Flow - 1')
    axes[2].axis('off')

    axes[3].imshow(gt_flow[:, :, 1], vmin=y_vmin, vmax=y_vmax)
    axes[3].set_title('GT Flow - 2')
    axes[3].axis('off')

    axes[4].imshow(pred_flow[:, :, 0], vmin=x_vmin, vmax=x_vmax)
    axes[4].set_title('Pred Flow - 1')
    axes[4].axis('off')

    axes[5].imshow(pred_flow[:, :, 1], vmin=y_vmin, vmax=y_vmax)
    axes[5].set_title('Pred Flow - 2')
    axes[5].axis('off')
    
    fig.tight_layout()

    # 이미지를 Tensorboard에 로깅하기 위해 버퍼에 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    plt.close()
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    return tf.expand_dims(image, 0)