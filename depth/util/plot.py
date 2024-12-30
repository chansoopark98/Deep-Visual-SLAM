import io
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_images(image: tf.Tensor, pred_depths: tf.Tensor, gt_depth: tf.Tensor) -> tf.Tensor:
    image = image[0]
    # Plot 설정
    depth_len = len(pred_depths)

    fig, axes = plt.subplots(1, 2 + depth_len, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(gt_depth[0].numpy(), vmin=0., vmax=10., cmap='plasma')
    axes[1].set_title('GT Depth')
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
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    return tf.expand_dims(image, 0)