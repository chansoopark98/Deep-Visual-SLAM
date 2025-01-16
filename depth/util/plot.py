import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_images(image: tf.Tensor,
                pred_depths: tf.Tensor,
                gt_depth: tf.Tensor,
                mode: str,
                depth_max: float) -> tf.Tensor:
    """
    Generates a visualization of input images, ground truth depth maps, and predicted depth maps.

    Args:
        image (tf.Tensor): Input RGB image tensor of shape [B, H, W, 3].
        pred_depths (tf.Tensor): List of predicted depth tensors, each of shape [B, H, W, 1].
        gt_depth (tf.Tensor): Ground truth depth tensor of shape [B, H, W, 1].
        mode (str): Evaluation mode, either 'relative' or 'metric'.
        depth_max (float): Maximum depth value for visualization (used in 'metric' mode).

    Returns:
        tf.Tensor: A single image tensor suitable for TensorBoard logging, of shape [1, H, W, 4].
    """
    if mode not in ['relative', 'metric']:
        raise ValueError("Mode must be either 'relative' or 'metric'.")

    if mode == 'relative':
        prefix = 'Relative GT Depth'
        depth_max = 1.0
    elif mode == 'metric':
        prefix = 'Metric GT Depth'

    # Extract the first image and depth maps for visualization
    image = image[0]
    gt_depth = tf.clip_by_value(gt_depth[0], 0.0, depth_max)

    # Plot settings
    depth_len = len(pred_depths)
    fig, axes = plt.subplots(1, 2 + depth_len, figsize=(20, 5))

    # Input image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Ground truth depth
    axes[1].imshow(gt_depth.numpy(), vmin=0.0, vmax=depth_max, cmap='plasma')
    axes[1].set_title(f'{prefix} ({depth_max})')
    axes[1].axis('off')

    # Predicted depth maps
    for idx, pred_depth in enumerate(pred_depths):
        pred_depth = tf.clip_by_value(pred_depth[0], 0.0, depth_max)
        axes[2 + idx].imshow(pred_depth.numpy(), vmin=0.0, vmax=depth_max, cmap='plasma')
        axes[2 + idx].set_title(f'Pred Depth Scale {idx}')
        axes[2 + idx].axis('off')

    fig.tight_layout()

    # Save the plot to a buffer and convert it to a TensorFlow tensor
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Decode the PNG buffer into a TensorFlow tensor
    image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image_tensor, 0)
