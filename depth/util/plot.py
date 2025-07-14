import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional, Callable
from PIL import Image

class PlotTool:
    def __init__(self, config: dict) -> None:
        self.batch_size = config['Train']['batch_size']
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_scales = config['Train']['num_scale']  # 4

    def plot_images(self, images: torch.Tensor, pred_depths: List[torch.Tensor], 
                   denorm_func: Callable) -> np.ndarray:
        """Plot images with multi-scale depth predictions
        
        Args:
            images: [B, 3, H, W] tensor
            pred_depths: List of [B, 1, H, W] tensors at different scales
            denorm_func: Function to denormalize images
            
        Returns:
            [H, W, 3] numpy array for visualization
        """
        # Get first image from batch and denormalize
        image = denorm_func(images[0].cpu())  # denorm_func returns [3, H, W] tensor

        
        # Convert to numpy and transpose to [H, W, 3] for matplotlib
        image = image.numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        pred_depths = [depth[0] for depth in pred_depths]  # Get first batch
        
        fig, axes = plt.subplots(1, 1 + self.num_scales, figsize=(10, 10))

        # Plot original image - now [H, W, 3]
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Image')
        axes[0].axis('off')
    
        # Plot depth maps at different scales
        for idx in range(self.num_scales):
            depth = pred_depths[idx].detach().cpu().numpy()[0]  # [H, W]
            axes[idx + 1].imshow(depth, vmin=0., vmax=10., cmap='plasma')
            axes[idx + 1].set_title(f'Scale {idx}')
            axes[idx + 1].axis('off')

        fig.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Load from buffer
        pil_image = Image.open(buf)
        image_array = np.array(pil_image)  # [H, W, 4] RGBA or [H, W, 3] RGB
        
        # Convert RGBA to RGB if needed
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]  # Remove alpha channel
        
        # Return [H, W, 3] for visualization
        return image_array