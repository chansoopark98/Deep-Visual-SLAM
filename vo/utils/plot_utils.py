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
        self.vis_batch_size = config['Train']['vis_batch_size']
        if self.vis_batch_size > self.batch_size:
            self.vis_batch_size = self.batch_size
        self.image_size = (config['Train']['img_h'], config['Train']['img_w'])
        self.num_source = config['Train']['num_source']  # 2
        self.num_scales = config['Train']['num_scale']  # 4
        self.min_depth = config['Train']['min_depth']
        self.max_depth = config['Train']['max_depth']

    def plot_images(self, target_images: dict, pred_depths: List[torch.Tensor], 
                denorm_func: Callable) -> np.ndarray:
        denormed_targets = []
        for scale in range(self.num_scales):
            target_image = target_images[('target_image', scale)][0]  # Get first batch
            target_image = target_image.detach().cpu()
            target_image = denorm_func(target_image)
            target_image = target_image.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            denormed_targets.append(target_image)

        pred_depths = [depth[0] for depth in pred_depths]  # Get first batch

        # 4줄 2칸 (왼: 이미지, 오: depth)
        fig, axes = plt.subplots(self.num_scales, 2, figsize=(8, 3 * self.num_scales))

        for scale in range(self.num_scales):
            # 왼쪽: 이미지
            axes[scale, 0].imshow(denormed_targets[scale])
            axes[scale, 0].set_title(f'Target Image - Scale {scale}')
            axes[scale, 0].axis('off')

            # 오른쪽: depth
            depth = pred_depths[scale].detach().cpu().numpy()[0]  # [H, W]
            axes[scale, 1].imshow(depth, vmin=self.min_depth, vmax=self.max_depth, cmap='plasma')
            axes[scale, 1].set_title(f'Predicted Depth - Scale {scale}')
            axes[scale, 1].axis('off')

        fig.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Load from buffer
        pil_image = Image.open(buf)
        image_array = np.array(pil_image)
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        return image_array