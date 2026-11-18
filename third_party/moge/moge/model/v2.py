# MODIFIED for Deep-Visual-SLAM (see ../../NOTICE.md): utils3d import and MoGeModel.infer removed;
# inference post-processing lives in dvslam/depth/moge.py.
from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.amp
import torch.version
from huggingface_hub import hf_hub_download

from ..utils.geometry import normalized_view_plane_uv
from .utils import wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing, wrap_module_with_autocast
from .modules.dinov2_encoder import DINOv2Encoder
from .modules.mlp import MLP
from .modules.conv_stack import ConvStack


class MoGeModel(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        num_tokens_range: List[int] = [1200, 3600],
        **deprecated_kwargs
    ):
        super(MoGeModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        
        self.encoder = DINOv2Encoder(**encoder) 
        self.neck = ConvStack(**neck)
        if points_head is not None:
            self.points_head = ConvStack(**points_head) 
            self.points_head.fp32_output_projection = True  # Use fp32 for the final output projection for better surface reconstruction
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
            self.normal_head.fp32_output_projection = True
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'MoGeModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)

        # NOTE: `strict=False` so that a checkpoint may legitimately omit optional heads. That also
        # means anything the checkpoint does not provide silently keeps its random initialization,
        # so report it -- e.g. loading a v2 checkpoint into a v3 config leaves the whole refiner
        # untrained, which is otherwise indistinguishable from a bad prediction.
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        if missing_keys:
            warnings.warn(
                f"{len(missing_keys)} parameter(s) are absent from the checkpoint and keep their random "
                f"initialization: {missing_keys}"
            )
        if unexpected_keys:
            warnings.warn(
                f"{len(unexpected_keys)} parameter(s) in the checkpoint have no counterpart in the model "
                f"and were ignored: {unexpected_keys}"
            )

        return model
    
    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_mixed_precision(self, dtype: torch.dtype = torch.bfloat16):
        """Enable fine-grained mixed precision: run the encoder in `dtype`, keep the neck and heads in fp32.

        Calling this repeatedly replaces the previous wrapping rather than stacking it.
        """
        for handle in getattr(self, '_autocast_handles', []):
            handle.remove()

        module_dtype_map = [
            (self.encoder, dtype),
            (self.neck, torch.float32),
            *((getattr(self, head, None), torch.float32) for head in ['points_head', 'normal_head', 'mask_head', 'scale_head']),
        ]
        self._autocast_handles = [
            wrap_module_with_autocast(module, device_type='cuda', dtype=module_dtype)
            for module, module_dtype in module_dtype_map if module is not None
        ]

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output =='sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points
    
    def forward(self, image: torch.Tensor, num_tokens: Union[int, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)

        # Heads decoding
        points, normal, mask = (getattr(self, head)(features)[-1] if hasattr(self, head) else None for head in ['points_head', 'normal_head', 'mask_head'])
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None
        
        # Resize
        points, normal, mask = (F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False) if v is not None else None for v in [points, normal, mask])
        
        # Remap output
        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(points)     # slightly improves the performance in case of very large output values
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'points': points, 
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict
