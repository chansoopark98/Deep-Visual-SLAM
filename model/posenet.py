# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict

try:
    from .resnet_encoder import ResnetEncoder
except:
    from resnet_encoder import ResnetEncoder


class PoseNet(nn.Module):
    """Pose estimation network with ResNet encoder and pose decoder"""
    
    def __init__(self, num_layers=18, pretrained=True, num_input_images=2, 
                 num_frames_to_predict_for=None, stride=1):
        super(PoseNet, self).__init__()
        
        self.num_input_images = num_input_images
        
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_images - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        
        # Encoders - one for each input image
        self.encoders = nn.ModuleList()
        for i in range(num_input_images):
            self.encoders.append(
                ResnetEncoder(num_layers=num_layers,
                            pretrained=pretrained,
                            num_input_images=1)
            )
        
        # Get number of channels from encoder
        self.num_ch_enc = self.encoders[0].num_ch_enc
        
        # Decoder layers
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_images * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, input_images):
        """
        Args:
            input_images: tensor of shape [B, num_input_images*3, H, W]
                         or list of tensors each of shape [B, 3, H, W]
        Returns:
            axisangle: tensor of shape [B, num_frames_to_predict_for, 1, 3]
            translation: tensor of shape [B, num_frames_to_predict_for, 1, 3]
        """
        # Handle input format
        if isinstance(input_images, torch.Tensor):
            # Split concatenated images
            if input_images.shape[1] == self.num_input_images * 3:
                input_images = torch.split(input_images, 3, dim=1)
            else:
                raise ValueError(f"Expected {self.num_input_images * 3} channels, got {input_images.shape[1]}")
        
        # Encode each image
        all_features = []
        for i, image in enumerate(input_images):
            features = self.encoders[i](image)
            all_features.append(features)
        
        # Decoder
        last_features = [f[-1] for f in all_features]
        
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        
        out = out.mean(3).mean(2)
        
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        axisangle = out[..., :3]
        translation = out[..., 3:]
        
        return axisangle, translation
    

if __name__ == "__main__":
    import numpy as np
    
    # Test 1: PoseNet with multiple encoders
    print("Test 1: PoseNet with multiple encoders")
    model1 = PoseNet(num_layers=18, pretrained=True, num_input_images=2)
    
    # Test with concatenated input
    dummy_input = torch.randn(4, 6, 256, 512)  # Batch=4, 2 images concatenated
    axisangle1, translation1 = model1(dummy_input)
    print(f"Axis-angle shape: {axisangle1.shape}")
    print(f"Translation shape: {translation1.shape}")
    
    # Test with list of images
    img1 = torch.randn(4, 3, 256, 512)
    img2 = torch.randn(4, 3, 256, 512)
    axisangle2, translation2 = model1([img1, img2])
    print(f"Axis-angle shape (list input): {axisangle2.shape}")
    print(f"Translation shape (list input): {translation2.shape}")
    

    # Test 3: Multiple frames prediction
    print("\nTest 3: PoseNet with 3 input images")
    model3 = PoseNet(num_layers=18, pretrained=True, num_input_images=3)
    dummy_input3 = torch.randn(4, 9, 256, 512)  # 3 images concatenated
    axisangle4, translation4 = model3(dummy_input3)
    print(f"Axis-angle shape (3 images): {axisangle4.shape}")
    print(f"Translation shape (3 images): {translation4.shape}")
    print(f"Number of poses predicted: {axisangle4.shape[1]}")