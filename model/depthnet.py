# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
try:
    from .layers import *
    from .resnet_encoder import ResnetEncoder
except:
    from layers import *
    from resnet_encoder import ResnetEncoder


class DepthNet(nn.Module):
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1,
                 scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthNet, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        # Encoder
        # Use integrated ResNet encoder
        self.encoder = ResnetEncoder(num_layers=num_layers,
                                    pretrained=pretrained,
                                    num_input_images=num_input_images)
        self.num_ch_enc = self.encoder.num_ch_enc
        self.use_encoder = True

        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input_data):
        """
        Args:
            input_data: Either image tensor [B, C, H, W] if using encoder,
                       or feature list if using decoder only
        Returns:
            outputs: dict with disparity (and optionally depth) predictions
        """
        # Encode if using integrated encoder
        if self.use_encoder:
            input_features = self.encoder(input_data)
        else:
            input_features = input_data
            
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                
                # Optionally compute depth
                if self.use_encoder and not self.training:
                    _, depth = disp_to_depth(self.outputs[("disp", i)], 
                                           self.min_depth, self.max_depth)
                    self.outputs[("depth", i)] = depth

        return self.outputs


if __name__ == "__main__":
    # Test 1: New integrated mode with encoder
    print("Test: DepthNet with integrated encoder")
    model = DepthNet(num_layers=18, pretrained=True, num_input_images=1)
    dummy_input = torch.randn(4, 3, 256, 512)
    outputs = model(dummy_input)
    print("Outputs with encoder:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\nTest: Backward compatibility check")
    print(f"Has encoder in model 1: {model.use_encoder}")