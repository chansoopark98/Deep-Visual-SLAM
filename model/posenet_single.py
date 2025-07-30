# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

try:
    from .resnet_encoder import ResnetEncoder
    from .raft.core.raft import SmallRAFT
except:
    from resnet_encoder import ResnetEncoder
    from raft.core.raft import SmallRAFT

class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        b, c, h, w = net.shape
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h*w).mean(-1).view(b, c, 1, 1)

        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + self.convq_glo(glo))

        net = (1-z) * net + z * q
        return net
    
class FlowUpdateModule(nn.Module):
    """
    RAFT flow + ConvGRU 기반 pose twist 회귀 모듈
    입력: net (B,128,H/8,W/8), inp (B,128,H/8,W/8),
           corr (B, Ccorr, H/8, W/8), flow (B,2,H/8,W/8)
    출력: pose6d (B,6)
    """
    def __init__(self, corr_planes):
        super().__init__()
        # Correlation encoder
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_planes, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        # Flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        # ConvGRU: hidden=128, input_size=128+128+64
        self.gru = ConvGRU(128, 128+128+64)
        # Pose head: global pooling -> 6D
        self.pose_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 6)
        )

    def forward(self, net, inp, corr, flow):
        # net, inp, corr already in (B, C, H, W) format
        corr_feat = self.corr_encoder(corr)
        flow_feat = self.flow_encoder(flow)
        # Concatenate inputs for GRU
        x = torch.cat([inp, corr_feat, flow_feat], dim=1)  # (B,320,H,W)
        # GRU update: net -> new hidden state
        net = self.gru(net, x)
        # Pose regression
        pose6d = self.pose_head(net)  # (B,6)
        return pose6d
    
class FlowPoseNet(nn.Module):
    def __init__(self):
        super(FlowPoseNet, self).__init__()
        
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.small = True
        args.alternate_corr = False
        args.mixed_precision = False
        corr_planes=4*(2*3+1)**2
        
        # Initialize RAFT for flow estimation
        self.flow_net = SmallRAFT(args)
        path = './raft-small.pth'
        state_dict = torch.load(path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.flow_net.load_state_dict(new_state_dict)
        self.flow_net.eval()

        self.update_module = FlowUpdateModule(corr_planes)

    @torch.no_grad()
    def extract_flow(self, x1, x2):
        """
        RAFT Estimation
        Args:
            x1: tensor of shape [B, 3, H, W] (left image)
            x2: tensor of shape [B, 3, H, W] (right image)
        Returns:
            flow: list of tensors, each of shape [B, 2, H, W]
        """
        flow_list = self.flow_net(x1, x2)
        flow_map = flow_list[-1]  # Use the last flow map
        return flow_map

    def forward(self, input_images):
        """
        Args:
            input_images: tensor of shape [B, 6, H, W] (concatenated left and right images)
        Returns:
            axis_angle: tensor of shape [B, 1, 1, 3]
            translation: tensor of shape [B, 1, 1, 3]
        """
        left_tensor = input_images[:, :3, :, :]
        left_tensor = 2 * left_tensor - 1.0  # Normalize to [-1, 1]
        right_tensor = input_images[:, 3:, :, :]
        right_tensor = 2 * right_tensor - 1.0

        with torch.no_grad():
            flow = self.extract_flow(left_tensor, right_tensor)   # (B,2,H,W)

        # TODO
        return axis_angle, translation

class PoseNet(nn.Module):
    """Pose estimation network with ResNet encoder and pose decoder"""
    
    def __init__(self, num_layers=18, pretrained=True, num_input_images=2, stride=1):
        super(PoseNet, self).__init__()
        
        self.num_input_images = num_input_images
        
        self.encoder = ResnetEncoder(num_layers=num_layers,
                                    pretrained=pretrained,
                                    num_input_images=num_input_images)
        
        # Get number of channels from encoder
        self.num_ch_enc = self.encoder.num_ch_enc

        # Decoder layers
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6, 1)
        
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, input_images):
        """
        Args:
            input_images: tensor of shape [B, 6, H, W]
        Returns:
            axisangle: tensor of shape [B, 1, 1, 3]
            translation: tensor of shape [B, 1, 1, 3]
        """
        feature = self.encoder(input_images)
            
        # Decoder
        last_features = feature[-1]

        cat_features = self.relu(self.convs["squeeze"](last_features))

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, 1, 1, 6)

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


    model = FlowPoseNet()
    axis_angle, translation = model(dummy_input)
    print(axis_angle.shape, translation.shape)