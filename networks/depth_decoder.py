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
from layers import *
from networks.flownet_decoder import DeconvModule
from networks.feature_refine.transformers import MultiHeadAttention, MultiHeadAttentionOne, GroupAttention


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, inter_output=False, drn=False, depth_att=False, depth_cv=False, depth_refine=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.inter_output = inter_output

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.num_ch_cv = np.array([None, 256, 512, 512, 1024])

        # dilated backbone
        self.drn = drn
        # attention for multi-stage depth decode
        self.depth_att = depth_att
        # cost volume for depth estimation 
        self.depth_cv = depth_cv
        # coarse2fine
        self.depth_refine = depth_refine

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # depth_att
            if self.depth_att:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                # self.convs[("att", i)] = ChannelAttention(num_ch_in)
                # self.convs[("att", i)] = SpatialAttention()
                self.convs[("att", i)] = CS_Block(num_ch_in)
                # self.convs[("att", i)] = MultiHeadAttentionOne(n_head=1, d_model=num_ch_in, d_k=num_ch_in, d_v=num_ch_in)
            
            # depth cv
            if self.depth_cv and i > 0:
            #     num_ch_in = self.num_ch_cv[i]
            #     num_ch_out = self.num_ch_enc[i - 1]
            #     self.convs[("cv_deconv", i)] = DeconvModule(in_channels=num_ch_in,
            #                                                 out_channels=num_ch_out,
            #                                                 kernel_size=4,
            #                                                 stride=2,
            #                                                 padding=1,
            #                                                 bias=True,
            #                                                 norm_cfg=None,
            #                                                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                    num_ch_mid = num_ch_in
                    # num_ch_in += self.num_ch_enc[i - 1]
                    num_ch_in += self.num_ch_cv[i]
                self.convs[("cv_conv", i)] = nn.Sequential(
                                                nn.Conv2d(num_ch_in, num_ch_mid, 1), # 1*1 conv
                                                ConvBlock(num_ch_mid, num_ch_mid)
                                                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
                # #TODO: for eval nodeconv / deconv
                # if self.depth_cv:
                #     # num_ch_in += self.num_ch_enc[i - 1]
                #     num_ch_in += self.num_ch_cv[i]
            if self.depth_refine and i < 3:
                num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            if self.depth_refine and s != self.scales[-1]: # not the first level
                self.convs[("pred_up", s)] = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True) # upsample using deconv
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, corr1=None, corr2=None):
        # corr1/corr2: Dict type, {"level$#": Tensor, ...}
        if self.depth_cv:
            assert corr1 is not None
            assert corr2 is not None
            # order of corr's is determined in trainer.py
            corr1 = [None] + [v for v in corr1.values()]
            corr2 = [None] + [v for v in corr2.values()]

        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if self.drn:
                x = [upsample(x)] if i not in [3, 4] else [x]
            else:
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            if self.depth_refine and i < 3:
                x += [self.convs[("pred_up", i)](self.outputs[("c2f", i+1)])] # use pred of prev (deeper) layer
        
            x = torch.cat(x, 1)
            
            if self.depth_cv and i > 0:
                # corr = torch.cat([corr1, corr2], dim=1)
                corr = torch.maximum(corr1[i], corr2[i]) # TODO: max / sum / concat
                # corr = self.convs[("cv_deconv", i)](corr)
                x = torch.cat([x, F.interpolate(corr, x.size()[-2:], mode="bilinear", align_corners=False)], dim=1)
                x = self.convs[("cv_conv", i)](x)

            if self.depth_att:
                # apply self-attention
                x = self.convs[("att", i)](x)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                tmp = self.convs[("dispconv", i)](x)
                self.outputs[("c2f", i)] = tmp
                self.outputs[("disp", i)] = self.sigmoid(tmp)

        return self.outputs
