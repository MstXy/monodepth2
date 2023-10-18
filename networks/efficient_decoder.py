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
from monodepth2.layers import *
from monodepth2.networks.attention_layers import *

import math
#EfficientNet: 6.78 M, 37.33G FLOPS 128
#EfficientNet: 6.25 M, 37.26G FLOPS
# EfficientNet: 4.60 M, 10.31G FLOPS
# from memory_profiler import profile
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature

class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    # @profile
    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
    # @profile
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SimpleFPN(nn.Module):
    def __init__(self, num_ch, top=False):
        super(SimpleFPN, self).__init__()
        self.top = top
        self.swish = MemoryEfficientSwish()

        if self.top is not True:
            self.down_conv = SeparableConvBlock(num_ch, stride=2)

        self.up_conv = SeparableConvBlock(num_ch)
        self.conv = SeparableConvBlock(num_ch)
        self.cat_conv = Conv3x3(num_ch, num_ch)
        self.iconv = SeparableConvBlock(num_ch)

    def forward(self, feat_up, feat, feat_down=None):
        feat_up = self.up_conv(upsample(feat_up))
        feat = self.conv(feat)
        if feat_down is not None:
            feat_down = self.down_conv(feat_down)
            out = self.iconv(self.swish(self.cat_conv(feat_up + feat + feat_down)))
        else:
            out = self.iconv(self.swish(self.cat_conv(feat_up + feat)))

        return out


class EfficientDecoder(nn.Module):
    def __init__(self, num_ch_enc, epsilon=1e-4):
        super(EfficientDecoder, self).__init__()
        bottleneck = 128
        self.do = nn.Dropout(p=0.5)

        self.iconv5_0 = SeparableConvBlock(num_ch_enc[4], bottleneck)
        self.iconv4_0 = SeparableConvBlock(num_ch_enc[3], bottleneck)
        self.iconv3_0 = SeparableConvBlock(num_ch_enc[2], bottleneck)
        self.iconv2_0 = SeparableConvBlock(num_ch_enc[1], bottleneck)
        self.iconv1_0 = SeparableConvBlock(num_ch_enc[0], bottleneck)

        self.simfpn4_0 = SimpleFPN(bottleneck)
        self.simfpn3_0 = SimpleFPN(bottleneck)
        self.simfpn2_0 = SimpleFPN(bottleneck)
        self.simfpn1_0 = SimpleFPN(bottleneck, top=True)

        self.simfpn3_1 = SimpleFPN(bottleneck)
        self.simfpn2_1 = SimpleFPN(bottleneck)
        self.simfpn1_1 = SimpleFPN(bottleneck, top=True)

        self.simfpn2_2 = SimpleFPN(bottleneck)
        self.simfpn1_2 = SimpleFPN(bottleneck, top=True)

        self.simfpn1_3 = SimpleFPN(bottleneck, top=True)

        self.upconv1 = SeparableConvBlock(bottleneck, bottleneck//2)
        self.conv1 = nn.Sequential(SeparableConvBlock(bottleneck//2),
                                   Conv3x3(bottleneck//2, bottleneck//2))
        self.swish = MemoryEfficientSwish()

        ## [Default]: Channel Attention
        # self.attn_3 = ChannelAttention(bottleneck)
        # self.attn_2 = ChannelAttention(bottleneck)
        # self.attn_1 = ChannelAttention(bottleneck)
        # self.attn_0 = ChannelAttention(bottleneck//2)

        # ## [Opt#1]: Multi Head Attention
        # self.attn_3 = MultiHeadAttention(n_head=1, d_model=bottleneck, d_k=bottleneck, d_v=bottleneck)
        # self.attn_2 = MultiHeadAttention(n_head=1, d_model=bottleneck, d_k=bottleneck, d_v=bottleneck)
        # self.attn_1 = MultiHeadAttention(n_head=1, d_model=bottleneck, d_k=bottleneck, d_v=bottleneck)
        # self.attn_0 = MultiHeadAttention(n_head=1, d_model=bottleneck//2, d_k=bottleneck//2, d_v=bottleneck//2)

        # ## [Opt#2]: EfficientViT 
        # self.attn_3 = LiteMLA(in_channels=bottleneck, out_channels=bottleneck)
        # self.attn_2 = LiteMLA(in_channels=bottleneck, out_channels=bottleneck)
        # self.attn_1 = LiteMLA(in_channels=bottleneck, out_channels=bottleneck)
        # self.attn_0 = LiteMLA(in_channels=bottleneck//2, out_channels=bottleneck//2)

        ## [Opt#3]: RVT
        self.attn_3 = MaxVitAttentionPairCl(dim=bottleneck, skip_first_norm=False)
        self.attn_2 = MaxVitAttentionPairCl(dim=bottleneck, skip_first_norm=False)
        self.attn_1 = MaxVitAttentionPairCl(dim=bottleneck, skip_first_norm=False)
        self.attn_0 = MaxVitAttentionPairCl(dim=bottleneck//2, skip_first_norm=False)

        # disp
        self.disp3 = nn.Sequential(SeparableConvBlock(bottleneck, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(SeparableConvBlock(bottleneck, 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(SeparableConvBlock(bottleneck, 1), nn.Sigmoid())
        self.disp0 = nn.Sequential(SeparableConvBlock(bottleneck//2, 1), nn.Sigmoid())

    # @profile
    def forward(self, input_features, f_i=0):
        self.outputs = {}
        l1, l2, l3, l4, l5 = input_features

        l5 = self.iconv5_0(self.do(l5))
        l4 = self.iconv4_0(self.do(l4))
        l3 = self.iconv3_0(l3)
        l2 = self.iconv2_0(l2)
        l1 = self.iconv1_0(l1)

        x4 = self.simfpn4_0(l5, l4, l3)
        x3 = self.simfpn3_0(l4, l3, l2)
        x2 = self.simfpn2_0(l3, l2, l1)
        x1 = self.simfpn1_0(l2, l1)

        l3 = self.simfpn3_1(x4, x3, x2)
        l2 = self.simfpn2_1(x3, x2, x1)
        l1 = self.simfpn1_1(x2, x1)

        x2 = self.simfpn2_2(l3, l2, l1)
        x1 = self.simfpn1_2(l2, l1)

        x1 = self.simfpn1_3(x2, x1)

        x0 = upsample(self.upconv1(x1))
        x0 = self.swish(self.conv1(x0))

        disp3 = self.disp3(self.attn_3(x3))
        disp2 = self.disp2(self.attn_2(x2))
        disp1 = self.disp1(self.attn_1(x1))
        disp0 = self.disp0(self.attn_0(x0))

        self.outputs[("disp", 3)] = disp3
        self.outputs[("disp", 2)] = disp2
        self.outputs[("disp", 1)] = disp1
        self.outputs[("disp", 0)] = disp0

        return self.outputs


if __name__ == '__main__':
    feat = []

    feat = []

    b = 1
    c = [16, 24, 40, 112, 320, 1280]
    h = 192
    w = 640
    disp = {}
    for n in range(5):
        h_ = h // 2 ** (n + 1)
        w_ = w // 2 ** (n + 1)
        c_ = c[n]
        feat.append(torch.randn(b, c_, h_, w_).cuda())
        # feat.append(torch.randn(b, c_, h_, w_))

    c_ = c[5]
    feat.append(torch.randn(b, c_, h_, w_).cuda())
    # feat.append(torch.randn(b, c_, h_, w_))

    former_decoder = EfficientDecoder(num_ch_enc=c).cuda()
    # former_decoder = EfficientDecoder(num_ch_enc=c)

    # from memonger import SublinearSequential
    #
    # net2 = SublinearSequential(
    #     *list(net1.children())
    # )

    total = sum([param.nelement() for param in former_decoder.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    import time
    init_time = time.time()
    for i in range(100):
        result = former_decoder(feat)
    end_time = time.time()
    inferring = end_time - init_time
    print(100 / inferring)
    print("down")
