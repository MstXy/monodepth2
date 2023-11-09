import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict

from modules.aspp import ASPP
from modules.basic_modules import upsample, ConvBlock, conv3x3pad


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_aspp=True, use_skips=True,
                 norm_layer=nn.BatchNorm2d):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.use_aspp = use_aspp
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ASPP
        if use_aspp:
            self.aspp = ASPP(in_channels=num_ch_enc[-1], out_channels=self.num_ch_dec[-1], atrous_rates=[12, 24, 36],
                             norm_layer=norm_layer)

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            setattr(self, "upconv_{}_0".format(i), ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, "upconv_{}_1".format(i), ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            setattr(self, "reconv_{}".format(s), conv3x3pad(self.num_ch_dec[s], self.num_output_channels))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        x = self.aspp(x)
        for i in range(4, -1, -1):
            # x = self.convs[("upconv", i, 0)](x)
            x = getattr(self, "upconv_{}_0".format(i))(x)
            x = [upsample(x, scale_factor=2, mode=self.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # x = self.convs[("upconv", i, 1)](x)
            x = getattr(self, "upconv_{}_1".format(i))(x)
            if i in self.scales:
                self.outputs["disp_{}".format(self.func, i)] = getattr(self, "reconv_{}".format(i))(x)

        return self.outputs