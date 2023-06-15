import torch
import torch.nn as nn
import torch.nn.functional as F


_BATCH_NORM = nn.BatchNorm2d

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, mid_ch, out_ch, rates):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, mid_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, mid_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, mid_ch))


        concat_ch = mid_ch * (len(rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, out_ch, 1, 1, 0, 1))


    def forward(self, x):
        for frame_id in x.keys():
            if not isinstance(frame_id, str): # not right view
                for idx in range(5):
                    if idx in [4]: # only the last layer?
                        tmp = x[frame_id][idx]

                        out = torch.cat([stage(tmp) for stage in self.stages.children()], dim=1)
                        out = self.fc1(out)

                        x[frame_id][idx] = out
        return x