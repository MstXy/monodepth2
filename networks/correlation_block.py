# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt
from typing import Dict, Optional, Sequence, Union
import torch
import torch.nn as nn
from monodepth2.networks.correlation import correlation


class CorrBlock(nn.Module):
    """Basic Correlation Block.

    A block used to calculate correlation.

    Args:
        corr (dict): Config dict for build correlation operator.
        act_cfg (dict): Config dict for activation layer.
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Default: False.
        scale_mode (str): How to scale correlation. The value includes
        `'dimension'` and `'sqrt dimension'`, but it doesn't work when
        scaled = True. Default to `'dimension'`.
    """

    def __init__(self,
                 corr_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 scaled: bool = False,
                 scale_mode: str = 'dimension') -> None:
        super(CorrBlock, self).__init__()

        assert scale_mode in ('dimension', 'sqrt dimension'), (
            'scale_mode must be \'dimension\' or \'sqrt dimension\' '
            f'but got {scale_mode}')

        if not act_cfg['type'] == 'LeakyReLU':
            raise NotImplementedError
        act = nn.LeakyReLU(negative_slope=act_cfg['negative_slope'])
        self.scaled = scaled
        self.scale_mode = scale_mode

        self.kernel_size = corr_cfg.get('kernel_size', 1) # correlation.kernel_size
        self.corr_block = [correlation.FunctionCorrelation, act]
        self.stride = corr_cfg.get('stride', 1)

    def forward(self, feat1: torch.Tensor,
                feat2: torch.Tensor) -> torch.Tensor:
        """Forward function for CorrBlock.

        Args:
            feat1 (Tensor): The feature from the first image.
            feat2 (Tensor): The feature from the second image.

        Returns:
            Tensor: The correlation between feature1 and feature2.
        """
        N, C, H, W = feat1.shape
        scale_factor = 1.

        if self.scaled:
            if 'sqrt' in self.scale_mode:
                scale_factor = sqrt(float(C * self.kernel_size**2))
            else:
                scale_factor = float(C * self.kernel_size**2)

        corr = self.corr_block[0](feat1, feat2) / scale_factor
        corr = corr.view(N, -1, H // self.stride, W // self.stride)
        out = self.corr_block[1](corr)
        return out

    def __repr__(self):
        s = super().__repr__()
        s += f'\nscaled={self.scaled}'
        s += f'\nscale_mode={self.scale_mode}'
        return s


class CorrEncoder(nn.Module):
    def __init__(self,
                 pyramid_levels: Sequence[str],
                 kernel_size: Sequence[int] = (3, 3, 3, 3),
                 out_channels: Sequence[int] = (128, 256, 512, 1024),
                 inter_channels=(64, 64, 64),
                 redir_in_channels: int = 256,
                 redir_channels: int = 32,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 paddings: Sequence[int] = (0, 0, 0, 0),
                 corr_cfg: dict = dict(
                     type='Correlation',
                     kernel_size=1,
                     max_displacement=10,
                     stride=1,
                     padding=0,
                     dilation_patch=2),
                 scaled: bool = False,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)) -> None:

        super(CorrEncoder, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.corr = CorrBlock(corr_cfg, act_cfg, scaled=scaled, scale_mode='dimension')
        self.conv_redir = nn.Sequential(
            nn.Conv2d(redir_in_channels, redir_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(act_cfg['negative_slope'])
        )
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(441 + redir_channels, out_channels[0], kernel_size[0], strides[0], paddings[0], dilations[0]),
                nn.LeakyReLU(act_cfg['negative_slope'])
            ),
            nn.Sequential(
                nn.Conv2d(out_channels[0], inter_channels[1], kernel_size[1], strides[1], paddings[1], dilations[1]),
                nn.LeakyReLU(act_cfg['negative_slope']),
                nn.Conv2d(inter_channels[1], out_channels[1], kernel_size[1], 1, paddings[1], dilations[1]),
                nn.LeakyReLU(act_cfg['negative_slope'])
            ),
            nn.Sequential(
                nn.Conv2d(out_channels[1], inter_channels[2], kernel_size[2], strides[2], paddings[2], dilations[2]),
                nn.LeakyReLU(act_cfg['negative_slope']),
                nn.Conv2d(inter_channels[2], out_channels[2], kernel_size[2], 1, paddings[2], dilations[2]),
                nn.LeakyReLU(act_cfg['negative_slope'])
            ),
            nn.Sequential(
                nn.Conv2d(out_channels[2], inter_channels[3], kernel_size[3], strides[3], paddings[3], dilations[3]),
                nn.LeakyReLU(act_cfg['negative_slope']),
                nn.Conv2d(inter_channels[3], out_channels[3], kernel_size[3], 1, paddings[3], dilations[3]),
                nn.LeakyReLU(act_cfg['negative_slope'])
            )
        )



    def forward(self, f1: torch.Tensor,
                f2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward function for CorrEncoder.
        Args:
            f1 (Tensor): The feature from the first input image.
            f2 (Tensor): The feature from the second input image.
        """

        corr_feat = self.corr(f1, f2)  # 441
        redir_feat = self.conv_redir(f1)
        x = torch.cat((redir_feat, corr_feat), dim=1)

        outs = dict()
        for i, convs_layer in enumerate(self.layers):
            x = convs_layer(x)
            # After correlation, the feature level starts at level3
            if 'level' + str(i + 3) in self.pyramid_levels:
                outs['level' + str(i + 3)] = x

        return outs



























