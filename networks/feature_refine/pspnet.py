import torch
from torch import nn
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, dropout=0.3):
        super(PPM, self).__init__()
        self.features = []
        self.bottleneck_dim = in_dim
        for bin in bins:
            self.features.append(
                nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True))
            )
        self.features = nn.ModuleList(self.features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim*2, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x):

        for frame_id in x.keys():
            if not isinstance(frame_id, str): # not right view
                for idx in range(5):
                    if idx in [4]: # only the last layer?
                        tmp = x[frame_id][idx]
                        tmp_size = tmp.size()
                        out = [tmp]
                        for f in self.features:
                            out.append(F.interpolate(f(tmp), tmp_size[2:], mode='bilinear', align_corners=True))
                        out = torch.cat(out, 1)
                        tmp = self.bottleneck(out)
                        x[frame_id][idx] = tmp
        return x