import torch
import torch.nn as nn
import torch.nn.functional as F


class CVConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv0 = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, 1),
                                    nn.BatchNorm2d(out_channels), 
                                    nn.ReLU()
                                )

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                    nn.BatchNorm2d(out_channels), 
                                    nn.ReLU()
                                )

        self.conv2 = nn.Sequential(
                                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                    nn.BatchNorm2d(out_channels), 
                                )

        # self.conv3 = nn.Sequential(
        #                             nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #                             nn.BatchNorm2d(out_channels), 
        #                             nn.ReLU()
        #                         )
        # self.conv4 = nn.Sequential(
        #                             nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #                             nn.BatchNorm2d(out_channels), 
        #                         )
    
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        residual1 = x
        x = self.conv2(self.conv1(x))

        x += residual1
        x = self.relu(x)

        # residual2 = x
        # x = self.conv4(self.conv3(x))

        # x += residual2
        # x = self.relu(x)
      
        return x









