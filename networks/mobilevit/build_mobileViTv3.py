import argparse
from networks.mobilevit.layers import arguments_nn_layers
from networks.mobilevit.classification import arguments_classification
from networks.mobilevit.misc.averaging_utils import arguments_ema
from networks.mobilevit.classification import build_classification_model
from networks.mobilevit.utils.opt_utils import load_config_file


import torch
import torch.nn as nn
import numpy as np


class MobileViT(nn.Module):

    def __init__(self, parser, pretrained=True, num_input_images=1) -> None:
        super(MobileViT, self).__init__()
        

        group = parser.add_argument_group(title="MobileViT arguments", description="MobileViT arguments")

        CONFIG_PATH = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/networks/mobilevit/config/classification/mobilevitv3_x_small_oneserver.yaml"
        PRETRAIN_PATH = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/networks/mobilevit/weights/MobileViTv3-v1/results_classification/mobilevitv3_XS_e300_7671/checkpoint_ema_best.pt"

        group.add_argument('--common.config-file', type=str, default=CONFIG_PATH) #TODO: default v3_xs

        # model related arguments
        group = arguments_nn_layers(parser=group)
        group = arguments_classification(parser=group)
        group = arguments_ema(parser=group)

        # no distributed
        group.add_argument('--ddp.rank', type=int, default=0)
        opts = parser.parse_args()
        opts = load_config_file(opts)

        # num_input_image:
        if num_input_images == 2: setattr(opts, "model.classification.num_input", 2)

        # pretrained
        if pretrained: setattr(opts, "model.classification.pretrained", PRETRAIN_PATH)

        self.encoder = build_classification_model(opts=opts)
        self.num_ch_enc = np.array([32,48,96,160,160])

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        x = self.encoder.conv_1(x) # first layer

        self.features.append(self.encoder.layer_1(x)) # B, 32, 96, 320
        self.features.append(self.encoder.layer_2(self.features[-1])) # B, 48, 48, 160
        self.features.append(self.encoder.layer_3(self.features[-1])) # B, 96, 24, 80

        self.features.append(self.encoder.layer_4(self.features[-1])) # B, 160, 12, 40
        self.features.append(self.encoder.layer_5(self.features[-1])) # B, 160, 6, 20

        # x = self.encoder.conv_1x1_exp(x)
        # print(sum(p.numel() for p in self.encoder.conv_1x1_exp.parameters() if p.requires_grad))
        # 103680

        return self.features


if __name__ == "__main__":

    # model = MobileNetV3(model_type="small")
    model = MobileViT()
    x = torch.randn((1,3,192,640))
    output = model(x)
    print("Finished")