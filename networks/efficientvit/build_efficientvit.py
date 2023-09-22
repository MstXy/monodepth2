import os

import torch
import torch.nn as nn
import numpy as np

from networks.efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1
from networks.efficientvit.models.nn.drop import apply_drop_func


class EfficientViT(nn.Module):
    def __init__(self, model_name="b1", pretrained=True, num_input_images=1) -> None:
        super(EfficientViT, self).__init__()

        if model_name == "b1":
            self.encoder = efficientvit_backbone_b1()
            apply_drop_func(self.encoder.stages, {"name":"droppath", "drop_prob":0.05, "linear_decay":"true"})
            self.num_ch_enc = np.array([16,32,64,128,256])
        
            weight_path = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/networks/efficientvit/weights/b1-r288.pt"
            
        else:
            raise NotImplementedError

        if pretrained:
            weight = torch.load(weight_path, map_location="cpu")
            weight = weight["state_dict"]
            model_dict = self.encoder.state_dict()
            for key in model_dict.keys():
                model_dict[key] = weight["backbone." + key]
            self.encoder.load_state_dict(model_dict, strict=True)


    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        self.features.append(self.encoder.input_stem(x))
        for stage_id, stage in enumerate(self.encoder.stages, 1):
            self.features.append(stage(self.features[-1]))
        return self.features
