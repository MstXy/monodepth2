import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from networks.feature_refine.transformers import MultiHeadAttentionOne, MultiHeadAttention

class MobileNetV3(nn.Module):

    def __init__(self, model_type="large") -> None:
        super(MobileNetV3, self).__init__()

        self.model_type = model_type

        if model_type == "large":
            self.encoder = models.mobilenet_v3_large(models.MobileNet_V3_Large_Weights)
            self.num_ch_enc = np.array([16,24,40,112,160])

        elif model_type == "small":
            self.encoder = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights)
            self.num_ch_enc = np.array([16,16,24,48,96]) 
    
    def forward(self, input_image):
        if self.model_type == "large":
            return self.forward_large(input_image)
        
        elif self.model_type == "small":
            return self.forward_small(input_image)
        
    def forward_large(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        x = self.encoder.features[0](x) # first layer

        self.features.append(self.encoder.features[1](x)) # 16
        self.features.append(self.encoder.features[3](self.encoder.features[2](self.features[-1]))) # 24
        self.features.append(self.encoder.features[6](self.encoder.features[5](self.encoder.features[4](self.features[-1])))) # 40
        self.features.append(self.encoder.features[12](self.encoder.features[11]( 
                                # 4 layers of c=80 ----
                                self.encoder.features[10](self.encoder.features[9](self.encoder.features[8](self.encoder.features[7](
                                self.features[-1]))))))) # 112
        self.features.append(self.encoder.features[15](self.encoder.features[14](self.encoder.features[13](self.features[-1])))) # 160

        return self.features

    def forward_small(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        self.features.append(self.encoder.features[0](x)) # 16

        self.features.append(self.encoder.features[1](self.features[-1])) # 16
        self.features.append(self.encoder.features[3](self.encoder.features[2](self.features[-1]))) # 24
        self.features.append(self.encoder.features[8](self.encoder.features[7](
                                self.encoder.features[6](self.encoder.features[5](self.encoder.features[4](self.features[-1])))))) # 48
        self.features.append(self.encoder.features[11](self.encoder.features[10](self.encoder.features[9](self.features[-1])))) # 96

        return self.features


class MobileNetV2(nn.Module):

    def __init__(self) -> None:
        super(MobileNetV2, self).__init__()
        
        self.encoder = models.mobilenet_v2(models.MobileNet_V2_Weights)
        self.num_ch_enc = np.array([16,24,32,96,160]) # TODO: rm17

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        x = self.encoder.features[0](x) # first layer

        self.features.append(self.encoder.features[1](x)) # 16
        self.features.append(self.encoder.features[3](self.encoder.features[2](self.features[-1]))) # 24
        self.features.append(self.encoder.features[6](self.encoder.features[5](self.encoder.features[4](self.features[-1])))) # 32
        self.features.append(self.encoder.features[13](self.encoder.features[12](self.encoder.features[11](
                                self.encoder.features[10](self.encoder.features[9](self.encoder.features[8](self.encoder.features[7](self.features[-1])))))))) # 96
        self.features.append(
            # self.encoder.features[17]( # TODO: rm17
                                self.encoder.features[16](self.encoder.features[15](self.encoder.features[14](self.features[-1])))) # 320
        # ) 

        return self.features
    

class MobileNetAtt(nn.Module):

    def __init__(self, n_head=1) -> None:
        super(MobileNetAtt, self).__init__()
        
        self.encoder = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights) #mobile net
        # self.att_1 = MultiHeadAttentionOne(n_head, 48,48,48) # d=48
        # self.att_2 = MultiHeadAttentionOne(n_head, 96,96,96) # d=96
        self.att_1 = MultiHeadAttention(n_head, 48,48,48) # d=48
        self.att_2 = MultiHeadAttention(n_head, 96,96,96) # d=96

        self.num_ch_enc = np.array([16,16,24,48,96]) 

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225


        self.features.append(self.encoder.features[0](x)) # 16

        self.features.append(self.encoder.features[1](self.features[-1])) # 16
        self.features.append(self.encoder.features[3](self.encoder.features[2](self.features[-1]))) # 24
        self.features.append(self.att_1(self.encoder.features[8](self.encoder.features[7](
                                self.encoder.features[6](self.encoder.features[5](self.encoder.features[4](self.features[-1]))))))) # 48
        self.features.append(self.att_2(self.encoder.features[11](self.encoder.features[10](self.encoder.features[9](self.features[-1]))))) # 96

        return self.features

class MobileNetAtt2(nn.Module):

    def __init__(self, n_head=1, pretrained=True) -> None:
        super(MobileNetAtt2, self).__init__()
        
        if pretrained: 
            self.encoder = models.mobilenet_v2(models.MobileNet_V2_Weights) #mobile net
        else:
            self.encoder = models.mobilenet_v2()

        # self.att_1 = MultiHeadAttentionOne(n_head, 96,96,96) # d=96
        # self.att_2 = MultiHeadAttentionOne(n_head, 160,160,160) # d=160 # TODO: rm17

        # # TODO: normal attention
        self.att_1 = MultiHeadAttention(n_head, 96,96,96) # d=96
        self.att_2 = MultiHeadAttention(n_head, 160,160,160) # d=160 # TODO: rm17

        self.num_ch_enc = np.array([16,24,32,96,160]) # TODO: rm17

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        x = self.encoder.features[0](x) # first layer

        self.features.append(self.encoder.features[1](x)) # 16
        self.features.append(self.encoder.features[3](self.encoder.features[2](self.features[-1]))) # 24
        self.features.append(self.encoder.features[6](self.encoder.features[5](self.encoder.features[4](self.features[-1])))) # 32
        self.features.append(self.att_1(self.encoder.features[13](self.encoder.features[12](self.encoder.features[11](
                                self.encoder.features[10](self.encoder.features[9](self.encoder.features[8](self.encoder.features[7](self.features[-1]))))))))) # 96
        self.features.append(self.att_2(
            # self.encoder.features[17]( # TODO: rm17
                                self.encoder.features[16](self.encoder.features[15](self.encoder.features[14](self.features[-1]))))) # 320
        # ) 

        return self.features


if __name__ == "__main__":

    # model = MobileNetV3(model_type="small")
    model = MobileNetV2()
    x = torch.randn((1,3,192,640))
    output = model(x)
    print("Finished")