from networks.mobilevit.build_mobileViTv3 import MobileViT

import torch
import os

# config_file_name = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/networks/mobilevit/config/classification/mobilevitv3_x_small_oneserver.yaml"
# print(os.path.isfile(config_file_name))

model = MobileViT(pretrained=True)
x = torch.randn((1,3,192,640))
output = model(x)

# 1, 32, 96, 320
# 1, 48, 48, 160
# 1, 96, 24, 80
# 1, 160, 12, 40
# 1, 160, 6, 20 

print("Finished")