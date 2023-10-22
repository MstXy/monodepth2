import os

from utils import readlines
from datasets import MotionDataset
from torch.utils.data import DataLoader



data_path = "/mnt/km-nfs/ns100002-share/zcy-exp/kitti_movements"

# file format: folder & frame_id & l

train_filenames = readlines(
    os.path.join("/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2", "splits/motion/train.txt"))

train_dataset = MotionDataset(data_path, train_filenames, height=192, width=640,
                            frame_idxs=[0, -1, 1], num_scales=4, is_train=True, img_ext='.jpg')


dataloader = DataLoader(train_dataset, 16, shuffle=False,
                        num_workers=12, pin_memory=True, drop_last=False)


for inputs in dataloader:
    print(inputs)
    break