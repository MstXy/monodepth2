import os
import random
import math

all_filepath = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/splits/motion/all.txt" 

train_filepath = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/splits/motion/train.txt"
val_filepath = "/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/splits/motion/val.txt"


all = open(all_filepath, 'r').readlines()

# random shuffle
random.seed(42)
random.shuffle(all)

split_r = 0.8
total = len(all)
train = all[:math.floor(split_r*total)]
val = all[math.floor(split_r*total):]

with open(train_filepath, 'w') as train_file:
    train_file.writelines(train)
with open(val_filepath, 'w') as val_file:
    val_file.writelines(val)