import torch
import time

import numpy as np
import torchvision.transforms as transforms

resize = {}
b = 64
h = 192
w = 640

for i in range(1, 4):
    s = 2 ** i
    resize[i] = transforms.Resize((h// s, w// s))
    
    
def time_test():
    x = torch.randn(b, 3, h, w)
    y = torch.randn(b, 3, h, w)
    inputs_x = {}
    inputs_y = {}

    t1 = time.time()
    x = x.cuda()
    inputs_x[0] = x
    for i in range(1, 4):
        inputs_x[i] = resize[i](inputs_x[i-1])
    t2 = time.time()
    delt_1 = t2 - t1

    t1 = time.time()
    inputs_y[0] = y
    for i in range(1, 4):
        inputs_y[i] = resize[i](inputs_y[i-1])
    for i in range(1, 4):
        inputs_y[i].cuda()
    t2 = time.time()
    delt_2 = t2 - t1

    return delt_1, delt_2

if __name__ == "__main__":
    sum1 = 0
    sum2 = 0
    n = 10
    for i in range(n):
        print(i)
        delt_1, delt_2 = time_test()
        sum1 += delt_1
        sum2 += delt_2
        
    print("delt_1: ", sum1/n)
    print("delt_2: ", sum2/n)
    
    