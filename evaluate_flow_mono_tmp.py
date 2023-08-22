import os.path

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils.utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


from PIL import Image
import flow_vis
import torchvision
from datetime import datetime
import torchvision.transforms as transforms

corr_feature_level = 2
import os
from monodepth2.UPFlow_pytorch.utils.tools import tools
import cv2
import numpy as np
from copy import deepcopy
import torch
import warnings  # ignore warnings
import torch.nn.functional as F
import torch.optim as optim
from monodepth2.UPFlow_pytorch.dataset.kitti_dataset import kitti_train, kitti_flow
from monodepth2.UPFlow_pytorch.model.upflow import UPFlow_net
from torch.utils.data import DataLoader
import time
if_cuda = True

from options import MonodepthOptions
options = MonodepthOptions()
opts = options.parse()

opts.load_weights_folder = '/home/liu/Downloads/weights_59'
opts.models_to_load = ["encoder", "pose", "corr", "flow"]
opts.optical_flow = 'flownet'
opts.log_dir = '/home/liu/Downloads/weights_59'

from trainer import Trainer
trainer = Trainer(opts)
save_path = os.path.join(opts.log_dir, 'flow_val')
os.makedirs(save_path, exist_ok=True)
t1 = time.time()
from datasets.flow_eval_datasets import KITTI as KITTI_flow_2015_dataset

trans = torchvision.transforms.Resize((opts.height, opts.width))
trans2 = torchvision.transforms.Resize((375, 1242))
val_dataset = KITTI_flow_2015_dataset(split='training', root=opts.val_data_root)
out_list, epe_list = [], []

def pred_gt_error_vis(flow, flow_gt, valid_gt):
    ## flow vis for debug
    if val_id % 200 == 0:
        out_flow = flow_vis.flow_to_color(flow.permute(1, 2, 0).clone().detach().numpy(), convert_to_bgr=False)
        gt_flow = flow_vis.flow_to_color(flow_gt.permute(1, 2, 0).clone().detach().numpy(), convert_to_bgr=False)
        gt_flow = Image.fromarray(gt_flow)
        out_flow = Image.fromarray(out_flow)
        result = trainer.merge_images(gt_flow, out_flow)
        path = os.path.join(save_path
                            , datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg')
        result.save(path)


class Test_model(tools.abs_test_model):
    def __init__(self, pretrain_path='./scripts/upflow_kitti2015.pth'):
        super(Test_model, self).__init__()
        param_dict = {
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'if_froze_pwc': False,
            'if_use_cor_pytorch': True,  # speed is very slow, just for debug when cuda correlation is not compiled
            'if_sgu_upsample': True,

        }
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net = net_conf()  # .to(device)
        net.load_model(pretrain_path, if_relax=True, if_print=True)
        if if_cuda:
            net = net.to(opts.device)
        net.eval()
        self.net_work = net

    def eval_forward(self, im1, im2, gt, *args):
        # === network output
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False, 'if_shared_features': False }
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = flow_fw
        return pred_flow

    def eval_forward2(self, im1, im2):
        # === network output
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False, 'if_shared_features': False }
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = flow_fw
        return pred_flow

    def eval_save_result(self, save_name, predflow, *args, **kwargs):
        # you can save flow results here
        print(save_name)


def kitti_2015_test():
    pretrain_path = '/home/liu/data16t/Projects/test0618_basedON0616/monodepth2/UPFlow_pytorch/scripts/upflow_kitti2015.pth'
    testmodel = Test_model(pretrain_path=pretrain_path)
    # note that eval batch size should be 1 for KITTI 2012 and KITTI 2015 (image size may be different for different sequence)
    bench = kitti_flow.Evaluation_bench(name='2015_train', if_gpu=if_cuda, batch_size=1)
    epe_all, f1, epe_noc, epe_occ = bench(testmodel)
    print('EPE All = %.2f, F1 = %.2f, EPE Noc = %.2f, EPE Occ = %.2f' % (epe_all, f1, epe_noc, epe_occ))

def flow_pred_using_flownet(image1, image2, flow_gt, valid_gt):
    color_aug = transforms.ColorJitter(
        trainer.brightness, trainer.contrast, trainer.saturation, trainer.hue)

    image1 = image1[None].to(trainer.device)
    image2 = image2[None].to(trainer.device)
    padder = InputPadder(image1.shape, mode='kitti')
    image1, image2 = padder.pad(image1, image2)

    # image1 = color_aug(image1)
    # image2 = color_aug(image2)

    image1 = trans(image1)
    image2 = trans(image2)
    flow_gt = trans(flow_gt)
    valid_gt = trans(valid_gt.unsqueeze(0))
    valid_gt = valid_gt.squeeze()

    # flow forward propagation
    all_color_aug = torch.cat((image1, image2, image1), )  # all images: ([i-1, i, i+1] * [L, R])
    all_features = trainer.models["encoder"](all_color_aug)
    all_features = [torch.split(f, 1) for f in all_features]  # separate by frame
    features = {}
    for i, k in enumerate(opts.frame_ids):
        features[k] = [f[i] for f in all_features]
    if opts.optical_flow in ["flownet", ]:
        outdict = trainer.predict_flow(features)
    if opts.optical_flow in ["upflow", ]:
        imputs_dic = {
            ("color_aug", 0, 0): image1,
            ("color_aug", 1, 0): image2,
        }
        outdict = trainer.predict_upflow(features, imputs_dic)
    out_flow = outdict['flow'][0].squeeze()
    flow = out_flow.detach().cpu()
    return flow, flow_gt, valid_gt


def flow_pred_using_upflow(image1, image2):
    pretrain_path = '/home/liu/data16t/Projects/test0618_basedON0616/monodepth2/UPFlow_pytorch/scripts/upflow_kitti2015.pth'
    test_model = Test_model(pretrain_path=pretrain_path)
    def get_process_img_only_img(img):
        mean = [104.920005, 110.1753, 114.785955]
        stddev = 1 / 0.0039216
        img = (img - mean) / stddev
        img = np.transpose(img, [2, 0, 1])
        return img

    image1 = get_process_img_only_img(image1)
    image2 = get_process_img_only_img(image2)

    image1 = image1.unsqueeze(dim=0).to(opts.device)
    image2 = image2.unsqueeze(dim=0).to(opts.device)
    predflow = test_model.eval_forward2(image1, image2)
    flow = predflow.squeeze(dim=0).detach().cpu()
    # vis
    flow_toshow = flow_vis.flow_to_color(flow.permute(1, 2, 0).clone().detach().numpy(), convert_to_bgr=False)
    flow_toshow = Image.fromarray(flow_toshow)
    flow_toshow.show()
    return flow


if __name__ == "__main__":
    epe_list = []
    out_list = []



    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        flow, flow_gt, valid_gt = flow_pred_using_flownet(image1, image2, flow_gt, valid_gt)
        # flow = flow_pred_using_upflow(image1, image2)
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        tmp = torch.sum(epe * valid_gt) / (torch.sum(valid_gt) + 1e-6)
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        val_num = torch.sum((1 * val))

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out[val].cpu().numpy()
        out_list.append(out[val].cpu().numpy())
        print(epe[val].mean().item(), out[val].mean().item())


    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    t2 = time.time()
    print("Validation KITTI:  epe: %f,   f1: %f, time_spent: %f" % (epe, f1, t2 - t1))
    print("KITTI_epe", epe)
    print("KITTI_f1", f1)


