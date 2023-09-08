import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import flow_vis
import json
import math
import os
import sys
import pdb
file_dir = os.path.dirname(__file__)
parent_project = os.path.dirname(file_dir)
sys.path.append(parent_project)

# ==== torch import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as tvu
import torchvision.transforms.functional as F

# ==== torch ddp import
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

# ==== monodepth module import
import monodepth2.datasets as datasets
import monodepth2.utils.utils as mono_utils
from monodepth2.UPFlow_pytorch.utils.tools import tools as uptools
import monodepth2.networks as networks
from monodepth2.layers import *

from monodepth2.networks.MonoFlowNet import MonoFlowNet
from monodepth2.networks.UnFlowNet import UnFlowNet
from monodepth2.options import MonodepthOptions
options = MonodepthOptions()
opt = options.parse()

fpath = os.path.join(os.path.dirname(__file__), "splits", opt.split, "{}_files.txt")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
cmap = plt.get_cmap('viridis')
import monodepth2.datasets.flow_eval_datasets as flow_eval_datasets
from monodepth2.utils import frame_utils
from monodepth2.utils.utils import InputPadder, forward_interpolate


IF_DEBUG = False
if IF_DEBUG:
    opt.batch_size = 3
    opt.num_workers = 0
    opt.log_frequency = 10
    # def debughook(etype, value, tb):
    #     pdb.pm()  # post-mortem debugger
    # sys.excepthook = debughook

print(torch.get_num_threads())
torch.set_num_threads(10)
print(torch.get_num_threads())

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class MonoFlowLoss():
    def __init__(self):
        self.opt = opt
        self.num_scales = len(self.opt.scales)
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            
        # frame_id and idx_pair_list
        if len(self.opt.frame_ids) == 3:
            self.idx_pair_list = [(-1, 0), (0, 1)]
        elif len(self.opt.frame_ids) == 2:
            self.idx_pair_list = [(-1, 0)]
        else:
            raise NotImplementedError 

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch, by depth info and flow info both.
        Generated images are saved into the `outputs` dictionary.
        """

        ## ==== image warping using depth and pose
        if self.opt.depth_branch:
            # compute warped images by depth
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = nn.functional.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                outputs[("depth", 0, scale)] = depth

                for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                    if isinstance(frame_id, str): # s_0, s_-1, s_1, ...
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)]

                    # from the authors of https://arxiv.org/abs/1712.00175
                    if self.opt.pose_model_type == "posecnn":

                        axisangle = outputs[("axisangle", 0, frame_id)]
                        translation = outputs[("translation", 0, frame_id)]

                        inv_depth = 1 / depth
                        mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                        T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                    cam_points = self.backproject_depth[source_scale].to(depth.device)(
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale].to(depth.device)(
                        cam_points, inputs[("K", source_scale)], T)

                    outputs[("sample", frame_id, scale)] = pix_coords

                    outputs[("color", frame_id, scale)] = nn.functional.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border",
                        align_corners=True)

                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]

    def _compute_flow_loss_paired(self, img1, img2, flow_1_2, flow_2_1, name="prev_curr"):
        flow_loss = {}
        flow_loss['all_flow_loss'] = 0
        # ===== flow warped paired:
        img1_warped = mono_utils.torch_warp(img2, flow_1_2)
        img2_warped = mono_utils.torch_warp(img1, flow_2_1)

        # ===== occ_1_2, occ_2_1:
        if opt.flow_occ_check:
            occ_1_2, occ_2_1 = mono_utils.cal_occ_map(flow_fwd=flow_1_2, flow_bwd=flow_2_1)
            if opt.stop_occ_gradient:
                occ_1_2, occ_2_1 = occ_1_2.clone().detach(), occ_2_1.clone().detach()
        else:
            occ_1_2 = mono_utils.create_border_mask(flow_1_2)
            # occ_1_2 = torch.zeros_like(flow_1_2[:, 0, :, :].unsqueeze(1))
            occ_2_1 = occ_1_2
        # ===== photo loss calculation:
        photo_loss_l1 = self.photo_loss_multi_type(img1, img1_warped, occ_1_2,
                                                photo_loss_type='L1',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check) + \
                     self.photo_loss_multi_type(img2, img2_warped, occ_2_1,
                                                photo_loss_type='L1',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check)

        photo_loss_ssim = self.photo_loss_multi_type(img1, img1_warped, occ_1_2,
                                                photo_loss_type='SSIM',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check) + \
                     self.photo_loss_multi_type(img2, img2_warped, occ_2_1,
                                                photo_loss_type='SSIM',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check)
        flow_loss["photo_loss_l1"] = photo_loss_l1
        flow_loss["photo_loss_ssim"] = photo_loss_ssim
        # ===== smooth loss calculation:
        smo_loss_o1 = self.edge_aware_smoothness_order1(img=img1, pred=flow_1_2) + \
                   self.edge_aware_smoothness_order1(img=img2, pred=flow_2_1)
        smo_loss_o2 = self.edge_aware_smoothness_order2(img=img1, pred=flow_1_2) + \
                   self.edge_aware_smoothness_order2(img=img2, pred=flow_2_1)
        # ===== census loss calculation:

        # ===== multi scale distillation loss calculation:

        flow_loss["smo_loss_o1"] = smo_loss_o1
        flow_loss["smo_loss_o2"] = smo_loss_o2
        return flow_loss, img1_warped, img2_warped, occ_1_2, occ_2_1

    def compute_losses(self, inputs, outputs):
        self.generate_images_pred(inputs, outputs)
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        losses["flow_loss"] = 0
        total_loss = 0
        if self.opt.optical_flow:
            # compute flow losses, warp img, calculate occ map
            for scale in self.opt.scales:
                tmp_smooth_o1_loss = 0
                tmp_smooth_o2_loss = 0
                tmp_photo_l1_loss = 0
                tmp_photo_ssim_loss = 0
                
                for idx_pair in self.idx_pair_list:
                    # prev(-1) to curr(0)
                    flow_loss, img1_warped, img2_warped, occ_1_2, occ_2_1 = self._compute_flow_loss_paired(
                        img1=inputs[("color_aug", idx_pair[0], scale)],
                        img2=inputs[("color_aug", idx_pair[1], scale)],
                        flow_1_2=outputs[('flow', idx_pair[0], idx_pair[1], scale)],  # flow -1 to 0 (flow_prev_to_curr)
                        flow_2_1=outputs[('flow', idx_pair[1], idx_pair[0], scale)],  # flow 0 to -1 (flow_curr_to_prev)
                        name=idx_pair)
                    outputs[('f_warped', idx_pair[0], idx_pair[1], scale)] = img1_warped  # prev_img warped by curr_img and flow_prev_to_curr
                    outputs[('f_warped', idx_pair[1], idx_pair[0], scale)] = img2_warped   # curr_img warped by prev_img and flow_curr_to_prev
                    outputs[('occ', idx_pair[0], idx_pair[1], scale)] = occ_1_2
                    outputs[('occ', idx_pair[1], idx_pair[0], scale)] = occ_2_1

                    tmp_smooth_o1_loss += flow_loss["smo_loss_o1"]
                    tmp_smooth_o2_loss += flow_loss["smo_loss_o2"]
                    tmp_photo_l1_loss += flow_loss["photo_loss_l1"]
                    tmp_photo_ssim_loss += flow_loss["photo_loss_ssim"]
                losses["smo_loss_o1", scale] = tmp_smooth_o1_loss * opt.loss_smo1_w if scale in [2, 3] else 0 
                losses["smo_loss_o2", scale] = tmp_smooth_o2_loss * opt.loss_smo2_w if scale in [2, 3] else 0 
                losses["photo_loss_l1", scale] = tmp_photo_l1_loss * opt.loss_l1_w * math.exp(-scale)
                losses["photo_loss_ssim", scale] = tmp_photo_ssim_loss * opt.loss_ssim_w * math.exp(-scale)
                losses["flow_loss"] += (losses["smo_loss_o1", scale] + losses["smo_loss_o2", scale] + \
                                        losses["photo_loss_l1", scale] + losses["photo_loss_ssim", scale])

            total_loss += losses["flow_loss"]


        # if self.opt.optical_flow in ['upflow']:
        #     self.flow_warp(inputs, outputs)
        #     total_loss += outputs['loss_by_upflow']
        #     raise NotImplementedError
        
        
        if self.opt.depth_branch:
            for scale in self.opt.scales:
                loss = 0
                reprojection_losses = []

                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    source_scale = 0

                disp = outputs[("disp", scale)]
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                elif self.opt.predictive_mask:
                    # use the predicted mask
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        mask = F.interpolate(
                            mask, [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                    reprojection_losses *= mask

                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape, device=identity_reprojection_loss.device) * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if not self.opt.disable_automasking:
                    outputs["identity_selection/{}".format(scale)] = (
                            idxs > identity_reprojection_loss.shape[1] - 1).float()

                loss += to_optimise.mean()

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)

                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                total_loss += loss
                losses["loss/scale_{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def photo_loss_multi_type(self, x, y, occ_mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                              photo_loss_delta=0.4, photo_loss_use_occ=False,
                              ):
        occ_weight = occ_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(photo_loss_delta)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)
        elif photo_loss_type == 'SSIM':
            loss_diff, occ_weight = self.weighted_ssim(x, y, occ_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)
        if photo_loss_use_occ:
            tmp = torch.sum(loss_diff[0], dim=0).unsqueeze(0) * occ_weight
            # mono_utils.stitching_and_show([loss_diff[0], occ_weight[0], tmp[0], torch.sum(loss_diff[0], dim=0).unsqueeze(0)], show=True)
            photo_loss = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
        else:
            photo_loss = torch.mean(loss_diff)
        return photo_loss

    def weighted_ssim(self, x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
        """Computes a weighted structured image similarity measure.
        Args:
          x: a batch of images, of shape [B, C, H, W].
          y:  a batch of images, of shape [B, C, H, W].
          weight: shape [B, 1, H, W], representing the weight of each
            pixel in both images when we come to calculate moments (means and
            correlations). values are in [0,1]
          c1: A floating point number, regularizes division by zero of the means.
          c2: A floating point number, regularizes division by zero of the second
            moments.
          weight_epsilon: A floating point number, used to regularize division by the
            weight.

        Returns:
          A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
          similarity loss per pixel per channel, and the second, of shape
          [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
          know how much to weigh each pixel in the first tensor. For example, if
          `'weight` was very small in some area of the images, the first tensor will
          still assign a loss to these pixels, but we shouldn't take the result too
          seriously.
        """

        def _avg_pool3x3(x):
            # tf kernel [b,h,w,c]
            return nn.functional.avg_pool2d(x, (3, 3), (1, 1))
            # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

        if c1 == float('inf') and c2 == float('inf'):
            raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                             'likely unintended.')
        average_pooled_weight = _avg_pool3x3(weight)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

        def weighted_avg_pool3x3(z):
            wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
            return wighted_avg * inverse_average_pooled_weight

        mu_x = weighted_avg_pool3x3(x)
        mu_y = weighted_avg_pool3x3(y)
        sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
        sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
        sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x ** 2 + mu_y ** 2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        result = ssim_n / ssim_d
        return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight

    def edge_aware_smoothness_order2(self, img, pred):
        def gradient_x(img, stride=1):
            gx = img[:, :, :-stride, :] - img[:, :, stride:, :]
            return gx

        def gradient_y(img, stride=1):
            gy = img[:, :, :, :-stride] - img[:, :, :, stride:]
            return gy

        pred_gradients_x = gradient_x(pred)
        pred_gradients_xx = gradient_x(pred_gradients_x)
        pred_gradients_y = gradient_y(pred)
        pred_gradients_yy = gradient_y(pred_gradients_y)
        image_gradients_x = gradient_x(img, stride=2)
        image_gradients_y = gradient_y(img, stride=2)
        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
        smoothness_x = torch.abs(pred_gradients_xx) * weights_x
        smoothness_y = torch.abs(pred_gradients_yy) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    def edge_aware_smoothness_order1(self, img, pred):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx
        def gradient_y(img):
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)
        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)
        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    def l1_plus_ssmi(self, img1, img1_warped, occ_1_2, alpha=0.85):
        l1_sum = torch.abs(img1_warped - img1 + 1e-6)
        l1_loss = torch.sum(l1_sum * occ_1_2) / (torch.sum(occ_1_2) + 1e-6)
        self.ssim.to(img1.device)
        ssim_sum = self.ssim(img1_warped, img1)
        ssim_loss = torch.sum(ssim_sum * occ_1_2) / (torch.sum(ssim_sum) + 1e-6)
        return alpha * ssim_loss + (1-alpha) * l1_loss


def load_train_objs():
    # ====== 1.train dataset: kitti flow and depth
    if opt.train_dataset in ['kitti']:
        train_filenames = mono_utils.readlines(fpath.format("train"))
        train_dataset = datasets.KITTIRAWDataset(
            opt.data_path, train_filenames, opt.height, opt.width,
            opt.frame_ids, 4, is_train=True, img_ext='.png', color_only=(not opt.depth_branch and opt.optical_flow))
    
    elif opt.train_dataset in ['kitti_odom']:
        train_filenames = mono_utils.readlines(fpath.format("train"))
        train_dataset = datasets.KITTIOdomDataset(
            opt.data_path, train_filenames, opt.height, opt.width,
            opt.frame_ids, 4, is_train=True, img_ext='.png', color_only=(not opt.depth_branch and opt.optical_flow))
        
    elif opt.train_dataset in ['kitti_mv2015']:
        train_dataset = datasets.KITTI_MV_2015()
    
    elif opt.train_dataset in ['FlyingChairs']:
        train_dataset = flow_eval_datasets.FlyingChairs(root=opt.data_path_FlyingChairs, aug_params=None, split='training')
        # opt.height = 384
        # opt.width = 512
        opt.frame_ids=[-1, 0]
    
    # ====== 2.train dataset: kitti

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
        sampler=(torch.utils.data.distributed.DistributedSampler(dataset) if opt.ddp else None),
        drop_last=True
    )

    # ====== 2.eval dataset: depth

    # ====== 3.eval dataset: flow

    return train_dataset, train_dataloader


def load_val_objs():
    if opt.val_dataset in ['kitti']:
        val_filenames = mono_utils.readlines(fpath.format("val"))
        val_dataset = datasets.KITTIRAWDataset(
            opt.data_path, val_filenames, opt.height, opt.width,
            opt.frame_ids, 4, is_train=False, img_ext='.png')
        val_loader = DataLoader(
            val_dataset, opt.batch_size, True,
            num_workers=opt.num_workers, pin_memory=False, drop_last=True)
    return val_dataset, val_loader


def prepare_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
        sampler=(torch.utils.data.distributed.DistributedSampler(dataset) if opt.ddp else None),
        drop_last=True
    )



class DDP_Trainer():
    def __init__(self, gpu_id, train_dataset, train_loader, val_dataset, val_loader):
        self.opt = opt
        self.gpu_id = gpu_id
        self.is_master_node = (self.gpu_id == str(os.environ["CUDA_VISIBLE_DEVICES"][0])) if opt.ddp else True

        self.train_loader = train_loader
        ############### val dataset ###############
        
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.val_iter = iter(self.val_loader)

        #################### input augmentation ####################
        self.num_scales = len(self.opt.scales)
        # frame_id and idx_pair_list
        if len(self.opt.frame_ids) == 3:
            self.idx_pair_list = [(-1, 0), (0, 1)]
        elif len(self.opt.frame_ids) == 2:
            self.idx_pair_list = [(-1, 0)]
        else:
            raise NotImplementedError 
        
        self.to_tensor = transforms.ToTensor()
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.opt.height // s, self.opt.width // s))
                                               # interpolation=Image.ANTIALIAS)
        self.norm_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)


        #################### model, optim, loss, loding and saving ####################
        if self.opt.model_name == "MonoFlowNet":
            self.model = MonoFlowNet(opt)
        elif self.opt.model_name == "UnFlowNet":
            self.model = UnFlowNet().to('cuda:' + str(self.gpu_id))
        else:
            raise NotImplementedError
        
        if self.opt.load_weights_folder is not None:
            self.load_ddp_model()
        if opt.ddp:
            self.ddp_model = DDP(self.model.to(self.gpu_id), device_ids=[self.gpu_id], find_unused_parameters=True)
        else:
            self.ddp_model = self.model.to('cuda:' + str(self.gpu_id))

        if opt.freeze_Resnet:
            paras = []
            for name, param in self.ddp_model.named_parameters():
                if 'ResEncoder' in name:
                    param.requires_grad = False
                elif 'FlowDecoder' in name:
                    paras.append(param)
                    param.requires_grad = True
                else:
                    raise NotImplementedError
            self.model_optimizer = optim.Adam(paras, self.opt.learning_rate)
        else:
            self.model_optimizer = optim.Adam(self.ddp_model.parameters(), self.opt.learning_rate)

        # self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.4)
        self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer,
                                                                       T_max=self.opt.num_epochs/2,
                                                                       verbose=self.is_master_node)

        self.mono_loss = MonoFlowLoss()
        self.writers = {}
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        if self.is_master_node:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        self.step = 0
        self.epoch = 0

        if self.is_master_node:
            self.save_opts()
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using: device cuda:", self.gpu_id)
        
    def load_ddp_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))
        chechpoint_path = os.path.join(self.opt.load_weights_folder, "monoFlow.pth")
        self.model.load_state_dict(
            torch.load(chechpoint_path, map_location='cpu'))

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def save_ddp_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        save_path = os.path.join(save_folder, "monoFlow.pth")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        torch.save(self.ddp_model.state_dict(), save_path)
        optim_save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_save_path)

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        loss = losses["loss"].cpu().data
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        curr_lr = self.model_optimizer.param_groups[0]['lr']
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {} | learning_rate: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  mono_utils.sec_to_hm_str(time_sofar), mono_utils.sec_to_hm_str(training_time_left),
                                  curr_lr))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if isinstance(l, tuple) and l[0] in ['smo_loss_o1', 'smo_loss_o2', 'photo_loss_l1', 'photo_loss_ssim']:
                continue
            else:
                writer.add_scalar("{}".format(l), v, self.step)
        # writer.add_scalar("flow_mean", torch.mean(
        #     torch.abs(outputs['flow', self.idx_pair_list[0][0], self.idx_pair_list[0][1], 0][0])
        #     ), self.step)

        for scale in opt.scales:
            writer.add_scalar("smo_loss_o1/scale{}".format(scale), losses["smo_loss_o1", scale], self.step)
            writer.add_scalar("smo_loss_o2/scale{}".format(scale), losses["smo_loss_o2", scale], self.step)
            writer.add_scalar("photo_loss_l1/scale{}".format(scale), losses["photo_loss_l1", scale], self.step)
            writer.add_scalar("photo_loss_ssim/scale{}".format(scale), losses["photo_loss_ssim", scale], self.step)


        for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
            if self.opt.optical_flow is not None:
                for idx_pair in self.idx_pair_list:
                    # top to bottom: 1.curr 2.occ_prev_curr 3.flow_prev_curr 4.prev
                    img1_img2_flow = mono_utils.log_vis_1(inputs, outputs, idx_pair[0], idx_pair[1], j)
                    # top to bottom:  diff(target, source), diff * mask, warped, source, flow_img1_img2
                    img1_warped_img1_diff = mono_utils.log_vis_2(inputs, outputs, idx_pair[0], idx_pair[1], j)
                    writer.add_image('1.Img2_||_2.OccImg1->Img2_||_3.FlowImg1->Img2_||_4.Img1._||_IdxPair{},{}/{}'.
                                     format(idx_pair[0], idx_pair[1], j),img1_img2_flow, self.step)
                    writer.add_image('1.Diff_||_2.DiffMasked_||_3.WarpedImg1_||_4.SourceImg1_||_5.FlowImg1->Img2_||_IdxPair{},{}/{}'.
                                     format(idx_pair[0], idx_pair[1], j), img1_warped_img1_diff, self.step)

            if self.opt.depth_branch:
                for s in self.opt.scales:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        mono_utils.normalize_image(outputs[("disp", s)][j]), self.step)

                    if self.opt.predictive_mask:
                        for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                            writer.add_image(
                                "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                                outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                                self.step)

                    elif not self.opt.disable_automasking:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def eval_depth_flow(self):
        """Flow and Depth Evaluation on a single minibatch
        Flow evaluation dataset:

        Depth evaluation dataset:
            monodepth2/splits/eigen_zhou/val_files.txt
        """
        if self.opt.depth_branch:
            try:
                inputs = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                inputs = next(self.val_iter)

            with torch.no_grad():
                outputs, losses = self._run_batch(inputs, batch_idx=1)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("val", inputs, outputs, losses)
                del inputs, outputs, losses

        if self.opt.optical_flow:
            self.ddp_model.eval()
            """ Peform validation using the KITTI-2015 (train) split """
            val_dataset = flow_eval_datasets.KITTI_2015_scene_flow(split='training', root=self.opt.val_data_root)
            out_list, epe_list = [], []
            save_path_dir = os.path.join(self.log_path, 'evaluate_flow_kitti')
            os.makedirs(save_path_dir, exist_ok=True)
            for val_id in range(len(val_dataset)):
                input_dict = val_dataset[val_id]
                # {('color', -1, 0): img1, ('color', 0, 0): img2, ('flow', -1, 0): flow, ('valid', -1, 0): valid}
                image1_ori, image2_ori, flow_gt, valid_gt = input_dict[('color', -1, 0)], input_dict[('color', 0, 0)], \
                                                            input_dict[('flow', -1, 0)], input_dict[('valid', -1, 0)]   
                
                image1_ori = image1_ori[None].to(self.gpu_id)
                image2_ori = image2_ori[None].to(self.gpu_id)
                padder = InputPadder(image1_ori.shape, mode='kitti', divided_by=64)
                image1, image2 = padder.pad(image1_ori, image2_ori)
                with torch.no_grad():
                    input_dict = {}
                    # image1 = F.resize(image1, size=[192, 640], antialias=False)
                    # image2 = F.resize(image2, size=[192, 640], antialias=False)
                    input_dict[("color_aug", -1, 0)], input_dict[("color_aug", 0, 0)], input_dict[("color_aug", 1, 0)] = \
                    image1, image2, image1
                    out_dict = self.ddp_model(input_dict)
                    flow_1_2 = out_dict['flow', -1, 0, 0][0]
                flow = padder.unpad(flow_1_2).cpu()
                epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
                mag = torch.sum(flow_gt**2, dim=0).sqrt()

                # vis
                err_map = torch.sum(torch.abs(flow - flow_gt) * valid_gt, dim=0)
                err_map_norm = colors.Normalize(vmin=0, vmax=torch.max(err_map))
                err_map_colored_tensor = mono_utils.plt_color_map_to_tensor(cmap(err_map_norm(err_map)))
                to_save = mono_utils.stitching_and_show(img_list=[image1[0], flow, flow_gt, err_map_colored_tensor, image2[0]],
                                                        ver=True, show=False)
                save_path = os.path.join(save_path_dir, str(self.epoch) + "th_epoch_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".png")
                to_save.save(save_path)
                epe = epe.view(-1)
                mag = mag.view(-1)
                val = valid_gt.view(-1) >= 0.5
                out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
                epe_list.append(epe[val].mean().item())
                out_list.append(out[val].cpu().numpy())

            epe_list = np.array(epe_list)
            out_list = np.concatenate(out_list)
            epe = np.mean(epe_list)
            f1 = 100 * np.mean(out_list)
            print("\n Validation KITTI \n: %f, %f" % (epe, f1))
            writer = self.writers['val']
            writer.add_scalar('kitti_epe', epe, self.step)
            writer.add_scalar('kitti_f1', f1, self.step)

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def preprocess(self, inputs):
        """Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        todo:
        https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image
        """
        # color_aug = transforms.ColorJitter(
        #     self.brightness, self.contrast, self.saturation, self.hue)

        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                if opt.height != inputs[k].shape[2] or opt.width != inputs[k].shape[3]:
                    inputs[k] = self.resize[0](inputs[k])
                    
                for i in range(1, self.num_scales):
                    inputs[(n, im, i)] =self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # inputs[(n, im, i)] = (self.to_tensor(f))
                inputs[(n + "_aug", im, i)] = f
        
        # visualize
        # for j in range(inputs['color_aug', -1, 0].size()[0]):
        #     mono_utils.stitching_and_show(img_list=[
        #         inputs['color_aug', -1, 0][j],
        #         inputs['color_aug', 0, 0][j],
        #         inputs['color_aug', 1, 0][j],
        #     ], ver=True, show=True)

    def _run_batch(self, inputs, batch_idx):
        self.model_optimizer.zero_grad()
        start_batch_time = time.time()
        self.preprocess(inputs)
        outputs = self.ddp_model(inputs)
        losses = self.mono_loss.compute_losses(inputs, outputs)
        losses['loss'].backward()
        self.model_optimizer.step()

        # ===== log
        early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000 and not self.opt.debug and self.is_master_node
        late_phase = self.step % 2000 == 0 and not self.opt.debug and self.is_master_node
        if early_phase or late_phase:
            self.log_time(batch_idx, time.time() - start_batch_time, losses)
            self.log("train", inputs, outputs, losses)
        self.step += 1
        return outputs, losses

    def _run_epoch(self):
        for batch_idx, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.gpu_id)
            self._run_batch(inputs=inputs, batch_idx=batch_idx)

    def train(self, ):
        self.start_time = time.time()
        for epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.epoch = epoch
            if self.epoch > self.opt.occ_start_epoch:
                opt.flow_occ_check = True
            self._run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.is_master_node:
                self.save_ddp_model()
                self.eval_depth_flow()


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    dataset = load_train_objs()
    train_loader = prepare_dataloader(dataset)
    trainer = DDP_Trainer(gpu_id=rank, train_loader=train_loader)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    if opt.ddp:
        # world_size = torch.cuda.device_count()
        mp.spawn(main, args=(opt.world_size,), nprocs=opt.world_size, join=True)
    else:
        train_dataset, train_loader = load_train_objs()
        val_dataset, val_loader = load_val_objs()
        trainer = DDP_Trainer(int(opt.device[-1]), train_dataset, train_loader,
                              val_dataset, val_loader)
        trainer.train()


