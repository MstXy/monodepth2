import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import flow_vis
import json
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
from monodepth2.options import MonodepthOptions
from monodepth2.networks.MonoFlowNet import MonoFlowNet
from monodepth2.networks.UnFlowNet import UnFlowNet
options = MonodepthOptions()
opt = options.parse()
fpath = os.path.join(os.path.dirname(__file__), "splits", opt.split, "{}_files.txt")

IF_DEBUG = False
if IF_DEBUG:
    opt.batch_size = 3
    opt.num_workers = 0
    opt.log_frequency = 10
    # def debughook(etype, value, tb):
    #     pdb.pm()  # post-mortem debugger
    # sys.excepthook = debughook


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
                                                photo_loss_type='abs_robust',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check) + \
                     self.photo_loss_multi_type(img2, img2_warped, occ_2_1,
                                                photo_loss_type='abs_robust',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check)

        photo_loss_ssim = self.photo_loss_multi_type(img1, img1_warped, occ_1_2,
                                                photo_loss_type='SSIM',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check) + \
                     self.photo_loss_multi_type(img2, img2_warped, occ_2_1,
                                                photo_loss_type='SSIM',  # abs_robust, charbonnier, L1, SSIM
                                                photo_loss_delta=0.4, photo_loss_use_occ=opt.flow_occ_check)
        flow_loss["photo_loss_l1"] = photo_loss_l1 * 0.15
        flow_loss["photo_loss_ssim"] = photo_loss_ssim * 0.85
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
                # prev(-1) to curr(0)
                flow_loss_1, img1_warped, img2_warped, occ_1_2, occ_2_1 = self._compute_flow_loss_paired(
                    img1=inputs[("color_aug", -1, scale)],
                    img2=inputs[("color_aug", 0, scale)],
                    flow_1_2=outputs[('flow', -1, 0, scale)],  # flow -1 to 0 (flow_prev_to_curr)
                    flow_2_1=outputs[('flow', 0, -1, scale)],  # flow 0 to -1 (flow_curr_to_prev)
                    name="prev_curr")
                outputs[('f_warped', -1, 0, scale)] = img1_warped  # prev_img warped by curr_img and flow_prev_to_curr
                outputs[('f_warped', 0, -1, scale)] = img2_warped   # curr_img warped by prev_img and flow_curr_to_prev
                outputs[('occ', -1, 0, scale)] = occ_1_2
                outputs[('occ', 0, -1, scale)] = occ_2_1

                # curr(0) to next(1)
                flow_loss_2, img1_warped, img2_warped, occ_1_2, occ_2_1 = self._compute_flow_loss_paired(
                    img1=inputs["color_aug", 0, scale],
                    img2=inputs["color_aug", 1, scale],
                    flow_1_2=outputs[('flow', 0, 1, scale)],  # flow 0 to 1 (flow_curr_to_next)
                    flow_2_1=outputs[('flow', 1, 0, scale)],  # flow 1 to 0 (flow_next_to_curr)
                    name="curr_next")
                outputs[('f_warped', 0, 1, scale)] = img1_warped  # curr_img warped by next_img and flow_curr_to_next
                outputs[('f_warped', 1, 0, scale)] = img2_warped   # next_img warped by curr_img and flow_next_to_curr
                outputs[('occ', 0, 1, scale)] = occ_1_2
                outputs[('occ', 1, 0, scale)] = occ_2_1

                losses["smo_loss_o1", scale] = flow_loss_1["smo_loss_o1"] + flow_loss_2["smo_loss_o1"]
                losses["smo_loss_o2", scale] = flow_loss_1["smo_loss_o2"] + flow_loss_2["smo_loss_o2"]
                losses["photo_loss_l1", scale] = flow_loss_1["photo_loss_l1"] + flow_loss_2["photo_loss_l1"]
                losses["photo_loss_ssim", scale] = flow_loss_1["photo_loss_ssim"] + flow_loss_2["photo_loss_ssim"]
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

    def depth_evaluation(self, inputs, outputs, losses):
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


class DDP_Trainer():
    def __init__(self, gpu_id, train_loader):
        self.opt = opt
        self.gpu_id = gpu_id
        self.is_master_node = (self.gpu_id == str(os.environ["CUDA_VISIBLE_DEVICES"][0])) if opt.ddp else True

        self.train_loader = train_loader
        ############### val dataset ###############
        
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[opt.dataset]
        self.train_filenames = mono_utils.readlines(fpath.format("train"))
        img_ext = '.png'
        num_train_samples = len(self.train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.val_filenames = mono_utils.readlines(fpath.format("val"))
        self.val_dataset = self.dataset(
            self.opt.data_path, self.val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
        self.val_iter = iter(self.val_loader)

        #################### input augmentation ####################
        self.num_scales = len(self.opt.scales)
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
            self.model = UnFlowNet()

        self.model_optimizer = optim.Adam(self.model.parameters(), self.opt.learning_rate)
        if self.opt.load_weights_folder is not None:
            self.load_ddp_model()
        if opt.ddp:
            self.ddp_model = DDP(self.model.to(self.gpu_id), device_ids=[self.gpu_id], find_unused_parameters=True)
        else:
            self.ddp_model = self.model.to('cuda:' + str(self.gpu_id))

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
        writer.add_scalar("flow_mean", torch.mean(torch.abs(outputs['flow', -1, 0, 0][0])), self.step)

        for scale in opt.scales:
            writer.add_scalar("smo_loss_o1/scale{}".format(scale), losses["smo_loss_o1", scale], self.step)
            writer.add_scalar("smo_loss_o2/scale{}".format(scale), losses["smo_loss_o2", scale], self.step)
            writer.add_scalar("photo_loss_l1/scale{}".format(scale), losses["photo_loss_l1", scale], self.step)
            writer.add_scalar("photo_loss_ssim/scale{}".format(scale), losses["photo_loss_ssim", scale], self.step)


        for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
            if self.opt.optical_flow is not None:
                # top to bottom: 1.curr 2.occ_prev_curr 3.flow_prev_curr 4.prev
                prev_curr_and_flow = mono_utils.log_vis_1(inputs, outputs, -1, 0, j)
                curr_next_and_flow = mono_utils.log_vis_1(inputs, outputs, 0, 1, j)

                # top to bottom:  diff(target, source), diff * mask, warped, source, flow_img1_img2
                prev_warped_prev_and_diff = \
                    mono_utils.log_vis_2(inputs, outputs, -1, 0, j)
                curr_warped_curr_and_diff = \
                    mono_utils.log_vis_2(inputs, outputs, 0, 1, j)

                writer.add_image('1.Curr_2.OccPrevCurr_3.FlowPrevCurr_4.Prev/{}'.format(j),
                                 prev_curr_and_flow, self.step)
                writer.add_image('1.Next_2.OccCurrNext_3.FlowCurrNext_4.Curr/{}'.format(j),
                                 curr_next_and_flow, self.step)
                writer.add_image('1.Diff_2.DiffMasked_3.WarpedPrev_4.Prev_5.FlowPrevCurr/{}'.format(j),
                                 prev_warped_prev_and_diff, self.step)
                writer.add_image('1.Diff_2.DiffMasked_3.WarpedCurr_4.Curr_5.FlowCurrNext/{}'.format(j),
                                 curr_warped_curr_and_diff, self.step)

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

    def eval(self):
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
                outputs, losses = self.process_batch(inputs)

                if "depth_gt" in inputs and self.opt.depth_branch:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("val", inputs, outputs, losses)
                del inputs, outputs, losses

        if self.opt.optical_flow:
            from monodepth2.evaluation.evaluate_flow import evaluate_flow_online
            with torch.no_grad():
                eval_flow_result = evaluate_flow_online(
                    self.log_path,
                    checkpoint_path=os.path.join(self.log_path, "models", "weights_{}".format(self.epoch), "monoFlow.pth"),
                    model_name=self.opt.model_name,
                    epoch_idx=self.epoch,
                    opt_main=self.opt)
                writer = self.writers['val']
                writer.add_scalar('kitti_epe', eval_flow_result['kitti_epe'], self.step)
                writer.add_scalar('kitti_f1', eval_flow_result['kitti_f1'], self.step)

    def preprocess(self, inputs):
        """Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        todo:
        https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image
        """
        color_aug = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, _ = k
                for i in range(1, self.num_scales):
                    if opt.norm_trans:
                        inputs[(n, im, i)] = self.norm_trans(
                            self.resize[i](inputs[(n, im, i - 1)])
                        )
                    else:
                        inputs[(n, im, i)] =self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # inputs[(n, im, i)] = (self.to_tensor(f))
                inputs[(n + "_aug", im, i)] = f

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
            if "depth_gt" in inputs and self.opt.depth_branch:
                self.mono_loss.depth_evaluation(inputs, outputs, losses)
            self.log("train", inputs, outputs, losses)
        self.step += 1

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
                self.eval()


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


def load_train_objs():
    # ====== 1.train dataset: flow and depth
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                     "kitti_odom": datasets.KITTIOdomDataset}
    dataset = datasets_dict[opt.dataset]
    train_filenames = mono_utils.readlines(fpath.format("train"))
    train_dataset = dataset(
        opt.data_path, train_filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=True, img_ext='.png')

    # ====== 2.eval dataset: depth

    # ====== 3.eval dataset: flow

    return train_dataset


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


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    dataset = load_train_objs()
    train_loader = prepare_dataloader(dataset)
    trainer = DDP_Trainer(gpu_id=rank, train_loader=train_loader)
    trainer.train()
    destroy_process_group()


#############################         test          ###############################
def model_test():
    opt.depth_branch = True
    opt.optical_flow = 'flownet'
    opt.batch_size = 10

    model = MonoFlowNet().to(opt.device)
    inputs = {}
    for i in opt.frame_ids:
        inputs[("color_aug", i, 0)] = torch.randn(opt.batch_size, 3, 192, 640).to(opt.device)

    ## params and flops evaluation
    from thop import profile
    flops, params = profile(model, inputs=(inputs,))
    print(flops / 1e9, 'GFLOP', params / 1e6, 'M parameters')
    t1 = time.time()
    outputs = model(inputs)
    print('time spent:', time.time() - t1)
    for k, v in outputs.items():
        print(k, v.shape)

def use_single_gpu(gpu, tmpdataset):
    t1 = time.time()
    mono_loss = MonoFlowLoss()
    model = MonoFlowNet().to(gpu)
    model_optimizer = optim.Adam(model.parameters(), 0.001)
    tmp_loader = DataLoader(
        tmpdataset, opt.batch_size, True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    t2 = 0

    def preprocess(self, inputs):

        """Resize colour images to the required scales and augment if required

                We create the color_aug object in advance and apply the same augmentation to all
                images in this item. This ensures that all images input to the pose network receive the
                same augmentation.
                """
        color_aug = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, _ = k
                for i in range(1, self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # inputs[(n, im, i)] = (self.to_tensor(f))
                inputs[(n + "_aug", im, i)] = color_aug(f)


    for idx, inputs in enumerate(tqdm(tmp_loader)):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(gpu)
        preprocess(inputs)
        outputs = model(inputs)
        losses = mono_loss.compute_losses(inputs, outputs)
        losses['loss'].backward()
        if idx % 30 == 0:
            print('using ddp, idx={}, loss={}, time spent={}'.format(idx * opt.batch_size, losses["loss"].cpu().data, time.time()-t2))
            t2 = time.time()
        model_optimizer.step()
    print(time.time() - t1)


if __name__ == "__main__":
    if opt.ddp:
        # world_size = torch.cuda.device_count()
        mp.spawn(main, args=(opt.world_size,), nprocs=opt.world_size, join=True)
    else:
        dataset = load_train_objs()
        train_loader = prepare_dataloader(dataset)
        trainer = DDP_Trainer(gpu_id=int(opt.device[-1]), train_loader=train_loader)
        trainer.train()




















































