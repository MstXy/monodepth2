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
file_dir = os.path.dirname(__file__)
parent_project = os.path.dirname(os.path.dirname(file_dir))
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

# monodepth module import
import monodepth2.datasets as datasets
import monodepth2.utils.utils as mono_utils
from monodepth2.UPFlow_pytorch.utils.tools import tools as uptools
import monodepth2.networks as networks
from monodepth2.layers import *
from monodepth2.options import MonodepthOptions
options = MonodepthOptions()
opt = options.parse()



import pdb
import sys


fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", opt.split, "{}_files.txt")


IF_DEBUG = False
if IF_DEBUG:
    opt.batch_size = 3
    opt.num_workers = 0
    opt.log_frequency = 10
    # def debughook(etype, value, tb):
    #     pdb.pm()  # post-mortem debugger
    # sys.excepthook = debughook


class MonoFlowNet(nn.Module):
    def __init__(self):
        super(MonoFlowNet, self).__init__()

        # configs
        self.opt = opt
        self.corr_feature_level = 2  # for debug
        self.device = self.opt.device
        self.feature_channels = [64, 164, 128, 256, 512]
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # modules
        self.ResEncoder = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        if self.opt.depth_branch:
            self.DepthDecoder = networks.DepthDecoder(self.ResEncoder.num_ch_enc, self.opt.scales)
            self.PoseDecoder = networks.PoseDecoder(self.ResEncoder.num_ch_enc, self.num_pose_frames)

            if self.opt.pose_model_type == "separate_resnet":
                self.PoseEncoder = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.PoseDecoder = networks.PoseDecoder(
                    self.PoseEncoder.num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

        if self.opt.optical_flow:
            self.Corr, self.FlowDecoder = self.build_flow_decoder()

    def forward(self, inputs):
        outputs = {}
        all_color_aug = torch.cat(
            [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])  # all images: ([i-1, i, i+1] * [L, R])
        all_features = self.ResEncoder(all_color_aug)
        all_features = [torch.split(f, self.opt.batch_size) for f in all_features]  # separate by frame
        features = {}
        for i, k in enumerate(self.opt.frame_ids):
            features[k] = [f[i] for f in all_features]
        if self.opt.depth_branch:
            outputs.update(self.DepthDecoder(features[0]))
            outputs.update(self.predict_poses(inputs, features))

        if self.opt.optical_flow:
            flow_prev_curr, flow_curr_next, flow_curr_prev, flow_next_curr, loss_by_upflow = self.predict_flow(inputs, features)
            outputs['loss_by_upflow'] = loss_by_upflow
            outputs["flow_fwd"] = (flow_prev_curr, flow_curr_next)
            # if IF_DEBUG:
            #     mono_utils.stitching_and_show(img_list=[flow_prev_curr[0], flow_curr_next[0]], ver=True)
            # outputs["flow_bwd"] = (flow_curr_prev, flow_next_curr)
            # ===== flow occlusion map
            if opt.flow_occ_check:
                occ_map_prev_curr, occ_map_curr_prev = mono_utils.cal_occ_map(flow_prev_curr, flow_curr_prev)
                occ_map_curr_next, occ_map_next_curr = mono_utils.cal_occ_map(flow_curr_next, flow_next_curr)
                outputs['flow_occ_fwd'] = (occ_map_prev_curr, occ_map_curr_next)
                outputs['flow_occ_bwd'] = (occ_map_curr_prev, occ_map_next_curr)
        return outputs

    def cal_fwd_flownet(self, feature_1, feature_2):
        ''' calculate forward flow of feat1 to feat2
        Args:
            feature_1: feature list of previous img (img_1)
            feature_2: feature list of next img (img_2)
        Returns: flow of img1->img2
        '''
        corr_1_2 = self.Corr(feature_1[self.corr_feature_level], feature_2[self.corr_feature_level])
        mod_1 = {"level" + str(i): feature_1[i] for i in range(1, len(feature_1))}
        flow_1_2 = self.FlowDecoder(mod_1, corr_1_2)
        return flow_1_2

    def cal_fwd_upflow(self, feature_1, feature_2, img_1, img_2):
        '''
        Args:
            feature_1:
            feature_2:
            img_1:
            img_2:
        Returns: flow_1_2
        '''
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().to(opt.device)
        input_dict_1_2 = {'x1_features': feature_1, 'x2_features': feature_2,
                               'im1': img_1, 'im2': img_2,
                               'im1_raw': img_1, 'im2_raw': img_2,
                               'im1_sp': img_1, 'im2_sp': img_2,
                               'start': start, 'if_loss': True, 'if_shared_features': True}
        output_dict_1_2 = self.FlowDecoder(input_dict_1_2)
        return output_dict_1_2

    def predict_flow(self, inputs, features):
        if self.opt.optical_flow in ['flownet', ]:
            flow_prev_curr = self.cal_fwd_flownet(features[-1], features[0])
            # if IF_DEBUG:
            #     mono_utils.stitching_and_show(img_list=[features[-1][0][0][0:2, :], features[0][0][0][0:2, :]], ver=True, show=True)
            #     mono_utils.stitching_and_show(img_list=[inputs['color_aug', -1, 0][0], inputs['color_aug', 0, 0][0], inputs['color_aug', 1, 0][0]],
            #                                   ver=True, show=True)

            # flow_curr_prev = self.cal_fwd_flownet(features[0], features[1])
            flow_curr_next = self.cal_fwd_flownet(features[0], features[1])
            # flow_next_curr = self.cal_fwd_flownet(features[1], features[0])
            return flow_prev_curr['level2_upsampled'], flow_curr_next['level2_upsampled'], 0, 0, 0 #\
                   # flow_curr_prev['level2_upsampled'], flow_next_curr['level2_upsampled']

        elif self.opt.optical_flow in ["upflow", ]:
            flow_prev_curr = self.cal_fwd_upflow(features[-1], features[0], inputs[("color_aug", -1, 0)], inputs[("color_aug", 0, 0)])
            flow_curr_prev = self.cal_fwd_upflow(features[0], features[-1], inputs[("color_aug", 0, 0)], inputs[("color_aug", -1, 0)])
            flow_curr_next = self.cal_fwd_upflow(features[0], features[1], inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)])
            flow_next_curr = self.cal_fwd_upflow(features[1], features[0], inputs[("color_aug", 1, 0)], inputs[("color_aug", 0, 0)])

            loss_by_upflow = flow_prev_curr['smooth_loss'] + flow_prev_curr['photo_loss'] + \
                             flow_prev_curr['census_loss'] + flow_prev_curr['msd_loss']
            return flow_prev_curr['flow_f_out'], flow_curr_next['flow_f_out'], \
                   flow_curr_prev['flow_f_out'], flow_next_curr['flow_f_out'], loss_by_upflow
        # todo: 确认两个flownet的dimension输出是一样的

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if not isinstance(f_i, str):  # not s_0, s_-1, s_1
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.PoseEncoder(torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.PoseDecoder(pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if not isinstance(i, str)], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.PoseEncoder(pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if not isinstance(i, str)]

            axisangle, translation = self.PoseDecoder(pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if not isinstance(i, str):
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def build_flow_decoder(self):
        if self.opt.optical_flow in ["flownet", ]:
            # TODO: dimesions change
            feature_channels = [64, 164, 128, 256, 512]
            Corr = networks.CorrEncoder(
                in_channels=473,
                pyramid_levels=['level3', 'level4', 'level5', 'level6'],
                kernel_size=(3, 3, 3, 3),
                num_convs=(1, 2, 2, 2),
                out_channels=(256, 512, 512, 1024),
                redir_in_channels=feature_channels[self.corr_feature_level],
                redir_channels=32,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                corr_cfg=dict(
                    type='Correlation',
                    kernel_size=1,
                    max_displacement=10,
                    stride=1,
                    padding=0,
                    dilation_patch=2),
                scaled=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
            )

            FlowNet = networks.FlowNetCDecoder(in_channels=dict(
                level6=1024, level5=1026, level4=770, level3=386, level2=130),
                out_channels=dict(level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                init_cfg=[
                    dict(
                        type='Kaiming',
                        layer=['Conv2d', 'ConvTranspose2d'],
                        a=0.1,
                        mode='fan_in',
                        nonlinearity='leaky_relu',
                        bias=0),
                    dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
                ])
            return Corr, FlowNet

        if self.opt.optical_flow in ["upflow", ]:
            from monodepth2.UPFlow_pytorch.model.upflow import UPFlow_net
            param_dict = dict(occ_type='for_back_check', alpha_1=0.1, alpha_2=0.5, occ_check_obj_out_all='all',
                              stop_occ_gradient=False, smooth_level='final', smooth_type='edge', smooth_order_1_weight=1,
                              smooth_order_2_weight=0, photo_loss_type='abs_robust', photo_loss_delta=0.4,
                              photo_loss_use_occ=opt.photo_loss_use_occ, photo_loss_census_weight=1, if_norm_before_cost_volume=True,
                              norm_moments_across_channels=False, norm_moments_across_images=False,
                              multi_scale_distillation_weight=1, multi_scale_distillation_style='upup',
                              multi_scale_photo_weight=1, multi_scale_distillation_occ=True, if_froze_pwc=False,
                              input_or_sp_input=1, if_use_boundary_warp=True, if_use_cor_pytorch=True)

            net_conf = UPFlow_net.config()
            net_conf.update(param_dict)
            net_conf.get_name(print_now=True)
            return None, net_conf()
        else:
            raise NotImplementedError


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



    def flow_warp(self, inputs, outputs):
        # ==== image warping using flow
        flow_prev_curr = outputs["flow_fwd"][0]
        flow_curr_next = outputs["flow_fwd"][1]
        prev_img_warped = uptools.torch_warp(inputs[('color_aug', 0, 0)], flow_prev_curr)
        curr_img_warped = uptools.torch_warp(inputs[('color_aug', 1, 0)], flow_curr_next)
        #
        # if IF_DEBUG and (count % (1e20) == 0):
        #     mono_utils.stitching_and_show([target[0], occ_fw[0], flow_fwd[0]], ver=True)

        # # todo: assume they are equal
        # if self.opt.optical_flow in ["upflow", ]:
        #     prev_img_warped = uptools.torch_warp(inputs[("color_aug", 0, 0)],
        #                                          flow_prev_curr)  # warped im1 by forward flow and im2
        #     curr_img_warped = uptools.torch_warp(inputs[("color_aug", 1, 0)], flow_curr_next)
        # elif self.opt.optical_flow in ["flownet", ]:
        #     # warp I_curr(inputs[('color', -1, 0)]) to prev_img_warped
        #     # warp I_next(inputs[('color', 0, 0)]) to curr_img_warped
        #     ## I_curr[:, i+flow_prev_curr[i,j][0], j+flow_prev_curr[i,j][1]] = I_prev[:, i, j]
        #     prev_img_warped = mono_utils.torch_warp(img2=inputs[('color', 0, 0)], flow=flow_prev_curr)
        #     curr_img_warped = mono_utils.torch_warp(img2=inputs[('color', 1, 0)], flow=flow_curr_next)

        outputs[("warped_flow", -1, "level2_upsampled")] = prev_img_warped  # warped previous to current by flow
        outputs[("warped_flow", 0, "level2_upsampled")] = curr_img_warped  # warped previous to current by flow

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

    def compute_losses(self, inputs, outputs):
        self.generate_images_pred(inputs, outputs)
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        flow_loss = 0
        if self.opt.optical_flow in ['flownet', ]:
            self.flow_warp(inputs, outputs)
            if opt.flow_occ_check:
                occ_map_prev_curr, occ_map_curr_next = outputs['flow_occ_fwd']
                occ_map_curr_prev, occ_map_next_curr = outputs['flow_occ_bwd']

            ## compute flow losses
            for k, pred in outputs.items():
                if 'warped_flow' in k and -1 in k:
                    #target=inputs["color_aug", -1, 0], pred=prev_warped_by_curr ('warped_flow', -1, 0), occ_fwd=occ_map_prev_curr
                    target = inputs["color_aug", -1, 0]
                    # occ_fw = occ_map_prev_curr
                    flow_fwd = outputs['flow_fwd'][0]
                    # count += 1

                elif 'warped_flow' in k and 0 in k:
                    # target=inputs["color_aug", 0, 0], pred=curr_warped_by_next ('warped_flow', 0, 0), occ_fwd=occ_map_curr_next
                    target = inputs[("color_aug", 0, 0)]
                    # occ_fw = occ_map_curr_next
                    flow_fwd = outputs['flow_fwd'][1]
                else:
                    continue

                # ==== todo: flow loss level
                # ==== 1. smooth loss
                # 1.1 First Order smoothness loss
                # smo_loss = mono_utils.edge_aware_smoothness_order1(
                #     img=target, pred=flow_fwd)
                smo_loss = 0
                # flow_loss += smo_loss
                # 1.2 Second Order smoothness loss

                # ==== 2. photo loss
                # if True:  # self.conf.stop_occ_gradient:
                #     occ_fw = occ_fw.clone().detach()
                photo_loss = 0
                # photo_loss = self.photo_loss_multi_type(target, pred, occ_fw,
                #                                         photo_loss_type='abs_robust',  # abs_robust, charbonnier, L1, SSIM
                #               photo_loss_delta=0.4, photo_loss_use_occ=False)
                # flow_loss += photo_loss


                l1_loss = torch.mean(torch.abs(pred - target + 1e-6))
                self.ssim.to(pred.device)
                ssim_loss = torch.mean(self.ssim(pred, target))
                flow_loss += 0.85 * ssim_loss + 0.15 * l1_loss


                # ==== 3. census loss

                # ==== 4. pyramid distillation loss

                # if IF_DEBUG and count % 10000 == 0:
                #     print("smo_loss={}, photo_loss={}".format(smo_loss, photo_loss))
            losses["flow_loss"] = flow_loss
            losses["smo_loss"] = smo_loss
            losses["photo_loss"] = photo_loss
            total_loss += flow_loss

        if self.opt.optical_flow in ['upflow']:
            self.flow_warp(inputs, outputs)
            total_loss += outputs['loss_by_upflow']
        
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

    def flow_evaluation(self, model):
        save_path = os.path.join(opt.log_dir, 'flow_val')
        os.makedirs(save_path, exist_ok=True)
        for m in model.values():
            m.eval()

        t1 = time.time()
        from monodepth2.datasets.flow_eval_datasets import KITTI as KITTI_flow_2015_dataset
        trans = transforms.Resize((opt.height, opt.width))
        val_dataset = KITTI_flow_2015_dataset(split='training', root=self.val_data_root)

        out_list, epe_list = [], []
        for val_id in range(len(val_dataset)):
            # ==== data preparation
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].to(opt.device)
            image2 = image2[None].to(opt.device)
            padder = mono_utils.InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)
            image1 = trans(image1)
            image2 = trans(image2)
            flow_gt = trans(flow_gt)
            valid_gt = trans(valid_gt.unsqueeze(0))
            valid_gt = valid_gt.squeeze()

            # ==== flow forward propagation:
            all_color_aug = torch.cat((image1, image2, image1), )  # all images: ([i-1, i, i+1] * [L, R])
            outdict = model(all_color_aug)
            out_flow = outdict['flow'][0].squeeze()
            flow = out_flow.cpu()

            # ==== flow vis for debug
            if val_id % 10 == 0:
                out_flow = flow_vis.flow_to_color(flow.permute(1, 2, 0).clone().detach().numpy(),
                                                  convert_to_bgr=False)
                gt_flow = flow_vis.flow_to_color(flow_gt.permute(1, 2, 0).clone().detach().numpy(),
                                                 convert_to_bgr=False)
                gt_flow = Image.fromarray(gt_flow)
                out_flow = Image.fromarray(out_flow)
                result = mono_utils.merge_images(gt_flow, out_flow)
                path = os.path.join(save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg')
                result.save(path)

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5
            val_num = torch.sum((1 * val))
            # def vis_val(val):
            #     val = (255 * val).numpy().reshape((192, 640))
            #     val_vis = Image.fromarray(np.uint8(val))
            #     val_vis.show()
            # vis_val(val)

            out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())

        for m in model.values():
            m.train()
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)
        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)

        t2 = time.time()
        print("Validation KITTI:  epe: %f,   f1: %f, time_spent: %f" % (epe, f1, t2 - t1))
        writer = self.writers['val']
        writer.add_scalar("KITTI_epe", epe, self.step)
        writer.add_scalar("KITTI_f1", f1, self.step)

        return {'kitti_epe': epe, 'kitti_f1': f1}

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
        breakpoint()
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


        #################### model, optim, loss, loding and saving ####################
        self.model = MonoFlowNet()
        if self.opt.load_weights_folder is not None:
            self.load_ddp_model()
        if opt.ddp:
            self.ddp_model = DDP(self.model.to(self.gpu_id), device_ids=[self.gpu_id], find_unused_parameters=True)
        else:
            self.ddp_model = self.model.to(self.gpu_id)
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

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def load_ddp_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))
        chechpoint_path = os.path.join(self.opt.load_weights_folder, "momoFlow.pth")
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
        save_path = os.path.join(save_folder, "momoFlow.pth")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        torch.save(self.ddp_model.state_dict(), save_path)

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        loss = losses["loss"].cpu().data
        # todo
        # smo_loss = losses["smo_loss"]
        # photo_loss = losses["photo_loss"]
        smo_loss = 0
        photo_loss = 0
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        curr_lr = self.model_optimizer.param_groups[0]['lr']
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {} | learning_rate: {} | smooth_loss: {} | photo loss: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  mono_utils.sec_to_hm_str(time_sofar), mono_utils.sec_to_hm_str(training_time_left),
                                  curr_lr, smo_loss, photo_loss))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            if self.opt.optical_flow is not None:
                tmp = outputs['flow_fwd'][0][j].permute(1, 2, 0).clone().detach().cpu().numpy()
                to_tensorboard_0 = transforms.functional.pil_to_tensor(
                    mono_utils.stitching_and_show(img_list=[
                    inputs['color_aug', -1, 0][j],  # prev
                    outputs['flow_fwd'][0][j],  # flow_prev_curr
                    inputs['color_aug', 0, 0][j]  # curr
                ], ver=True, show=False)
                )

                to_tensorboard_diff = transforms.functional.pil_to_tensor(
                    mono_utils.stitching_and_show(img_list=[
                    mono_utils.img_diff_show(outputs[("warped_flow", -1, 'level2_upsampled')][j], inputs['color_aug', -1, 0][j]),  # diff
                    outputs[("warped_flow", -1, 'level2_upsampled')][j],  # prev warped by flow
                    inputs['color_aug', -1, 0][j]  # prev
                ], ver=True, show=False)
                )

                flow_tmp = torch.from_numpy(flow_vis.flow_to_color(tmp, convert_to_bgr=False)).permute(2, 0, 1)
                writer.add_image('flow_prev_curr/{}'.format(j), to_tensorboard_0, self.step)
                tmp = outputs['flow_fwd'][1][j].permute(1, 2, 0).clone().detach().cpu().numpy()
                flow_tmp = torch.from_numpy(flow_vis.flow_to_color(tmp, convert_to_bgr=False)).permute(2, 0, 1)
                writer.add_image('flow_curr_next/{}'.format(j), flow_tmp, self.step)
                writer.add_image('prev_warped_by_flow', to_tensorboard_diff, self.step)
                writer.add_image('curr_warped_by_flow', outputs[("warped_flow", 0, 'level2_upsampled')][j], self.step)
                if opt.flow_occ_check:
                    writer.add_image('flow_occ_fwd', outputs['flow_occ_fwd'][0][j], self.step)

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
        self.set_eval()
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


        self.set_train()

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

        # for j in range(inputs['color_aug', -1, 0].size()[0]):
        #     mono_utils.stitching_and_show(img_list=[
        #         inputs['color', -1, 0][j],
        #         inputs['color', 0, 0][j],
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
        self.step = 0
        for epoch in range(self.opt.num_epochs):
            self.epoch = epoch
            self._run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.is_master_node:
                # self.val()
                # todo: eval flow
                # if self.opt.optical_flow:
                #     self.kitti_val_result = self.val_flow()
                self.save_ddp_model()


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
        shuffle=False,
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

if __name__ == "__main__":
    if opt.ddp:
        # world_size = torch.cuda.device_count()
        mp.spawn(main, args=(opt.world_size,), nprocs=opt.world_size, join=True)
    else:
        dataset = load_train_objs()
        train_loader = prepare_dataloader(dataset)
        trainer = DDP_Trainer(gpu_id=int(opt.device[-1]), train_loader=train_loader)
        trainer.train()




















































