# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

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
from UPFlow_pytorch.utils.tools import tools as uptools
import monodepth2.utils.utils as mono_utils

corr_feature_level = 2

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.val_data_root = self.opt.val_data_root

        if self.opt.optical_flow in ["upflow",]:
            from UPFlow_pytorch.model.upflow import UPFlow_net


        ##<<<<<<<< for input augmentation and resize >>>>>>>>>>>>
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
        ##>>>>>>>>>>>>> for input augmentation and resize <<<<<<<<<<<<<<<<



        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else self.opt.device)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s_0")

        if self.opt.full_stereo:
            self.opt.frame_ids += ["s_" + str(i) for i in self.opt.frame_ids]

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.depth_branch:
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.optical_flow in ["flownet",]:
            # TODO: dimesions change
            feature_channels = [64, 164, 128, 256, 512]

            self.models["corr"] = networks.CorrEncoder(
                in_channels=473,
                pyramid_levels=['level3', 'level4', 'level5', 'level6'],
                kernel_size=(3, 3, 3, 3),
                num_convs=(1, 2, 2, 2),
                out_channels=(256, 512, 512, 1024),
                redir_in_channels=feature_channels[corr_feature_level],
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

            self.models["corr"].to(self.device)
            self.parameters_to_train += list(self.models["corr"].parameters())
            
            self.models["flow"] = networks.FlowNetCDecoder(in_channels=dict(
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
            
            self.models["flow"].to(self.device)
            self.parameters_to_train += list(self.models["flow"].parameters())

        if self.opt.optical_flow in ["upflow",]:
            self.upflow_param_dict = {
                'occ_type': 'for_back_check',
                'alpha_1': 0.1,
                'alpha_2': 0.5,
                'occ_check_obj_out_all': 'all',
                'stop_occ_gradient': True,
                'smooth_level': 'final',  # final or 1/4
                'smooth_type': 'edge',  # edge or delta
                'smooth_order_1_weight': 1,
                # smooth loss
                'smooth_order_2_weight': 0,
                # photo loss type add SSIM
                'photo_loss_type': 'abs_robust',  # abs_robust, charbonnier,L1, SSIM
                'photo_loss_delta': 0.4,
                'photo_loss_use_occ': True,
                'photo_loss_census_weight': 1,
                # use cost volume norm
                'if_norm_before_cost_volume': True,
                'norm_moments_across_channels': False,
                'norm_moments_across_images': False,
                'multi_scale_distillation_weight': 1,
                'multi_scale_distillation_style': 'upup',
                'multi_scale_photo_weight': 1,  # 'down', 'upup', 'updown'
                'multi_scale_distillation_occ': True,  # if consider occlusion mask in multiscale distilation
                'if_froze_pwc': False,
                'input_or_sp_input': 1,
                'if_use_boundary_warp': True,
                'if_sgu_upsample': False,  # if use sgu upsampling
                'if_use_cor_pytorch': True,
            }

            net_conf = UPFlow_net.config()
            net_conf.update(self.upflow_param_dict)
            net_conf.get_name(print_now=True)
            self.models['upflow'] = net_conf()
            self.models["upflow"].to(self.device)
            self.parameters_to_train += list(self.models["upflow"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.4)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.train_dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        # TODO: for debugging
        # img_ext = '.png' if self.opt.png else '.jpg'
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs



        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=False, drop_last=True,
            sampler=(torch.utils.data.distributed.DistributedSampler(train_dataset) if self.opt.ddp else None))

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()


        # ==== loss, occ check
        # self.occ_check_model = uptools.occ_check_model(occ_type=self.upflow_param_dict['occ_type'],
        #                                                occ_alpha_1=self.upflow_param_dict['alpha_1'],
        #                                                occ_alpha_2=self.upflow_param_dict['alpha_2'],
        #                                                obj_out_all=self.upflow_param_dict['occ_check_obj_out_all'])

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        # self.load_model()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def preprocess(self, inputs):
        color_aug = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)
        """Resize colour images to the required scales and augment if required

                We create the color_aug object in advance and apply the same augmentation to all
                images in this item. This ensures that all images input to the pose network receive the
                same augmentation.
                """

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

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        end_time = time.time()
        data_loading_time_consumption = 0
        forward_inference_time_consumption = 0
        backword_time_consumption = 0

        print_time = False
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            if print_time:
                print('==============================================self.step:', self.step)
                print("data loading time consumption :{}".format(before_op_time - end_time))
                data_loading_time_consumption += before_op_time - end_time


            outputs, losses = self.process_batch(inputs)

            if print_time:
                inference_time = time.time()
                print("Forward inference time consumption :{}".format(inference_time - before_op_time))
                forward_inference_time_consumption += inference_time - before_op_time

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            end_time = time.time()
            duration = end_time - before_op_time
            if print_time:
                print("Backword time consumption: {}".format(end_time - inference_time))
                backword_time_consumption += end_time - inference_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000 and not self.opt.debug
            late_phase = self.step % 2000 == 0 and not self.opt.debug

            if early_phase or late_phase:
                if print_time:
                    print('\n', data_loading_time_consumption, forward_inference_time_consumption, backword_time_consumption, '\n')
                    data_loading_time_consumption = 0
                    forward_inference_time_consumption = 0
                    backword_time_consumption = 0
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs and self.opt.depth_branch:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            if (self.step+1) % 1000 == 0 and self.opt.optical_flow:
                self.kitti_val_result = self.val_flow()

            self.step += 1
        
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        outputs = {}
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        self.preprocess(inputs)
        
        # for k, v in self.models.items():
        #     self.models[k] = torch.nn.DataParallel(v, device_ids=list(range(torch.cuda.device_count())))

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids]) # all images: ([i-1, i, i+1] * [L, R])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features] # separate by frame
            features = {}
            
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            if self.opt.depth_branch:
                outputs = self.models["depth"](features[0]) # only predict depth for current frame (monocular)
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            if self.opt.depth_branch:
                outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        if self.opt.optical_flow in ["flownet",]:
            outputs.update(self.predict_flow(features))
        
        if self.opt.optical_flow in ["upflow",]:
            outputs.update(self.predict_upflow(features, inputs))
 
        self.generate_images_pred(inputs, outputs)

        
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

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
                if not isinstance(f_i, str): # not s_0, s_-1, s_1
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
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
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if not isinstance(i, str)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if not isinstance(i, str):
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def show_tensor(self, input_tensor):
        """
        Args:
            input_tensor: CHW
        Returns:

        """
        assert len(input_tensor.shape) == 3
        trans = torchvision.transforms.ToPILImage()
        out = trans(input_tensor)
        return out

    def merge_images(self, image1, image2):
        """Merge two images into one, displayed side by side
        """
        (width1, height1) = image1.size
        (width2, height2) = image2.size

        result_width = max(width1, width2)
        result_height = height1+height2

        result = Image.new('RGB', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(0, height1))
        return result

    def predict_upflow(self, features, inputs):

        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().to(self.device)

        input_dict_pre_curr = {'x1_features': features[-1], 'x2_features': features[0],
                               'im1': inputs[("color_aug", -1, 0)], 'im2': inputs[("color_aug", 0, 0)],
                               'im1_raw': inputs[("color_aug", -1, 0)], 'im2_raw': inputs[("color_aug", 0, 0)],
                               'im1_sp': inputs[("color_aug", -1, 0)], 'im2_sp': inputs[("color_aug", 0, 0)],
                               'start': start, 'if_loss': False, 'if_shared_features': True}
        
        input_dict_curr_next = {'x1_features': features[0], 'x2_features': features[1],
                                'im1': inputs[("color_aug", 0, 0)], 'im2': inputs[("color_aug", 1, 0)],
                                'im1_raw': inputs[("color_aug", 0, 0)], 'im2_raw': inputs[("color_aug", 1, 0)],
                                'im1_sp': inputs[("color_aug", 0, 0)], 'im2_sp': inputs[("color_aug", 1, 0)],
                                'start': start, 'if_loss': False, 'if_shared_features': True}
        output_dict_pre_curr = self.models["upflow"](input_dict_pre_curr)
        output_dict_curr_next = self.models["upflow"](input_dict_curr_next)

        outputs = {}
        outputs["flow"] = [output_dict_pre_curr['flow_f_out'], output_dict_curr_next['flow_f_out']]
        # todo : add multi level flow output to calculate multilevel flow loss
        return outputs

    def predict_flow(self, features):
        """Predict flow between ??.
        """
        # TODO: multi image
        outputs = {}
        # print(type(features[-1])) # Python list
        # print(len(features[-1])) # 5

        # use D=256, i.e., level=3
        corr_prev_curr = self.models["corr"](features[-1][corr_feature_level], features[0][corr_feature_level])  # mono view, for now
        corr_curr_next = self.models["corr"](features[0][corr_feature_level], features[1][corr_feature_level])


        if self.opt.debug:
            # for debug; feature_dim explicitly
            feature_shape_list = []
            for i in range(len(features[0])):
                feature_shape_list.append(
                    features[0][i].shape
                )
            # for debug; corr shape
            corr_shape = {}
            for level in corr_prev_curr:
                corr_shape[str(level)] = corr_prev_curr[level].shape

        mod_prev = {"level"+str(i) : features[-1][i] for i in range(1, len(features[-1]))}
        mod_curr = {"level"+str(i) : features[0][i] for i in range(1, len(features[0]))}

        flow_prev_curr = self.models["flow"](mod_prev, corr_prev_curr)
        flow_curr_next = self.models["flow"](mod_curr, corr_curr_next)
        outputs["flow"] = [flow_prev_curr['level2_upsampled'], flow_curr_next['level2_upsampled']]

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
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

    def val_flow(self):
        save_path = os.path.join(self.opt.log_dir, 'flow_val')
        os.makedirs(save_path, exist_ok=True)
        self.set_eval()
        t1 = time.time()
        from datasets.flow_eval_datasets import KITTI as KITTI_flow_2015_dataset
        trans = torchvision.transforms.Resize((self.opt.height, self.opt.width))
        val_dataset = KITTI_flow_2015_dataset(split='training', root=self.val_data_root)
        out_list, epe_list = [], []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].to(self.device)
            image2 = image2[None].to(self.device)
            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)
            image1 = trans(image1)
            image2 = trans(image2)
            flow_gt = trans(flow_gt)
            valid_gt = trans(valid_gt.unsqueeze(0))
            valid_gt = valid_gt.squeeze()

            # flow forward propagation:
            all_color_aug = torch.cat((image1, image2, image1), )  # all images: ([i-1, i, i+1] * [L, R])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, 1) for f in all_features]  # separate by frame
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            if self.opt.optical_flow in ["flownet", ]:
                outdict = self.predict_flow(features)

            if self.opt.optical_flow in ["upflow", ]:
                imputs_dic = {
                    ("color_aug", -1, 0): image1,
                    ("color_aug", 0, 0): image2,
                    ("color_aug", 1, 0): image1,
                }
                outdict = self.predict_upflow(features, imputs_dic)

            out_flow = outdict['flow'][0].squeeze()
            flow = out_flow.cpu()

            ## flow vis for debug
            if val_id % 10 == 0:
                out_flow = flow_vis.flow_to_color(flow.permute(1, 2, 0).clone().detach().numpy(), convert_to_bgr=False)
                gt_flow = flow_vis.flow_to_color(flow_gt.permute(1, 2, 0).clone().detach().numpy(), convert_to_bgr=False)
                gt_flow = Image.fromarray(gt_flow)
                out_flow = Image.fromarray(out_flow)
                result = self.merge_images(gt_flow, out_flow)
                path = os.path.join(save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg')
                result.save(path)

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5
            val_num = torch.sum((1*val))
            # def vis_val(val):
            #     val = (255 * val).numpy().reshape((192, 640))
            #     val_vis = Image.fromarray(np.uint8(val))
            #     val_vis.show()
            # vis_val(val)

            out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())

        self.set_train()
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)
        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)

        t2 = time.time()
        print("Validation KITTI:  epe: %f,   f1: %f, time_spent: %f" % (epe, f1, t2-t1))
        writer = self.writers['val']
        writer.add_scalar("KITTI_epe", epe, self.step)
        writer.add_scalar("KITTI_f1", f1, self.step)

        return {'kitti_epe': epe, 'kitti_f1': f1}

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        if self.opt.depth_branch:
            # compute warped images by depth
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = F.interpolate(
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

                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

                    outputs[("sample", frame_id, scale)] = pix_coords

                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border",
                        align_corners=True)

                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        flow_loss = 0




        if self.opt.optical_flow:
            ## ==== image warping using flow
            flow_prev_curr = outputs["flow"][0]
            flow_curr_next = outputs["flow"][1]
            # todo: assume they are equal
            if self.opt.optical_flow in ["upflow", ]:
                prev_img_warped = uptools.torch_warp(inputs[("color_aug", 0, 0)],
                                                     flow_prev_curr)  # warped im1 by forward flow and im2
                curr_img_warped = uptools.torch_warp(inputs[("color_aug", 1, 0)], flow_curr_next)

            elif self.opt.optical_flow in ["flownet", ]:
                # warp I_curr(inputs[('color', -1, 0)]) to prev_img_warped
                # warp I_next(inputs[('color', 0, 0)]) to curr_img_warped
                ## psudo:
                ## I_curr[:, i+flow_prev_curr[i,j][0], j+flow_prev_curr[i,j][1]] = I_prev[:, i, j]
                prev_img_warped = mono_utils.torch_warp(img2=inputs[('color_aug', 0, 0)], flow=flow_prev_curr)
                curr_img_warped = mono_utils.torch_warp(img2=inputs[('color_aug', 1, 0)], flow=flow_curr_next)

            outputs[("warped_flow", -1, "level2_upsampled")] = prev_img_warped  # warped previous to current by flow
            outputs[("warped_flow", 0, "level2_upsampled")] = curr_img_warped  # warped previous to current by flow


            ## compute flow losses
            for k, pred in outputs.items():
                if 'warped_flow' in k and -1 in k:
                    target = inputs["color_aug", -1, 0]

                elif 'warped_flow' in k and 0 in k:
                    target = inputs[("color_aug", 0, 0)]
                else:
                    continue
                l1_loss = torch.mean(torch.abs(pred - target + 1e-6))
                ssim_loss = torch.mean(self.ssim(pred, target))
                flow_loss += 0.85 * ssim_loss + 0.15 * l1_loss
            losses["flow_loss"] = flow_loss
            total_loss += flow_loss

        
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
                        identity_reprojection_loss.shape, device=self.device) * 0.00001

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

    def photo_loss_multi_type(self, x, y, occ_mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier, L1, SSIM
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
            return F.avg_pool2d(x, (3, 3), (1, 1))
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

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        curr_lr = self.model_optimizer.param_groups[0]['lr']
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {} | learning_rate: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),
                                  curr_lr))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            if self.opt.optical_flow is not None:
                flow_tmp = torch.from_numpy(
                    flow_vis.flow_to_color(outputs['flow'][0][j].permute(1, 2, 0).clone().detach().cpu().numpy(),
                                           convert_to_bgr=False)).permute(2, 0, 1)
                writer.add_image('flow_prev_curr/{}'.format(j), flow_tmp, self.step)
                flow_tmp = torch.from_numpy(
                    flow_vis.flow_to_color(outputs['flow'][1][j].permute(1, 2, 0).clone().detach().cpu().numpy(),
                                           convert_to_bgr=False)).permute(2, 0, 1)
                writer.add_image('flow_curr_next/{}'.format(j), flow_tmp, self.step)
                writer.add_image('prev_warped_by_flow', outputs[("warped_flow", -1, 'level2_upsampled')][j], self.step)
                writer.add_image('curr_warped_by_flow', outputs[("warped_flow", 0, 'level2_upsampled')][j], self.step)
            
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
                        normalize_image(outputs[("disp", s)][j]), self.step)

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

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
                to_save['full_stereo'] = self.opt.full_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

