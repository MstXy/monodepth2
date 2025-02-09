# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from easydict import EasyDict 

import cupy as cp

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from networks.feature_refine import APNB, AFNB, ASPP, PPM, SelfAttention
from networks.dilated_resnet import dilated_resnet18
from networks.mobilenet_encoder import MobileNetV3, MobileNetV2, MobileNetAtt, MobileNetAtt2
from networks.mobilevit.build_mobileViTv3 import MobileViT
from networks.mobilevit.misc.averaging_utils import EMA
from networks.mobilevit.utils.checkpoint_utils import copy_weights

from networks.efficientvit.build_efficientvit import EfficientViT

from networks.ARFlow_losses.flow_loss import unFlowLoss

from IPython import embed


class Trainer:
    def __init__(self, options, parser=None):
        alt_parser = copy.deepcopy(parser)

        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.DEVICE_NUM = 4 # change for # of GPU
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:{}".format(self.DEVICE_NUM))

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

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s_0")
        
        if self.opt.full_stereo:
            self.opt.frame_ids += ["s_" + str(i) for i in self.opt.frame_ids]

        # frame_id and idx_pair_list
        if len(self.opt.frame_ids) == 3:
            self.idx_pair_list = [(-1, 0)] # TODO: was [(-1, 0), (0, 1)]
        elif len(self.opt.frame_ids) == 2:
            self.idx_pair_list = [(-1, 0)]
        else:
            raise NotImplementedError 

        # move utils up
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        
        if self.opt.encoder == "mobilenetv3-large":
            print("using MobileNetV3-large as backbone")
            self.models["encoder"] = MobileNetV3(model_type="large")
            self.opt.mobile_backbone = "v3l"
        elif self.opt.encoder == "mobilenetv3-small":
            print("using MobileNetV3-small as backbone")
            self.models["encoder"] = MobileNetV3(model_type="small")
            self.opt.mobile_backbone = "v3s"
        elif self.opt.encoder == "mobilenetv2":
            print("using MobileNetV2 as backbone")
            self.models["encoder"] = MobileNetV2()
            self.opt.mobile_backbone = "v2"
        elif self.opt.encoder == "mobilenetatt":
            print("using MobileNet+Attention as backbone")
            self.models["encoder"] = MobileNetAtt(self.opt.nhead)
            self.opt.mobile_backbone = "vatt"
        elif self.opt.encoder == "mobilenetatt2":
            print("using MobileNet+Attention as backbone")
            self.models["encoder"] = MobileNetAtt2(self.opt.nhead)
            self.opt.mobile_backbone = "vatt2"
        elif self.opt.encoder == "mobilevitv3_xs":
            print("using MobileViTv3_XS as backbone")
            self.models["encoder"] = MobileViT(parser, self.opt.encoder)
            self.opt.mobile_backbone = "mbvitv3_xs"
            self.use_ema = self.models["encoder"].use_ema
            if self.use_ema:
                print("using EMA")
                self.model_ema = EMA(
                    model=self.models["encoder"],
                    ema_momentum=self.models["encoder"].ema_momentum,
                    device=self.device
                )
            else:
                self.model_ema = None
        elif self.opt.encoder ==  "mobilevitv3_s":
            print("using MobileViTv3_S as backbone")
            self.models["encoder"] = MobileViT(parser, self.opt.encoder)
            self.opt.mobile_backbone = "mbvitv3_s"
            self.use_ema = self.models["encoder"].use_ema
            if self.use_ema:
                print("using EMA")
                self.model_ema = EMA(
                    model=self.models["encoder"],
                    ema_momentum=self.models["encoder"].ema_momentum,
                    device=self.device
                )
            else:
                self.model_ema = None
        elif self.opt.encoder == "efficientvit":
            self.models["encoder"] = EfficientViT(model_name="b1")
            self.opt.mobile_backbone = "effvit-b1"
            self.use_ema = False
        elif self.opt.encoder == "efficientnet":
            self.models["encoder"] = networks.EfficientEncoder()
            self.opt.mobile_backbone = "eff-b0"
            self.use_ema = False
        else:
            self.opt.mobile_backbone = None
            # dilated ResNet?
            if self.opt.drn:
                # default to res18
                self.models["encoder"] = dilated_resnet18(
                    self.opt.weights_init == "pretrained")
            else:
                self.models["encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.self_att:
            dropout = 0.1
            self.models["self_att"] = SelfAttention(num_heads=1, dropout=dropout)
            self.models["self_att"].to(self.device)
            self.parameters_to_train += list(self.models["self_att"].parameters())
        if self.opt.psp:
            fea_dim = 512
            bins = [1,2,3,6]
            dropout = 0.3
            self.models["psp"] = PPM(in_dim=fea_dim, reduction_dim=int(fea_dim/len(bins)), bins=bins, dropout=dropout, only_current_frame=self.opt.pose_model_type=="separate_resnet")
            self.models["psp"].to(self.device)
            self.parameters_to_train += list(self.models["psp"].parameters())
        if self.opt.aspp:
            fea_dim = 512
            atrous_rates=[6, 12, 18]
            self.models["aspp"] = ASPP(in_ch=fea_dim, mid_ch=fea_dim//2, out_ch=fea_dim, rates=atrous_rates, only_current_frame=self.opt.pose_model_type=="separate_resnet")
            self.models["aspp"].to(self.device)
            self.parameters_to_train += list(self.models["aspp"].parameters())
            # print(sum(p.numel() for p in self.models["aspp"].parameters() if p.requires_grad))
        if self.opt.apnb:
            fea_dim = 512
            dropout = 0.05
            self.models["apnb"] = APNB(in_channels=fea_dim, out_channels=fea_dim, key_channels=256, value_channels=256, dropout=dropout, norm_type="batchnorm")
            self.models["apnb"].to(self.device)
            self.parameters_to_train += list(self.models["apnb"].parameters())
        if self.opt.afnb:
            low_fea_dim = 256
            high_fea_dim = 512
            fea_dim = 512
            kv_dim = 128
            dropout = 0.05
            self.models["afnb"] = AFNB(low_in_channels=low_fea_dim, high_in_channels=high_fea_dim, 
                                       out_channels=fea_dim, key_channels=kv_dim, value_channels=kv_dim, dropout=dropout, norm_type="batchnorm")
            self.models["afnb"].to(self.device)
            self.parameters_to_train += list(self.models["afnb"].parameters())
        if self.opt.ann:
            # AFNB: fusion
            low_fea_dim = 256
            high_fea_dim = 512
            fea_dim = 512
            kv_dim = 128
            dropout = 0.05
            self.models["afnb"] = AFNB(low_in_channels=low_fea_dim, high_in_channels=high_fea_dim, 
                                       out_channels=fea_dim, key_channels=kv_dim, value_channels=kv_dim, dropout=dropout, norm_type="batchnorm")
            self.models["afnb"].to(self.device)
            self.parameters_to_train += list(self.models["afnb"].parameters())
            # APNB: self-att
            fea_dim = 512
            dropout = 0.05
            self.models["apnb"] = APNB(in_channels=fea_dim, out_channels=fea_dim, key_channels=256, value_channels=256, dropout=dropout, norm_type="batchnorm")
            self.models["apnb"].to(self.device)
            self.parameters_to_train += list(self.models["apnb"].parameters())

        # alternative depth decoder:
        if self.opt.decoder == "default":
            print("using default depth decoder")
            print("Using depth channel att fusion: %s" % self.opt.depth_att)
            print("Using down sample: %s" % self.opt.updown)
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales, drn=self.opt.drn, 
                    depth_att=self.opt.depth_att, depth_cv=self.opt.depth_cv, depth_refine=self.opt.coarse2fine, updown=self.opt.updown, 
                    corr_levels=self.opt.all_corr_levels, n_head=self.opt.nhead,
                    cv_reproj=self.opt.cv_reproj, backproject_depth=self.backproject_depth, project_3d=self.project_3d,
                    mobile_backbone=self.opt.mobile_backbone)
        elif self.opt.decoder == "efficient":
            print("using efficient decoder")
            self.models["depth"] = networks.EfficientDecoder(self.models["encoder"].num_ch_enc)
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
            elif self.opt.pose_model_type == "separate_backbone":
                self.models["pose_encoder"] = MobileViT(alt_parser, num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames, 
                    # inter_output=self.opt.refine_pred
                    )

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            # if self.opt.refine_pred:
            #     self.models["att_pose"] = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.5, device=self.device)
            #     self.parameters_to_train += list(self.models["att_pose"].parameters())
        
        self.use_flow = self.opt.optical_flow in ["flownet","PWCLiteWithResNet"]
        
        if self.opt.optical_flow in ["flownet",] or self.opt.depth_cv:
            # set cupy device
            cp.cuda.Device(self.DEVICE_NUM).use()

            # TODO: dimesions change
            # self.models["corr"] = networks.CorrEncoder(
            #     in_channels=473,
            #     pyramid_levels=['level3', 'level4', 'level5', 'level6'],
            #     kernel_size=(3, 3, 3, 3),
            #     num_convs=(1, 2, 2, 2),
            #     out_channels=(256, 512, 512, 1024),
            #     redir_in_channels=256,
            #     redir_channels=32,
            #     strides=(1, 2, 2, 2),
            #     dilations=(1, 1, 1, 1),
            #     corr_cfg=dict(
            #         type='Correlation',
            #         kernel_size=1,
            #         max_displacement=10,
            #         stride=1,
            #         padding=0,
            #         dilation_patch=2),
            #     scaled=False,
            #     conv_cfg=None,
            #     norm_cfg=None,
            #     act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
            # )
            
            # self.models["corr"] = networks.corr_encoder.CorrEncoderSimple(levels=self.opt.all_corr_levels)

            self.models["corr"] = networks.corr_encoder.CorrEncoderAtt(levels=self.opt.all_corr_levels, n_head=self.opt.nhead)

            self.models["corr"].to(self.device)
            self.parameters_to_train += list(self.models["corr"].parameters())
            
        if self.use_flow:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'networks/ARFlow_losses/kitti_raw.json')) as f:
                self.flow_cfg = EasyDict(json.load(f))
            self.models["flow"] = networks.PWCLiteWithResNet(self.flow_cfg.model)
            self.arflow_loss = unFlowLoss(cfg=self.flow_cfg.loss)

            self.models["flow"].to(self.device)
            self.parameters_to_train += list(self.models["flow"].parameters())

            

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
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # log number of parameters
        print(sum([sum(p.numel() for p in self.models[k].parameters() if p.requires_grad) for k in self.models.keys() if k in ["encoder", "depth"]]))

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

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
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

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
        for self.epoch in range(self.opt.num_epochs):
            
            # self.eval_depth_flow()
            if 'stage1' in self.flow_cfg.train:
                if self.epoch >= self.flow_cfg.train.stage1.epoch:
                    self.arflow_loss.cfg.update(self.flow_cfg.train.stage1.loss)
                    print('\n ==========update loss function to stage1 loss========== \n')
                    
            if self.epoch > self.opt.occ_start_epoch:
                self.opt.flow_occ_check = True

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

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            if self.use_ema:
                self.model_ema.update_parameters(self.models["encoder"])

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
        
        self.model_lr_scheduler.step()
        # self.models["encoder"] = copy_weights(model_tgt=self.models["encoder"], model_src=self.model_ema)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        self.preprocess(inputs)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids]) # all images: ([i-1, i, i+1] * [L, R])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features] # separate by frame

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]


            # psp to refine feature
            if self.opt.psp:
                features = self.models["psp"](features)
            # aspp to refine feature
            if self.opt.aspp:
                features = self.models["aspp"](features)
            # apnb to refine feature
            if self.opt.apnb:
                features = self.models["apnb"](features)
            # afnb to refine feature
            if self.opt.afnb:
                features = self.models["afnb"](features)
            if self.opt.ann:
                features = self.models["afnb"](features)
                features = self.models["apnb"](features)
            # self attention to refine feature
            if self.opt.self_att:
                features = self.models["self_att"](features)

            if self.opt.depth_cv:
                # using cost volume for depth prediction
                ## use corr encoder
                # corr_prev_curr = self.models["corr"](features[-1][3], features[0][3]) # mono view, for now
                # corr_next_curr = self.models["corr"](features[1][3], features[0][3]) # TODO: keep order or not??
                # outputs = self.models["depth"](features[0], corr_prev_curr, corr_next_curr)

                ## use corr simple & multi corr
                corrs = self.models["corr"](features)
                outputs = self.models["depth"](features[0], corrs)
                
            elif self.opt.cv_reproj:
                cp.cuda.Device(self.DEVICE_NUM).use()
                # predict pose before depth
                outputs = self.predict_poses(inputs, features)
                outputs.update(self.models["depth"](features[0], inputs=inputs, outputs=outputs, adjacent_features={-1:features[-1],1:features[1]})) # only predict depth for current frame (monocular)
            else:
                # normal
                outputs = self.models["depth"](features[0]) # only predict depth for current frame (monocular)
        else:
            # If we are using a separate encoder for depth and pose, then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids]) # all images: ([i-1, i, i+1] * [L, R])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features] # separate by frame

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            if self.opt.depth_cv:
                # using cost volume for depth prediction
                ## use corr encoder
                # corr_prev_curr = self.models["corr"](features[-1][3], features[0][3]) # mono view, for now
                # corr_next_curr = self.models["corr"](features[1][3], features[0][3]) # TODO: keep order or not??
                # outputs = self.models["depth"](features[0], corr_prev_curr, corr_next_curr)

                ## use corr simple & multi corr
                corrs = self.models["corr"](features)
                outputs = self.models["depth"](features[0], corrs)
                
            elif self.opt.cv_reproj:
                cp.cuda.Device(self.DEVICE_NUM).use()
                # predict pose before depth
                outputs = self.predict_poses(inputs, features)
                outputs.update(self.models["depth"](features[0], inputs=inputs, outputs=outputs, adjacent_features={-1:features[-1],1:features[1]})) # only predict depth for current frame (monocular)
            
            elif self.use_flow:
                # if use flow branch:
                imgs = [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids] # all images: ([i-1, i, i+1])
                features = [self.models["encoder"](img) for img in imgs]
                outputs = self.models["depth"](features[0]) # depth only takes in current frame
            else:
                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        
        if not self.opt.cv_reproj and self.use_pose_net: ## IF cv_reproj, apply before depth decoder 
            outputs.update(self.predict_poses(inputs, features))

        if self.use_flow:
            outputs.update(self.predict_flow(imgs, features))

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

                    if self.opt.pose_model_type == "separate_resnet" or self.opt.pose_model_type == "separate_backbone":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    

                    # if self.opt.refine_pred:
                    #     pose = self.models["pose"](pose_inputs)
                    #     # TODO: use attention
                        
                    #     axisangle = axisangle
                    #     translation = translation
                    # else:
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    
                # elif self.opt.full_stereo: # is right view & full_stereo: s_0, s_-1, s_1
                #     if int(f_i.split("_")[1]) == 0: # s_0
                #         outputs[("cam_T_cam", 0, f_i)] = inputs["stereo_T"]
                #     else: # s_-1, s_1
                #         if int(f_i.split("_")[1]) < 0:
                #             pose_inputs = [pose_feats[f_i], pose_feats[0]]
                #         else:
                #             pose_inputs = [pose_feats[0], pose_feats[f_i]]

                #         axisangle, translation = self.models["pose"](pose_inputs)

                #         outputs[("axisangle", 0, f_i)] = axisangle
                #         outputs[("translation", 0, f_i)] = translation

                #         # Invert the matrix if the frame id is negative
                #         outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                #             axisangle[:, 0], translation[:, 0], invert=(int(f_i.split("_")[1]) < 0))


        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn", "separate_backbone"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if not isinstance(i, str)], 1)

                if self.opt.pose_model_type == "separate_resnet" or self.opt.pose_model_type == "separate_backbone":
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
    

    def predict_flow(self, imgs, features):
        """Predict flow between ??.
        """
        return self.models["flow"](imgs, features)


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

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
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
                
                # if self.opt.full_stereo:
                #     T = outputs[("cam_T_cam", 0, frame_id)]
                # else: # normal one frame stereo
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

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

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
            losses["loss/{}".format(scale)] = loss

            if scale == 0:
                losses["loss/ident"] = identity_reprojection_loss.mean()
                losses["loss/reproj"] = reprojection_loss.mean()
                losses["loss/combined"] = to_optimise.mean()

        total_loss /= self.num_scales

        # =============== FLOW LOSS =================
        flow_loss = 0
        if self.use_flow:
            pyramid_flows = []
            for idx_pair in self.idx_pair_list:
                for scale in self.opt.scales:
                    pyramid_flows.append(
                        torch.cat(
                        (outputs[('flow', idx_pair[0], idx_pair[1], scale)], outputs[('flow', idx_pair[1], idx_pair[0], scale)]), 
                        dim=1)
                        )
                target = torch.cat((inputs['color_aug', idx_pair[0], 0], inputs['color_aug', idx_pair[1], 0]), dim=1)
            
            flow_loss, l_ph_pyramid, l_sm_pyramid, flow_mean, pyramid_occ_mask1, pyramid_occ_mask2, \
                pyramid_im1_recons, pyramid_im2_recons = self.arflow_loss(pyramid_flows, target)
            
            for scale in self.opt.scales:
                losses['smo_loss_o1', scale] = l_sm_pyramid[scale]
                losses['smo_loss_o2', scale] = l_sm_pyramid[scale]
                losses['photo_loss_l1', scale] = l_ph_pyramid[scale]
                losses['photo_loss_ssim', scale] = l_ph_pyramid[scale]
                
            for scale in self.opt.scales:
                for idx_pair in self.idx_pair_list:
                    outputs[('f_warped', idx_pair[0], idx_pair[1], scale)] = pyramid_im1_recons[scale]  # prev_img warped by curr_img and flow_prev_to_curr
                    outputs[('f_warped', idx_pair[1], idx_pair[0], scale)] = pyramid_im2_recons[scale]   # curr_img warped by prev_img and flow_curr_to_prev
                    outputs[('occ', idx_pair[0], idx_pair[1], scale)] = pyramid_occ_mask1[scale]
                    outputs[('occ', idx_pair[1], idx_pair[0], scale)] = pyramid_occ_mask2[scale]
        
        losses["loss"] = total_loss + flow_loss * self.opt.flow_loss_weight
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

        depth_gt_vis = depth_gt * mask
        depth_pred_vis = depth_pred * mask
        
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        # add gt, pred diff vis
        depth_pred_vis *= torch.median(depth_gt) / torch.median(depth_pred) # median, so use depth_gt, which contains no 0
        depth_pred_vis = torch.clamp(depth_pred_vis, min=1e-3, max=80)
        outputs["depth_diff"] = (depth_gt_vis - depth_pred_vis) ** 2

        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    writer.add_image(
                        "color_aug_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color_aug", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                
                # visualize pred & gt diff
                writer.add_image(
                    "depth_diff/{}".format(j), outputs["depth_diff"][j].data, self.step)

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
        
        if self.use_ema:
            self.models["encoder"] = copy_weights(model_tgt=self.models["encoder"], model_src=self.model_ema)

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
