# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                              #    default=os.path.join(file_dir, "kitti_data")
                                 default="/mnt/km-nfs/ns100002-share/KITTI_raw"
                                 # default="/home/zcy/data/win_id4_share/KITTI/raw/data/raw_dataset"
                                 )
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="/mnt/km-nfs/ns100002-share/zcy-exp/tmp"
                                 # default=os.path.join(os.path.expanduser("~"), "tmp")
                                 )

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192) # 192
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640) # 640
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true",
                                 default=False)
        # dilated resnet --------------------
        self.parser.add_argument("--drn",
                                 type=bool,
                                 help="use dilated resnet as backbone",
                                 default=False)
        # -----------------------------
        # backbone switch -----------
        self.parser.add_argument("--encoder",
                                 type=str,
                                 help="alternative encoder choices",
                                 default="efficientnet",
                                 choices=["mobilenetv3-large", "mobilenetv3-small", "mobilenetv2", "mobilenetatt", "mobilenetatt2", 
                                          "mobilevitv3_xs","mobilevitv3_s",
                                          "efficientvit", "efficientnet"
                                          "None"])
        # -----------------------------
      #   # backbone EMA -----------
      #   self.parser.add_argument("--ema",
      #                            type=bool,
      #                            help="EMA encoder weight update",
      #                            default=True)
      #   # -----------------------------
        # decoder switch -----------
        self.parser.add_argument("--decoder",
                                 type=str,
                                 help="alternative decoder choices",
                                 default="efficient",
                                 choices=["default", "efficient"])
        # -----------------------------
        # heads for transformer -----------
        self.parser.add_argument("--nhead",
                                 type=int,
                                 help="number of heads for transformers",
                                 default=1)
        # -----------------------------  
        # improved depth decoder -------------
        self.parser.add_argument('--updown',
                                 type=bool,
                                 help="use down sample in depth decoder",
                                 default=False)
        self.parser.add_argument('--depth_att',
                                 type=bool,
                                 help="use attention in depth decoder",
                                 default=False)
        self.parser.add_argument('--depth_cv',
                                 type=bool,
                                 help="use attention in depth decoder",
                                 default=False)
        self.parser.add_argument('--all_corr_levels',
                                 nargs="+",
                                 type=int,
                                 help="coorelation levels used in depth decoder",
                                 default=[3,4])
        self.parser.add_argument('--cv_reproj',
                                 type=bool,
                                 help="use cost volume on warped image on computed depth",
                                 default=False)
        # ------------------------------------
        # prediction refine (coarse-to-fine) --------------------
        self.parser.add_argument("--coarse2fine",
                                 type=bool,
                                 help="use coarse-to-fine to refine prediction",
                                 default=False)
        # -----------------------------
        # feature refine --------------------
        self.parser.add_argument("--self_att",
                                 type=bool,
                                 help="use transformer block to refine feature",
                                 default=False)

        self.parser.add_argument("--psp",
                                 type=bool,
                                 help="use ppm block to refine feature",
                                 default=False)
        
        self.parser.add_argument("--aspp",
                                 type=bool,
                                 help="use aspp block to refine feature",
                                 default=False)
        
        self.parser.add_argument("--apnb",
                                 type=bool,
                                 help="use apnb block to refine feature",
                                 default=False)
        
        self.parser.add_argument("--afnb",
                                 type=bool,
                                 help="use afnb block to refine feature",
                                 default=False)
        
        self.parser.add_argument("--ann",
                                 type=bool,
                                 help="use afnb+apnb block to refine feature",
                                 default=False)
        # -----------------------------
        # full stereo ------------------------
        self.parser.add_argument("--full_stereo",
                                 help="if set, uses stereo pair for all adjacent frames (if any) in training",
                                 action="store_true",
                                 default=False)
        # ------------------------
        # optical flow branch ----------------
        self.parser.add_argument("--optical_flow",
                                 type=str,
                                 help="optical flow model",
                                 choices=["PWCLiteWithResNet", "flownet"],
                                 default="PWCLiteWithResNet")
        self.parser.add_argument("--flow_loss_weight",
                                 type=float,
                                 help="weight for flow loss",
                                 default=1)
        
        self.parser.add_argument("--flow_occ_check",
                                 default=False,
                                 type=bool)

        self.parser.add_argument("--occ_start_epoch",
                                 type=int,
                                 default=10)

        self.parser.add_argument("--stop_occ_gradient",
                                 type=bool,
                                 default=True)

        self.parser.add_argument("--norm_trans",
                                 type=bool,
                                 default=False)
        # ------------------------

        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=40) # 20
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=30)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 # default="shared",
                                 choices=["posecnn", "separate_resnet", "shared", "separate_backbone"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 # default=["encoder", "depth", "pose_encoder", "pose"],
                                 default=["encoder", "depth", "corr", "pose"],
                                 )

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        # refiner--------------------
        self.parser.add_argument("--eval_refiner",
                                 help="if set evaluates with refiner module",
                                 action="store_true", 
                                 default=False)
        self.parser.add_argument("--single_refiner",
                                 help="if set evaluates with refiner module",
                                 action="store_true", 
                                 default=False)
        # -----------------------
        
        
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true",
                                 default=False # true for full stereo
                                 )
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
