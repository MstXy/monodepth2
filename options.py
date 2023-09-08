# Copyright Niantic 2019. Patent Pending. All rights reserved.
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import argparse
from datetime import datetime
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


def str2bool(stri_):
    return True if stri_.lower() == 'true' else False


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 # default=os.path.join(file_dir, "kitti_data")
                                 default="/mnt/km-nfs/ns100002-share/KITTI_raw"
                                 )
        self.parser.add_argument("--data_path_FlyingChairs",
                                 help="paht to flyingchairs dataset",
                                 type=str,
                                 default="/mnt/km-nfs/ns100002-share/FlyingChairs_release/data")
        self.parser.add_argument("--data_path_FlyingThings3D",
                                    type=str,
                                    default="/mnt/km-nfs/ns100002-share/FlyingThings3D_subset/train")
        
        self.parser.add_argument("--data_path_MpiSintel",
                                    type=str,
                                    default="/mnt/km-nfs/ns100002-share/MPI-Sintel-complete")
        
        self.parser.add_argument("--data_path_KITTI_mv15",
                                    type=str, 
                                    default="/home/wangshuo/LAB-Backup/Data/kitti/data_scene_flow_multiview")
    
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.dirname(os.path.dirname(file_dir)), 'mono_log',
                                                      datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.parser.add_argument("--val_data_root",
                                 type=str,
                                 help='flow valuation kitti dataset root dir',
                                 default='/mnt/km-nfs/ns100002-share/data_scene_flow')
        
        self.parser.add_argument("--train_dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "FlyingChairs"])
        
        self.parser.add_argument("--val_dataset",
                                 type=str,
                                 help="dataset to validate on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "FlyingChairs"])
        
        self.parser.add_argument("--freeze_Resnet",
                                    type=str2bool,
                                    help="freeze Resnet",
                                    default='False')
        
        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 # default="2023-06-21_21-14-49/mdp/models/weights_19")
                                 default=None)

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose", "corr", "flow"])

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="MonoFlowNet")  # MonoFlowNet or UnFlowNet

        # ==== ddp settings
        self.parser.add_argument("--ddp",
                                 type=bool,
                                 help='whether use ddp',
                                 default=False)

        self.parser.add_argument("--world_size",
                                 type=int,
                                 default=4)
        # ==== training settings
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=15)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=40)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="start_epoch",
                                 default=0)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=200)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        # LOSS WEIGHTS
        self.parser.add_argument("--loss_smo1_w",
                                 type=float,
                                 help="smoothness weight order 1",
                                 default=10.0)
        
        self.parser.add_argument("--loss_smo2_w",
                                    type=float,
                                    help="smoothness weight order2",
                                    default=10.0)
        
        self.parser.add_argument("--loss_l1_w",
                                 type=float,
                                 help="weight for l1 loss",
                                 default=0.25)
        
        self.parser.add_argument("--loss_ssim_w",
                                 type=float,
                                 help="weight for ssim loss",
                                 default=0.75)
        
        # TRAINING options
        self.parser.add_argument("--debug",
                                 type=bool,
                                 default=False)
        self.parser.add_argument("--device",
                                 type=str,
                                 default='cuda:0'
                                 )

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

        

        # optical flow branch ----------------
        self.parser.add_argument("--optical_flow",
                                 type=str,
                                 help="optical flow model",
                                 # default="upflow")
                                 default="flownet")
                                 # default=None)

        self.parser.add_argument("--feature_type",
                                 type=int,
                                 default=0)

        self.parser.add_argument("--flow_occ_check",
                                 default=False,
                                 type=str2bool)

        self.parser.add_argument("--occ_start_epoch",
                                 type=int,
                                 default=10)

        self.parser.add_argument("--stop_occ_gradient",
                                 type=str2bool,
                                 default=True)

        self.parser.add_argument("--norm_trans",
                                 type=str2bool,
                                 default='False')

        self.parser.add_argument("--depth_branch",
                                 type=str2bool,
                                 help="predict depth or not",
                                 default='False')
        
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

        self.parser.add_argument("--flow_scales",
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
                                 action="store_true")
        # full stereo ------------------------
        self.parser.add_argument("--full_stereo",
                                 help="if set, uses stereo pair for all adjacent frames (if any) in training",
                                 action="store_true")
        # ------------------------

        # ------------------------

        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])



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
                                 # default="pretrained",
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
                                 # default="separate_resnet",
                                 default="shared",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")



        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=100)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=3)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
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


'''
--data_path
--data_path_FlyingChairs
--data_path_FlyingThings3D
--data_path_MpiSintel
--data_path_KITTI_mv15
'''
