from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import os
import sys
file_dir = os.path.dirname(__file__)
parent_project = os.path.dirname(file_dir)
sys.path.append(os.path.dirname(parent_project))




import copy

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from monodepth2.layers import disp_to_depth
from monodepth2.utils.utils import readlines
from monodepth2.options import MonodepthOptions
import monodepth2.datasets as datasets
import monodepth2.networks as networks



from monodepth2.layers import BackprojectDepth, Project3D, transformation_from_parameters

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
from monodepth2.networks.MonoFlowNet import MonoFlowNet
from thop.vision.basic_hooks import count_convNd, zero_ops
from thop import profile

class TestModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TestModel, self).__init__()
        self.encoder = encoder
        self.depth_decoder = decoder
    def forward(self, x):
        feat = self.encoder(x)
        result = self.depth_decoder(feat)
        return(result)
    

splits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

DEVICE_NUM = 1
DEVICE = torch.device("cuda:{}".format(DEVICE_NUM))

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt, parser):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        print(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))


        all_corr_levels = [2,3,4]
        opt.depth_cv = False


        opt.cv_reproj = False

        BATCH_SIZE = 16

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width, 
                                           [0] if (not opt.depth_cv and not opt.cv_reproj) else [0, -1, 1], 4, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=True)


        
        opt.mobile_backbone = "eff-b0"

        ## warping utils for reprojection
        backproject_depth = {}
        project_3d = {}
        for scale in opt.scales:
            h = 192 // (2 ** scale)
            w = 640 // (2 ** scale)

            backproject_depth[scale] = BackprojectDepth(BATCH_SIZE, h, w)
            backproject_depth[scale].cuda(device="cuda:{}".format(DEVICE_NUM))

            project_3d[scale] = Project3D(BATCH_SIZE, h, w)
            project_3d[scale].cuda(device="cuda:{}".format(DEVICE_NUM))





        test_model = MonoFlowNet(opt)
        # check params and FLOPs
        verbose = False
        x = torch.randn(1, 3, 192, 640)
        args = dict(inputs=(x,), verbose=verbose)
        macs, params = profile(test_model, **args)
        print(f'Params/FLOP: {params * 1e-6:.2f} M, {macs * 1e-9:.2f}G FLOPS')
        total = sum([p.numel() for p in test_model.parameters() if p.requires_grad])
        print("%.2fM" % (total / 1e6))




        test_model_dict_path = os.path.join(opt.load_weights_folder, "monoFlow.pth")
        test_model_dict = torch.load(test_model_dict_path, map_location='cpu')
        model_dict = test_model.state_dict()
        test_model.load_state_dict({k: v for k, v in test_model_dict.items() if k in model_dict})

        test_model_dict.cuda(device="cuda:{}".format(DEVICE_NUM))
        test_model_dict.eval()


        if opt.cv_reproj:
            # use pose model
            if opt.pose_model_type == "separate_resnet":
                pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
                pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
                pose_model_dict = pose_encoder.state_dict()
                pose_encoder_dict = torch.load(pose_encoder_path, map_location='cpu')
                pose_encoder.load_state_dict({k: v for k, v in pose_encoder_dict.items() if k in pose_model_dict})
                pose_encoder.cuda(device="cuda:{}".format(DEVICE_NUM))
                pose_encoder.eval()
                print("--- pose encoder loaded")
                pose_decoder = networks.PoseDecoder(np.array([64, 64, 128, 256, 512]), num_input_features=1,
                    num_frames_to_predict_for=2)
            else:
                pose_decoder = networks.PoseDecoder(np.array([64, 64, 128, 256, 512]), 2)
            pose_path = os.path.join(opt.load_weights_folder, "pose.pth")
            pose_decoder.load_state_dict(torch.load(pose_path, map_location='cpu'))
            pose_decoder.cuda(device="cuda:{}".format(DEVICE_NUM))
            pose_decoder.eval()
            print("--- pose loaded")

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(opt.height, opt.width))

        with torch.no_grad():
            for data in dataloader:
                for key, ipt in data.items():
                    data[key] = ipt.to(DEVICE)


                input_color = data[("color", 0, 0)]
                input_dict = {}
                for i in [-1, 0, 1]:
                    input_dict[("color_aug", i, 0)] = data[("color", i, 0)]

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = test_model(input_dict)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    DEBUG_FLAG = False
    if DEBUG_FLAG:
        opts.eval_mono = True
        opts.load_weights_folder =  "~/tmp/repcv_c/models/weights_14"
    # weights_folder = "/mnt/km-nfs/ns100002-share/zcy-exp/tmp/efficientnet_eff_test_flow/models/weights_"
    # weights_folder = "/mnt/km-nfs/ns100002-share/zcy-exp/tmp/efficientnet_eff/models/weights_"
    weights_folder = "/mnt/km-nfs/ns100002-share/Projects/mono_log/2023-10-24_22-32-44/MonoFlowNet/models/weights_"

    for i in range(89, 80, -1):
        opts.load_weights_folder = weights_folder + str(i)
        evaluate(opts, options.parser)
