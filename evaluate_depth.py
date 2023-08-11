from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

from networks.feature_refine import APNB, AFNB, ASPP, PPM, SelfAttention
from networks.dilated_resnet import dilated_resnet18
from networks.mobilenet_encoder import MobileNetV3, MobileNetV2, MobileNetAtt, MobileNetAtt2

from layers import BackprojectDepth, Project3D, transformation_from_parameters

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

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


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")

        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location='cpu')

        all_corr_levels = [2,3,4]
        if opt.load_weights_folder.split("/")[-3].split("_")[0] == "depcv":
            print("using cost volume for depth")
            opt.depth_cv = True
            import cupy as cp
            cp.cuda.Device(DEVICE_NUM).use()
            # corrEncoder = networks.CorrEncoder(
            #         in_channels=473,
            #         pyramid_levels=['level3', 'level4', 'level5', 'level6'],
            #         kernel_size=(3, 3, 3, 3),
            #         num_convs=(1, 2, 2, 2),
            #         out_channels=(256, 512, 512, 1024),
            #         redir_in_channels=256,
            #         redir_channels=32,
            #         strides=(1, 2, 2, 2),
            #         dilations=(1, 1, 1, 1),
            #         corr_cfg=dict(
            #             type='Correlation',
            #             kernel_size=1,
            #             max_displacement=10,
            #             stride=1,
            #             padding=0,
            #             dilation_patch=2),
            #         scaled=False,
            #         conv_cfg=None,
            #         norm_cfg=None,
            #         act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
            #     )
            ## all corrs
            # corrEncoder = networks.corr_encoder.CorrEncoderSimple(levels=[1,2,3])
            corrEncoder = networks.corr_encoder.CorrEncoderAtt(levels=all_corr_levels, n_head=opt.nhead)
            corr_path = os.path.join(opt.load_weights_folder, "corr.pth")
            corrEncoder.load_state_dict(torch.load(corr_path, map_location='cpu'))
            corrEncoder.cuda(device="cuda:{}".format(DEVICE_NUM))
            corrEncoder.eval()
        else:
            opt.depth_cv = False

        if opt.load_weights_folder.split("/")[-3].split("_")[0] == "repcv":
            print("using reprojection cv")
            opt.cv_reproj = True
            import cupy as cp
            cp.cuda.Device(DEVICE_NUM).use()

            if opt.load_weights_folder.split("/")[-3].split("_")[1] == "c2f":
                print("using coarse-2-fine")
                opt.coarse2fine = True
            else:
                opt.coarse2fine = False
        else:
            opt.cv_reproj = False

        BATCH_SIZE = 16

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0] if (not opt.depth_cv and not opt.cv_reproj) else [0, -1, 1], 4, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=True)

        # alternate backbone
        if opt.load_weights_folder.split("/")[-3].split("_")[0] == "mb":
            print("using mobilenetv3")
            if opt.load_weights_folder.split("/")[-3].split("_")[1] == "3s":
                print("using: Small")
                encoder = MobileNetV3(model_type="small")
                opt.mobile_backbone = "v3s"
            elif opt.load_weights_folder.split("/")[-3].split("_")[1] == "3l":
                print("using: Large")
                encoder = MobileNetV3(model_type="large")
                opt.mobile_backbone = "v3l"
            elif opt.load_weights_folder.split("/")[-3].split("_")[1] == "2":
                print("using: Large")
                encoder = MobileNetV2()
                opt.mobile_backbone = "v2"
            elif opt.load_weights_folder.split("/")[-3].split("_")[1] == "att":
                print("using +Attention")
                encoder = MobileNetAtt(opt.nhead)
                opt.mobile_backbone = "vatt"
            elif opt.load_weights_folder.split("/")[-3].split("_")[1] == "att2":
                print("using v2+Attention")
                encoder = MobileNetAtt2(opt.nhead)
                opt.mobile_backbone = "vatt2"
        else:
            opt.mobile_backbone = None
            if opt.load_weights_folder.split("/")[-3].split("_")[0] == "drn":
                print("using dilated ResNet")
                opt.drn = True
                encoder = dilated_resnet18()
            else:
                opt.drn = False
                encoder = networks.ResnetEncoder(opt.num_layers, False)
        
        if opt.load_weights_folder.split("/")[-3].split("_")[0] == "depatt":
            print("using depth attention")
            opt.depth_att = True
        else:
            opt.depth_att = False

        if opt.load_weights_folder.split("/")[-3].split("_")[0] == "c2f":
            print("using coarse-2-fine")
            opt.coarse2fine = True
        else:
            opt.coarse2fine = False

        # refiner: ---------
        if opt.eval_refiner:
            print("Eval with refiner module.")
            
            refiner_name = opt.load_weights_folder.split("/")[-3].split("_")[0]
            print(refiner_name)
            if refiner_name == "att":
                refiner_name = "self_att"
            if refiner_name == "drn":
                print("we are using dilated resnet")
                refiner_name = opt.load_weights_folder.split("/")[-3].split("_")[1]
                print("now use {} refiner".format(refiner_name))

            if opt.single_refiner:
                refiner = None

                if refiner_name == "self_att":
                    print("evaluting with self att")
                    dropout = 0.1
                    refiner = SelfAttention(num_heads=1, dropout=dropout)

                if refiner_name == "psp":
                    print("evaluting with psp")
                    fea_dim = 512
                    bins = [1,2,3,6]
                    dropout = 0.3
                    refiner = PPM(in_dim=fea_dim, reduction_dim=int(fea_dim/len(bins)), bins=bins, dropout=dropout)

                if refiner_name == "aspp":
                    print("evaluting with aspp")
                    fea_dim = 512
                    atrous_rates=[6, 12, 18]
                    refiner = ASPP(in_ch=fea_dim, mid_ch=fea_dim//2, out_ch=fea_dim, rates=atrous_rates)

                if refiner_name == "apnb":
                    print("evaluting with apnb")
                    fea_dim = 512
                    dropout = 0.05
                    refiner = APNB(in_channels=fea_dim, out_channels=fea_dim, key_channels=256, value_channels=256, dropout=dropout, norm_type="batchnorm")

                if refiner_name == "afnb":
                    print("evaluting with afnb")
                    low_fea_dim = 256
                    high_fea_dim = 512
                    fea_dim = 512
                    kv_dim = 128
                    dropout = 0.05
                    refiner = AFNB(low_in_channels=low_fea_dim, high_in_channels=high_fea_dim, 
                                            out_channels=fea_dim, key_channels=kv_dim, value_channels=kv_dim, dropout=dropout, norm_type="batchnorm")
                
                refiner_path = os.path.join(opt.load_weights_folder, "{}.pth".format(refiner_name))
                refiner.load_state_dict(torch.load(refiner_path, map_location='cpu'))
                refiner.cuda(device="cuda:{}".format(DEVICE_NUM))
                refiner.eval()

            else:
                if refiner_name == "ann":
                    # AFNB: fusion
                    low_fea_dim = 256
                    high_fea_dim = 512
                    fea_dim = 512
                    kv_dim = 128
                    dropout = 0.05
                    afnb = AFNB(low_in_channels=low_fea_dim, high_in_channels=high_fea_dim, 
                                            out_channels=fea_dim, key_channels=kv_dim, value_channels=kv_dim, dropout=dropout, norm_type="batchnorm")

                    # APNB: self-att
                    fea_dim = 512
                    dropout = 0.05
                    apnb = APNB(in_channels=fea_dim, out_channels=fea_dim, key_channels=256, value_channels=256, dropout=dropout, norm_type="batchnorm")

                    afnb_path = os.path.join(opt.load_weights_folder, "afnb.pth")
                    apnb_path = os.path.join(opt.load_weights_folder, "apnb.pth")
                    afnb.load_state_dict(torch.load(afnb_path, map_location='cpu'))
                    apnb.load_state_dict(torch.load(apnb_path, map_location='cpu'))
                    refiner = nn.Sequential(afnb, apnb)
                    refiner.cuda(device="cuda:{}".format(DEVICE_NUM))
                    refiner.eval()
                
                if refiner_name == "asppatt":
                    print("eval using aspp + att")
                    fea_dim = 512
                    atrous_rates=[6, 12, 18]
                    aspp = ASPP(in_ch=fea_dim, mid_ch=fea_dim//2, out_ch=fea_dim, rates=atrous_rates)

                    dropout = 0.1
                    att = SelfAttention(num_heads=1, dropout=dropout)

                    aspp_path = os.path.join(opt.load_weights_folder, "aspp.pth")
                    att_path = os.path.join(opt.load_weights_folder, "self_att.pth")
                    aspp.load_state_dict(torch.load(aspp_path, map_location='cpu'))
                    att.load_state_dict(torch.load(att_path, map_location='cpu'))
                    refiner = nn.Sequential(aspp, att)
                    refiner.cuda(device="cuda:{}".format(DEVICE_NUM))
                    refiner.eval()

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


        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, drn=opt.drn, 
                                              depth_att=opt.depth_att, depth_cv=opt.depth_cv, depth_refine=opt.coarse2fine,
                                              corr_levels = all_corr_levels, n_head=opt.nhead,
                                              cv_reproj=opt.cv_reproj, backproject_depth=backproject_depth, project_3d=project_3d,
                                              mobile_backbone=opt.mobile_backbone) 

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

        encoder.cuda(device="cuda:{}".format(DEVICE_NUM))
        encoder.eval()
        depth_decoder.cuda(device="cuda:{}".format(DEVICE_NUM))
        depth_decoder.eval()

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

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                for key, ipt in data.items():
                    data[key] = ipt.to(DEVICE)


                input_color = data[("color", 0, 0)]

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = None
                if not opt.eval_refiner:
                    if opt.depth_cv:
                        x_1_m = encoder(data[("color", -1, 0)])
                        x_0 = encoder(input_color)
                        x_1 = encoder(data[("color", 1, 0)])
                        # output = depth_decoder(x_0, corrEncoder(x_1_m[3], x_0[3]), corrEncoder(x_1[3], x_0[3]))

                        ## use corr simple & multi corr
                        features = {}
                        features[-1] = x_1_m
                        features[0] = x_0
                        features[1] = x_1
                        corrs = corrEncoder(features)
                        output = depth_decoder(x_0, corrs)
                    
                    elif opt.cv_reproj:
                        x_1_m = encoder(data[("color", -1, 0)])
                        x_0 = encoder(input_color)
                        x_1 = encoder(data[("color", 1, 0)])

                        features = {}
                        features[-1] = x_1_m
                        features[0] = x_0
                        features[1] = x_1

                        outputs = {}
                        frame_ids = [0,1,-1]
                        if opt.pose_model_type == "shared":
                            pose_feats = {f_i: features[f_i] for f_i in frame_ids}
                        else:
                            pose_feats = {f_i: data["color_aug", f_i, 0] for f_i in frame_ids}

                        for f_i in frame_ids[1:]:
                            if not isinstance(f_i, str): # not s_0, s_-1, s_1
                                # To maintain ordering we always pass frames in temporal order
                                if f_i < 0:
                                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                                else:
                                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                                if opt.pose_model_type == "separate_resnet":
                                    pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]

                                axisangle, translation = pose_decoder(pose_inputs)

                                outputs[("axisangle", 0, f_i)] = axisangle
                                outputs[("translation", 0, f_i)] = translation

                                # Invert the matrix if the frame id is negative
                                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                                
                        print("--- pose predicted")
                        output = depth_decoder(features[0], inputs=data, outputs=outputs, adjacent_features={-1:features[-1],1:features[1]}) # only predict depth for current frame (monocular)
                        # output = depth_decoder(encoder(input_color))

                    else:
                        output = depth_decoder(encoder(input_color))
                else:
                    # refiner:
                    print("using: " + refiner_name)
                    output = encoder(input_color)
                    output = {0: output}
                    output = depth_decoder(refiner(output)[0])

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
    evaluate(opts)
