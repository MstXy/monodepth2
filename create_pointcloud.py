import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from pointcloud.ply_utils import PLYSaver, normalize_image

import torch.nn.functional as F

from options import MonodepthOptions
from datasets import KITTIOdomDataset
from networks import EfficientEncoder, EfficientDecoder
from utils import readlines


def main(opt):

    output_dir = Path("/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2/pointcloud/output")

    output_dir.mkdir(exist_ok=True, parents=True)

    file_name = "pc_00_192_640.ply"
    use_mask = True
    roi = [
            40,
            256,
            48,
            464
        ] # or None
    # roi = None

    max_d = 20
    min_d = 3

    opt.width = 640 # 512
    opt.height = 192 # 256

    # change roi
    roi = [
        25,
        160,
        40,
        600, # we are using left view, so shift right a bit
    ]

    # setup data_loader instances
    ## Odometry dataset
    opt.data_path = "/mnt/km-nfs/ns100002-share/KITTI_raw/odometry/dataset"
    opt.eval_split = "odom_0" # odom_9 | odom_10 | odom_0
    sequence_id = int(opt.eval_split.split("_")[1])
    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))
    ds = KITTIOdomDataset(opt.data_path, filenames,
                                           opt.height, opt.width,
                                           [0], 4, is_train=False, img_ext='.png')
    data_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)

    # build model architecture
    ## depth
    depth_encoder = EfficientEncoder()
    depth_decoder = EfficientDecoder(depth_encoder.num_ch_enc)


    # prepare model for testing
    DEVICE_NUM = 1
    device = torch.device("cuda:{}".format(DEVICE_NUM))

    ## depth encoder weights
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location='cpu')
    model_dict = depth_encoder.state_dict()
    depth_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    depth_encoder = depth_encoder.to(device)
    depth_encoder.eval()

    ## depth decoder weights
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    depth_decoder = depth_decoder.to(device)
    depth_decoder.eval()

    # Pose 
    ## gt pose
    poses = []
    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    with open(gt_poses_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            poses.append(T_w_cam0)
    poses = torch.from_numpy(np.array(poses)).float().to(device)

    # gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    # gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    # gt_global_poses = np.concatenate(
    #     (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    # gt_global_poses[:, 3, 3] = 1
    # gt_global_poses = torch.from_numpy(gt_global_poses).float().to(device)

    mask_fill = 32

    n = data_loader.batch_size # 1

    target_image_size = (opt.height,opt.width)

    plysaver = PLYSaver(target_image_size[0], target_image_size[1], min_d=min_d, max_d=max_d, batch_size=n, roi=roi, dropout=.75)
    plysaver.to(device)

    pose_buffer = []
    intrinsics_buffer = []
    mask_buffer = []
    keyframe_buffer = []
    depth_buffer = []

    buffer_length = 5
    min_hits = 1
    key_index = buffer_length // 2

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            for key, ipt in data.items():
                    data[key] = ipt.to(device)

            # if not torch.any(pose_distance_thresh(data, spatial_thresh=1)):
            #     continue
            input_color = data[("color", 0, 0)]
            result = depth_decoder(depth_encoder(input_color))

            pred_disp = result[("disp", 0)]
            
            # ## visualize pred
            # plt.imshow(  normalize_image(pred_disp[0].cpu()).permute(1, 2, 0)  )
            # plt.savefig('foo.png')
            # raise ValueError

            ## motion mask --------------
            if "cv_mask" not in result:
                cv_mask = pred_disp.new_zeros(pred_disp.shape)
            # mask = ((result["cv_mask"] >= .1) & (output >= 1 / max_d)).to(dtype=torch.float32)
            mask = (cv_mask >= .1).to(dtype=torch.float32)
            mask = (F.conv2d(mask, mask.new_ones((1, 1, mask_fill+1, mask_fill+1)), padding=mask_fill // 2) < 1).to(dtype=torch.float32)

            # TODO
            pose_buffer += [poses[i]] # gt_global_poses[i] | gt_local_poses[i] 
            intrinsics_buffer += [data[("K", 0)]]
            mask_buffer += [mask]
            keyframe_buffer += [input_color]
            depth_buffer += [pred_disp]

            if len(pose_buffer) >= buffer_length:
                pose = pose_buffer[key_index]
                intrinsics = intrinsics_buffer[key_index]
                keyframe = keyframe_buffer[key_index]
                depth = depth_buffer[key_index]

                mask = (torch.sum(torch.stack(mask_buffer), dim=0) > buffer_length - min_hits).to(dtype=torch.float32)
                if use_mask:
                    depth *= mask

                plysaver.add_depthmap(depth, keyframe, intrinsics, pose)

                del pose_buffer[0]
                del intrinsics_buffer[0]
                del mask_buffer[0]
                del keyframe_buffer[0]
                del depth_buffer[0]

        f = output_dir / file_name
        plysaver.save(f)


if __name__ == '__main__':
    options = MonodepthOptions()
    opts = options.parse()

    weights_folder = "/mnt/km-nfs/ns100002-share/zcy-exp/tmp/efficientnet_eff/models/weights_39"
    opts.load_weights_folder = weights_folder
    main(opts)