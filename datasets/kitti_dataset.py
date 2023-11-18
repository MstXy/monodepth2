# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

import torch

# util for intrinsic retrieval
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        # handle first image's prev frame:
        if int(frame_index) < 0:
            frame_index = 1
        
        # handle last image's next frame:
        try:
            color = self.loader(self.get_image_path(folder, frame_index, side))
        except FileNotFoundError:
            color = self.loader(self.get_image_path(folder, frame_index - 2 , side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)
        
        # Load the calibration file for image_2
        line = self.filenames[0].split()
        folder = line[0]
        calib_filepath = os.path.join(self.data_path, "sequences/{:02d}".format(int(folder)), 'calib.txt')
        filedata = read_calib_file(calib_filepath)
        # Create 3x4 projection matrices
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        # Compute the camera intrinsics
        self.K = P_rect_20[0:3, 0:3]
        self.Ks, self.inv_Ks = [], []
        # Because of cropping and resizing of the frames, we need to recompute the intrinsics
        self.process_intrinsics()
        self.odom = True

    def process_intrinsics(self):
        # Because of cropping and resizing of the frames, we need to recompute the intrinsics
        P_cam = self.K
        orig_size = (self.full_res_shape[1], self.full_res_shape[0])

        for scale in range(self.num_scales):
            target_image_size = ((self.height  // (2 ** scale)), (self.width  // (2 ** scale)))
            r_orig = orig_size[0] / orig_size[1]
            r_target = target_image_size[0] / target_image_size[1]

            if r_orig >= r_target:
                new_height = r_target * orig_size[1]
                c_x = P_cam[0, 2] / orig_size[1]
                c_y = (P_cam[1, 2] - (orig_size[0] - new_height) / 2) / new_height
                rescale = orig_size[1] / target_image_size[1]
            else:
                new_width = orig_size[0] / r_target
                c_x = (P_cam[0, 2] - (orig_size[1] - new_width) / 2) / new_width
                c_y = P_cam[1, 2] / orig_size[0]
                rescale = orig_size[0] / target_image_size[0]

            f_x = P_cam[0, 0] / target_image_size[1] / rescale
            f_y = P_cam[1, 1] / target_image_size[0] / rescale

            intrinsics_mat = torch.zeros((4, 4))
            intrinsics_mat[0, 0] = f_x * target_image_size[1]
            intrinsics_mat[1, 1] = f_y * target_image_size[0]
            intrinsics_mat[0, 2] = c_x * target_image_size[1]
            intrinsics_mat[1, 2] = c_y * target_image_size[0]
            intrinsics_mat[2, 2] = 1
            intrinsics_mat[3, 3] = 1
            K = intrinsics_mat

            inv_K = torch.linalg.pinv(K)

            self.Ks.append(K)
            self.inv_Ks.append(inv_K)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
