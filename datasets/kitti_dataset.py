# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch.utils.data
import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torchvision.transforms as transforms
from monodepth2.utils.kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import numpy as np
from PIL import Image
import monodepth2.utils.utils as mono_utils
from monodepth2.options import MonodepthOptions
options = MonodepthOptions()
opt = options.parse()


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
        color = self.loader(self.get_image_path(folder, frame_index, side))

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



import random
from monodepth2.utils.utils import Tools as tools
class KITTI_MV_2015(torch.utils.data.Dataset):
    def __init__(self, mv_data_dir, frame_ids=[-1, 0], height=192, witdth=640, mv_type='2015'):  # todo mv_type='2012' or '2015'
        super(KITTI_MV_2015, self).__init__()

        self.to_tensor = transforms.ToTensor()
        self.is_train = True
        self.frame_ids = frame_ids
        self.mv_data_dir = mv_data_dir
        self.filenames = self.mv15_data_get_file_names()[mv_type]
        self.resize_0 = transforms.Resize((height, witdth), interpolation=Image.ANTIALIAS)
    
    def __len__(self):
        return len(self.filenames) - 1
    
    def __getitem__(self, index):
        inputs = {}
        # do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        inputs[("color", -1, -1)] = self.get_color(index, 0, do_flip)
        inputs[("color", 0, -1)] = self.get_color(index, 1, do_flip)
            
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.to_tensor(self.resize_0(inputs[(n, im, -1)]))
                del inputs[("color", im, -1)]

        return inputs

    def get_color(self, index,  frame_id, do_flip=False):
        color = self.pil_loader(self.filenames[index][frame_id])
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    
    def mv15_data_get_file_names(self,):
        file_names_save_path = os.path.join(self.mv_data_dir, 'kitti_flow_2015_multiview_file_names.pkl')
        if os.path.isfile(file_names_save_path):
            data = tools.pickle_saver.load_picke(file_names_save_path)
            return data
        else:
            # mv_2012_file_name = 'data_stereo_flow_multiview.zip'
            # mv_2012_zip_file = os.path.join(kitti_flow_dir_2012, mv_2012_file_name)
            # mv_2012_dir = os.path.join(kitti_flow_dir_2012, mv_2012_file_name[:-4])
            # if os.path.isdir(mv_2012_dir):
            #     pass
            # else:
            #     tools.extract_zip(mv_2012_zip_file, mv_2012_dir)

            mv_2015_file_name = 'data_scene_flow_multiview.zip'
            mv_2015_zip_file = os.path.join(self.mv_data_dir, mv_2015_file_name)
            mv_2015_dir = os.path.join(self.mv_data_dir, mv_2015_file_name[:-4])
            if os.path.isdir(mv_2015_dir):
                pass
            else:
                tools.extract_zip(mv_2015_zip_file, mv_2015_dir)

            def read_mv_data(d_path):
                sample_ls = []
                for sub_dir in ['testing', 'training']:
                    img_dir = os.path.join(d_path, sub_dir, 'image_2')
                    file_ls = os.listdir(img_dir)
                    file_ls.sort()
                    print(' ')
                    for ind in range(len(file_ls) - 1):
                        name = file_ls[ind]
                        nex_name = file_ls[ind + 1]
                        id_ = int(name[-6:-4])
                        id_nex = int(nex_name[-6:-4])
                        if id_ != id_nex - 1 or 12 >= id_ >= 9 or 12 >= id_nex >= 9:
                            pass
                        else:
                            file_path = os.path.join(img_dir, name)
                            file_path_nex = os.path.join(img_dir, nex_name)
                            sample_ls.append((file_path, file_path_nex))
                return sample_ls

            filenames = {}
            # filenames['2012'] = read_mv_data(mv_2012_dir)
            filenames['2015'] = read_mv_data(mv_2015_dir)
            tools.pickle_saver.save_pickle(files=filenames, file_path=file_names_save_path)
            return filenames


    def pil_loader(self, path):
        return Image.open(path).convert('RGB')

























