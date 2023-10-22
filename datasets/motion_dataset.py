import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torchvision.transforms.functional as F # for aug
from torchvision.transforms.functional import InterpolationMode
import torch.utils.data as data
from torchvision import transforms

import skimage.transform

from utils import readlines

from .mono_dataset import MonoDataset
from .kitti_dataset import KITTIDataset

class MotionDataset(KITTIDataset):

    def __init__(self, *args, **kwargs):
        super(MotionDataset, self).__init__(*args, **kwargs)

        self.load_motionmask = True
       
    def get_image_path(self, folder, frame_index, side):
        # we only have left views, i.e. side = l or 2
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "{:02d}".format(int(folder)),
            "JPEGImages",
            f_str)
        return image_path

    def get_motion(self, folder, frame_index, side, do_flip):

        f_str = "{:06d}{}".format(frame_index, ".npy")
        mask_path = os.path.join(
            self.data_path,
            "{:02d}".format(int(folder)),
            "SegmentationClassNpy",
            f_str)
        
        
        motion_mask = np.load(mask_path)
        # motion_mask = motion_mask.resize(self.full_res_shape, Image.NEAREST)
        
        if do_flip:
            motion_mask = np.fliplr(motion_mask).copy()
        
        motion_mask = np.expand_dims(motion_mask, 0)
        motion_mask = torch.from_numpy(motion_mask)
        motion_mask = F.resize(motion_mask, self.full_res_shape, interpolation=InterpolationMode.NEAREST)

        return motion_mask

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "motion_mask"                           for ground truth motion masks.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if isinstance(i, str):
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + int(i.split("_")[1]), other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # if do_color_aug:
        #     color_aug_params = transforms.ColorJitter.get_params(
        #         self.brightness, self.contrast, self.saturation, self.hue)
        #     color_aug = ColorJitterAug(color_aug_params)
        # else:
        #     color_aug = (lambda x: x)

        # self.preprocess(inputs, color_aug)

        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.to_tensor(self.resize[0](inputs[(n, im, -1)]))
                del inputs[("color", im, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        
        if self.load_motionmask:
            # load motion mask
            motion_mask = self.get_motion(folder, frame_index, side, do_flip)
            inputs["motion_mask"] = motion_mask


        if "s_0" in self.frame_idxs: # s_0 is the base case of using stereo image
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs



if __name__ == "__main__":

    data_path = "/mnt/km-nfs/ns100002-share/zcy-exp/kitti_movements"

    # file format: folder & frame_id & l

    train_filenames = readlines(
        os.path.join("/mnt/km-nfs/ns100002-share/zcy-exp/monodepth2", "splits/motion/train.txt"))
    
    train_dataset = MotionDataset(data_path, train_filenames, height=192, width=640,
                               frame_idxs=[0, -1, 1], num_scales=4, is_train=True, img_ext='.jpg')

    print(train_dataset[0])