# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import torch
import numpy as np
from scipy import interpolate
import torch.nn.functional as F
from PIL import Image
import torchvision
import flow_vis

class InputPadder:
    """ Pads images such that dimensions are divisible by 'divided_by' :int """
    def __init__(self, dims, mode='sintel', divided_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divided_by) + 1) * divided_by - self.ht) % divided_by
        pad_wd = (((self.wd // divided_by) + 1) * divided_by - self.wd) % divided_by
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]
        #  padding for the left, top, right and bottom borders respectively

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


def image_grads(image_batch, stride=1):
    # image_batch: [batch_size, 3, width, height]
    # image_batch_grad: [batch_size, 3, width-1, height-1]
    # image_batch_grad: [batch_size, 3, width, height]
    image_batch_grad = torch.zeros_like(image_batch)
    image_batch_grad[:, :, stride:, :] = image_batch[:, :, stride:, :] - image_batch[:, :, :-stride, :]
    image_batch_grad[:, :, :, stride:] = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
    return image_batch_grad


def torch_warp(img2, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    relation:
        img_2[x + flow[0], y + flow[1]] = img_1[x, y]

    img: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow

    """
    device = img2.device
    B, _, H, W = img2.size()

    meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
    meshgrid = torch.Tensor(np.stack(meshgrid, axis=0).astype(np.float32)).to(device)
    img2_coor = meshgrid.repeat(B, 1, 1, 1) + flow

    # scale to [0,1]
    img2_coor[:, 0, :, :] = 2.0 * img2_coor[:, 0, :, :] / max(W - 1, 1) - 1.0
    img2_coor[:, 1, :, :] = 2.0 * img2_coor[:, 1, :, :] / max(H - 1, 1) - 1.0

    img1_warped = F.grid_sample(img2, img2_coor.permute(0, 2, 3, 1),  # (B, H, W, C)
                                padding_mode="zeros",
                                align_corners=True)
    return img1_warped


def cal_occ_map(flow_fwd, flow_bwd, scale=1, occ_alpha_1=0.7, occ_alpha_2=0.05):
    def sum_func(x):
        '''sqrt(flow[0]^2 + flow[1]^2)
        '''
        temp = torch.sum(x ** 2, dim=1, keepdim=True)
        temp = torch.pow(temp, 0.5)
        return temp
    mag_sq = sum_func(flow_fwd) + sum_func(flow_bwd)
    flow_fwd_warped = torch_warp(flow_bwd, flow_fwd)  # (img, flow)
    flow_bwd_warped = torch_warp(flow_fwd, flow_bwd)
    flow_fwd_diff = flow_fwd + flow_fwd_warped
    flow_bwd_diff = flow_bwd + flow_bwd_warped
    occ_thresh = occ_alpha_1 * mag_sq + occ_alpha_2 / scale
    occ_fw = sum_func(flow_fwd_diff) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
    occ_bw = sum_func(flow_bwd_diff) < occ_thresh
    return occ_fw.float(), occ_bw.float()

############################################
####           img, flow, vis utils      ###
############################################


def merge_images(image1, image2):
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


def tensor_to_pil(input_tensor):
    """
    Args:
        input_tensor: CHW
    Returns:

    """
    assert len(input_tensor.shape) == 3
    trans = torchvision.transforms.ToPILImage()
    out = trans(input_tensor)
    return out


def flow_to_pil(flow_to_vis, to_pil=True):
    flow_to_vis_dup = flow_to_vis.permute(1, 2, 0).clone().detach().cpu().numpy()
    vis_flow = torch.from_numpy(
        flow_vis.flow_to_color(flow_to_vis_dup,
                               convert_to_bgr=False)).permute(2, 0, 1)
    torch_to_pil = torchvision.transforms.ToPILImage()
    if to_pil:
        return torch_to_pil(vis_flow)
    else:
        return vis_flow


def stitching_and_show(img_list, ver=False, show=True):
    '''Stitching imgs , if img in img_list got C=2, process as flow
    Args:
        img_list: list of tensors with dim=[C, H, W]
        ver: vertical or horizontal
    Returns:
    '''
    torch_to_pil = torchvision.transforms.ToPILImage()
    if isinstance(img_list[0], torch.Tensor):
        C, H, W = img_list[0].size()
    else:
        raise TypeError("Not tensor type")
    img_num = len(img_list)
    if not ver:
        stitching = Image.new('RGB', (img_num * W, H))
        i = 0
        for img in img_list:
            if img.size()[0] == 2:  # flow
                to_pil = flow_to_pil
            else:
                to_pil = torch_to_pil
            img = to_pil(img)
            stitching.paste(im=img, box=(i * W, 0))
            i += 1
    else:
        stitching = Image.new('RGB', (W, img_num * H))
        i = 0
        for img in img_list:
            if img.size()[0] == 2:  # flow
                to_pil = flow_to_pil
            else:
                to_pil = torch_to_pil
            img = to_pil(img)
            stitching.paste(im=img, box=(0, i * H))
            i += 1
    if show:
        stitching.show()

    return stitching


def img_diff_show(img1:torch.Tensor, img2:torch.Tensor):
    diff = torch.unsqueeze(torch.sum(torch.abs(img1 - img2), dim=0), dim=0)
    return diff

def plt_color_map_to_tensor(plt_color_map):
    array = (plt_color_map[:, :, :3] * 255).astype(np.uint8).transpose((2, 0, 1))
    tensor = torch.Tensor(array)
    return tensor

def add_img_weighted(img1, img2, alpha1=0.5):
    assert type(img1) == type(img2)
    if isinstance(img1, torch.Tensor):
        if img1.size() != img2.size():
            raise ValueError("Input images must have the same size and number of channels.")
    elif isinstance(img1, np.ndarray):
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same size and number of channels.")
    return img1 * alpha1 + img2 * (1-alpha1)


def log_vis_1(inputs, outputs, occ_dict, img1_idx, img2_idx, j):
    ''' top to bottom: 1.curr 2.occ_prev_curr 3.flow_prev_curr 4.prev
    Args:
        img1_idx, img2_idx: (-1,0): prev_and_curr; (0,1) curr_and_next
        j: the j th img in batch
    '''
    img_1_img_2_and_flow = torchvision.transforms.functional.pil_to_tensor(
        stitching_and_show(img_list=[
            inputs['color_aug', img2_idx, 0][j],  # curr
            occ_dict[(img1_idx, img2_idx)][j].repeat(3, 1, 1),  # occ_prev_curr
            # occ_dict[(img2_idx, img1_idx)][j].repeat(3, 1, 1),  # occ_curr_prev
            outputs['flow_fwd'][img2_idx][j],  # flow_prev_curr(img2_idx=0); flow_curr_next(img2_idx=1)
            # outputs['flow_bwd'][img2_idx][j],  # flow_curr_prev
            inputs['color_aug', img1_idx, 0][j] # prev
        ], ver=True, show=False))
    return img_1_img_2_and_flow

def log_vis_2(inputs, outputs, occ_dict, img1_idx, img2_idx, f_warped_dict, j):
    ''' diff(target, source), diff * mask, warped, source, flow_img1_img2
    Args:
        img1_idx, img2_idx: (-1,0): prev_and_curr; (0,1) curr_and_next
        j: the j th img in batch
    '''
    diff = img_diff_show(f_warped_dict[(img1_idx, img2_idx)][j], inputs['color_aug', img1_idx, 0][j])
    diff_mask = diff * occ_dict[(img1_idx, img2_idx)][j].repeat(3, 1, 1)
    aa = f_warped_dict[(img1_idx, img2_idx)][j]

    source = inputs['color_aug', img1_idx, 0][j]
    flow_img1_img2 = outputs['flow_fwd'][img2_idx][j]

    results = torchvision.transforms.functional.pil_to_tensor(
        stitching_and_show(img_list=[
            diff, diff_mask, aa, source, flow_img1_img2
        ], ver=True, show=False)
    )
    return results

def edge_aware_smoothness_order1(img, pred):

    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    pred_gradients_x = gradient_x(pred)
    pred_gradients_y = gradient_y(pred)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)