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
    meshgrid = torch.tensor(np.stack(meshgrid, axis=0).astype(np.float32), device=device)
    img2_coor = meshgrid.repeat(B, 1, 1, 1) + flow

    # scale to [0,1]
    img2_coor[:, 0, :, :] = 2.0 * img2_coor[:, 0, :, :] / max(W - 1, 1) - 1.0
    img2_coor[:, 1, :, :] = 2.0 * img2_coor[:, 1, :, :] / max(H - 1, 1) - 1.0

    img1_warped = F.grid_sample(img2, img2_coor.permute(0, 2, 3, 1),  # (B, H, W, C)
                                padding_mode="zeros",
                                align_corners=True)
    return img1_warped


def cal_occ_map(flow_fwd, flow_bwd, scale=1, occ_alpha_1=0.8, occ_alpha_2=0.05, border_mask=True):
    def sum_func(x):
        '''sqrt(flow[0]^2 + flow[1]^2)
        '''
        temp = torch.sum(x ** 2, dim=1, keepdim=True)
        # temp = torch.pow(temp, 0.5)
        return temp

    flow_fwd_warped = torch_warp(flow_bwd, flow_fwd)  # (img, flow)
    flow_bwd_warped = torch_warp(flow_fwd, flow_bwd)
    mag_sq = (sum_func(flow_fwd) + sum_func(flow_bwd) + sum_func(flow_fwd_warped) + sum_func(flow_fwd_warped))/2
    
    flow_fwd_diff = flow_fwd + flow_fwd_warped
    flow_bwd_diff = flow_bwd + flow_bwd_warped
    occ_thresh = 0.01 * mag_sq / 2 + 0.5
    occ_fw = sum_func(flow_fwd_diff) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
    occ_bw = sum_func(flow_bwd_diff) < occ_thresh
    if not border_mask:
        mask_fw = create_outgoing_mask(flow_fwd)
        mask_bw = create_outgoing_mask(flow_bwd)
    else:
        mask_fw = create_border_mask(flow_fwd)
        mask_bw = create_border_mask(flow_fwd)
    fw = mask_fw * occ_fw
    bw = mask_bw * occ_bw

    return fw.float(), bw.float()


def create_outgoing_mask(flow):
    device = flow.device
    num_batch, channel, height, width = flow.shape

    grid_x = torch.arange(width, device=device).view(1, 1, width)
    grid_x = grid_x.repeat(num_batch, height, 1)
    grid_y = torch.arange(height, device=device).view(1, height, 1)
    grid_y = grid_y.repeat(num_batch, 1, width)

    flow_u, flow_v = torch.unbind(flow, 1)
    pos_x = grid_x.type(torch.FloatTensor).to(device) + flow_u
    pos_y = grid_y.type(torch.FloatTensor).to(device) + flow_v
    inside_x = (pos_x <= (width - 1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (height - 1)) & (pos_y >= 0.0)
    inside = inside_x & inside_y
    return inside.type(torch.FloatTensor).unsqueeze(1).to(device)


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = torch_warp(flow21, flow12)
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()




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


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u *= u_scale
        v *= v_scale
        res = torch.cat([u, v], dim=1)
    return res


def log_vis_1(inputs, outputs, img1_idx, img2_idx, j, scale=0):
    ''' top to bottom: 1.curr 2.occ_prev_curr 3.flow_prev_curr 4.prev
    Args:
        img1_idx, img2_idx: (-1,0): prev_and_curr; (0,1) curr_and_next
        j: the j th img in batch
    '''
    img_1_img_2_and_flow = torchvision.transforms.functional.pil_to_tensor(
        stitching_and_show(img_list=[
            inputs['color_aug', img2_idx, scale][j],  # curr
            outputs[('occ', img1_idx, img2_idx, scale)][j].repeat(3, 1, 1),  # occ_prev_curr
            # occ_dict[(img2_idx, img1_idx)][j].repeat(3, 1, 1),  # occ_curr_prev
            outputs[('flow', img1_idx, img2_idx, scale)][j],  # flow_prev_curr(img2_idx=0); flow_curr_next(img2_idx=1)
            # outputs['flow_bwd'][img2_idx][j],  # flow_curr_prev
            inputs['color_aug', img1_idx, scale][j] # prev
        ], ver=True, show=False))
    return img_1_img_2_and_flow

def log_vis_2(inputs, outputs, img1_idx, img2_idx, j, scale=0):
    ''' diff(target, source), diff * mask, warped, source, flow_img1_img2
    Args:
        img1_idx, img2_idx: (-1,0): prev_and_curr; (0,1) curr_and_next
        j: the j th img in batch
    '''
    diff = img_diff_show(outputs[('f_warped', img1_idx, img2_idx, scale)][j], inputs['color_aug', img1_idx, 0][j])
    diff_mask = diff * outputs[('occ', img1_idx, img2_idx, scale)][j].repeat(3, 1, 1)
    aa = outputs[('f_warped', img1_idx, img2_idx, scale)][j]
    source = inputs['color_aug', img1_idx, 0][j]
    flow_img1_img2 = outputs[('flow', img1_idx, img2_idx, scale)][j]

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


def create_mask(tensor, paddings):
    shape = tensor.shape  # B, C, H, W
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner = torch.ones((inner_height, inner_width), device=tensor.device)

    mask2d = torch.nn.ZeroPad2d((paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]))(inner)
    # padding_left, padding_right, padding_top, padding_bottom
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)  # B, 1, H, W
    return mask4d.detach()


def create_border_mask(tensor, border_ratio=0.1):
    num_batch, _, height, width = tensor.shape
    sz = np.ceil(height * border_ratio).astype(np.int).item(0)
    border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
    return border_mask.detach()


def length_sq(x):
    return torch.sum(x**2, 1, keepdim=True)




import pickle
class Tools():
    class pickle_saver():

        @classmethod
        def save_pickle(cls, files, file_path):
            with open(file_path, 'wb') as data:
                pickle.dump(files, data)

        @classmethod
        def load_picke(cls, file_path):
            with open(file_path, 'rb') as data:
                data = pickle.load(data)
            return data