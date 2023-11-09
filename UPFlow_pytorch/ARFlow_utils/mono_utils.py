import os
import hashlib
import zipfile
import torch
import numpy as np
from scipy import interpolate
import torch.nn.functional as F
from PIL import Image
import torchvision
import flow_vis



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