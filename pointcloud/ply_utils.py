from array import array

import torch
import torch.nn as nn
import numpy as np

import plyfile

##
def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

## Pose Utils
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


class Backprojection(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Backprojection, self).__init__()

        self.N, self.H, self.W = batch_size, height, width

        yy, xx = torch.meshgrid([torch.arange(0., float(self.H)), torch.arange(0., float(self.W))])
        yy = yy.contiguous().view(-1)
        xx = xx.contiguous().view(-1)
        self.ones = nn.Parameter(torch.ones(self.N, 1, self.H * self.W), requires_grad=False)
        self.coord = torch.unsqueeze(torch.stack([xx, yy], 0), 0).repeat(self.N, 1, 1)
        self.coord = nn.Parameter(torch.cat([self.coord, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K) :
        cam_p_norm = torch.matmul(inv_K[:, :3, :3], self.coord[:depth.shape[0], :, :])
        cam_p_euc = depth.view(depth.shape[0], 1, -1) * cam_p_norm
        cam_p_h = torch.cat([cam_p_euc, self.ones[:depth.shape[0], :, :]], 1)

        return cam_p_h

class PLYSaver(torch.nn.Module):
    def __init__(self, height, width, min_d=3, max_d=400, batch_size=1, roi=None, dropout=0):
        super(PLYSaver, self).__init__()
        self.min_d = min_d
        self.max_d = max_d
        self.roi = roi
        self.dropout = dropout
        self.data = []

        self.projector = Backprojection(batch_size, height, width)

    def save(self, file):
        vertices = np.array(list(map(tuple, self.data)),  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertex_el = plyfile.PlyElement.describe(vertices, 'vertex')
        plyfile.PlyData([vertex_el]).write(file)

    def add_depthmap(self, depth: torch.Tensor, image: torch.Tensor, intrinsics: torch.Tensor,
                     extrinsics: torch.Tensor):
        # depth transform
        self.inv_depth_min_max = (0.33, 0.0025)
        depth = (1-depth) * self.inv_depth_min_max[1] + depth * self.inv_depth_min_max[0]
        depth = 1 / depth
        image = (image) * 255 #  (image + .5) * 255
        mask = (self.min_d <= depth) & (depth <= self.max_d)
        if self.roi is not None:
            mask[:, :, :self.roi[0], :] = False
            mask[:, :, self.roi[1]:, :] = False
            mask[:, :, :, self.roi[2]] = False
            mask[:, :, :, self.roi[3]:] = False
        if self.dropout > 0:
            mask = mask & (torch.rand_like(depth) > self.dropout)

        coords = self.projector(depth, torch.inverse(intrinsics))
        coords = extrinsics @ coords
        coords = coords[:, :3, :]
        data_batch = torch.cat([coords, image.view_as(coords)], dim=1).permute(0, 2, 1)
        data_batch = data_batch.view(-1, 6)[mask.view(-1), :]

        self.data.extend(data_batch.cpu().tolist())