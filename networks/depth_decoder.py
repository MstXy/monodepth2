# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from networks.flownet_decoder import DeconvModule
from networks.feature_refine.transformers import MultiHeadAttention, MultiHeadAttentionOne, GroupAttention
from networks.utils.correlation_block import CorrBlock

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, inter_output=False, 
                 drn=False, depth_att=False, depth_cv=False, depth_refine=False, corr_levels=[2,3], 
                 cv_reproj=False, backproject_depth=None, project_3d=None):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.inter_output = inter_output

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # self.num_ch_cv = np.array([None, 256, 512, 512, 1024])
        ## TODO: uncomment for corrs eval
        # self.num_ch_cv = np.array([0, 441, 441, 441, 441]) 

        ## TODO: comment for corrs eval
        self.num_ch_cv_in = np.array([0, 441, 441, 441, 441])
        self.num_ch_cv_out = np.array([0, 64, 64, 128, 256])

        # dilated backbone
        self.drn = drn
        # attention for multi-stage depth decode
        self.depth_att = depth_att
        # cost volume for depth estimation 
        self.depth_cv = depth_cv
        # coarse2fine
        self.depth_refine = depth_refine
        # all corrs levels
        self.corr_levels = corr_levels
        # use cv on warp
        self.cv_reproj = cv_reproj

        if self.cv_reproj:
            self.backproject_depth = backproject_depth
            self.project_3d = project_3d
            self.corrblock = CorrBlock(corr_cfg = dict(
                     type='Correlation',
                     kernel_size=1,
                     max_displacement=10,
                     stride=1,
                     padding=0,
                     dilation_patch=2),
                    scaled = False,
                    act_cfg = dict(type='LeakyReLU', negative_slope=0.1))

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # depth_att
            if self.depth_att:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                # self.convs[("att", i)] = ChannelAttention(num_ch_in)
                # self.convs[("att", i)] = SpatialAttention()
                self.convs[("att", i)] = CS_Block(num_ch_in)
                # self.convs[("att", i)] = MultiHeadAttentionOne(n_head=1, d_model=num_ch_in, d_k=num_ch_in, d_v=num_ch_in)
            
            # cv reproj
            if self.cv_reproj and i + 1 in self.corr_levels:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_mid = num_ch_in
                # num_ch_in += 441
                ## for attention like:
                num_ch_in += self.num_ch_enc[i - 1]
                self.convs[("cv_conv", i)] = ConvBlock(num_ch_in, num_ch_mid)


            # depth cv
            if self.depth_cv and i > 0:
            #     num_ch_in = self.num_ch_cv[i]
            #     num_ch_out = self.num_ch_enc[i - 1]
            #     self.convs[("cv_deconv", i)] = DeconvModule(in_channels=num_ch_in,
            #                                                 out_channels=num_ch_out,
            #                                                 kernel_size=4,
            #                                                 stride=2,
            #                                                 padding=1,
            #                                                 bias=True,
            #                                                 norm_cfg=None,
            #                                                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
                # # TODO: uncomment for corrs eval
                # num_ch_in = self.num_ch_dec[i]
                # if self.use_skips:
                #     num_ch_in += self.num_ch_enc[i - 1]
                # num_ch_mid = num_ch_in
                # # num_ch_in += self.num_ch_enc[i - 1]
                # if i-1 in self.corr_levels:
                #     num_ch_in += self.num_ch_cv[i]
                # self.convs[("cv_conv", i)] = nn.Sequential(
                #                                 nn.Conv2d(num_ch_in, num_ch_mid, 1), # 1*1 conv
                #                                 ConvBlock(num_ch_mid, num_ch_mid)
                #                                 # ConvBlock(num_ch_in, num_ch_mid)
                #                             )

                # TODO: comment for corrs eval
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_mid = num_ch_in
                # num_ch_in += self.num_ch_enc[i - 1]
                if i-1 in self.corr_levels:
                    num_ch_in += self.num_ch_cv_out[i]
                    self.convs[("cv_conv", i, 0)] = nn.Sequential(
                                                    nn.Conv2d(self.num_ch_cv_in[i], self.num_ch_cv_out[i], 1),
                                                    nn.ReLU(inplace=True),
                                                    ConvBlock(self.num_ch_cv_out[i], self.num_ch_cv_out[i])
                                                    # ConvBlock(self.num_ch_cv_in[i], self.num_ch_cv_out[i])
                                                )
                    self.convs[("cv_conv", i, 1)] = nn.Sequential(
                                                    ConvBlock(num_ch_in, num_ch_mid)
                                                    # ConvBlock(num_ch_mid, num_ch_mid)
                                                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
                # #TODO: for eval nodeconv / deconv
                # if self.depth_cv:
                #     # num_ch_in += self.num_ch_enc[i - 1]
                #     num_ch_in += self.num_ch_cv[i]
            if self.depth_refine and i < 3:
                num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            ## IF use simple upsample(interpolate?), comment out below line
            # if self.depth_refine and s != self.scales[-1]: # not the (3rd) level
            #     self.convs[("pred_up", s)] = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True) # upsample using deconv 
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        ## additional depth prediction on level 4
        self.convs[("dispconv", 4)] = Conv3x3(self.num_ch_dec[4], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, corrs=None, inputs=None, outputs=None, adjacent_features=None):
        # # corr1/corr2: Dict type, {"level$#": Tensor, ...}
        # if self.depth_cv:
        #     assert corr1 is not None
        #     assert corr2 is not None
        #     # order of corr's is determined in trainer.py
        #     corr1 = [None] + [v for v in corr1.values()]
        #     corr2 = [None] + [v for v in corr2.values()]

        if self.cv_reproj:
            assert inputs is not None
            assert outputs is not None
            assert adjacent_features is not None

        ## corr: Dict type, {0: Tensor, 1: Tensor, ..., 3: Tensor}

        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if self.drn:
                x = [upsample(x)] if i not in [3, 4] else [x]
            else:
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            if self.depth_refine and i < 3:
                ## deconv
                # x += [self.convs[("pred_up", i)](self.outputs[("c2f", i+1)])] # use pred of prev (deeper) layer
                ## simple upsample
                x += [upsample(self.outputs[("c2f", i+1)])]
        
            if self.cv_reproj and i + 1 in self.corr_levels:
                self.generate_image_pred(inputs, outputs, scale=i+1, adjacent_features=adjacent_features)
                f_m1 = outputs[("color_f", -1, i+1)] # warped feature (actually is two level up)
                f_1 = outputs[("color_f", 1, i+1)]
                f_0 = input_features[i - 1]

                # f_m1 = upsample(f_m1)
                # f_1 = upsample(f_1)
                ## instead, downsample f_0

                corr_m1 = self.corrblock(f_0, f_m1)
                corr_1 = self.corrblock(f_0, f_1)

                # cv = torch.maximum(corr_m1, corr_1)
                cv = (corr_m1 + corr_1) / (2 + 1e-7)

                ## attentin-like process
                B, C, H, W = f_0.shape
                cv = F.softmax(cv.view(B, 441, -1), dim=-1)
                cv = torch.matmul(cv, f_0.view(B, C, -1).permute(0, 2, 1))
                cv = cv.reshape(B,C,21,21)
                cv = F.interpolate(cv, size=(H,W), mode="bilinear", align_corners=False)

                x += [cv]

            x = torch.cat(x, 1)
            
            if self.cv_reproj and i + 1 in self.corr_levels:
                x = self.convs[("cv_conv", i)](x)


            # if self.depth_cv and i > 0:
            #     # TODO: max / sum / concat
            #     # corr = torch.cat([corr1, corr2], dim=1)
            #     # corr = torch.maximum(corr1[i], corr2[i]) 

            #     ## deconv
            #     # corr = self.convs[("cv_deconv", i)](corr)
                
            #     x = torch.cat([x, F.interpolate(corr, x.size()[-2:], mode="bilinear", align_corners=False)], dim=1)
            #     x = self.convs[("cv_conv", i)](x)

            ## all corrs
            if self.depth_cv and i-1 in self.corr_levels: # 4,3,2,1
                # # TODO: uncomment for corrs eval
                # x = torch.cat([x, corrs[i-1]], dim=1)
                # x = self.convs[("cv_conv", i)](x)

                # TODO: comment for corrs eval
                x = torch.cat([x, self.convs[("cv_conv", i, 0)](corrs[i-1])], dim=1)
                x = self.convs[("cv_conv", i, 1)](x)

            if self.depth_att:
                # apply self-attention
                x = self.convs[("att", i)](x)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                tmp = self.convs[("dispconv", i)](x)
                self.outputs[("c2f", i)] = tmp
                self.outputs[("disp", i)] = self.sigmoid(tmp)
                if self.cv_reproj:
                    outputs.update(self.outputs)

            ## additional depth prediction on level 4
            if i == 4:
                tmp = self.convs[("dispconv", i)](x)
                self.outputs[("c2f", i)] = tmp
                self.outputs[("disp", i)] = self.sigmoid(tmp)
                if self.cv_reproj:
                    outputs.update(self.outputs)

        return self.outputs


    def generate_image_pred(self, inputs, outputs, scale, adjacent_features):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        disp = outputs[("disp", scale)]

        disp = upsample(disp) # 24*80 -> 48*160

        # disp = F.interpolate(
        #     disp, [192, 640], mode="bilinear", align_corners=False)
        
        source_scale = scale - 1
        # source_scale = 0

        _, depth = disp_to_depth(disp, 0.1, 100.0)

        frame_ids = [-1, 1] # TODO: mono currently
        for i, frame_id in enumerate(frame_ids):
            
            # if self.opt.full_stereo:
            #     T = outputs[("cam_T_cam", 0, frame_id)]
            # else: # normal one frame stereo
            if isinstance(frame_id, str): # s_0, s_-1, s_1, ...
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](
                cam_points, inputs[("K", source_scale)], T)
       
            outputs[("color_f", frame_id, scale)] = F.grid_sample(
                # use original image
                # inputs[("color", frame_id, source_scale)], 
                # use feature instead
                adjacent_features[frame_id][scale-2],
                pix_coords,
                padding_mode="border",
                align_corners=True)
