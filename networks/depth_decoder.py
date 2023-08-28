# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import math

from collections import OrderedDict
from layers import *
from networks.flownet_decoder import DeconvModule
from networks.feature_refine.transformers import MultiHeadAttention, MultiHeadAttentionOne, GroupAttention, ScaledDotProductAttention
from networks.utils.correlation_block import CorrBlock

# import pdb 
# pdb.set_trace()

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, inter_output=False, 
                 drn=False, depth_att=False, depth_cv=False, depth_refine=False, updown=False, corr_levels=[2,3], n_head=1,
                 cv_reproj=False, backproject_depth=None, project_3d=None, mobile_backbone=None):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.LAST_LAYER_OUTPUT = False # layer 4, for corrs and corrs reprojection

        self.inter_output = inter_output

        self.num_ch_enc = num_ch_enc
        if mobile_backbone == "v3l":
            self.num_ch_dec = np.array([8, 12, 20, 56, 80])
        elif mobile_backbone == "v3s":
            self.num_ch_dec = np.array([8, 8, 12, 24, 48])
        elif mobile_backbone == "v2":
            self.num_ch_dec = np.array([8,12,16,48,80]) # TODO: rm17
        elif mobile_backbone == "vatt":
            self.num_ch_dec = np.array([8, 8, 12, 24, 48])
        elif mobile_backbone == "vatt2":
            self.num_ch_dec = np.array([8,12,16,48,160])
        elif mobile_backbone == "mbvitv3_xs":
            self.num_ch_dec = np.array([16,24,48,80,80])
        else:
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
        # down sample
        self.updown = updown
        # coarse2fine
        self.depth_refine = depth_refine
        # all corrs levels
        self.corr_levels = corr_levels
        # use cv on warp
        self.cv_reproj = cv_reproj

        # decoder
        self.convs = OrderedDict()

        if self.cv_reproj:
            self.backproject_depth = backproject_depth
            self.project_3d = project_3d
            # self.corrblock = CorrBlock(corr_cfg = dict(
            #          type='Correlation',
            #          kernel_size=1,
            #          max_displacement=10,
            #          stride=1,
            #          padding=0,
            #          dilation_patch=2),
            #         scaled = False,
            #         act_cfg = dict(type='LeakyReLU', negative_slope=0.1))
            
            dims=[64,64,128,256,512]
            for i in self.corr_levels:
                self.convs[("corr", i)] = MultiHeadAttention(n_head=n_head, d_model=dims[i-2], d_k=dims[i-2], d_v=dims[i-2])
                # self.convs[("corr", i)] = ScaledDotProductAttention(temperature=np.power(dims[i-2], 0.5))

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # down sample
            if self.updown and i in [4,3,2]:
                self.convs[("down_conv", i)] = Conv3x3(self.num_ch_enc[i-2], self.num_ch_dec[i-1]//2, stride=2)
                # self.convs[("down_conv", i)] = SeparableConv(self.num_ch_enc[i-2], self.num_ch_dec[i-1]//2, stride=2)
                # self.convs[("down_conv", i)] = SeparableConv(self.num_ch_enc[i-2], 64, stride=2)

            # depth_att
            if self.depth_att:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                    if self.updown and i in [4,3,2]:
                        num_ch_in += self.num_ch_dec[i - 1]//2
                        # num_ch_in += 64
                    self.convs[("att", i)] = ChannelAttention(num_ch_in)
                # self.convs[("att", i)] = SpatialAttention()
                # self.convs[("att", i)] = CS_Block(num_ch_in)
                # self.convs[("att", i)] = MultiHeadAttentionOne(n_head=1, d_model=num_ch_in, d_k=num_ch_in, d_v=num_ch_in)
            

            # # TODO: ensemble simple
            # cv reproj
            if not self.depth_att and self.cv_reproj and i + 1 in self.corr_levels:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = num_ch_in
                num_ch_in += self.num_ch_enc[i - 1]
                if self.depth_refine and i < 4:
                    num_ch_in += 1
                self.convs[("cv_conv", i)] = nn.Sequential(
                                                # ConvBlock(num_ch_in, num_ch_out)
                                                nn.Conv2d(num_ch_in,num_ch_out,1),
                                                nn.ELU(inplace=True)
                                            )
                
            if self.depth_att and self.cv_reproj and i + 1 in self.corr_levels:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = num_ch_in
                num_ch_in += self.num_ch_enc[i - 1]
                self.convs[("att", i)] = CS_Block(num_ch_in)
                self.convs[("cv_conv", i)] = nn.Sequential(
                                                ConvBlock(num_ch_in, num_ch_out)
                                            )


            # for corrs concat
            if self.depth_cv and i-1 in self.corr_levels:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = num_ch_in
                num_ch_in += self.num_ch_enc[i - 1]
                self.convs[("cv_conv", i)] = nn.Sequential(
                                                ConvBlock(num_ch_in, num_ch_out)
                                                # nn.Conv2d(num_ch_in,num_ch_out,1),
                                                # nn.ELU(inplace=True)
                                            )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # if self.depth_refine and i < 4:
            #     num_ch_in += 1
            if self.updown and i in [4,3,2]:
                num_ch_in += self.num_ch_dec[i - 1]//2
                # num_ch_in += 64
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            ## IF use simple upsample(interpolate?), comment out below line
            if self.depth_refine: # not the (4th) level
                self.convs[("pred_up", s)] = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True) # upsample using deconv 
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        if self.LAST_LAYER_OUTPUT:
            ## additional depth prediction on level 4
            self.convs[("dispconv", 4)] = Conv3x3(self.num_ch_dec[4], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, corrs=None, inputs=None, outputs=None, adjacent_features=None):

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

            if self.updown and i in [4,3,2]:
                x += [self.convs[("down_conv", i)](input_features[i - 2])] # self.num_ch_dec[i-1]//2

            # if self.depth_refine and i < 4:
            #     ## deconv
            #     x += [self.convs[("pred_up", i)](self.outputs[("c2f", i+1)])] # use pred of prev (deeper) layer
                ## simple upsample
                # x += [upsample(self.outputs[("c2f", i+1)])]
        
            if self.cv_reproj and i + 1 in self.corr_levels:
                if self.depth_refine and i < 4:
                    ## deconv
                    x += [self.convs[("pred_up", i)](self.outputs[("c2f", i+1)])] # use pred of prev (deeper) layer
                self.generate_image_pred(inputs, outputs, scale=i+1, adjacent_features=adjacent_features)
                f_m1 = outputs[("color_f", -1, i+1)] # warped feature (actually is two level up)
                f_1 = outputs[("color_f", 1, i+1)]
                f_0 = input_features[i - 1]
                
                # ## print cosine similarity of two features
                # B,C,H,W = f_0.shape
                # print(F.cosine_similarity(f_0, f_m1, dim=1).sum() / (B*H*W), F.cosine_similarity(f_0, f_1, dim=1).sum() / (B*H*W))

                # f_m1 = upsample(f_m1)
                # f_1 = upsample(f_1)
                ## instead, downsample f_0


                # ## scaled attention
                # B,C,H,W = f_0.shape
                # corr_m1, attn1, _ = self.convs[("corr", i+1)](q=f_0.view(B, C, H*W), k=f_m1.view(B, C, H*W), v=f_m1.view(B, C, H*W))
                # corr_1, attn2, _ = self.convs[("corr", i+1)](q=f_0.view(B, C, H*W), k=f_1.view(B, C, H*W), v=f_1.view(B, C, H*W))

                # # corr_m1, attn1, _ = self.convs[("corr", i+1)](q=f_m1.view(B, C, H*W), k=f_0.view(B, C, H*W), v=f_0.view(B, C, H*W))
                # # corr_1, attn2, _ = self.convs[("corr", i+1)](q=f_1.view(B, C, H*W), k=f_0.view(B, C, H*W), v=f_0.view(B, C, H*W))
                # corr_m1, corr_1 = corr_m1.view(B,C,H,W), corr_1.view(B,C,H,W)

                ## transformer block
                corr_m1, attn1 = self.convs[("corr", i+1)](q=f_0, k=f_m1, v=f_m1)
                corr_1, attn2 = self.convs[("corr", i+1)](q=f_0, k=f_1, v=f_1)

                # ## print cosine similarity of two features
                # print("after:", end='')
                # print(F.cosine_similarity(corr_m1, f_0, dim=1).sum() / (B*H*W), F.cosine_similarity(corr_1, f_0, dim=1).sum() / (B*H*W))

                # cv = torch.maximum(corr_m1, corr_1)
                corr_m1,corr_1 = f_m1,f_1
                cv = (corr_m1 + corr_1) / (2 + 1e-7)

                # import matplotlib.pyplot as plt
                # attn1 = attn1[0].reshape(48, 160, 48, 160) # H, W, H, W
                # attn2 = attn2[0].reshape(48, 160, 48, 160) # H, W, H, W
                # for ci in range(0, attn1.shape[0], 8):
                #     for cj in range(0, attn1.shape[1], 8):
                #         plt.imshow(attn1[ci,cj,:,:].unsqueeze(0).permute(1,2,0).cpu())
                #         plt.savefig("att/attn1_{:02d}{:03d}.png".format(ci,cj))
                #         plt.imshow(attn2[ci,cj,:,:].unsqueeze(0).permute(1,2,0).cpu())
                #         plt.savefig("att/attn2_{:02d}{:03d}.png".format(ci,cj))
                # print("saved. Now halt the program")

                x += [cv]
                # weight = 0.6
                # x[-1] = x[-1] * weight + cv * (1-weight) ## TODO: ensemble simple

            if self.depth_cv and i-1 in self.corr_levels:
                f = x[-1]
                cv = corrs[i-1]
                 
                # for corrs concat
                x += [cv]
                # weight = 0.6
                # x[-1] = x[-1] * weight + cv * (1-weight) ## TODO: ensemble simple

            x = torch.cat(x, 1)
            
            # if self.depth_att and i + 1 in self.corr_levels:
            if self.depth_att and i > 0:
                # apply self-attention
                x = self.convs[("att", i)](x)

            # # TODO: ensemble simple
            if self.cv_reproj and i + 1 in self.corr_levels:
                x = self.convs[("cv_conv", i)](x)

            # For corrs concat
            # all corrs 
            if self.depth_cv and i - 1 in self.corr_levels: # 4,3,2,1
                x = self.convs[("cv_conv", i)](x)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                tmp = self.convs[("dispconv", i)](x)
                self.outputs[("c2f", i)] = tmp
                self.outputs[("disp", i)] = self.sigmoid(tmp)
                if self.cv_reproj:
                    outputs.update(self.outputs)
            
            if self.LAST_LAYER_OUTPUT and i == 4:
                ## additional depth prediction on level 4
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
