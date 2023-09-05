"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

Modified from 
https://github.com/NVlabs/PWC-Net

}

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
# os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory
from .correlation import corr_pwc as correlation
import numpy as np

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class sgu_model(nn.Module):
    def __init__(self):
        super(sgu_model, self).__init__()

        class FlowEstimatorDense_temp(nn.Module):

            def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), ch_out=2):
                super(FlowEstimatorDense_temp, self).__init__()
                N = 0
                ind = 0
                N += ch_in
                self.conv1 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv2 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv3 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv4 = conv(N, f_channels[ind])
                N += f_channels[ind]

                ind += 1
                self.conv5 = conv(N, f_channels[ind])
                N += f_channels[ind]
                self.num_feature_channel = N
                ind += 1
                self.conv_last = conv(N, ch_out, isReLU=False)

            def forward(self, x):
                x1 = torch.cat([self.conv1(x), x], dim=1)
                x2 = torch.cat([self.conv2(x1), x1], dim=1)
                x3 = torch.cat([self.conv3(x2), x2], dim=1)
                x4 = torch.cat([self.conv4(x3), x3], dim=1)
                x5 = torch.cat([self.conv5(x4), x4], dim=1)
                x_out = self.conv_last(x5)
                return x5, x_out
        f_channels_es = (32, 32, 32, 16, 8)
        in_C = 64
        self.warping_layer = WarpingLayer_no_div()
        self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
        self.upsample_output_conv = nn.Sequential(conv(3, 16, kernel_size=3, stride=1, dilation=1),
                                                    conv(16, 16, stride=2),
                                                    conv(16, 32, kernel_size=3, stride=1, dilation=1),
                                                    conv(32, 32, stride=2), )

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        n, c, h, w = flow_init.shape
        n_f, c_f, h_f, w_f = feature_1.shape
        if h != h_f or w != w_f:
            flow_init = upsample2d_flow_as(flow_init, feature_1, mode="bilinear", if_rate=True)
        feature_2_warp = self.warping_layer(feature_2, flow_init)
        input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
        feature, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, :2, :, :]
        inter_mask = x_out[:, 2, :, :]
        inter_mask = torch.unsqueeze(inter_mask, 1)
        inter_mask = torch.sigmoid(inter_mask)
        n_, c_, h_, w_ = inter_flow.shape
        if output_level_flow is not None:
            inter_flow = upsample2d_flow_as(inter_flow, output_level_flow, mode="bilinear", if_rate=True)
            inter_mask = upsample2d_flow_as(inter_mask, output_level_flow, mode="bilinear")
            flow_init = output_level_flow
        flow_up = tools.torch_warp(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
        return flow_init, flow_up, inter_flow, inter_mask
    def output_conv(self, x):
        return self.upsample_output_conv(x)

class PWCDecoder(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4, training=True):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(PWCDecoder, self).__init__()
        self.training = training

        # self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        # self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        # self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        # self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        # self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        # self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        # self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        # self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        # self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        # self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        # self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        # self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        # self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        # self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        # self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        # self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        # self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        # self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.conv_feat_scale1 = conv(64, 16, kernel_size=3, stride=1, padding=1)  # 1->2
        self.conv_feat_scale2 = conv(64, 32, kernel_size=3, stride=1, padding=1)  # 2->3
        self.conv_feat_scale3 = conv(128, 64, kernel_size=3, stride=1, padding=1)  # 3->4
        self.conv_feat_scale4 = conv(256, 96, kernel_size=3, stride=1, padding=1)  # 4->5
        self.conv_feat_scale5 = conv(512, 128, kernel_size=3, stride=1, padding=1)  # 5->6
        self.conv_scale5_to_scale_6 = conv(128, 196, kernel_size=3, stride=2)  # 6->5


        nd = (2*md+1)**2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        self.pred_flow_up1 = conv(in_planes=2, out_planes=16, kernel_size=3, stride=1, padding=1)
        self.pred_flow_up2 = conv(in_planes=16, out_planes=2, kernel_size=3, stride=1, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self, resnet_feat1, resnet_feat2):
        # im1 = x[:,:3,:,:]
        # im2 = x[:,3:,:,:]
        #
        # c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        # c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        # c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        # c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        # c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        # c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        # c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        # c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        # c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        # c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        # c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        # c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        c11, c21 = self.conv_feat_scale1(resnet_feat1[0]), self.conv_feat_scale1(resnet_feat2[0])  # feature_1&2_level_1; (16, 96, 320), 1/2 scale
        c12, c22 = self.conv_feat_scale2(resnet_feat1[1]), self.conv_feat_scale2(resnet_feat2[1])  # feature_1&2_level_2; (32, 48, 160), 1/4 scale
        c13, c23 = self.conv_feat_scale3(resnet_feat1[2]), self.conv_feat_scale3(resnet_feat2[2])  # feature_1&2_level_3; (64, 24, 80), 1/8 scale
        c14, c24 = self.conv_feat_scale4(resnet_feat1[3]), self.conv_feat_scale4(resnet_feat2[3])  # feature_1&2_level_4; (96, 12, 40), 1/16 scale
        c15, c25 = self.conv_feat_scale5(resnet_feat1[4]), self.conv_feat_scale5(resnet_feat2[4])  # feature_1&2_level_5; (128, 6, 20), 1/32 scale
        c16, c26 = self.conv_scale5_to_scale_6(c15), self.conv_scale5_to_scale_6(c25)  # feature_1&2_level_6; (196, 3, 10), 1/64 scale


        corr6 = correlation.FunctionCorrelation(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = correlation.FunctionCorrelation(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = correlation.FunctionCorrelation(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = correlation.FunctionCorrelation(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = correlation.FunctionCorrelation(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        flow2 = self.upsample_conv(flow2)
        flow3 = self.upsample_conv(flow3)
        flow4 = self.upsample_conv(flow4)
        flow5 = self.upsample_conv(flow5)
        flow6 = self.upsample_conv(flow6)
        out = {'level0':flow2, 'level1':flow3, 'level2':flow4, 'level3':flow5, 'level4':flow6}
        return out
    
        # if self.training:
        #     out = {'level0':flow2, 'level1':flow3, 'level2':flow4, 'level3':flow5, 'level4':flow6}
        #     return out
        # else:
        #     return flow2


    def upsample_conv(self, flow_init):
        '''get upsampled output flow'''
        flow_up_1 = F.interpolate(flow_init, scale_factor=2, mode='bilinear', align_corners=True)
        flow_up_1 = self.pred_flow_up2(self.pred_flow_up1(flow_up_1))
        flow_up_2 = F.interpolate(flow_up_1, scale_factor=2, mode='bilinear', align_corners=True)
        flow_up_2 = self.pred_flow_up2(self.pred_flow_up1(flow_up_2))
        return flow_up_2






