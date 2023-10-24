import os
import sys
file_dir = os.path.dirname(__file__)
parent_project = os.path.dirname(os.path.dirname(file_dir))
sys.path.append(parent_project)

# ==== torch import
import torch
import torch.nn as nn
import monodepth2.networks as networks
from monodepth2.layers import *
import monodepth2.utils.utils as mono_utils
from monodepth2.options import MonodepthOptions
options = MonodepthOptions()
initial_opt = options.parse()

class MonoFlowNet(nn.Module):
    def __init__(self, opt=initial_opt):
        super(MonoFlowNet, self).__init__()
        # configs
        self.opt = opt
        self.corr_feature_level = 2  # for debug
        self.device = self.opt.device
        self.feature_channels = [64, 164, 128, 256, 512]
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # modules
        if self.opt.encoder == "efficientnet":
            self.Encoder = networks.EfficientEncoder()
            self.opt.mobile_backbone = "eff-b0"
            self.use_ema = False
            
        elif self.opt.encoder == "resnet":
            self.Encoder = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        
        if self.opt.depth_branch:
            print("using efficient decoder")
            self.DepthDecoder = networks.EfficientDecoder(self.Encoder.num_ch_enc)

            if self.opt.pose_model_type == "separate_resnet":
                self.PoseEncoder = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.PoseDecoder = networks.PoseDecoder(
                    self.PoseEncoder.num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

        if self.opt.optical_flow:
            self.Corr, self.FlowDecoder = self.build_flow_decoder()
        
    def forward(self, inputs):
        outputs = {}
        curr_bs = inputs[("color_aug", 0, 0)].size()[0]
        all_color_aug = torch.cat(
            [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])  # all images: ([i-1, i, i+1] * [L, R])
        all_features = self.Encoder(all_color_aug)
        all_features = [torch.split(f, curr_bs) for f in all_features]  # separate by frame
        
        features = {}
        for i, k in enumerate(self.opt.frame_ids):
            features[k] = [f[i] for f in all_features]

        if self.opt.depth_branch:
            outputs.update(self.DepthDecoder(features[0]))
            outputs.update(self.predict_poses(inputs, features))

        if self.opt.optical_flow:
            outputs.update(self.predict_flow(inputs, features))
        return outputs

    def cal_fwd_flownet(self, feature_1, feature_2):
        ''' calculate forward flow of feat1 to feat2
        Args:
            feature_1: feature list of previous img (img_1)
            feature_2: feature list of next img (img_2)
            scale: flow output scale
        Returns: flow of img1->img2
        '''
        corr_1_2 = self.Corr(feature_1[self.corr_feature_level], feature_2[self.corr_feature_level])
        mod_1 = {"level" + str(i+1): feature_1[i] for i in range(0, len(feature_1))}
        # features have 5 levels in total: feature_0 is level_1(W/2 * H/2) to level_5(W/32, H/32), level_0 is the initial input size(W, H)
        flow_1_2 = self.FlowDecoder(mod_1, corr_1_2)  # ['level'+str(scale)]
        return flow_1_2

    def cal_fwd_pwcnet(self, feature_1, feature_2):
        return self.FlowDecoder(feature_1, feature_2)

    def cal_fwd_upflow(self, feature_1, feature_2, img_1, img_2):
        '''
        Args:
            feature_1:
            feature_2:
            img_1:
            img_2:
        Returns: flow_1_2
        '''
        start = np.zeros((1, 2, 1, 1))
        start = torch.from_numpy(start).float().to(initial_opt.device)
        input_dict_1_2 = {'x1_features': feature_1, 'x2_features': feature_2,
                               'im1': img_1, 'im2': img_2,
                               'im1_raw': img_1, 'im2_raw': img_2,
                               'im1_sp': img_1, 'im2_sp': img_2,
                               'start': start, 'if_loss': True, 'if_shared_features': True}
        output_dict_1_2 = self.FlowDecoder(input_dict_1_2)
        return output_dict_1_2

    def predict_flow(self, inputs, features):
        outputs = {}
        if len(self.opt.frame_ids) == 3:
            idx_pair_list = [(-1, 0), (0, 1)]
        elif len(self.opt.frame_ids) == 2:
            idx_pair_list = [(-1, 0)]
        else:
            raise NotImplementedError
        
        if self.opt.optical_flow in ['flownet', ]:
            for (img1_idx, img2_idx) in idx_pair_list:
                out_1 = self.cal_fwd_flownet(features[img1_idx], features[img2_idx])
                out_2 = self.cal_fwd_flownet(features[img2_idx], features[img1_idx])
                for scale in self.opt.scales:
                    outputs[('flow', img1_idx, img2_idx, scale)] = out_1['level'+str(scale)]
                    outputs[('flow', img2_idx, img1_idx, scale)] = out_2['level'+str(scale)]
            return outputs

        elif self.opt.optical_flow in ["upflow", ]:
            # todo: DEBUG this part 
            for scale in self.opt.scales:
                for (img1_idx, img2_idx) in idx_pair_list:
                    out_fwd = self.cal_fwd_upflow(
                        features[img1_idx], features[img2_idx], inputs[("color_aug", img1_idx, scale)], 
                        inputs[("color_aug", img2_idx, scale)])
                    out_bwd = self.cal_fwd_upflow(
                        features[img2_idx], features[img1_idx], inputs[("color_aug", img2_idx, scale)],
                        inputs[("color_aug", img1_idx, scale)])
                    outputs[('flow', img1_idx, img2_idx, scale)] = out_fwd['flow', scale]
                    outputs[('flow', img2_idx, img1_idx, scale)] = out_bwd['flow', scale]
                    
            loss_by_upflow += out_fwd['smooth_loss'] + out_fwd['photo_loss'] + out_fwd['census_loss'] + out_fwd['msd_loss'] + \
                                 out_bwd['smooth_loss'] + out_bwd['photo_loss'] + out_bwd['census_loss'] + out_bwd['msd_loss']
            return outputs

        elif self.opt.optical_flow in ["pwc", ]:
            for (img1_idx, img2_idx) in idx_pair_list:
                out_1 = self.cal_fwd_pwcnet(features[img1_idx], features[img2_idx])
                out_2 = self.cal_fwd_pwcnet(features[img2_idx], features[img1_idx])
                for scale in initial_opt.scales:
                    outputs[('flow', img1_idx, img2_idx, scale)] = out_1['level'+str(scale)]
                    outputs[('flow', img2_idx, img1_idx, scale)] = out_2['level'+str(scale)]
            return outputs

        elif self.opt.optical_flow in ["arflow", ]:
            imgs = [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids] # all images: ([i-1, i, i+1])
            features = [self.Encoder(img) for img in imgs]
            outputs = self.FlowDecoder(imgs, features)
            return outputs            
            
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if not isinstance(f_i, str):  # not s_0, s_-1, s_1
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.PoseEncoder(torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.PoseDecoder(pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if not isinstance(i, str)], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.PoseEncoder(pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if not isinstance(i, str)]

            axisangle, translation = self.PoseDecoder(pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if not isinstance(i, str):
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs
        
    def build_flow_decoder(self):
        if self.opt.optical_flow in ["flownet", ]:
            feature_channels = [64, 64, 128, 256, 512]
            Corr = networks.CorrEncoder(
                pyramid_levels=['level3', 'level4', 'level5', 'level6'],
                kernel_size=(3, 3, 3, 3),
                out_channels=(128, 256, 512, 1024),
                inter_channels=(64, 64, 64, 64),
                redir_in_channels=feature_channels[self.corr_feature_level],  # self.corr_feature_level=2, scale at 1/8
                redir_channels=32,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                paddings=(1, 1, 1, 1),
                corr_cfg=dict(
                    type='Correlation',
                    kernel_size=1,
                    max_displacement=10,
                    stride=1,
                    padding=0,
                    dilation_patch=2),
                scaled=False,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
            )


            if initial_opt.feature_type == 0:
                usmc = [1024, 2 + 512 * 3, 2 + 256 * 3, 2 + 128 * 3, 2 + 64 * 2, 2 + 64 * 2],
            elif initial_opt.feature_type == 1:
                usmc = [1024, 2 + 512 * 2, 2 + 256 * 2, 2 + 128 * 2, 2 + 64 * 2, 2 + 64 * 2],
            elif initial_opt.feature_type == 2:
                usmc = [1024, 2 + 512 * 3, 2 + 256 * 3, 2 + 128 * 3, 2 + 64 * 1, 2 + 64 * 1],

            FlowNet = networks.MonoFlowDecoder(
                upsample_module_in_channels=usmc[0],
                upsample_module_out_channels=[512, 256, 128, 64, 64, 32]
            )

            # FlowNet = networks.FlowNetCDecoder(
            #     in_channels=dict(level6=1024, level5=1026, level4=770, level3=386, level2=130),
            #     out_channels=dict(level6=512, level5=256, level4=128, level3=64),
            #     deconv_bias=True,
            #     pred_bias=True,
            #     upsample_bias=True,
            #     norm_cfg=None,
            #     act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            #     init_cfg=[
            #         dict(
            #             type='Kaiming',
            #             layer=['Conv2d', 'ConvTranspose2d'],
            #             a=0.1,
            #             mode='fan_in',
            #             nonlinearity='leaky_relu',
            #             bias=0),
            #         dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
            #     ])

            return Corr, FlowNet

        elif self.opt.optical_flow in ["upflow", ]:
            from monodepth2.UPFlow_pytorch.model.upflow import UPFlow_net
            param_dict = dict(occ_type='for_back_check', alpha_1=0.1, alpha_2=0.5, occ_check_obj_out_all='all',
                              stop_occ_gradient=False, smooth_level='final', smooth_type='edge', smooth_order_1_weight=1,
                              smooth_order_2_weight=0, photo_loss_type='abs_robust', photo_loss_delta=0.4,
                              photo_loss_use_occ=True, photo_loss_census_weight=1, 
                              if_norm_before_cost_volume=True,
                              norm_moments_across_channels=False, norm_moments_across_images=False,
                              multi_scale_distillation_weight=1, multi_scale_distillation_style='upup',
                              multi_scale_photo_weight=1, multi_scale_distillation_occ=True, if_froze_pwc=False,
                              input_or_sp_input=1, if_use_boundary_warp=True, if_use_cor_pytorch=True,
                              if_sgu_upsample=True)

            net_conf = UPFlow_net.config()
            net_conf.update(param_dict)
            net_conf.get_name(print_now=True)
            return None, net_conf()

        elif self.opt.optical_flow in ["pwc", ]:
            return None, networks.PWCDecoder()

        elif self.opt.optical_flow in ["arflow", ]:
            return None, networks.PWCLiteWithResNet().to(self.device)
        
        else: 
            raise NotImplementedError

def model_test():
    import time
    from thop import profile

    initial_opt.depth_branch = True
    initial_opt.optical_flow = 'arflow'
    initial_opt.batch_size = 1

    model = MonoFlowNet().to(initial_opt.device)
    inputs = {}
    for i in initial_opt.frame_ids:
        inputs[("color_aug", i, 0)] = torch.randn(initial_opt.batch_size, 3, 192, 640).to(initial_opt.device)

    verbose = False
    # check params and FLOPs
    # test_model = TestModel(encoder_test, depth_decoder_test)
    
    macs, params = profile(model, inputs=(inputs,), verbose=verbose)
    print(f'Params/FLOP: {params * 1e-6:.2f} M, {macs * 1e-9:.2f}G FLOPS')
    
    total = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("model param counts total: %.2fM" % (total / 1e6))
    
    total = sum([p.numel() for p in model.Encoder.parameters() if p.requires_grad])
    print("model param counts Encoder: %.2fM" % (total / 1e6))
    
    total = sum([p.numel() for p in model.FlowDecoder.parameters() if p.requires_grad])
    print("model param counts FlowDecoder: %.2fM" % (total / 1e6))
    
    total = sum([p.numel() for p in model.DepthDecoder.parameters() if p.requires_grad])
    print("model param counts DepthDecoder: %.2fM" % (total / 1e6))

    total += sum([p.numel() for p in model.PoseEncoder.parameters() if p.requires_grad])
    print("model param counts DepthDecoder + PoseDecoder: %.2fM" % (total / 1e6))


    # ## params and flops evaluation
    # from thop import profile
    # flops, params = profile(model, inputs=(inputs,))
    # print(flops / 1e9, 'GFLOP', params / 1e6, 'M parameters')
    # t1 = time.time()
    # outputs = model(inputs)
    # print('time spent:', time.time() - t1)
    # for k, v in outputs.items():
    #     print(k, v.shape)

if __name__ =="__main__":
    
    # initial_opt.optical_flow='pwc'
    # net = MonoFlowNet()
    # paras = []
    # for name, param in net.named_parameters():
    #     if 'ResEncoder' in name:
    #         print('not update', name)
    #         param.requires_grad = False
            
    #     elif 'FlowDecoder' in name:
    #         print('update only', name)
    #         paras.append(param)
    #         param.requires_grad = True
    #     else:
    #         raise NotImplementedError
            
    # net.cuda()
    # for name, p in net.named_parameters():
    #     print(name, p.requires_grad)
    # # print(net)
    
    model_test()
    






































