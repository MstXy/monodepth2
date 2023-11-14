import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from monodepth2.utils.utils import torch_warp as flow_warp
import monodepth2.utils.utils as mono_utils

from monodepth2.ARFlow_utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg, opt):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg
        self.opt = opt

    def update_opt(self, new_opt):
        self.opt = new_opt
    
    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1, scale=0):
        '''
        params:
            im1_scaled: B x 3 x H x W
            im1_recons: B x 3 x H x W
            occu_mask1: B x 1 x H x W
            scale: int, usually 0,1,2,3
        output:
            photo_loss: float
            l1_loss
            ssim_loss
            ternary_loss
            visulization of l1, ssim, ternary
        '''
        loss = []
        loss += [(im1_scaled - im1_recons).abs() * occu_mask1 * self.opt.loss_l1_w[scale]]
        loss += [SSIM(im1_recons * occu_mask1,  im1_scaled * occu_mask1) * self.opt.loss_ssim_w[scale]]
        loss += [TernaryLoss(im1_recons * occu_mask1, im1_scaled * occu_mask1) * self.opt.loss_ternary_w[scale]]

        # debug
        # print(torch.sum(loss[2][0]), torch.max(torch.sum(loss[1][0])), torch.min(torch.sum(loss[0][0])))
        tmp1 = torch.sum(loss[0][0], dim=0, keepdim=True) / torch.max(torch.sum(loss[0][0], dim=0))
        tmp2 = torch.sum(loss[1][0], dim=0, keepdim=True) / torch.max(torch.sum(loss[1][0], dim=0))
        tmp3 = torch.sum(loss[2][0], dim=0, keepdim=True) / torch.max(torch.sum(loss[2][0], dim=0))
        
        l1_loss = loss[0].mean() / (occu_mask1.mean() + 1e-6)
        ssim_loss = loss[1].mean()/ (occu_mask1.mean() + 1e-6)
        ternary_loss = loss[2].mean()/ (occu_mask1.mean() + 1e-6)
        
        
        
        # l1_loss = loss[0].mean() / occu_mask1.mean()
        # ssim_loss = loss[1].mean() / occu_mask1.mean()
        # ternary_loss = loss[2].mean() / occu_mask1.mean()
        
        # nan check
        if torch.isnan(l1_loss):
            print("l1_loss is nan")
            print(" \n occu_mask1.mean(), loss[0].mean() \n ", occu_mask1.mean(), loss[0].mean())
            raise ValueError("l1_loss is nan")
        
        
        photo_loss = l1_loss + ssim_loss + ternary_loss
        
        tmp_all = mono_utils.stitching_and_show(
            img_list=[tmp1, tmp2, tmp3], 
            show=False, 
            ver=True)
        return photo_loss, l1_loss, ssim_loss, ternary_loss, tmp_all


    def loss_smooth(self, flow, im1_scaled, scale=0):
        # if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
        #     func_smooth = smooth_grad_2nd
        # else:
        #     func_smooth = smooth_grad_1st
        
        func_smooth = smooth_grad_2nd
        
        
        loss = []
        loss_x, loss_y = func_smooth(flow, im1_scaled, self.cfg.alpha)
        loss += [
            (loss_x.mean() / 2. + loss_y.mean() / 2.) * self.opt.loss_smo2_w[scale] 
            ]
        
        # debug
        tmp1 = torch.sum(loss_x[0], dim=0, keepdim=True) / torch.max(torch.sum(loss_x[0], dim=0))
        tmp2 = torch.sum(loss_y[0], dim=0, keepdim=True) / torch.max(torch.sum(loss_y[0], dim=0))
        # tensor detach
        tmp3 = torch.resize_as_(tmp1.clone().detach(), tmp2.clone().detach()) + tmp2.clone().detach()



        tmp_all = mono_utils.stitching_and_show(
            img_list=[
                im1_scaled[0], tmp1, tmp2, tmp3, 
                (1 - tmp3.resize_as_(im1_scaled[0][0, :].clone().detach())) * im1_scaled[0],
            ],
            show=False, 
            ver=True)
        
        return sum([l.mean() for l in loss]), tmp_all

    def forward(self, pyramid_flows, target, start_occ=False):
        """
        :param pyramid_flows: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]
        
        occ_loss = 0
        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        pyramid_l1_losses = []
        pyramid_ssim_losses = []
        pyramid_ternary_losses = []
        
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []
        self.pyramid_im1_recons = []
        self.pyramid_im2_recons = []
        self.pyramid_photo_loss_vis = []
        self.pyramid_smooth_loss_vis = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):            
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            # im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            # im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)
            im1_recons = flow_warp(im2_scaled, flow[:, :2])
            im2_recons = flow_warp(im1_scaled, flow[:, 2:])
            self.pyramid_im1_recons.append(im1_recons)
            self.pyramid_im2_recons.append(im2_recons)
            use_occ_from_scale0 = False
            
            if use_occ_from_scale0:
                if i == 0:
                    # if self.cfg.occ_from_back:
                    #     occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    #     occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                    # else:
                    #     occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    #     occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
                    # occu_mask1_ = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    # occu_mask2_ = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
                    
                    # debug : chekck the occu_mask
                    for idx in range(3):
                        ttmp = mono_utils.stitching_and_show(
                            img_list=[
                                im1_scaled[idx], im2_scaled[idx], occu_mask1[idx], occu_mask1_[idx]
                            ],
                            show=False,
                            ver=True
                        )
                        
                        tttmp = mono_utils.stitching_and_show(
                            img_list=[
                                im1_scaled[idx], im2_scaled[idx], 
                                im1_scaled[idx] * occu_mask1[idx], 
                                im2_scaled[idx] * occu_mask1_[idx]
                            ],
                            show=False,
                            ver=True
                        )

                else:
                    occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                            (h, w), mode='nearest')
                    occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                            (h, w), mode='nearest')

            else:
                # occu_mask1_ = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                # occu_mask2_ = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                if start_occ:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
                else:
                    occu_mask1 = occu_mask2 = torch.ones_like(im1_scaled[:, 0:1])
                
                # # debug : chekck the occu_mask
                # for idx in range(3):
                #     ttmp = mono_utils.stitching_and_show(
                #         img_list=[
                #             im1_scaled[idx], im2_scaled[idx], occu_mask1[idx], occu_mask1_[idx]
                #         ],
                #         show=False,
                #         ver=True
                #     )
                    
                #     tttmp = mono_utils.stitching_and_show(
                #         img_list=[
                #             im1_scaled[idx], im2_scaled[idx], 
                #             im1_scaled[idx] * occu_mask1[idx], 
                #             im2_scaled[idx] * occu_mask1_[idx]
                #         ],
                #         show=False,
                #         ver=True
                #     )


            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp, l1_loss, ssim_loss, ternary_loss, photo_tmp_all = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1, scale=i)

            if i == 0:
                s = min(h, w)

            loss_smooth, smo_tmp_all = self.loss_smooth(flow[:, :2] / s, im1_scaled, scale=i)
            self.pyramid_smooth_loss_vis.append(smo_tmp_all)
            self.pyramid_photo_loss_vis.append(photo_tmp_all)
            

            # if self.cfg.bk
            loss_warp_bk, l1_loss_bk, ssim_loss_bk, ternary_loss_bk, _ = self.loss_photomatric(im2_scaled, im2_recons, occu_mask2, scale=i)
            loss_smooth_bk, tmp_all = self.loss_smooth(flow[:, 2:] / s, im2_scaled, scale=i)

            loss_warp = (loss_warp + loss_warp_bk) /2
            loss_smooth = (loss_smooth + loss_smooth_bk) /2
            l1_loss = (l1_loss + l1_loss_bk) /2
            ssim_loss = (ssim_loss + ssim_loss_bk) /2
            ternary_loss = (ternary_loss + ternary_loss_bk) /2
            
            pyramid_l1_losses.append(l1_loss)
            pyramid_ssim_losses.append(ssim_loss)
            pyramid_ternary_losses.append(ternary_loss)
            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        # pyramid_warp_losses = [l * w for l, w in
        #                        zip(pyramid_warp_losses, self.opt.loss_l1_w[i])]
        # pyramid_smooth_losses = [l * w for l, w in
        #                          zip(pyramid_smooth_losses,self.opt.loss_smo2_w[i])]
        
        
        # test, add occ_loss
        # occ_loss += torch.mean(2- occu_mask1 - occu_mask2) * 10
        
        
        
        
        # check if grad
        # print("occ_grad", occ_loss, start_occ, occ_loss.requires_grad, occu_mask1.requires_grad, occu_mask2.requires_grad)
        
        
        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = sum(pyramid_smooth_losses)
        total_loss = warp_loss + smooth_loss + occ_loss 

        return total_loss, occ_loss, pyramid_warp_losses,pyramid_l1_losses, pyramid_ssim_losses, pyramid_ternary_losses, \
                pyramid_smooth_losses, pyramid_flows[0].abs().mean(), \
                self.pyramid_smooth_loss_vis, self.pyramid_photo_loss_vis, \
                self.pyramid_occu_mask1, self.pyramid_occu_mask2, \
                self.pyramid_im1_recons, self.pyramid_im2_recons
