import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.utils as tvu
import torchvision.transforms.functional as F
import imageio
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
cmap = plt.get_cmap('viridis')
import monodepth2.datasets.flow_eval_datasets as datasets

from monodepth2.utils import frame_utils
from monodepth2.utils.utils import InputPadder, forward_interpolate
import monodepth2.utils.utils as mono_utils
from monodepth2.options import MonodepthOptions
options = MonodepthOptions()
opt = options.parse()

norm_trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)


@torch.no_grad()
def create_sintel_submission(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img = tvu.flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(f'vis_test/RAFT/{dstype}/'):
                os.makedirs(f'vis_test/RAFT/{dstype}/flow')

            if not os.path.exists(f'vis_test/ours/{dstype}/'):
                os.makedirs(f'vis_test/ours/{dstype}/flow')

            if not os.path.exists(f'vis_test/gt/{dstype}/'):
                os.makedirs(f'vis_test/gt/{dstype}/image')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            image.save(f'vis_test/RAFT/{dstype}/flow/{test_id}.png')
            imageio.imwrite(f'vis_test/gt/{dstype}/image/{test_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_kitti_submission_vis(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Visualizations
        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')
            os.makedirs(f'vis_kitti/image')

        image.save(f'vis_kitti/flow/{test_id}.png')
        imageio.imwrite(f'vis_kitti/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(f'vis_kitti/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())


@torch.no_grad()
def validate_chairs(model, iters=6):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation')
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results



@torch.no_grad()
def validate_sintel(model, inference_func):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_1_2 = inference_func(model, image1, image2)

            flow = padder.unpad(flow_1_2).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_occ(model, iters=6):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def separate_inout_sintel_occ():
    """ Peform validation using the Sintel (train) split """
    dstype = 'clean'
    val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
    # coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    # coords = torch.stack(coords[::-1], dim=0).float()
    # return coords[None].expand(batch, -1, -1, -1)

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, occ, occ_path = val_dataset[val_id]
        _, h, w = image1.size()
        coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords = torch.stack(coords[::-1], dim=0).float()

        coords_img_2 = coords + flow_gt
        out_of_frame = (coords_img_2[0] < 0) | (coords_img_2[0] > w) | (coords_img_2[1] < 0) | (coords_img_2[1] > h)
        occ_union = out_of_frame | occ
        in_frame = occ_union ^ out_of_frame

        # Generate union of occlusions and out of frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'occ_plus_out'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, occ_union.int().numpy() * 255)

        # Generate out-of-frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'out_of_frame'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, out_of_frame.int().numpy() * 255)

        # # Generate in-frame occlusions
        # path_list = occ_path.split('/')
        # path_list[-3] = 'in_frame_occ'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, in_frame.int().numpy() * 255)




@torch.no_grad()
def validate_kitti(log_dir, model, inference_func, epoch_idx=0, opt_main=opt):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training',
                                 root=opt_main.val_data_root)

    out_list, epe_list = [], []
    save_path_dir = os.path.join(log_dir, 'evaluate_flow_kitti')
    os.makedirs(save_path_dir, exist_ok=True)
    for val_id in range(len(val_dataset)):
        image1_ori, image2_ori, flow_gt, valid_gt = val_dataset[val_id]
        image1_ori = image1_ori[None].cuda()
        image2_ori = image2_ori[None].cuda()
        padder = InputPadder(image1_ori.shape, mode='kitti', divided_by=64)
        image1, image2 = padder.pad(image1_ori, image2_ori)
        # if opt_main.norm_trans:
        #     image1 = norm_trans(image1)
        #     image2 = norm_trans(image2)

        flow_ori = inference_func(model, image1, image2)
        flow = padder.unpad(flow_ori).cpu()
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        # vis
        err_map = torch.sum(torch.abs(flow - flow_gt) * valid_gt, dim=0)
        err_map_norm = colors.Normalize(vmin=0, vmax=torch.max(err_map))
        err_map_colored_tensor = mono_utils.plt_color_map_to_tensor(cmap(err_map_norm(err_map)))
        to_save = mono_utils.stitching_and_show(img_list=[image1[0], flow, flow_gt, err_map_colored_tensor, image2[0]],
                                                ver=True, show=False)
        save_path = os.path.join(save_path_dir, str(epoch_idx) + "th_epoch_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".png")
        to_save.save(save_path)

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}

def evaluate_RAFTGMA():
    # from network import RAFTGMA
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()

    if args.dataset == 'separate':
        separate_inout_sintel_occ()
        sys.exit()

    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model, warm_start=True)
    # create_sintel_submission_vis(model, warm_start=True)
    # create_kitti_submission(model)
    # create_kitti_submission_vis(model)

    def RAFTGMA_inference_func(model, image1, image2):
        _, flow_pr = model(image1, image2, iters=args.iters, test_mode=True)
        return flow_pr[0]


    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters)

        elif args.dataset == 'things':
            validate_things(model.module, iters=args.iters)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, RAFTGMA_inference_func)

        elif args.dataset == 'sintel_occ':
            validate_sintel_occ(model.module, iters=args.iters)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, RAFTGMA_inference_func)



def evaluate_flow_MonoFlow(flow_branch='flownet'):
    from monodepth2.networks.MonoFlowNet import MonoFlowNet
    from monodepth2.options import MonodepthOptions
    options = MonodepthOptions()
    opt = options.parse()
    opt.depth_branch = False
    opt.optical_flow = flow_branch  # or 'upflow'
    opt.batch_size = 1

    model = MonoFlowNet(opt)
    # checkpoint_path ='/home/liu/data16t/Projects/test0618_basedON0616/' \
    #                 'monodepth2/networks/log/2023-08-21_13-05-33/mdp/models/weights_98plus29plus86/momoFlow.pth'
    checkpoint_path = '/home/wangshuo/LAB-Backup/Codes/mono_log/2023-09-02_02-17-17/MonoFlowNet/models/weights_71/monoFlow.pth'
    
    print('loading from', checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

    def MonoFlow_inference_func(model, image1, image2):
        input_dict = {}
        # image1 = F.resize(image1, size=[192, 640], antialias=False)
        # image2 = F.resize(image2, size=[192, 640], antialias=False)
        input_dict[("color_aug", -1, 0)], input_dict[("color_aug", 0, 0)], input_dict[("color_aug", 1, 0)] = \
            image1, image2, image1
        out_dict = model(input_dict)
        flow_1_2 = out_dict['flow', -1, 0, 0][0]
        return flow_1_2
    
    # validate_kitti(model, MonoFlow_inference_func, epoch_idx=38)
    validate_kitti('/home/wangshuo/LAB-Backup/Codes/mono_log/flow_eveal_test_2',model, MonoFlow_inference_func, epoch_idx=38)


def evaluate_flow_online(log_dir, checkpoint_path, model_name, epoch_idx, opt_main):
    if model_name == 'MonoFlowNet':
        from monodepth2.networks.MonoFlowNet import MonoFlowNet
        model = MonoFlowNet()
        print('loading from', checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == 'UnFlowNet':
        from monodepth2.networks.UnFlowNet import UnFlowNet
        model = UnFlowNet()
        print('loading from', checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        raise NotImplementedError
    model.cuda()

    def flow_inference_func(model, image1, image2):
        input_dict = {}
        # image1 = F.resize(image1, size=[192, 640], antialias=False)
        # image2 = F.resize(image2, size=[192, 640], antialias=False)
        input_dict[("color_aug", -1, 0)], input_dict[("color_aug", 0, 0)], input_dict[("color_aug", 1, 0)] = \
            image1, image2, image1
        out_dict = model(input_dict)
        flow_1_2 = out_dict['flow', -1, 0, 0][0]
        return flow_1_2

    vaL_kitti_out = validate_kitti(log_dir, model, flow_inference_func, epoch_idx, opt_main)

    return vaL_kitti_out


def evaluate_raft_pytorch_kitti():
    device='cuda'
    si = [192*4, 640*4]
    model = torchvision.models.optical_flow.raft_large(
        weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2,
        progress=True)
    model = model.to(device)
    model = model.eval()
    
    
    trans_raft = torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2.transforms()
    import monodepth2.datasets.flow_eval_datasets as flow_eval_datasets
    from monodepth2.options import MonodepthOptions
    options = MonodepthOptions()
    opt = options.parse()
    from torchvision.utils import flow_to_image
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    import monodepth2.utils.utils as mono_utils

    plt.rcParams["savefig.bbox"] = "tight"
    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]
        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.tight_layout()
    
    
  

    log_path =  os.path.join("/home/liu/tmp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    epe_list = []
    out_list = []
    epe_list_noc = []
    out_list_noc = []
    epe_list_occ = []
    out_list_occ = []
    for occ_noc in ['flow_occ', 'flow_noc']:
        """ Peform validation using the KITTI-2015 (train) split """
        val_dataset = flow_eval_datasets.KITTI_2015_scene_flow(split='training', root=opt.val_data_root, occ_noc=occ_noc)
        save_path_dir = os.path.join(log_path, occ_noc+'evaluate_flow_kitti')
        print(save_path_dir)
        
        os.makedirs(save_path_dir, exist_ok=True)
        for val_id in range(len(val_dataset)):
            input_dict = val_dataset[val_id]
            # {('color', -1, 0): img1, ('color', 0, 0): img2, ('flow', -1, 0): flow, ('valid', -1, 0): valid}
            image1_ori, image2_ori, flow_gt, valid_gt = input_dict[('color', -1, 0)], input_dict[('color', 0, 0)], \
                                                        input_dict[('flow', -1, 0)], input_dict[('valid', -1, 0)]   
            
            
            
            _, H, W = image1_ori.shape
            image1_ori = image1_ori[None].to(device)
            image2_ori = image2_ori[None].to(device)
            flow_gt = flow_gt.to(device)
            valid_gt = valid_gt.to(device)


            
            image1_ori = F.resize(image1_ori, size=si, antialias=False)
            image2_ori = F.resize(image2_ori, size=si, antialias=False)
            image1, image2 = trans_raft(image1_ori, image2_ori)
            
            with torch.no_grad():
                list_of_flows = model(image1, image2)
                predicted_flows = list_of_flows[-1]

                
                flow_imgs = flow_to_image(predicted_flows)
                # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
                image1 = [(img1 + 1) / 2 for img1 in image1]
                # grid = [[img1, flow_img] for (img1, flow_img) in zip(image1, flow_imgs)]
                # plot(grid)

            # flow_1_2[0, :, :] = flow_1_2[0, :, :] / self.opt.width * W
            # flow_1_2[1, :, :] = flow_1_2[1, :, :] / self.opt.height * H 
            # resize_to_val_size = transforms.Resize((H, W), antialias=True)
            # flow = resize_to_val_size(flow_1_2[None])[0]

            
            
            pred_flow = F.resize(predicted_flows[0], size=flow_gt.shape[1:3], antialias=False)
            pred_flow[1, :, :] = pred_flow[1, :, :]  / si[0] * H 
            pred_flow[0, :, :] = pred_flow[0, :, :]  / si[1] * W 
            
            
            epe = torch.sum((pred_flow - flow_gt)**2, dim=0).sqrt()
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            del input_dict

            
            # vis
            err_map = torch.sum(torch.abs(pred_flow - flow_gt) * valid_gt, dim=0).cpu()
            err_map_norm = colors.Normalize(vmin=0, vmax=torch.max(err_map))
            err_map_colored_tensor = mono_utils.plt_color_map_to_tensor(cmap(err_map_norm(err_map)))
            to_save = mono_utils.stitching_and_show(img_list=[pred_flow, flow_gt, err_map_colored_tensor,flow_imgs[0], image1_ori[0], image2_ori[0]],
                                                    ver=True, show=False)
            save_path = os.path.join(save_path_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".png")
            to_save.save(save_path)
            
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            epe_tmp = epe[val].mean().item()
            out_tmp = out[val].cpu().numpy()
            print(occ_noc, ": epe_tmp", epe_tmp, "f1_tmp",  100 * np.mean(out_tmp)) 
            
            
            epe_list.append(epe_tmp)
            out_list.append(out_tmp)
            if occ_noc == 'flow_occ':
                epe_list_occ.append(epe_tmp)
                out_list_occ.append(out_tmp)
            elif occ_noc == 'flow_noc':
                epe_list_noc.append(epe_tmp)
                out_list_noc.append(out_tmp)
            
            

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe_all = np.mean(epe_list)
    f1_all = 100 * np.mean(out_list)
    
    epe_list_occ = np.array(epe_list_occ)
    out_list_occ = np.concatenate(out_list_occ)
    epe_occ = np.mean(epe_list_occ)
    f1_occ = 100 * np.mean(out_list_occ)
    
    epe_list_noc = np.array(epe_list_noc)
    out_list_noc = np.concatenate(out_list_noc)            
    epe_noc = np.mean(epe_list_noc)
    f1_noc = 100 * np.mean(out_list_noc)
    
    print("\n Validation KITTI epe_all, f1_all: %f, %f" % (epe_all, f1_all))
    print(" Validation KITTI epe_occ, f1_occ: %f, %f" % (epe_occ, f1_occ))
    print(" Validation KITTI epe_noc, f1_noc\n: %f, %f" % (epe_noc, f1_noc))
   
 
def evaluate_flow_MonoFlow2(h=256, w=832):
    device = "cuda:0"
    from monodepth2.networks.MonoFlowNet import MonoFlowNet
    import torchvision.transforms as transforms
    import monodepth2.datasets.flow_eval_datasets as flow_eval_datasets
    opt.optical_flow = "arflow"
    opt.depth_branch = False
    opt.height = h
    opt.width = w

    ddp_model = MonoFlowNet(opt)
    load_weights_folder = "/home/liu/Downloads"
    def convert_to_non_ddp(test_model_dict_path):
        # original saved file with DataParallel
        state_dict = torch.load(test_model_dict_path, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        return new_state_dict 




    test_model_dict_path = os.path.join(load_weights_folder, "monoFlow.pth")
    new_state_dict = convert_to_non_ddp(test_model_dict_path)
    ddp_model.load_state_dict(new_state_dict)
    ddp_model.to(device)
    
    
    
    ddp_model.eval()
    out_list, epe_list = [], []
    out_list_occ, epe_list_occ = [], []
    out_list_noc, epe_list_noc = [], []
    
    
    resize_to_train_size = transforms.Resize((h, w), antialias=True)
    print("using h={}, w={}".format(h, w))
    for occ_noc in ['flow_occ', 'flow_noc']:
        """ Peform validation using the KITTI-2015 (train) split """
        val_dataset = flow_eval_datasets.KITTI_2015_scene_flow(split='training', root=opt.val_data_root, occ_noc=occ_noc)
        log_path =  os.path.join("/home/liu/tmp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'MonoFlowNet')
        save_path_dir = os.path.join(log_path, occ_noc+'evaluate_flow_kitti')
        
        os.makedirs(save_path_dir, exist_ok=True)
        for val_id in range(len(val_dataset)):
            input_dict = val_dataset[val_id]
            # {('color', -1, 0): img1, ('color', 0, 0): img2, ('flow', -1, 0): flow, ('valid', -1, 0): valid}
            image1_ori, image2_ori, flow_gt, valid_gt = input_dict[('color', -1, 0)], input_dict[('color', 0, 0)], \
                                                        input_dict[('flow', -1, 0)], input_dict[('valid', -1, 0)]   
            _, H, W = image1_ori.shape
            image1_ori = image1_ori[None].to(device)
            image2_ori = image2_ori[None].to(device)
            flow_gt = flow_gt.to(device)
            valid_gt = valid_gt.to(device)

            
            image1 = resize_to_train_size(image1_ori)
            image2 = resize_to_train_size(image2_ori)
            
            with torch.no_grad():
                input = {}
                if len(opt.frame_ids)==3:
                    input[("color_aug", -1, 0)], input[("color_aug", 0, 0)], input[("color_aug", 1, 0)] = \
                    image1, image2, image1
                elif len(opt.frame_ids)==2:
                    input[("color_aug", -1, 0)], input[("color_aug", 0, 0)] = image1, image2
                    

                out_dict = ddp_model(input)
                flow_1_2 = out_dict['flow', -1, 0, 0][0]
                    
            # flow = padder.unpad(flow_1_2).cpu()
            flow_1_2[0, :, :] = flow_1_2[0, :, :] / w * W
            flow_1_2[1, :, :] = flow_1_2[1, :, :] / h * H 
            resize_to_val_size = transforms.Resize((H, W), antialias=True)
            flow = resize_to_val_size(flow_1_2[None])[0]

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            del input_dict, input, out_dict


            err_map = torch.sum(torch.abs(flow - flow_gt) * valid_gt, dim=0).cpu()
            err_map_norm = colors.Normalize(vmin=0, vmax=torch.max(err_map))
            err_map_colored_tensor = mono_utils.plt_color_map_to_tensor(cmap(err_map_norm(err_map)))
            to_save = mono_utils.stitching_and_show(img_list=[image1_ori[0], flow, flow_gt, err_map_colored_tensor, image2_ori[0]],
                                                    ver=True, show=False)
            save_path = os.path.join(save_path_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".png")
            to_save.save(save_path)
            
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            epe_tmp = epe[val].mean().item()
            out_tmp = out[val].cpu().numpy()
            epe_list.append(epe_tmp)
            out_list.append(out_tmp)

            if occ_noc == 'flow_occ':
                epe_list_occ.append(epe_tmp)
                out_list_occ.append(out_tmp)
            elif occ_noc == 'flow_noc':
                epe_list_noc.append(epe_tmp)
                out_list_noc.append(out_tmp)
            
            

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe_all = np.mean(epe_list)
    f1_all = 100 * np.mean(out_list)
    
    epe_list_occ = np.array(epe_list_occ)
    out_list_occ = np.concatenate(out_list_occ)
    epe_occ = np.mean(epe_list_occ)
    f1_occ = 100 * np.mean(out_list_occ)
    
    epe_list_noc = np.array(epe_list_noc)
    out_list_noc = np.concatenate(out_list_noc)            
    epe_noc = np.mean(epe_list_noc)
    f1_noc = 100 * np.mean(out_list_noc)
    
    print(" Validation KITTI epe_all, f1_all: %f, %f" % (epe_all, f1_all))
    print(" Validation KITTI epe_occ, f1_occ: %f, %f" % (epe_occ, f1_occ))
    print(" Validation KITTI epe_noc, f1_noc\n: %f, %f" % (epe_noc, f1_noc))
    


if __name__ == '__main__':
    # evaluate_flow_MonoFlow()
    # evaluate_raft_pytorch_kitti()
    for i in [-1, 0, 1, 2]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            evaluate_flow_MonoFlow2(h = 256 + i * 64, w = 832 + j * 64)





























