import torch.utils.data as data
import torchvision.transforms as transforms

import os
import skimage.transform
import PIL.Image as pil
import numpy as np
from PIL import Image
# import monodepth2.utils.kitti_utils as kitti_utils
import monodepth2.utils.utils as mono_utils

class DiFintDataset(data.Dataset):
    def __init__(self,data_root:str='/home/liu/AD_Data/DiFint/win_id4_share/id4_data/0614/20230614-10-39-43/input/etc',
                    frame_ids:list=[-1, 0, 1],
                    num_scales:int=4,
                    interval_between_sample=1,
                    cam_idxs_list:list=['E', 'F', 'G', 'H'],
                    is_train:bool=True,
                    get_depth:bool=False,
                    get_cam_info:bool=False,
                    do_aug:bool=False,
                    img_height:int=256,
                    img_width:int=640,
                    iterval_between_sample:int=1,
                    interp=Image.BILINEAR
                 ) -> None:
        super(DiFintDataset).__init__()
        self.data_root = data_root
        self.frame_ids = frame_ids
        self.num_frames = len(self.frame_ids)
        self.num_scales = num_scales
        self.is_train = is_train
        self.get_depth = get_depth
        self.get_cam_info = get_cam_info
        self.do_aug = do_aug
        self.img_height = img_height
        self.img_width = img_width
        self.interval_between_sample = interval_between_sample
        self.interp = interp
        self.cam_idxs_list = cam_idxs_list
        self.iterval_between_sample = iterval_between_sample
        
        
        self.mv_data_dir = os.path.join(self.data_root, 'images')
        self.lidar_data_dir = os.path.join(self.data_root, 'pcd')
        self.to_tensor = transforms.ToTensor()
        
        self.init_trans = transforms.Compose([
            # transforms.Resize((self.img_height, self.img_width), interpolation=interp),
            transforms.PILToTensor()])
        
        self.aug_trans = transforms.Compose([
            transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ])
        self.resize=[]
        for i in range(0, self.num_scales):
            h = self.img_height // (2 ** i)
            w = self.img_width // (2 ** i)
            self.resize.append(transforms.Resize((h, w), interpolation=interp, antialias=True))
        self.filenames = self.get_filenames()

    def get_filenames(self):
        pkl_file_name = ''
        for i in self.cam_idxs_list:
            pkl_file_name += i+'_'
        file_names_save_path = os.path.join(self.mv_data_dir, pkl_file_name + '.pkl')
        if os.path.isfile(file_names_save_path):
            data = mono_utils.pickle_saver.load_picke(file_names_save_path)
            return data
        else:
            mv_file_name = 'images.zip'
            mv_dir = os.path.join(self.data_root, mv_file_name[:-4])
            if os.path.isdir(mv_dir):
                pass
            else:
                raise NotImplementedError
                # mv_zip_file = os.path.join(mv_data_dir, mv_file_name)
                # mono_utils.extract_zip(mv_zip_file, mv_dir)

            sample_ls_all_cam = {}
            for cam_idx in self.cam_idxs_list:
                sample_ls_all_cam[cam_idx] = []
                img_dir = os.path.join(mv_dir, cam_idx)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for idx in range(0,len(file_ls) - self.num_frames + 1, self.iterval_between_sample ):
                    name_ls = [None] * self.num_frames
                    for i_th_out_frame in range(self.num_frames):
                        name_ls[i_th_out_frame] = os.path.join(img_dir, file_ls[idx + i_th_out_frame])
                    sample_ls_all_cam[cam_idx].append(name_ls)

            mono_utils.pickle_saver.save_pickle(files=sample_ls_all_cam, file_path=file_names_save_path)
            return sample_ls_all_cam
        
    def __getitem__(self, index):
        '''Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        '''

        if not self.cam_idxs_list == ['E']:
            return self.get_all_mv_data(index)
        else:
            return self.get_monodepth_trainset(index)
        
    def get_monodepth_trainset(self, index):
        out_dict = {}
        for cam_idx in self.cam_idxs_list:
            # if self.get_cam_info:
            #     raise NotImplementedError
            #     out_dict[(cam_idx, 'intrinsics')] = self.get_intrinsics[cam_idx]
            #     out_dict[(cam_idx, 'extrinsics')] = self.get_extrinsics[cam_idx]
            
            name_ls = self.filenames[cam_idx][index]
            for i in range(self.num_frames):
                img = self.pil_loader(name_ls[i])
                # monodepth-like-output-dictioary
                # img = self.init_trans(img)
                img = self.to_tensor(img)
                
                if self.do_aug:
                    img = self.aug_trans(img)
                out_dict['color', self.frame_ids[i], 0] = img
                for scale in range(0, self.num_scales):
                    out_dict['color', self.frame_ids[i], scale] = self.resize[scale](img)
        return out_dict
    
    def get_all_mv_data(self, index):
        out_dict = {}

        for cam_idx in self.cam_idxs_list:
            # if self.get_cam_info:
            #     raise NotImplementedError
            #     out_dict[(cam_idx, 'intrinsics')] = self.get_intrinsics[cam_idx]
            #     out_dict[(cam_idx, 'extrinsics')] = self.get_extrinsics[cam_idx]
            
            name_ls = self.filenames[cam_idx][index]
            for i in range(self.num_frames):
                img = self.pil_loader(name_ls[i])
                # monodepth-like-output-dictioary
                # img = self.init_trans(img)
                img = self.to_tensor(img)
                
                if self.do_aug:
                    img = self.aug_trans(img)
                out_dict[cam_idx, self.frame_ids[i], 0] = img
                for scale in range(0, self.num_scales):
                    out_dict[cam_idx, self.frame_ids[i], scale] = self.resize[scale](img)
        return out_dict

    def get_depth(self,):
        raise NotImplementedError
        
    def pil_loader(self, path):
        return Image.open(path).convert('RGB')
        
    def __len__(self):
        return len(self.filenames[self.cam_idxs_list[0]])//self.iterval_between_sample - self.num_frames
    
if __name__ == "__main__":
    # DiFintDataset test
    data_root = '/home/liu/AD_Data/DiFint/win_id4_share/id4_data/0614/20230614-10-39-43/input/etc'
    dataset = DiFintDataset(data_root=data_root)
    a = (len(dataset)) 
    print(a)
    for data in iter(dataset):
        input_dict={('color', -1, 0): data['E', -1, 0],
                    ('color', 0, 0): data['E', 0, 0],
                    ('color', 1, 0): data['E', 1, 0]}
        
    
