# ------------------------------------------------------------------------
# Modified from (https://github.com/TimoStoff/events_contrast_maximization)
# ------------------------------------------------------------------------
from torch.utils import data as data
import pandas as pd
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from torch.utils.data.dataloader import default_collate
import h5py
# local modules
from basicsr.data.h5_augment import *
from torch.utils.data import ConcatDataset


"""
    Data augmentation functions.
    modified from https://github.com/TimoStoff/events_contrast_maximization

    @InProceedings{Stoffregen19cvpr,
    author = {Stoffregen, Timo and Kleeman, Lindsay},
    title = {Event Cameras, Contrast Maximization and Reward Functions: An Analysis},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    } 
"""


def concatenate_h5_datasets(dataset, opt):
    """
    file_path: path that contains the h5 file
    """
    file_folder_path = opt['dataroot']
    

    if os.path.isdir(file_folder_path):
        h5_file_path = [os.path.join(file_folder_path, s) for s in os.listdir(file_folder_path)]
    elif os.path.isfile(file_folder_path):
        h5_file_path = pd.read_csv(file_folder_path, header=None).values.flatten().tolist()
    else:
        raise Exception('{} must be data_file.txt or base/folder'.format(file_folder_path))
    print('Found {} h5 files in {}'.format(len(h5_file_path), file_folder_path))
    datasets = []
    for h5_file in h5_file_path:
        datasets.append(dataset(opt, h5_file))
    return ConcatDataset(datasets)


class H5ImageReconRestorationDataset(data.Dataset):

    def get_frame(self, index):
        """
        Get frame at index
        @param index The index of the frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_gt_frame(self, index):
        """
        Get gt frame at index
        @param index: The index of the gt frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['sharp_images']['image{:09d}'.format(index)][:]

    def get_voxel(self, index):
        """
        Get voxels at index
        @param index The index of the voxels to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['voxels']['voxel{:09d}'.format(index)][:]

    
    def get_gen_event(self, index):
        """
        Get event mask at index
        @param index The index of the event mask to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['gen_event_latent']['image{:09d}'.format(index)][:]


    def __init__(self, opt, data_path, return_voxel=True, return_frame=True, return_gt_frame=True,
            return_mask=False, norm_voxel=True):

        super(H5ImageReconRestorationDataset, self).__init__()
        self.opt = opt
        self.data_path = data_path
        self.seq_name = os.path.basename(self.data_path)
        self.seq_name = self.seq_name.split('.')[0]
        self.return_format = 'torch'

        self.return_gen_event = opt['return_gen_event']
        self.return_voxel = return_voxel
        self.return_frame = return_frame
        self.return_gt_frame = opt.get('return_gt_frame', return_gt_frame)
        self.return_voxel = opt.get('return_voxel', return_voxel)
        self.return_mask = opt.get('return_mask', return_mask)
        
        self.norm_voxel = norm_voxel # -MAX~MAX -> -1 ~ 1 
        self.h5_file = None
        self.transforms={}
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.threshold = 0.01


        if self.opt['norm_voxel'] is not None:
            self.norm_voxel = self.opt['norm_voxel']   # -MAX~MAX -> -1 ~ 1 
        
        if self.opt['return_voxel'] is not None:
            self.return_voxel = self.opt['return_voxel']

        if self.opt['crop_size'] is not None:
            self.transforms["RandomCrop"] = {"size": self.opt['crop_size']}
        
        if self.opt['use_flip']:
            self.transforms["RandomFlip"] = {}

        if 'LegacyNorm' in self.transforms.keys() and 'RobustNorm' in self.transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in self.transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]
                del (self.transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)

        if not self.normalize_voxels:
            self.vox_transform = self.transform

        with h5py.File(self.data_path, 'r') as file:
            self.dataset_len = len(file['images'].keys())

    def __getitem__(self, index):
        # 1) bounds check & seed
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} out of range')
        seed = random.randint(0, 2**32 - 1)

        # 2) load raw numpy arrays
        frame_np    = self.get_frame(index)        # shape [H, W, 3] 또는 [3, H, W]
        frame_gt_np = self.get_gt_frame(index)     # shape [H, W, 3]
        voxel_np    = self.get_voxel(index)        # [C_v, H, W]
        gen_event_np= self.get_gen_event(index)    # [C_e, H_l, W_l]

        # 3) to-tensor & 기본 변환 (정규화, flip 등)    
        frame    = self.transform_frame(frame_np,    seed, transpose_to_CHW=False)
        frame_gt = self.transform_frame(frame_gt_np, seed, transpose_to_CHW=False)
        voxel    = self.transform_voxel(voxel_np,    seed, transpose_to_CHW=False)
        gen_event= torch.from_numpy(gen_event_np)    # [C_e, H_l, W_l]
        gen_event = self.transform_gen_event(gen_event, seed)
        

        # 4) 랜덤 크롭 좌표 계산
        crop_img = 256               # ex) 256
        C_img, H, W = frame.shape
        C_e, H_l, W_l = gen_event.shape

        # ensure alignment
        ratio = H // H_l
        assert H % H_l == 0 and W % W_l == 0, \
            f'Image size ({H},{W}) must be integer multiple of latent size ({H_l},{W_l})'
        assert crop_img % ratio == 0, \
            f'crop_size ({crop_img}) must be divisible by downsample ratio ({ratio})'

        crop_lat = crop_img // ratio                       # ex) 256//8=32

        # latent 도메인에서 크롭 시작점 샘플
        top_l  = random.randint(0, H_l - crop_lat)
        left_l = random.randint(0, W_l - crop_lat)

        # 이미지 도메인 좌표로 변환
        top  = top_l * ratio
        left = left_l * ratio

        # 5) 실제 크롭
        frame     = frame[:,    top: top+crop_img,    left: left+crop_img]
        frame_gt  = frame_gt[:, top: top+crop_img,    left: left+crop_img]
        voxel     = voxel[:,    top: top+crop_img,    left: left+crop_img]
        gen_event = gen_event[:, top_l:top_l+crop_lat, left_l:left_l+crop_lat]

        # 6) 최종 item 구성
        item = {
            'frame':    frame,
            'frame_gt': frame_gt,
            'voxel':    voxel,
            'gen_event':gen_event,
            'seq':      self.seq_name,
            'path':     os.path.join(self.seq_name, f'image{index:06d}')
        }
        return item



    def __len__(self):
        return self.dataset_len

    def transform_frame(self, frame, seed, transpose_to_CHW=False):
        """
        Augment frame and turn into tensor
        @param frame Input frame
        @param seed  Seed for random number generation
        @returns Augmented frame
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255  # H,W,C -> C,H,W

            else:
                frame = torch.from_numpy(frame).float() / 255 # 0-1
            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed, transpose_to_CHW):
        """
        Augment voxel and turn into tensor
        @param voxel Input voxel
        @param seed  Seed for random number generation
        @returns Augmented voxel
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()# H,W,C -> C,H,W

            else:
                if self.norm_voxel:
                    voxel = torch.from_numpy(voxel).float() / abs(max(voxel.min(), voxel.max(), key=abs))  # -1 ~ 1
                else:
                    voxel = torch.from_numpy(voxel).float()

            if self.vox_transform:
                random.seed(seed)
                voxel = self.vox_transform(voxel)
        return voxel
    
    def transform_gen_event(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        @param voxel Input voxel
        @param seed  Seed for random number generation
        @returns Augmented voxel
        """
        
        # normalize voxel to [-1,1]
        # max_val = torch.max(torch.abs(voxel))
        # voxel = voxel / max_val

        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)

        return voxel


    @staticmethod
    def collate_fn(data, event_keys=['events'], idx_keys=['events_batch_indices']):
        """
        Custom collate function for pyTorch batching to allow batching events
        """
        collated_events = {}
        events_arr = []
        end_idx = 0
        batch_end_indices = []
        for idx, item in enumerate(data):
            for k, v in item.items():
                if not k in collated_events.keys():
                    collated_events[k] = []
                if k in event_keys:
                    end_idx += v.shape[0]
                    events_arr.append(v)
                    batch_end_indices.append(end_idx)
                else:
                    collated_events[k].append(v)
        for k in collated_events.keys():
            try:
                i = event_keys.index(k)
                events = torch.cat(events_arr, dim=0)
                collated_events[event_keys[i]] = events
                collated_events[idx_keys[i]] = batch_end_indices
            except:
                collated_events[k] = default_collate(collated_events[k])
        return collated_events
    

