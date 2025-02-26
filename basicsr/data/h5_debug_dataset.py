from torch.utils import data as data
import pandas as pd
import os

from torch.utils.data.dataloader import default_collate
import h5py
# local modules
from basicsr.data.h5_augment import *
from torch.utils.data import ConcatDataset



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


class H5DebugImageDataset(data.Dataset):

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
        return self.h5_file['gen_event']['image{:09d}'.format(index)][:]

    def __init__(self, opt, data_path, return_voxel=True, return_frame=True, return_gt_frame=True,
                 return_mask=False, norm_voxel=True):
        super(H5DebugImageDataset, self).__init__()
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
        
        self.norm_voxel = norm_voxel  # -MAX~MAX -> -1 ~ 1 
        self.h5_file = None
        self.transforms = {}
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None


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

        # diff_weight: 가중치 값을 통해 GT event에 추가할 차이의 비율을 결정 (실험을 위한 옵션)
        self.diff_weight = opt.get('diff_weight', 0.0)

        with h5py.File(self.data_path, 'r') as file:
            self.dataset_len = len(file['images'].keys())

    def __getitem__(self, index, seed=None):
        if index < 0 or index >= self.__len__():
            raise IndexError
        seed = random.randint(0, 2 ** 32) if seed is None else seed
        item = {}
        frame = self.get_frame(index)
        if self.return_gt_frame:
            frame_gt = self.get_gt_frame(index)
            frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=False)

        voxel = self.get_voxel(index)
        gt_voxel = self.transform_voxel(voxel, seed, transpose_to_CHW=False)

        frame = self.transform_frame(frame, seed, transpose_to_CHW=False)
        gen_event = self.get_gen_event(index)  # shape: (6, H, W)

        if self.return_gen_event:
            gen_event = torch.from_numpy(gen_event)
            # transform_gen_event()에서 정규화 및 augmentation 적용
            item['gen_event'] = self.transform_gen_event(gen_event, seed)
            
        if self.return_frame:
            item['frame'] = frame

    
            
        if self.diff_weight != 0.0 and self.return_gen_event:
            # 이미 transform_gen_event를 통해 gen_event가 torch.Tensor 형태로 변환되었으므로 사용
            # 두 텐서는 모두 [-1,1] 범위로 정규화되어 있다고 가정합니다.
            diff = item['gen_event'] - gt_voxel
            mod_voxel = gt_voxel + self.diff_weight * diff

            # gen_evenet 변경!
            item['gen_event'] = mod_voxel


        if self.return_gt_frame:
            item['frame_gt'] = frame_gt

        item['seq'] = self.seq_name
        item['path'] = os.path.join(self.seq_name, 'image{:06d}'.format(index))
        return item


    def __len__(self):
        return self.dataset_len



    def transform_frame(self, frame, seed, transpose_to_CHW=False):
        if self.return_format == "torch":
            if transpose_to_CHW:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255
            else:
                frame = torch.from_numpy(frame).float() / 255
            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)
        return frame
    


    def transform_voxel(self, voxel, seed, transpose_to_CHW):
        if self.return_format == "torch":
            if transpose_to_CHW:
                voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()
            else:
                if self.norm_voxel:
                    voxel = torch.from_numpy(voxel).float() / abs(max(voxel.min(), voxel.max(), key=abs))
                else:
                    voxel = torch.from_numpy(voxel).float()
            if self.vox_transform:
                random.seed(seed)
                voxel = self.vox_transform(voxel)
        return voxel
    
    def transform_gen_event(self, voxel, seed):
        # max_val = torch.max(torch.abs(voxel))
        # voxel = voxel / max_val

        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel
    

    @staticmethod
    def collate_fn(data, event_keys=['events'], idx_keys=['events_batch_indices']):
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
