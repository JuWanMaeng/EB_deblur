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


class H5ImageDataset(data.Dataset):

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

    def get_mask(self, index):
        """
        Get event mask at index
        @param index The index of the event mask to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['masks']['mask{:09d}'.format(index)][:]
    
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

        super(H5ImageDataset, self).__init__()
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
        self.threshold = 0.005


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


        self.get_peak_point()


    def __getitem__(self, index, seed=None):

        if index < 0 or index >= self.__len__():
            raise IndexError
        seed = random.randint(0, 2 ** 32) if seed is None else seed
        item={}
        frame = self.get_frame(index)
        if self.return_gt_frame:
            frame_gt = self.get_gt_frame(index)
            frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=False)

        # voxel = self.get_voxel(index)
        # item['voxel'] = self.transform_voxel(voxel, seed, transpose_to_CHW=False)

    
        frame = self.transform_frame(frame, seed, transpose_to_CHW=False)  # to tensor

        # gen_event = self.get_voxel(index)
        gen_event = self.get_gen_event(index)  


        # gen_event[np.abs(gen_event) <= self.threshold] = 0
        

        # normalize RGB
        if self.mean is not None or self.std is not None:
            normalize(frame, self.mean, self.std, inplace=True)
            if self.return_gt_frame:
                normalize(frame_gt, self.mean, self.std, inplace=True)

        if self.return_gen_event:
            gen_event = torch.from_numpy(gen_event)
            item['gen_event'] = self.transform_gen_event(gen_event,seed)
        if self.return_frame:
            item['frame'] = frame
        if self.return_gt_frame:
            item['frame_gt'] = frame_gt
        
            
        item['seq'] = self.seq_name
        item['path'] = os.path.join(self.seq_name, 'image{:06d}'.format(index))


        return item
    
    ############ noise debugging #################
    # def __getitem__(self, index, seed=None):
    #     if index < 0 or index >= self.__len__():
    #         raise IndexError
    #     seed = random.randint(0, 2 ** 32) if seed is None else seed
    #     item = {}
        
    #     # 프레임과 GT 프레임 불러오기
    #     frame = self.get_frame(index)
    #     if self.return_gt_frame:
    #         frame_gt = self.get_gt_frame(index)
    #         frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=False)
    #     frame = self.transform_frame(frame, seed, transpose_to_CHW=False)  # to tensor

    #     # 이벤트(여기서는 voxel) 불러오기
    #     gen_event = self.get_voxel(index)
        
        
    #     # 이후 이벤트 처리: tensor 변환 등
    #     if self.return_gen_event:
    #         gen_event = torch.from_numpy(gen_event)
    #         item['gen_event'] = self.transform_gen_event(gen_event, seed)
    #     if self.return_frame:
    #         item['frame'] = frame
    #     if self.return_gt_frame:
    #         item['frame_gt'] = frame_gt

    #     item['seq'] = self.seq_name
    #     item['path'] = os.path.join(self.seq_name, 'image{:06d}'.format(index))

    #     return item




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
        max_val = torch.max(torch.abs(voxel))
        voxel = voxel / max_val

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
    

    def discretize_values_bidirectional(self, values, peaks):
        """
        values: ND numpy array of real values (예: shape = (6, H, W))
        peaks:  1D array of 구간 경계 (예: [-1, -0.5, 0, 0.5, 1]), 오름차순 정렬 가정

        규칙:
        - x < 0  => 구간의 '상위 경계값'으로 매핑 (값을 올림)
        - x >= 0 => 구간의 '하위 경계값'으로 매핑 (값을 내림)
        """
        # 결과를 담을 배열(동일 shape) 초기화
        mapped_values = np.zeros_like(values, dtype=float)

        # 음수/양수 마스크
        neg_mask = (values < 0)
        pos_mask = (values >= 0)

        # -------------------------------
        # (A) 음수 영역: 상위 경계로 매핑
        def map_to_upper(vals_neg, boundary):
            """
            vals_neg: 1D or ND array (음수 부분만 슬라이싱)
            boundary: 1D array (peaks)
            """
            # 1) boundary[0]보다 작으면 boundary[0], boundary[-1]보다 크면 boundary[-1]로 클램핑
            vals_left_clamped = np.where(vals_neg < boundary[0], boundary[0], vals_neg)
            vals_clamped = np.where(vals_left_clamped >= boundary[-1], boundary[-1], vals_left_clamped)

            # 2) 구간 인덱스 찾기
            indices = np.searchsorted(boundary, vals_clamped, side='right') - 1
            # 3) valid range로 클램핑
            indices = np.clip(indices, 0, len(boundary) - 2)

            # 4) 상위 경계값
            return boundary[indices + 1]

        # -------------------------------
        # (B) 양수 영역: 하위 경계로 매핑
        def map_to_lower(vals_pos, boundary):
            """
            vals_pos: 1D or ND array (양수 부분만 슬라이싱)
            boundary: 1D array (peaks)
            """
            # 1) 클램핑
            vals_left_clamped = np.where(vals_pos < boundary[0], boundary[0], vals_pos)
            vals_clamped = np.where(vals_left_clamped >= boundary[-1], boundary[-1], vals_left_clamped)

            # 2) 구간 인덱스 찾기
            indices = np.searchsorted(boundary, vals_clamped, side='right') - 1
            indices = np.clip(indices, 0, len(boundary) - 2)

            # 3) 하위 경계값
            return boundary[indices]

        # 음수/양수 부분 각각 매핑 후, 결과에 반영
        mapped_values[neg_mask] = map_to_upper(values[neg_mask], peaks)
        mapped_values[pos_mask] = map_to_lower(values[pos_mask], peaks)

        return mapped_values

    def get_peak_point(self):

        self.x_peak = np.array([-0.949     , -0.941     , -0.93700004, -0.899     , -0.895     ,
            -0.889     , -0.883     , -0.875     , -0.849     , -0.843     ,
            -0.833     , -0.823     , -0.81299996, -0.799     , -0.78900003,
            -0.777     , -0.765     , -0.749     , -0.737     , -0.73300004,
            -0.723     , -0.705     , -0.699     , -0.685     , -0.667     ,
            -0.649     , -0.643     , -0.63100004, -0.625     , -0.611     ,
            -0.599     , -0.589     , -0.579     , -0.571     , -0.56299996,
            -0.555     , -0.549     , -0.533     , -0.527     , -0.499     ,
            -0.491     , -0.487     , -0.481     , -0.473     , -0.467     ,
            -0.463     , -0.45499998, -0.449     , -0.445     , -0.43699998,
            -0.42900002, -0.421     , -0.411     , -0.399     , -0.389     ,
            -0.385     , -0.375     , -0.36900002, -0.357     , -0.353     ,
            -0.34899998, -0.333     , -0.329     , -0.315     , -0.29900002,
            -0.29500002, -0.28500003, -0.277     , -0.26700002, -0.263     ,
            -0.249     , -0.235     , -0.223     , -0.215     , -0.211     ,
            -0.199     , -0.187     , -0.177     , -0.167     , -0.157     ,
            -0.149     , -0.133     , -0.125     , -0.117     , -0.111     ,
            -0.105     , -0.099     , -0.067     , -0.063     , -0.059     ,
            -0.053     , -0.049     ,  0.001     ,  0.051     ,  0.059     ,
                0.063     ,  0.067     ,  0.07099999,  0.101     ,  0.105     ,
                0.111     ,  0.117     ,  0.125     ,  0.133     ,  0.151     ,
                0.157     ,  0.167     ,  0.177     ,  0.187     ,  0.201     ,
                0.211     ,  0.215     ,  0.223     ,  0.235     ,  0.251     ,
                0.263     ,  0.26700002,  0.277     ,  0.28500003,  0.29500002,
                0.301     ,  0.315     ,  0.333     ,  0.351     ,  0.357     ,
                0.36900002,  0.375     ,  0.38099998,  0.389     ,  0.393     ,
                0.40100002,  0.411     ,  0.41500002,  0.421     ,  0.42900002,
                0.43699998,  0.44099998,  0.445     ,  0.45099998,  0.467     ,
                0.473     ,  0.487     ,  0.49699998,  0.501     ,  0.527     ,
                0.533     ,  0.551     ,  0.555     ,  0.56299996,  0.571     ,
                0.579     ,  0.589     ,  0.601     ,  0.611     ,  0.625     ,
                0.63100004,  0.643     ,  0.647     ,  0.651     ,  0.667     ,
                0.685     ,  0.701     ,  0.705     ,  0.723     ,  0.737     ,
                0.751     ,  0.765     ,  0.777     ,  0.78900003,  0.801     ,
                0.81299996,  0.823     ,  0.833     ,  0.843     ,  0.851     ,
                0.875     ,  0.883     ,  0.889     ,  0.895     ,  0.901     ,
                0.929     ,  0.93700004,  0.941     ,  0.947     ,  0.951     ])