from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.data.transforms import augment
from ptlflow.utils import flow_utils

import random
import numpy as np
import torch
import cv2, os
import torchvision.transforms.functional as TF


#### Multi Flow input dataset ####
######################################################################################################
class Multi_Flow_Blur_dataset(data.Dataset):

    def __init__(self, opt):
        super(Multi_Flow_Blur_dataset, self).__init__()
   
        self.opt = opt
        self.max_magnitude = 100
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_txt = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = []

        with open(self.lq_txt, 'r') as file:
            for line in file:
                # Add each line to the list, stripping the newline character
                self.paths.append(line.strip())

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        #blur_path = self.paths[index]        
        flow_path = self.paths[index]

        phase = self.opt['phase']
        scale = self.opt['scale']

        scene = flow_path.split('/')[-3]

        if phase == 'train':
            lq_path = os.path.join(f'/workspace/data/Gopro_my25/{phase}', scene,'blur', f'{scene}.png')
            gt_path = os.path.join(f'/workspace/data/Gopro_my25/{phase}',scene,'sharp',f'{scene}.png')
        else:
            lq_path = os.path.join(f'/workspace/data/Gopro_my25/test',scene,'blur',f'{scene}.png')
            gt_path = os.path.join(f'/workspace/data/Gopro_my25/test',scene,'sharp',f'{scene}.png')

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)


        # Optical flow array
        # flow_path = blur_path.replace('blur', 'flow/flows')
        # flow_path = flow_path.replace('png', 'flo')
        
        max_flow = 10000
        flow = flow_utils.flow_read(flow_path)  # H,W,10

        nan_mask = np.isnan(flow)
        flow[nan_mask] = max_flow + 1
        flow[nan_mask] = 0
        flow = np.clip(flow, -max_flow, max_flow)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, flow = self.padding(img_gt, img_lq, flow ,gt_size)

            # random crop
            img_gt, img_lq, flow = self.paired_random_crop(img_gt, img_lq, flow ,gt_size, scale,
                                                gt_path)


            ### augmentation ###
            # Randomly select an augmentation command
            commands = ['hflip', 'vflip' , 'noop', 'rot90']
            command = np.random.choice(commands)

            # augment flow 
            if command != 'noop':
                flow = self.augment_flow(flow, command)

            # augment blur 
            if command == 'hflip':
                img_lq = np.fliplr(img_lq).copy()
                img_gt = np.fliplr(img_gt).copy()

            elif command == 'vflip':
                img_lq = np.flipud(img_lq).copy()
                img_gt = np.flipud(img_gt).copy()

            elif command == 'rot90':
                img_lq = np.rot90(img_lq)
                img_gt = np.rot90(img_gt)



        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        ### normalize_flow_to_tensor ###
        n_flow = []
        flows = [flow[..., i:i+2] for i in range(0, flow.shape[2], 2)]
        for f in flows:
            n_f = self.normalize_flow_to_tensor(f)
            n_flow.append(n_f)
        flow = np.concatenate(n_flow,axis=-1)  # H,W,15

        flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()  # [3,H,W]

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'flow_path': flow_path,
            'flow' : flow
        }


    def __len__(self):
        return len(self.paths)
    
    def padding(self, img_lq, img_gt, flow, gt_size):
        h, w, _ = img_lq.shape

        h_pad = max(0, gt_size - h)
        w_pad = max(0, gt_size - w)
        
        if h_pad == 0 and w_pad == 0:
            return img_lq, img_gt, flow

        img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        flow = cv2.copyMakeBorder(flow, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        # print('img_lq', img_lq.shape, img_gt.shape)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=2)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
        if flow.ndim == 2:
            flow = np.expand_dims(flow, axis=2)

        return img_lq, img_gt, flow

    def paired_random_crop(self,img_gts, img_lqs, flow, lq_patch_size, scale, gt_path):
        """Paired random crop.

        It crops lists of lq and gt images with corresponding locations.

        Args:
            img_gts (list[ndarray] | ndarray): GT images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            lq_patch_size (int): LQ patch size.
            scale (int): Scale factor.
            gt_path (str): Path to ground-truth.

        Returns:
            list[ndarray] | ndarray: GT images and LQ images. If returned results
                only have one element, just return ndarray.
        """

        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(flow, list):
            flow = [flow]

        h_lq, w_lq, _ = img_lqs[0].shape
        h_gt, w_gt, _ = img_gts[0].shape
        h_flow, w_flow, _ = flow[0].shape
        gt_patch_size = int(lq_patch_size * scale)

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                            f'({lq_patch_size}, {lq_patch_size}). '
                            f'Please remove {gt_path}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # crop lq patch
        img_lqs = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in img_lqs
        ]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        img_gts = [
            v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            for v in img_gts
        ]

        # crop corresponding flow patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        flow = [
            v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            for v in flow
        ]

        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(flow) == 1:
            flow = flow[0]
        return img_gts, img_lqs, flow
    
    def augment_flow(self, flow, command):
        """
        Augment a concatenated flow tensor with shape (H, W, 10) based on the specified command.

        Parameters:
        - flow: 3D array representing concatenated optical flows with shape (H, W, 10).
        - command: String specifying the augmentation ('hflip', 'vflip', 'rot90').

        Returns:
        - flow: Augmented flow tensor with shape (H, W, 10).
        """
        # Split the flow into individual components
        flows = [flow[..., i:i+2] for i in range(0, flow.shape[2], 2)]

        # Apply augmentation to each flow component
        augmented_flows = []
        for f in flows:
            if command == 'hflip':  # horizontal
                f = cv2.flip(f, 1)
                f[:, :, 0] *= -1

            elif command == 'vflip':  # vertical
                f = cv2.flip(f, 0)
                f[:, :, 1] *= -1

            elif command == 'rot90':
                f = np.rot90(f)
                f = f[:, :, [1, 0]]
                f[:, :, 1] *= -1

            else:
                raise ValueError('Wrong command!')

            augmented_flows.append(f)

        # Concatenate the augmented flows back into a single tensor
        flow = np.concatenate(augmented_flows, axis=-1)

        return flow

    def normalize_flow_to_tensor(self, flow):
        """
        Normalize the optical flow and compute the 3D tensor C with x, y, and z components.

        Parameters:
        - flow: 2D array representing optical flow.

        Returns:
        - C: 3D tensor with shape (H, W, 3), where C[..., 0] is x, C[..., 1] is y, and C[..., 2] is z.
        """
        # Calculate the magnitude of the flow vectors
        u, v = flow[:,:,0], flow[:,:,1]
        magnitude = np.sqrt(u**2 + v**2)
        
        # Avoid division by zero by setting small magnitudes to a minimal positive value
        magnitude[magnitude == 0] = 1e-8
        
        # Normalize u and v components to get unit vectors for x and y
        x = u / magnitude
        y = v / magnitude

        # Normalize the magnitude to [0, 1] range for the z component
        z = magnitude / self.max_magnitude
        z = np.clip(z,0,1)
        z = z * 2 - 1

        # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
        C = np.stack((x, y, z), axis=-1)

        return C
    
    def tensor_to_original_flow(self,C):
        """
        Convert the normalized 3D tensor C back to the original 2D optical flow.

        Parameters:
        - C: 3D tensor with shape (H, W, 3), where C[..., 0] is x, C[..., 1] is y, and C[..., 2] is z.

        Returns:
        - flow: 2D array representing optical flow with shape (H, W, 2).
        """
        # Extract x, y, and z components
        x, y, z = C[..., 0], C[..., 1], C[..., 2]

        # Convert z from [-1, 1] range back to [0, 1] range
        z = (z + 1) / 2

        # Denormalize z to get the magnitude of the flow vectors
        magnitude = z * self.max_magnitude

        # Compute original u and v components using magnitude and x, y unit vectors
        u = x * magnitude
        v = y * magnitude

        # Stack u and v to create the original 2D optical flow with shape (H, W, 2)
        flow = np.stack((u, v), axis=-1)

        return flow

######################################################################################################
class Flow_Blur_dataset_C(data.Dataset):

    def __init__(self, opt):
        super(Flow_Blur_dataset_C, self).__init__()
   
        self.opt = opt
        self.max_magnitude = 100
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_txt = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = []

        with open(self.lq_txt, 'r') as file:
            for line in file:
                # Add each line to the list, stripping the newline character
                self.paths.append(line.strip())

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        blur_path = self.paths[index]        

        phase = self.opt['phase']
        scale = self.opt['scale']

        scene = blur_path.split('/')[-3]

        if phase == 'train':
            lq_path = blur_path
            gt_path = os.path.join(f'/workspace/data/Gopro_my/{phase}',scene,'sharp',f'{scene}.png')
        else:
            lq_path = blur_path
            gt_path = os.path.join(f'/workspace/data/Gopro_my/test',scene,'sharp',f'{scene}.png')

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)


        # Optical flow array
        flow_path = blur_path.replace('blur', 'flow/flows')
        flow_path = flow_path.replace('png', 'flo')
        
        max_flow = 10000
        flow = flow_utils.flow_read(flow_path)

        nan_mask = np.isnan(flow)
        flow[nan_mask] = max_flow + 1
        flow[nan_mask] = 0
        flow = np.clip(flow, -max_flow, max_flow)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, flow = self.padding(img_gt, img_lq, flow ,gt_size)

            # random crop
            img_gt, img_lq, flow = self.paired_random_crop(img_gt, img_lq, flow ,gt_size, scale,
                                                gt_path)


            ### augmentation ###
            # Randomly select an augmentation command
            commands = ['hflip', 'vflip' , 'noop', 'rot90']
            command = np.random.choice(commands)

            # augment flow 
            if command != 'noop':
                flow = self.augment_flow(flow, command)

            # augment blur 
            if command == 'hflip':
                img_lq = np.fliplr(img_lq).copy()
                img_gt = np.fliplr(img_gt).copy()

            elif command == 'vflip':
                img_lq = np.flipud(img_lq).copy()
                img_gt = np.flipud(img_gt).copy()
            elif command == 'rot90':
                img_lq = np.rot90(img_lq)
                img_gt = np.rot90(img_gt)



        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        ### normalize_flow_to_tensor ###
        flow = self.normalize_flow_to_tensor(flow)
        flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()  # [3,H,W]

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'flow_path': flow_path,
            'flow' : flow
        }


    def __len__(self):
        return len(self.paths)
    
    def padding(self, img_lq, img_gt, flow, gt_size):
        h, w, _ = img_lq.shape

        h_pad = max(0, gt_size - h)
        w_pad = max(0, gt_size - w)
        
        if h_pad == 0 and w_pad == 0:
            return img_lq, img_gt, flow

        img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        flow = cv2.copyMakeBorder(flow, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        # print('img_lq', img_lq.shape, img_gt.shape)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=2)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
        if flow.ndim == 2:
            flow = np.expand_dims(flow, axis=2)

        return img_lq, img_gt, flow

    def paired_random_crop(self,img_gts, img_lqs, flow, lq_patch_size, scale, gt_path):
        """Paired random crop.

        It crops lists of lq and gt images with corresponding locations.

        Args:
            img_gts (list[ndarray] | ndarray): GT images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            lq_patch_size (int): LQ patch size.
            scale (int): Scale factor.
            gt_path (str): Path to ground-truth.

        Returns:
            list[ndarray] | ndarray: GT images and LQ images. If returned results
                only have one element, just return ndarray.
        """

        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(flow, list):
            flow = [flow]

        h_lq, w_lq, _ = img_lqs[0].shape
        h_gt, w_gt, _ = img_gts[0].shape
        h_flow, w_flow, _ = flow[0].shape
        gt_patch_size = int(lq_patch_size * scale)

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                            f'({lq_patch_size}, {lq_patch_size}). '
                            f'Please remove {gt_path}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # crop lq patch
        img_lqs = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in img_lqs
        ]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        img_gts = [
            v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            for v in img_gts
        ]

        # crop corresponding flow patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        flow = [
            v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            for v in flow
        ]

        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(flow) == 1:
            flow = flow[0]
        return img_gts, img_lqs, flow
    
    def augment_flow(self, flow, command):

        if command == 'hflip':  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1

        elif command ==  'vflip':  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1

        elif command ==  'rot90':
            flow = np.rot90(flow)
            flow = flow[:, :, [1, 0]]
            flow[:, :, 1] *= -1

        else:
            assert 'Wrong command!'

        return flow

    def normalize_flow_to_tensor(self, flow):
        """
        Normalize the optical flow and compute the 3D tensor C with x, y, and z components.

        Parameters:
        - flow: 2D array representing optical flow.

        Returns:
        - C: 3D tensor with shape (H, W, 3), where C[..., 0] is x, C[..., 1] is y, and C[..., 2] is z.
        """
        # Calculate the magnitude of the flow vectors
        u, v = flow[:,:,0], flow[:,:,1]
        magnitude = np.sqrt(u**2 + v**2)
        
        # Avoid division by zero by setting small magnitudes to a minimal positive value
        magnitude[magnitude == 0] = 1e-8
        
        # Normalize u and v components to get unit vectors for x and y
        x = u / magnitude
        y = v / magnitude

        # Normalize the magnitude to [0, 1] range for the z component
        z = magnitude / self.max_magnitude
        z = np.clip(z,0,1)
        z = z * 2 - 1

        # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
        C = np.stack((x, y, z), axis=-1)

        return C
    


class Flow_Blur_dataset(data.Dataset):

    def __init__(self, opt):
        super(Flow_Blur_dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_txt = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = []

        with open(self.lq_txt, 'r') as file:
            for line in file:
                # Add each line to the list, stripping the newline character
                self.paths.append(line.strip())


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        flow_path = self.paths[index]         # /workspace/data/Gopro_my/flows/train/000001/0.png 
                                              # /workspace/data/results/Gopro/tens/000003/4.png
                                              
        flow_bytes = self.file_client.get(flow_path, 'flow')
        flow = imfrombytes(flow_bytes, float32=True)

        scene = flow_path.split('/')[-2]

        phase = self.opt['phase']
        if phase == 'train':
            lq_path = os.path.join(f'/workspace/data/Gopro_my/{phase}',scene,'blur',f'{scene}.png')
            gt_path = os.path.join(f'/workspace/data/Gopro_my/{phase}',scene,'sharp',f'{scene}.png')
        else:
            lq_path = os.path.join(f'/workspace/data/Gopro_my/test',scene,'blur',f'{scene}.png')
            gt_path = os.path.join(f'/workspace/data/Gopro_my/test',scene,'sharp',f'{scene}.png')
        

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, flow = self.padding(img_gt, img_lq, flow ,gt_size)

            # random crop
            img_gt, img_lq, flow = self.paired_random_crop(img_gt, img_lq, flow ,gt_size, scale,
                                                gt_path)
            
            img_gt, img_lq, flow = augment([img_gt,img_lq,flow],self.opt['use_flip'], self.opt['use_rot'])
            
            # angle = random.random() < 0.3

            # if angle:  # rotate only 180
            #     img_gt, img_lq, flow = self.rotate(img_gt, img_lq, flow, 180)


            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        flow = cv2.cvtColor(flow,cv2.COLOR_BGR2RGB)
        flow = torch.from_numpy(flow.transpose(2, 0, 1))
        flow = flow.float()
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'flow_path': flow_path,
            'flow' : flow
        }

    def __len__(self):
        return len(self.paths)
        

    def paired_random_crop(self,img_gts, img_lqs, flow, lq_patch_size, scale, gt_path):
        """Paired random crop.

        It crops lists of lq and gt images with corresponding locations.

        Args:
            img_gts (list[ndarray] | ndarray): GT images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
                should have the same shape. If the input is an ndarray, it will
                be transformed to a list containing itself.
            lq_patch_size (int): LQ patch size.
            scale (int): Scale factor.
            gt_path (str): Path to ground-truth.

        Returns:
            list[ndarray] | ndarray: GT images and LQ images. If returned results
                only have one element, just return ndarray.
        """

        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(flow, list):
            flow = [flow]

        h_lq, w_lq, _ = img_lqs[0].shape
        h_gt, w_gt, _ = img_gts[0].shape
        h_flow, w_flow, _ = flow[0].shape
        gt_patch_size = int(lq_patch_size * scale)

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                            f'({lq_patch_size}, {lq_patch_size}). '
                            f'Please remove {gt_path}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # crop lq patch
        img_lqs = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in img_lqs
        ]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        img_gts = [
            v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            for v in img_gts
        ]

        # crop corresponding flow patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        flow = [
            v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            for v in flow
        ]

        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(flow) == 1:
            flow = flow[0]
        return img_gts, img_lqs, flow
    


    def padding(self, img_lq, img_gt, flow, gt_size):
        h, w, _ = img_lq.shape

        h_pad = max(0, gt_size - h)
        w_pad = max(0, gt_size - w)
        
        if h_pad == 0 and w_pad == 0:
            return img_lq, img_gt,flow

        img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        flow = cv2.copyMakeBorder(flow, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        # print('img_lq', img_lq.shape, img_gt.shape)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=2)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
        if flow.ndim == 2:
            flow = np.expand_dims(flow, axis=2)

        return img_lq, img_gt, flow
    
    # def rotate(self, img_gt, img_lq, flow_rgb, angle):
    #     # 이미지를 회전
    #     k = angle // 90
    #     img_gt = np.rot90(img_gt,k=k)
    #     img_lq = np.rot90(img_lq,k=k)
    #     flow_rgb = np.rot90(flow_rgb,k=k)

    #     # BGR 이미지를 HSV로 변환
    #     hsv_image = cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2HSV)

    #     # 회전 각도에 따라 Hue를 반시계 방향으로 변경
    #     hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + angle ) % 180
    #     hsv_image[:, :, 0] = hsv_image[:, :, 0].astype(np.uint8)

    #     # HSV 이미지를 다시 BGR로 변환
    #     rotated_and_hue_adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    #     return img_gt, img_lq, rotated_and_hue_adjusted_image
    


if __name__ == '__main__':

    # Sample options dictionary to simulate `opt` (you might want to adjust these paths)
    opt = {
        'io_backend': {'type': 'disk'},
        'dataroot_gt': '/workspace/data/Gopro_my/train.txt',
        'dataroot_lq': '/workspace/data/Gopro_my/train.txt',  # Update with your actual path to LQ text file
        'phase': 'train',          # or 'test' depending on phase
        'scale': 1,
        'gt_size': 256
    }

    # Create the dataset
    dataset = Flow_Blur_dataset_C(opt)

    # Check dataset length
    print(f"Total samples in dataset: {len(dataset)}")

    # Load a sample item (change index as needed)
    sample_index = 5
    sample = dataset[sample_index]

    # Print out some of the sample's components to check for correctness
    print("Sample data:")
    print(f"  LQ Path: {sample['lq_path']}")
    print(f"  GT Path: {sample['gt_path']}")
    print(f"  Flow Path: {sample['flow_path']}")
    print(f"  LQ Image Shape: {sample['lq'].shape}")
    print(f"  GT Image Shape: {sample['gt'].shape}")
    print(f"  Flow Shape: {sample['flow'].shape}")