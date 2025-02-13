## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
import Motion_Deblurring.utils as utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.models.archs.EBR_fftformer_arch import fftformer
from basicsr.models.archs.fftformer_cross_arch import fftformer_cross
from basicsr.models.archs.EFNet_flow_arch import EFNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from metric import caculate_PSNR,caculate_PSNR_from_tensor
from val_utils import AverageMeter
from ptlflow.utils import flow_utils

def normalize_flow_to_tensor(flow):

    # Calculate the magnitude of the flow vectors
    u, v = flow[:,:,0], flow[:,:,1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Avoid division by zero by setting small magnitudes to a minimal positive value
    magnitude[magnitude == 0] = 1e-8
    
    # Normalize u and v components to get unit vectors for x and y
    x = u / magnitude
    y = v / magnitude

    # Normalize the magnitude to [0, 1] range for the z component
    z = magnitude / 100
    z = np.clip(z,0,1)
    z = z * 2 - 1

    # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
    C = np.stack((x, y, z), axis=-1)

    return C


def main():
    parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

    parser.add_argument('--result_dir', default='./results/Gopro', type=str, help='Directory for results')
    parser.add_argument('--weights', default='/workspace/data/FFTformer/experiments/FFTformer_cross_C_pusdo/models/net_g_680000.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='FFTformer_cross_680k_train', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = '/workspace/FFTformer/options/train/FFTformer_cross.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    # model_restoration = NAFNet(**x['network_g'])
    model_restoration = fftformer_cross(**x['network_g'])

    checkpoint = torch.load(args.weights)
    model_restoration.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ",args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()


    factor = 8
    dataset = args.dataset
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)


    
    files = []
    with open('/workspace/data/Gopro_my/train_into_future.txt', 'r') as file:
        for line in file:
            files.append(line.strip())
    print(len(files))


    psnr = AverageMeter()


    with torch.no_grad():
        for idx,file_ in enumerate(tqdm(files, ncols=80)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_tensor = img.unsqueeze(0).cuda()

            img_name = file_.split('/')[-1]
            scene = img_name[:-4]
            next_scene = str(int(scene) + 1111).zfill(6)
            sharp_path = file_.replace('blur', 'sharp')
            # sharp_path = os.path.join('/raid/joowan/GoPro/test/sharp', img_name)
            sharp_img = np.float32(utils.load_img(sharp_path))/255.
            # sharp_img = torch.from_numpy(sharp_img).permute(2,0,1)
            # sharp = img.unsqueeze(0).cuda()

            tmp_psnr = 0
            for i in range(2):
                flow_path = file_.replace('blur','flow/flows')
                flow_path = flow_path.replace('png','flo')
                flow_path = flow_path.replace('png','flo')
                if i == 1:
                    flow_path = flow_path.replace(f'{scene}', f'{next_scene}')
                max_flow = 10000
                flow = flow_utils.flow_read(flow_path)
                nan_mask = np.isnan(flow)
                flow[nan_mask] = max_flow + 1
                flow[nan_mask] = 0
                flow = np.clip(flow, -max_flow, max_flow)
                flow = normalize_flow_to_tensor(flow)


                flow = torch.from_numpy(flow.transpose(2, 0, 1)).unsqueeze(0)
                flow = flow.float()
                flow = flow.cuda()

                input_ = torch.cat([input_tensor,flow],dim=1)

                # Padding in case images are not multiples of 8
                h,w = input_.shape[2], input_.shape[3]

                # 1. Restormer, NAFNet
                # H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                # padh = H-h if h%factor!=0 else 0
                # padw = W-w if w%factor!=0 else 0
                # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                # 2. FFTformer
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(input_, (0, w_n, 0, h_n), mode='reflect')

                restored = model_restoration(in_tensor)

                # Unpad images to original dimensions
                if isinstance(restored, list):
                    restored = restored[-1]
                    mid_out = restored[0]

                restored = restored[:,:,:h,:w]
                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                restored_psnr = caculate_PSNR_from_tensor(sharp_img, restored)

                if tmp_psnr < restored_psnr:
                    final_out = restored
                    tmp_psnr = restored_psnr

            psnr.update(tmp_psnr, idx+1)

            save_path = os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')
            utils.save_img(save_path, img_as_ubyte(final_out))

    print(psnr.avg)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()