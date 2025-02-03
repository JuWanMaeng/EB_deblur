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
from basicsr.models.archs.fftformer_arch import fftformer
from skimage import img_as_ubyte
from pdb import set_trace as stx
from metric import caculate_PSNR,caculate_PSNR_from_tensor
from val_utils import AverageMeter
from ptlflow.utils import flow_utils


def main():
    parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

    parser.add_argument('--result_dir', default='./results/REDS', type=str, help='Directory for results')
    parser.add_argument('--weights', default='pretrain_model/FFTformer.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='FFTformer_test', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = '/workspace/FFTformer/options/train/FFTformer.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    # model_restoration = NAFNet(**x['network_g'])
    model_restoration = fftformer(**x['network_g'])

    checkpoint = torch.load(args.weights)
    model_restoration.load_state_dict(checkpoint)
    print("===>Testing using weights: ",args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()


    factor = 8
    dataset = args.dataset
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)


    
    files = []
    with open('/workspace/FFTformer/datasets/event_gen/REDS.txt', 'r') as file:
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


            input_ = input_tensor

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
            restored = restored[:,:,:h,:w]
            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored_psnr = caculate_PSNR_from_tensor(sharp_img, restored)

            psnr.update(restored_psnr, idx+1)

            # save_path = os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')
            # utils.save_img(save_path, img_as_ubyte(restored))

    print(psnr.avg)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()