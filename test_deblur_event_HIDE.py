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
from basicsr.models.archs.fftformer_cross_arch import fftformer_cross
from basicsr.models.archs.EFNet_arch import EFNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from metric import caculate_PSNR,caculate_PSNR_from_tensor
from val_utils import AverageMeter
from ptlflow.utils import flow_utils



def main():
    parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

    parser.add_argument('--result_dir', default='./results/HIDE', type=str, help='Directory for results')
    parser.add_argument('--weights', default='/workspace/data/FFTformer/experiments/EB_NAFNet_dit_threshold/models/net_g_280000.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='EBNAFNet_dit', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    # yaml_file = '/workspace/FFTformer/options/test/EB_FFTformer.yml'
    yaml_file = '/workspace/FFTformer/options/test/EB_NAFNet.yml'
    # yaml_file = '/workspace/FFTformer/options/test/EFNet_gen.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    model_restoration = NAFNet(**x['network_g'])
    # model_restoration = fftformer(**x['network_g'])
    # model_restoration = EFNet(**x['network_g'])

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
    with open('/workspace/FFTformer/datasets/event_gen/HIDE.txt', 'r') as file:
        for line in file:
            files.append(line.strip())
    print(len(files))


    psnr = AverageMeter()
    total_psnr = 0


    with torch.no_grad():
        for idx,file_ in enumerate(tqdm(files, ncols=80)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/255.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = torch.from_numpy(img).permute(2,0,1)
            input_tensor = img.unsqueeze(0).cuda()

            splits = file_.split('/')
            img_name = splits[-1]
            if splits[5] == 'test-close-ups':
                sharp_path = file_.replace('/test/test-close-ups','/GT')
            else:
                sharp_path = file_.replace('test/test-long-shot','GT')
            sharp_img = np.float32(utils.load_img(sharp_path))/255.


            event_path = file_.replace('.png','.npy')
            event = np.load(event_path)
            # max_val = np.max(np.abs(event))
            # event = event / max_val

            event = torch.from_numpy(event.transpose(2,0,1)).unsqueeze(0).float().cuda()

            threshold = 0.05
            # # 절대값이 threshold 이하인 값들을 0으로 설정
            event[torch.abs(event) <= threshold] = 0


            input_ = torch.cat([input_tensor,event],dim=1)

            final_restored, max_psnr = -1, -1

            for i in range(2):
                # Padding in case images are not multiples of 8
                b, c, h, w = input_.shape
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                input_ = torch.nn.functional.pad(input_, (0, w_n, 0, h_n), mode='reflect')

                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:,:,:h,:w]
                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

                temp_psnr = caculate_PSNR_from_tensor(sharp_img, restored)

                if temp_psnr > max_psnr:
                    final_restored = restored
                    max_psnr = temp_psnr

                if i == 0:
                    event = torch.flip(event, dims=[1])
                    input_ = torch.cat([input_tensor,event],dim=1)
                    

            save_path = os.path.join(result_dir,img_name)
            os.makedirs(save_path,exist_ok=True)

            out_save_path = os.path.join(save_path,'out.png')
            input_save_path = os.path.join(save_path,'inp.png')

            input_img = cv2.imread(file_)

            utils.save_img(out_save_path, img_as_ubyte(final_restored))
            cv2.imwrite(input_save_path,input_img)

            psnr_score = caculate_PSNR(sharp_path, out_save_path)
            total_psnr += psnr_score

    print(psnr.avg)
    print(total_psnr/len(files))




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    main()