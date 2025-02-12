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


from glob import glob
from basicsr.models.archs.NAFNet_arch import NAFNet
from skimage import img_as_ubyte





def main():
    parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

    parser.add_argument('--result_dir', default='./results/RealBlur-J', type=str, help='Directory for results')
    parser.add_argument('--weights', default='pretrained_model/ER_NAFNet/60K.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='ER_NAFNet', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = 'options/test/ER_NAFNet.yml'

    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    model_restoration = NAFNet(**x['network_g'])


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
    with open('datasets/event_gen/realblur_j.txt', 'r') as file:
        for line in file:
            files.append(line.strip())
    print(len(files))

    total_psnr = 0

    with torch.no_grad():
        for idx,file_ in enumerate(tqdm(files, ncols=80)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    
            splits = file_.split('/')
            scene = splits[-3]
            img_name = splits[-1]
            img_name = f'{scene}_{img_name}'


            event_path = file_.replace('blur','event')
            event_path = event_path.replace('.png','.npy')
            event = np.load(event_path)
            #[0,255] to [-1,1]
            event = event / 127.5 - 1
            # max_val = np.max(np.abs(event))
            # event = event / max_val

            event = torch.from_numpy(event.transpose(2,0,1)).unsqueeze(0).float().cuda()

            input_ = event

            restored = model_restoration(input_)
            restored = restored.cpu().detach().squeeze(0).numpy()


            save_path = os.path.join(result_dir,img_name)
            os.makedirs(save_path,exist_ok=True)
            np.save(f'{save_path}/out.npy', restored)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()