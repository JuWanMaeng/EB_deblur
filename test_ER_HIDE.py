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

from basicsr.models.archs.NAFNet2_arch import NAFNet2




def main():
    parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

    parser.add_argument('--result_dir', default='./results/HIDE', type=str, help='Directory for results')
    parser.add_argument('--weights', default='pretrained_model/ER_best_width32.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='ER_NAFNet', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = 'options/test/ER_NAFNet2.yml'

    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    s = x['network_g'].pop('type')
    ##########################

    model_restoration = NAFNet2(**x['network_g'])

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

            event_path = file_.replace('.png','.npy')
            event = np.load(event_path)
            # max_val = np.max(np.abs(event))
            # event = event / max_val
            event = torch.from_numpy(event.transpose(2,0,1)).unsqueeze(0).float().cuda()


            input_ = torch.cat([event,input_tensor],dim=1)

            b, c, h, w = input_.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_ = torch.nn.functional.pad(input_, (0, w_n, 0, h_n), mode='reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]
            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)



            save_path = os.path.join(result_dir,img_name)
            os.makedirs(save_path,exist_ok=True)

            out_save_path = os.path.join(save_path,'out.png')
   





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()