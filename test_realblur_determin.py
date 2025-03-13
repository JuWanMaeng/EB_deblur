import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch

import Motion_Deblurring.utils as utils

from basicsr.models.archs.NAFNet3_arch import NAFNet3


from skimage import img_as_ubyte

from metric import caculate_PSNR,caculate_PSNR_from_tensor
from val_utils import AverageMeter
from ptlflow.utils import flow_utils



def main():
    parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

    parser.add_argument('--result_dir', default='./results/RealBlur-J', type=str, help='Directory for results')
    parser.add_argument('--weights', default='/workspace/FFTformer/pretrain_model/Determin.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='Determin', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = '/workspace/FFTformer/options/test/NAFNet3.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    # model_restoration = NAFNet(**x['network_g'])
    # model_restoration = fftformer(**x['network_g'])
    model_restoration = NAFNet3(**x['network_g'])

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


    psnr = AverageMeter()
    total_psnr = 0

    with torch.no_grad():
        for idx,file_ in enumerate(tqdm(files, ncols=80)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/127.5 - 1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # input BGR
            img = torch.from_numpy(img).permute(2,0,1)
            input_tensor = img.unsqueeze(0).cuda()

            splits = file_.split('/')
            scene = splits[-3]
            img_name = splits[-1]
            img_name = f'{scene}_{img_name}'
            input_ = input_tensor

            # Padding in case images are not multiples of 8
            b, c, h, w = input_.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_ = torch.nn.functional.pad(input_, (0, w_n, 0, h_n), mode='reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]
            restored = torch.clamp(restored,-1,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            save_path = os.path.join(result_dir,img_name)
            os.makedirs(save_path,exist_ok=True)

            out_save_path = os.path.join(save_path,'out.npy')
            input_save_path = os.path.join(save_path,'inp.png')

            input_img = cv2.imread(file_)

            cv2.imwrite(input_save_path,input_img)
            np.save(out_save_path, restored)

            
    print(psnr.avg)
    print(total_psnr/len(files))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    main()