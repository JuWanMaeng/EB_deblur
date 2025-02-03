## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import Motion_Deblurring.utils as utils

# from natsort import natsorted
import glob
from basicsr.models.archs.NAFNet_cross_arch import NAFNet_cross
from skimage import img_as_ubyte
from pdb import set_trace as stx
from metric import caculate_PSNR, caculate_PSNR_from_tensor
from val_utils import AverageMeter
from ptlflow.utils import flow_utils
import cv2


def main():
    parser = argparse.ArgumentParser(description='Gopro inference start')
    parser.add_argument('--result_dir', default='./results/Gopro', type=str, help='Directory for results')
    parser.add_argument('--weights', default='pretrain_model/NAFNet_cross_640k.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='NAFNet_cross_680k', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = 'options/train/NAFNet_cross.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    model_restoration = NAFNet_cross(**x['network_g'])

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
    with open('/workspace/FFTformer/datasets/Gopro_test.txt', 'r') as file:
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

            splits = file_.split('/')
            img_name = splits[-1]
            img_num = splits[-3]
            sharp_path = file_.replace('blur', 'sharp')
            sharp_img = np.float32(utils.load_img(sharp_path))/255.

            flow_path = f'/workspace/data/results/B/GoPro/{img_num}/*.png'
            flow_paths = glob.glob(flow_path)
            flow_paths.sort()
            flow_paths = flow_paths[:-1]  # remove input file

            final_restored, max_psnr_flow, max_psnr = -1, -1, -1
            for flow_path in flow_paths:

                flow_img = utils.load_img(flow_path)
                flow_img = flow_img[:,:,::-1]
                flow_rgb = np.float32(flow_img)/ 255.


                flow = torch.from_numpy(flow_rgb.transpose(2, 0, 1)).unsqueeze(0)
                flow = flow.float()
                flow = flow.cuda()

                input_ = torch.cat([input_tensor,flow],dim=1)

                # Padding in case images are not multiples of 8
                b, c, h, w = input_.shape
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                input_ = torch.nn.functional.pad(input_, (0, w_n, 0, h_n), mode='reflect')

                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:,:,:h,:w]
                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                temp_psnr = caculate_PSNR_from_tensor(sharp_img, restored)

                if temp_psnr > max_psnr:
                    final_restored = restored
                    max_psnr_flow = flow_img
                    max_psnr = temp_psnr


            psnr.update(temp_psnr, idx+1)

            save_path = os.path.join(result_dir,img_num)
            os.makedirs(save_path,exist_ok=True)

            out_save_path = os.path.join(save_path,'out.png')
            input_save_path = os.path.join(save_path,'inp.png')
            flow_save_path = os.path.join(save_path,'flow.png')
            input_img = cv2.imread(file_)

            utils.save_img(out_save_path, img_as_ubyte(final_restored))
            cv2.imwrite(flow_save_path,max_psnr_flow)
            cv2.imwrite(input_save_path,input_img)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()