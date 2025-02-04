import glob,os
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm



def caculate_PSNR(sharp_path, blur_path):
    img1 = Image.open(sharp_path).convert('RGB')
    img2 = Image.open(blur_path).convert('RGB')

    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)

    

    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # 두 이미지가 완전히 동일한 경우
        return float('inf')

    # 최대 픽셀 값
    max_pixel = 1. if img1.max() <= 1 else 255.

    # PSNR 계산
    psnr = 20. * np.log10(max_pixel / np.sqrt(mse))
    
    # print(psnr)
    return psnr


if __name__ == '__main__':

    #### GoPro ####
    mother_path = '/workspace/data/FFTformer/results/EB_NAFNet_dit/visualization/EB_NAFNet_dit'
    output_folders = os.listdir(mother_path)
    output_folders.sort()
    total_psnr = 0

    for scene in tqdm(output_folders,ncols=80):
        if 'gt' in scene:
            continue
        else:
            output_img = os.path.join(mother_path,scene)
            sharp_path = output_img.replace('.png', '_gt.png')
            total_psnr+= caculate_PSNR(sharp_path, output_img)
    print(total_psnr / (len(output_folders)//2))


    #### GoPro ####
    # mother_path = '/workspace/FFTformer/results/Gopro/EFNet_C_test'
    # output_folders = os.listdir(mother_path)
    # output_folders.sort()
    # total_psnr = 0

    # for scene in tqdm(output_folders,ncols=80):
    #     output_img = os.path.join(mother_path,scene,'out.png')
    #     sharp_path = f'/workspace/data/Gopro_my/test/{scene}/sharp/{scene}.png'
    #     total_psnr+= caculate_PSNR(sharp_path, output_img)

    # print(f'psnr:{round(total_psnr,3) / len(output_folders)}')
    

    ### HIDE ###
    # path = '/workspace/FFTformer/results/HIDE/NAFNet_cross_640k'
    # folders = os.listdir(path)
    # folders.sort()

    # sharp_mother_path = '/workspace/data/HIDE_dataset/GT'
    # total_psnr = 0

    # for folder in tqdm(folders,ncols=80):
    #     out_path = os.path.join(path,folder,'out.png')
    #     sharp_path = os.path.join(sharp_mother_path,folder)

    #     total_psnr+= caculate_PSNR(sharp_path, out_path)

    # print(f'psnr:{round(total_psnr,3) / len(folders)}')

