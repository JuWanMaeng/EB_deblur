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

def caculate_PSNR_from_tensor(img1, img2):

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
