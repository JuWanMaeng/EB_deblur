## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures

def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    min_dimension = min(tar_img.shape[0], tar_img.shape[1])
    win_size = min(7, min_dimension)  # Ensure win_size does not exceed the image dimensions

    ssim_pre, ssim_map = structural_similarity(
        tar_img, prd_img, multichannel=True, gaussian_weights=True,
        use_sample_covariance=False, data_range=1.0, full=True, win_size=win_size, channel_axis=-1
    )
    ssim_map = ssim_map * cr1

    r = int(3.5 * 1.5 + 0.5)  # Radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2

    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0
    
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    with open('realblur-J_result.txt', 'a') as f:
       img_name = filename[0].split('/')[-1]
       f.write(f'{img_name}_{PSNR}\n')

    return (PSNR, SSIM)





with open('/workspace/FFTformer/datasets/event_gen/realblur_j_gt.txt', 'r') as file:
    gt_list = [line.strip() for line in file]

out_mother_path = '/workspace/FFTformer/results/RealBlur-J/EFNet'
path_list = []
for gt_path in gt_list:
    out_path = gt_path.replace('gt','blur')
    out_path_split = out_path.split('/')
    scene = out_path_split[-3]
    img_name = out_path_split[-1]
    out_folder = f'{scene}_{img_name}'
    
    output_path = os.path.join(out_mother_path, out_folder, 'out.png')
    path_list.append(output_path)



assert len(path_list) != 0, "Predicted files not found"
assert len(gt_list) != 0, "Target files not found"

psnr, ssim = [], []
img_files =[(i, j) for i,j in zip(gt_list,path_list)]
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(proc, img_files), total=len(img_files)))

for PSNR_SSIM in results:
    psnr.append(PSNR_SSIM[0])
    ssim.append(PSNR_SSIM[1])

avg_psnr = sum(psnr)/len(psnr)
avg_ssim = sum(ssim)/len(ssim)

print('PSNR: {:f} SSIM: {:f}\n'.format(avg_psnr, avg_ssim))
