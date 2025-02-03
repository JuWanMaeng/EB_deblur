import os
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import struct
from warnings import warn
from tqdm import tqdm

from ptlflow.utils.flow_utils import flow_read, flow_to_rgb

def calculate_rmse(image1, image2):
    """
    두 이미지 사이의 RMSE 계산
    :param image1: 첫 번째 이미지 (NumPy 배열)
    :param image2: 두 번째 이미지 (NumPy 배열)
    :return: RMSE 값
    """
    # 두 이미지 크기 확인 (크기가 같아야 함)
    if image1.shape != image2.shape:
        raise ValueError("이미지 크기가 일치하지 않습니다.")
    
    # RMSE 계산
    mse = np.mean((image1 - image2) ** 2)  # MSE 계산
    rmse = np.sqrt(mse)  # RMSE 계산
    return rmse

# root_path = '/workspace/data/results/A/Gopro/tens'
# gt_path = '/workspace/data/Gopro_my/train'

# scenes = os.listdir(root_path)
# scenes.sort()

# metric = []
# for idx,scene in enumerate(tqdm(scenes, ncols=80)):
#     rmses = []
#     gt_flow_path = os.path.join(gt_path,scene,'flow/flows',f'{scene}.flo')
#     gt_flow = flow_read(gt_flow_path)
#     gt_flow_rgb = flow_to_rgb(gt_flow)
#     gt_flow_rgb = gt_flow_rgb/255.0

#     next_scene = int(scene) + 2103
#     next_scene = str(next_scene).zfill(6)
    
    
#     gt_flow_path_2 = os.path.join(gt_path,next_scene,'flow/flows',f'{next_scene}.flo')
#     gt_flow_2 = flow_read(gt_flow_path_2)
#     gt_flow_rgb_2 = flow_to_rgb(gt_flow_2)
#     gt_flow_rgb_2 = gt_flow_rgb_2/255.0

#     outputs = os.listdir(os.path.join(root_path,scene))
#     outputs.sort()
#     outs = outputs[:-1]

#     for out in outs:
#         out_flow_path = os.path.join(root_path,scene,out)
#         out_flow = np.array(Image.open(out_flow_path).convert('RGB')) / 255.0
#         rmse = calculate_rmse(gt_flow_rgb, out_flow)
#         rmse_2 = calculate_rmse(gt_flow_rgb_2, out_flow)
#         rmses.append(rmse)
#         rmses.append(rmse_2)

#     metric.append(min(rmses))

# print(sum(metric))


#########################

root_path = '/workspace/data/results/A/Gopro/single'
gt_path = '/workspace/data/Gopro_my/test'

scenes = os.listdir(root_path)
scenes.sort()

metric = []
for idx,scene in enumerate(tqdm(scenes, ncols=80)):
    rmses = []
    gt_flow_path = os.path.join(gt_path,scene,'flow/flows',f'{scene}.flo')
    gt_flow = flow_read(gt_flow_path)
    gt_flow_rgb = flow_to_rgb(gt_flow)
    gt_flow_rgb = gt_flow_rgb/255.0

    next_scene = int(scene) + 1111
    next_scene = str(next_scene).zfill(6)
    
    
    gt_flow_path_2 = os.path.join(gt_path,next_scene,'flow/flows',f'{next_scene}.flo')
    gt_flow_2 = flow_read(gt_flow_path_2)
    gt_flow_rgb_2 = flow_to_rgb(gt_flow_2)
    gt_flow_rgb_2 = gt_flow_rgb_2/255.0

    outputs = os.listdir(os.path.join(root_path,scene))
    outputs.sort()
    outs = outputs[:-1]

    for out in outs:
        out_flow_path = os.path.join(root_path,scene,out)
        out_flow = np.array(Image.open(out_flow_path).convert('RGB')) / 255.0
        rmse = calculate_rmse(gt_flow_rgb, out_flow)
        rmse_2 = calculate_rmse(gt_flow_rgb_2, out_flow)
        rmses.append(rmse)
        rmses.append(rmse_2)

    metric.append(min(rmses))
    

print(sum(metric))
