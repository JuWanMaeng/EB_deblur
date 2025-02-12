import numpy as np
import torch

def calculate_event_rmse(img1, img2, crop_border=0, input_order='CHW'):
    """
    6채널 event data에 대해 RMSE (Root Mean Squared Error)를 계산합니다.
    입력 데이터는 [-1, 1] 범위이며, 내부에서 [0, 1]로 정규화합니다.
    
    Args:
        img1 (torch.Tensor or np.ndarray): 첫 번째 이미지. (H, W, 6) 또는 (6, H, W)
        img2 (torch.Tensor or np.ndarray): 두 번째 이미지. (H, W, 6) 또는 (6, H, W)
        crop_border (int): 가장자리에서 제거할 픽셀 수 (기본값: 0)
        input_order (str): 입력 데이터 순서 ('HWC' 또는 'CHW', 기본값: 'HWC')
        
    Returns:
        float: 계산된 RMSE 값.
    """
    # torch.Tensor인 경우 numpy 배열로 변환
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
        img1 = img1[0]
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        img2 = img2[0]
    
    # 입력 순서가 CHW이면 HWC로 변환
    if input_order == 'CHW':
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    elif input_order != 'HWC':
        raise ValueError("input_order는 'HWC' 또는 'CHW'여야 합니다.")
    
    # crop_border 적용 (가장자리 픽셀 제거)
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
    
    # [-1, 1] 범위를 [0, 1]로 정규화
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0
    
    # 전체 채널에 대해 RMSE 계산
    rmse = np.sqrt(np.mean((img1 - img2) ** 2))
    return rmse