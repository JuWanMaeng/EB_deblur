import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)




class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda() 


        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(ievemg, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y,weight=None, **kwargs):
        loss = l1_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CustomMaskL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, lambda_mask=5.0, epsilon=0.01, reduction='mean'):
        """
        Custom Masked L1 Loss

        Args:
            loss_weight (float): 전체 손실에 곱할 가중치.
            lambda_mask (float): GT 이벤트가 0에 가까운 영역에 추가적으로 적용할 페널티 강도.
            epsilon (float): GT 이벤트가 0로 간주할 임계값.wat
            reduction (str): 'mean'만 지원.
        """
        super(CustomMaskL1Loss, self).__init__()
        assert reduction == 'mean', "Only 'mean' reduction is supported."
        self.loss_weight = loss_weight
        self.lambda_mask = lambda_mask
        self.epsilon = epsilon

    def forward(self, pred, target):
        # GT 이벤트가 0에 가까운 픽셀에 대해 마스크 생성: |target| < epsilon 인 경우 1
        mask = (target.abs() < self.epsilon).float()
        # 해당 영역에 대해 (1 + lambda_mask) 만큼의 가중치 부여
        weight = 1.0 + self.lambda_mask * mask
        # L1 손실 계산
        loss = (weight * (pred - target).abs()).mean()
        return self.loss_weight * loss


class CustomMaskL2Loss(nn.Module):
    def __init__(self, loss_weight=1.0, lambda_mask=2.0,reduction='mean'):
        """
        Custom Masked L2 Loss (MSE Loss)

        Args:
            loss_weight (float): 전체 손실에 곱할 가중치.
            lambda_mask (float): GT 이벤트가 0에 가까운 영역에 추가적으로 적용할 페널티 강도.
            epsilon (float): GT 이벤트가 0로 간주할 임계값.
            reduction (str): 'mean'만 지원.
        """
        super(CustomMaskL2Loss, self).__init__()
        assert reduction == 'mean', "Only 'mean' reduction is supported."
        self.loss_weight = loss_weight
        self.lambda_mask = lambda_mask
    
    def forward(self, pred, target):
        # GT 이벤트가 0에 가까운 픽셀에 대해 마스크 생성
        mask = (target.abs() == 0).float()
        weight = 1.0 + self.lambda_mask * mask
        # L2 손실 (제곱 오차) 계산
        loss = (weight * (pred - target) ** 2).mean()
        return self.loss_weight * loss


class CustomMaskPSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, lambda_mask=5.0, epsilon=0.01, reduction='mean', toY=False):
        """
        Custom Masked PSNR Loss

        PSNR을 계산할 때, GT 이벤트가 0에 가까운 영역에 대해 추가 가중치를 적용하여
        weighted MSE를 구한 뒤 로그 변환하여 손실로 사용.

        Args:
            loss_weight (float): 전체 손실에 곱할 가중치.
            lambda_mask (float): GT 이벤트가 0에 가까운 영역에 추가적으로 적용할 페널티 강도.
            epsilon (float): GT 이벤트가 0로 간주할 임계값.
            reduction (str): 'mean'만 지원.
            toY (bool): True이면, 입력을 Y 채널로 변환 후 계산 (예: YUV 변환).
        """
        super(CustomMaskPSNRLoss, self).__init__()
        assert reduction == 'mean', "Only 'mean' reduction is supported."
        self.loss_weight = loss_weight
        self.lambda_mask = lambda_mask
        self.epsilon = epsilon
        self.toY = toY
        self.scale = 10 / np.log(10)
        # Y 채널 변환을 위한 계수 (예: ITU-R BT.601)
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        # Y 채널로 변환할 경우 (예: RGB -> Y)
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1, keepdim=True) + 16.
            target = (target * self.coef).sum(dim=1, keepdim=True) + 16.
            pred, target = pred / 255., target / 255.
        
        # GT 이벤트가 0에 가까운 영역에 대해 마스크 생성 및 가중치 계산
        mask = (target.abs() < self.epsilon).float()
        weight = 1.0 + self.lambda_mask * mask
        
        # 각 배치별로 weighted MSE를 계산:
        # (sum(weight * (pred-target)^2) / sum(weight)) 의 형태로 계산
        weighted_sum = (weight * (pred - target) ** 2).sum(dim=(1,2,3))
        normalization = weight.sum(dim=(1,2,3)) + 1e-8  # 0 나누는 경우 방지
        weighted_mse = weighted_sum / normalization
        
        # PSNR loss: scale * log(weighted_mse + 1e-8)
        loss = self.loss_weight * self.scale * torch.log(weighted_mse + 1e-8).mean()
        return loss