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
        return F.conv2d(img, self.kernel, groups=n_channels)

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



class WaveletDomainLoss(nn.Module):
    """
    Computes L1 loss in the wavelet domain using a differentiable Haar wavelet transform.
    For inputs of shape (B, C, H, W) (e.g., C=6) with data in [-1,1],
    it applies a 2D Haar DWT (level 1) via group convolution and computes the L1 loss
    between the wavelet coefficients of pred and target.
    """
    def __init__(self, loss_weight=1.0, wavelet='haar', reduction='mean'):
        """
        Args:
            loss_weight (float): Loss에 곱할 가중치.
            wavelet (str): 사용할 웨이블릿 종류 (현재는 'haar'만 지원).
            reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(WaveletDomainLoss, self).__init__()
        if wavelet != 'haar':
            raise ValueError("Only 'haar' wavelet is supported in this implementation.")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {["none", "mean", "sum"]}')
        self.loss_weight = loss_weight
        self.wavelet = wavelet
        self.reduction = reduction

        # Haar wavelet filters for 2D DWT (level 1)
        # These filters are defined for a 2x2 patch.
        # Standard normalization: each coefficient is scaled by 0.5.
        # LL (Approximation): captures low-frequency content.
        # LH, HL, HH (Details): capture horizontal, vertical, and diagonal details.
        haar_filters = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],   # LL
            [[[0.5, 0.5], [-0.5, -0.5]]],  # LH
            [[[0.5, -0.5], [0.5, -0.5]]],  # HL
            [[[0.5, -0.5], [-0.5, 0.5]]]   # HH
        ], dtype=torch.float32)  # shape: (4, 1, 2, 2)
        # Register as a buffer so that it's moved to the correct device.
        self.register_buffer('haar_filters', haar_filters)
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): shape (B, C, H, W). 예측 이미지 또는 event.
            target (Tensor): shape (B, C, H, W). GT 이미지 또는 event.
        Returns:
            A scalar tensor representing the L1 loss between the Haar wavelet coefficients.
        """
        B, C, H, W = pred.shape

        # Expand Haar filters for each channel using group convolution.
        # Original haar_filters: (4, 1, 2, 2). We replicate them for each channel.
        # Final filters shape: (4 * C, 1, 2, 2)
        filters = self.haar_filters.repeat(C, 1, 1, 1)  # shape: (4*C, 1, 2, 2)

        # Apply 2D convolution with stride=2 and groups=C to simulate level-1 DWT.
        # This applies the 4 filters independently to each channel.
        pred_coeffs = F.conv2d(pred, filters, stride=2, groups=C)  # shape: (B, 4*C, H//2, W//2)
        target_coeffs = F.conv2d(target, filters, stride=2, groups=C)

        # Compute L1 loss between the wavelet coefficients.
        loss = F.l1_loss(pred_coeffs, target_coeffs, reduction=self.reduction)
        return self.loss_weight * loss

