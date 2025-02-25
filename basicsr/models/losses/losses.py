import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import math

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

# class FFTLoss(nn.Module):
#     """L1 loss in frequency domain using log1p of FFT magnitude.

#     Args:
#         loss_weight (float): Loss weight for FFT loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(FFTLoss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {["none", "mean", "sum"]}')
#         self.loss_weight = loss_weight
#         self.reduction = reduction

#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (..., C, H, W). Predicted tensor.
#             target (Tensor): of shape (..., C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (..., C, H, W). Element-wise weights. Default: None.
#         Returns:
#             L1 loss between the log1p(FFT magnitude) of pred and target.
#         """
#         # 1) FFT 변환
#         pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
#         target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
#         # 2) Magnitude 계산
#         pred_mag = torch.abs(pred_fft)
#         target_mag = torch.abs(target_fft)

#         # 3) log1p로 동적 범위 압축
#         #    log1p(x) = log(1 + x)
#         pred_log = torch.log1p(pred_mag)
#         target_log = torch.log1p(target_mag)

#         # 4) L1 Loss 계산
#         loss = l1_loss(pred_log, target_log, reduction=self.reduction)
#         return self.loss_weight * loss

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


class KLDivLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='batchmean', scale_factor=5.0):
        super(KLDivLoss, self).__init__()
        assert reduction in ['batchmean', 'sum', 'mean']
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.scale_factor = scale_factor  # Scaling Factor 추가

    def forward(self, pred, target):
        assert len(pred.size()) == 4  # [B, C, H, W] 형태여야 함

        # Scale Factor 적용하여 Softmax 변환
        pred_prob = F.softmax(pred * self.scale_factor, dim=1) + 1e-8
        target_prob = F.softmax(target * self.scale_factor, dim=1) + 1e-8

        # KL Divergence 계산
        loss = F.kl_div(pred_prob.log(), target_prob, reduction=self.reduction)

        return self.loss_weight * loss


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



class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False, window_size=11, channel=6):
        """
        Args:
            loss_weight (float): Loss에 곱할 가중치.
            reduction (str): 현재는 'mean'만 지원합니다.
            toY (bool): True이면 RGB 이미지를 Y 채널로 변환하여 SSIM 계산.
            window_size (int): Gaussian 윈도우 크기 (보통 11).
            channel (int): 입력 이미지의 채널 수.
        """
        super(SSIMLoss, self).__init__()
        assert reduction == 'mean', "Currently only 'mean' reduction is supported."
        self.loss_weight = loss_weight
        self.toY = toY
        self.window_size = window_size
        self.channel = channel
        self.first = True
        # Y 채널 변환을 위한 계수 (Rec.601)
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        # Gaussian 윈도우 생성
        self.window = self.create_window(window_size, channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)  # (window_size, 1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # (1,1,window_size,window_size)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # img1, img2: (B, channel, H, W)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        # 상수 C1, C2 (입력이 [0,1] 범위일 때)
        C1 = (0.01)**2
        C2 = (0.03)**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): shape (B, 3, H, W) (or (B,1,H,W) if toY is True) RGB image.
            target (Tensor): shape (B, 3, H, W) (or (B,1,H,W)) RGB image.
        Returns:
            A scalar tensor representing the SSIM loss (1 - SSIM).
        """
        # 입력 이미지의 최소값이 음수라면, [-1,1] 범위로 가정하고 [0,1]로 변환합니다.
        if pred.min() < 0:
            pred = (pred + 1) / 2
        if target.min() < 0:
            target = (target + 1) / 2

        # SSIM 계산 (입력이 [0,1] 범위라고 가정)
        ssim_val = self.ssim(pred, target, self.window.to(pred.device), self.window_size, self.channel, size_average=True)
        loss = self.loss_weight * (1 - ssim_val)
        return loss


class WaveletDomainLoss(nn.Module):
    """
    Computes L1 loss in the wavelet domain using a differentiable Haar wavelet transform.
    For inputs of shape (B, C, H, W), it applies a 2D Haar DWT (level 1) via group convolution
    and computes the weighted L1 loss between the wavelet coefficients of pred and target.
    Each subband (LL, LH, HL, HH) can have a different weight.
    """
    def __init__(self, 
                 loss_weight=1.0, 
                 wavelet='haar', 
                 reduction='mean',
                 subband_weights=(1.0, 1.0, 1.0, 1.0)):
        """
        Args:
            loss_weight (float): Global loss weight.
            wavelet (str): Wavelet type ('haar' only in this example).
            reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
            subband_weights (tuple of floats): (wLL, wLH, wHL, wHH) subband weights.
        """
        super(WaveletDomainLoss, self).__init__()
        if wavelet != 'haar':
            raise ValueError("Only 'haar' wavelet is supported in this implementation.")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {["none", "mean", "sum"]}')
        self.loss_weight = loss_weight
        self.wavelet = wavelet
        self.reduction = reduction
        # 서브밴드 가중치 저장
        if len(subband_weights) != 4:
            raise ValueError("subband_weights must be a tuple of length 4 (LL, LH, HL, HH).")
        self.subband_weights = subband_weights

        # Define Haar wavelet filters for 2D DWT (level 1)
        haar_filters = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],   # LL
            [[[0.5, 0.5], [-0.5, -0.5]]], # LH
            [[[0.5, -0.5], [0.5, -0.5]]], # HL
            [[[0.5, -0.5], [-0.5, 0.5]]]  # HH
        ], dtype=torch.float32)  # shape: (4, 1, 2, 2)
        # Register filters as a buffer so that they move with the model to GPU.
        self.register_buffer('haar_filters', haar_filters)
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): shape (B, C, H, W). Predicted event/image.
            target (Tensor): shape (B, C, H, W). GT event/image.
        Returns:
            Weighted L1 loss over subbands (LL, LH, HL, HH).
        """
        B, C, H, W = pred.shape

        # Expand Haar filters for each channel.
        # Original shape: (4, 1, 2, 2) -> repeated for C channels -> (4*C, 1, 2, 2)
        filters = self.haar_filters.repeat(C, 1, 1, 1)

        # Apply 2D convolution with stride=2 and groups=C to simulate level-1 DWT.
        # Output shape: (B, 4*C, H//2, W//2)
        pred_coeffs = F.conv2d(pred, filters, stride=2, groups=C)
        target_coeffs = F.conv2d(target, filters, stride=2, groups=C)

        # Now, pred_coeffs and target_coeffs are chunked by channel in blocks of 4.
        # subband i in [0..3], channel c in [0..C-1].
        # Index = c*4 + i
        # We'll compute L1 loss for each subband separately and multiply by subband_weights.
        total_loss = 0.0
        # We might do a loop over i=0..3 for each subband
        for i in range(4):
            # slice the i-th subband from all channels: shape (B, C, H//2, W//2)
            # subband index: i, i+4, i+8, ... => stride=4
            # but easier to chunk in groups of 4 along dim=1
            sub_pred = pred_coeffs[:, i::4, :, :]  # shape (B, ?, H//2, W//2)
            sub_tgt = target_coeffs[:, i::4, :, :]
            # L1 loss for this subband
            sub_loss = F.l1_loss(sub_pred, sub_tgt, reduction=self.reduction)
            # multiply by subband weight
            w = self.subband_weights[i]
            total_loss += w * sub_loss

        # subband_losses를 모두 합산한 뒤, 전체에 loss_weight 곱함
        return self.loss_weight * total_loss
    


