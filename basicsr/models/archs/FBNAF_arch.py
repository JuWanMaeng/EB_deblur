
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from einops import rearrange
# from basicsr.models.archs.Flow_arch import KernelPrior
# from basicsr.models.archs.my_module import code_extra_mean_var

import numpy as np
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# class ResBlock(nn.Module):
#     def __init__(self, ch):
#         super(ResBlock, self).__init__()
#         self.body = nn.Sequential(
#                         nn.ReLU(),
#                         nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
#                         nn.ReLU(),
#                         nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

#     def forward(self, input):
#         res = self.body(input)
#         output = res + input
#         return output


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class flow_attention(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super(flow_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_flow = nn.Sequential(
                        nn.Conv2d(dim, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, flow):
        x = self.conv_1(input)
        flow = self.conv_flow(flow)
        att = torch.cat([x, flow], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class NAFBlock_flow(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dim=48):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = flow_attention(dim, in_ch=c, out_ch=c)

    def forward(self, inp, flow):
        x = inp

        # flow [B, C, H, W], C = dim, dim*2, dim*4
        x = self.kernel_atttion(x, flow)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

###########################################################################
## cross attention between flow and blur 

class FBCA(nn.Module):
    def __init__(self, dim, bias):
        super(FBCA, self).__init__()

        self.q = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8
        

    def forward(self, x, y):
        
        # x: image feature, y: flow feature
        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # flow
        v = self.v(y) # flow

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        out = self.norm(out)

        output = v * out
        output = self.project_out(output)
        
        return output

class MultiLayerFBCA(nn.Module):
    def __init__(self, dim, bias, num_layers):
        super(MultiLayerFBCA, self).__init__()
        # 여러 개의 Cross-Attention 레이어 쌓기
        self.layers = nn.ModuleList([FBCA(dim, bias) for _ in range(num_layers)])

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x, y)
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


#######################################################################
## Flow UNetEncoder

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # If input and output channels differ, add a convolution for the skip connection
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_conv(x)
        x = self.conv_block(x)
        return x + residual  # Add skip connection (residual path)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=48):
        super(UNetEncoder, self).__init__()
        
        self.encoder1 = ResBlock(in_channels, base_filters)
        self.down1 = Downsample(base_filters)
        
        self.encoder2 = ResBlock(base_filters * 2, base_filters * 2)
        self.down2 = Downsample(base_filters * 2)
        
        self.encoder3 = ResBlock(base_filters * 4, base_filters * 4)
        self.down3 = Downsample(base_filters * 4)
        
        # 필요시 추가
        # self.encoder4 = ResBlock(base_filters * 8, base_filters * 8)
        # self.down4 = Downsample(base_filters * 8)

    def forward(self, x):
        # Encoder 1
        enc1 = self.encoder1(x)  
        x = self.down1(enc1)     
        
        # Encoder 2
        enc2 = self.encoder2(x)  
        x = self.down2(enc2)     
        
        # Encoder 3
        enc3 = self.encoder3(x)  

        return enc1, enc2, enc3


# def generate_k(model, code, n_row=1):
#     model.eval()

#     # unconditional model
#     # for a random Gaussian vector, its l2norm is always close to 1.
#     # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

#     u = code  # [B, 19*19]
#     samples, _ = model.inverse(u)

#     samples = model.post_process(samples)

#     return samples


class FBNAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dim=96):
        super().__init__()

        self.dim = dim
        # Flow UNetEncoder
        self.FUNetEncoder = UNetEncoder(in_channels=3, base_filters=self.dim)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.kernel_down = nn.ModuleList()


        chan = width

        for num in enc_blk_nums:
            if num== 1:
                self.encoders.append(nn.Sequential(*[NAFBlock_flow(chan, dim=self.dim) for _ in range(num)]))
                self.dim = self.dim * 2
            else:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, y):
        B, C, H, W = y.shape

        inp = y[:,0:3,:,:]
        flow = y[:,3:,:,:]

        flow_enc1, flow_enc2, flow_enc3 = self.FUNetEncoder(flow)  # B, dim, 256, 256  # B, dim*2, 128, 128  # B, dim*4, 64, 64
        flow_features = [flow_enc1, flow_enc2, flow_enc3, None]

        x = self.intro(inp)

        encs = []

        for encoder, down, flow in zip(self.encoders, self.downs, flow_features):
            if len(encoder) == 1:
                x = encoder[0](x, flow)
                # kernel = kernel_down(kernel)
            else:
                x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        x = torch.clamp(x, 0.0, 1.0)

        return x[:, :, :H, :W]



