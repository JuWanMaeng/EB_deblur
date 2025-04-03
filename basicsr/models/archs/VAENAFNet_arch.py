# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)

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


class VAENAFNet(nn.Module):
    def __init__(self, img_channel=6, width=16, middle_blk_num=1, enc_blk_nums=[1,1,1,14], dec_blk_nums=[]):
        """
        Args:
            img_channel: 입력 이미지 채널 수
            width: 초기 feature channel 수
            middle_blk_num: Middle block의 총 개수 (짝수여야 합니다)
            enc_blk_nums: 각 encoder stage에서 NAFBlock의 개수 리스트
            dec_blk_nums: 각 decoder stage에서 NAFBlock의 개수 리스트
        """
        super().__init__()

        self.intro = nn.Conv2d(in_channels=6, out_channels=width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan *= 2

        # middle block를 반으로 분할 (짝수인지 확인)
        assert middle_blk_num % 2 == 0, "middle_blk_num must be even to split equally."
        half_middle = middle_blk_num // 2
        self.middle_encoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_middle)])
        self.middle_decoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_middle)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def encode(self, y):
        """Encoder 부분: intro, encoder stages, downsampling, 그리고 middle_encoder"""
        B, C, H, W = y.shape
        if y.shape[1] != 3:
            inp = y
            inp_img = y[:, 0:3, :, :]  # 예를 들어, 이벤트 데이터와 이미지를 사용하는 경우
        else:
            inp = y
            inp_img = y

        x = self.intro(inp)
        encs = []  # skip connection 저장

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        latent = self.middle_encoder(x)
        return latent, encs, inp_img, (H, W)

    def decode(self, latent, encs, inp_img, orig_size):
        """Decoder 부분: middle_decoder, upsampling, skip connection 및 최종 복원"""
        x = self.middle_decoder(latent)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp_img
        H, W = orig_size
        return x[:, :, :H, :W]

    def forward(self, y):
        latent, encs, inp_img, orig_size = self.encode(y)
        return self.decode(latent, encs, inp_img, orig_size)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



if __name__ == '__main__':
    # 디버깅을 위한 임의 입력 데이터 생성 (batch_size, 채널, height, width)
    # 입력 채널 수는 intro의 입력에 맞게 9로 설정 (예: 이벤트와 이미지 채널 포함)
    dummy_input = torch.randn(1, 6, 256, 256)

    # 모델 생성: 예시로 encoder와 decoder stage에 각각 2개의 NAFBlock을 사용합니다.
    model = VAENAFNet(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
    
    # 모델의 forward 경로 테스트
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # 추가적으로, encode와 decode를 별도로 테스트할 수도 있습니다.
    latent, encs, inp_img, orig_size = model.encode(dummy_input)
    recon = model.decode(latent, encs, inp_img, orig_size)
    print("Reconstructed output shape:", recon.shape)