import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import DeformConv2d

# 아래 두 모듈은 이미 정의되어 있다고 가정합니다.
#   EdgeAwareSharpening_ChannelAttentionTransformerBlock
#   MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class fftformer_EM(nn.Module):
    """
    FFTformer + SCER edge/motion fusion
    입력: y ∈ ℝ^{B, 3+6, H, W}  (RGB + 6-channel SCER)
    """
    def __init__(self,
                 inp_channels=9,      # 3 RGB + 6 event
                 out_channels=3,
                 dim=48,
                 num_blocks=[6,6,12,8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False):
        super().__init__()

        # 0) Event→SCER priors 분리용 (no learnable)
        #    SCER 채널: [0..5]
        #    -> edge = chans 2,3
        #    -> motion = chans 0,5

        # 1) 입력 임베딩
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 2) per-level edge/motion 프로젝션 (→ dim, 2·dim, 4·dim)
        self.edge_projs = nn.ModuleList([
            nn.Conv2d(2,      dim,     3, padding=1, bias=bias),
            nn.Conv2d(2,      dim*2,   3, padding=1, bias=bias),
            nn.Conv2d(2,      dim*4,   3, padding=1, bias=bias),
        ])
        self.motion_projs = nn.ModuleList([
            nn.Conv2d(2,      dim,     3, padding=1, bias=bias),
            nn.Conv2d(2,      dim*2,   3, padding=1, bias=bias),
            nn.Conv2d(2,      dim*4,   3, padding=1, bias=bias),
        ])

        # 3) encoder 레벨별 edge/motion 모듈
        self.edge_modules   = nn.ModuleList([
            EdgeAwareSharpening_ChannelAttentionTransformerBlock(dim,   num_heads=1) for _ in range(3)
        ])
        self.motion_modules = nn.ModuleList([
            MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(dim,   num_heads=1) for _ in range(3)
       ])

        # 4) 기존 fftformer UNet-style 인코더/디코더 준비
        #   (이름만 그대로 가져왔습니다)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim*2,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim*2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim*4,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias)
            for _ in range(num_blocks[2])
        ])

        # deform-conv 이용해 motion-driven 보정할 Conv 모듈
        self.dcn1 = DeformConv2d(dim,   dim,   3, padding=1, bias=False)
        self.dcn2 = DeformConv2d(dim*2, dim*2, 3, padding=1, bias=False)
        self.dcn3 = DeformConv2d(dim*4, dim*4, 3, padding=1, bias=False)

        # middle + decoder (기존과 동일)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim*4,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             att=True)
            for _ in range(num_blocks[2])
        ])
        self.up3_2 = Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim*2,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             att=True)
            for _ in range(num_blocks[1])
        ])
        self.up2_1 = Upsample(dim*2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             att=True)
            for _ in range(num_blocks[0])
        ])
        self.refinement   = nn.Sequential(*[
            TransformerBlock(dim=dim,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             att=True)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    def forward(self, y):
        B,C,H,W = y.shape
        # 1) split RGB / event
        rgb = y[:, :3]
        evt = y[:, 3:]

        # 2) SCER edge/motion priors
        edge   = evt[:, 2:4]                          # [B,2,H,W]
        motion = torch.cat([evt[:,0:1], evt[:,-1:]],1)# [B,2,H,W]

        # 3) patch embedding (rgb+evt→dim)
        x = self.patch_embed(y)

        # 4) level1
        e1 = self.encoder_level1(x)
        # 4-1) edge/motion 프로젝션
        ef1 = self.edge_projs[0](edge)
        mf1 = self.motion_projs[0](motion)
        # 4-2) edge-aware
        e1 = self.edge_modules[0](e1, e1, ef1)
        # 4-3) motion-driven deform
        offs1 = self.motion_modules[0](mf1, e1)
        e1   = self.dcn1(e1, offs1)

        # 5) down → level2
        x2 = self.down1_2(e1)
        e2 = self.encoder_level2(x2)
        ef2 = F.interpolate(ef1,   scale_factor=0.5, mode='bilinear', align_corners=False)
        mf2 = F.interpolate(mf1,   scale_factor=0.5, mode='bilinear', align_corners=False)
        # 5-1) edge/motion
        e2  = self.edge_modules[1](e2, e2, ef2)
        offs2 = self.motion_modules[1](mf2, e2)
        e2  = self.dcn2(e2, offs2)

        # 6) down → level3
        x3 = self.down2_3(e2)
        e3 = self.encoder_level3(x3)
        ef3 = F.interpolate(ef2,   scale_factor=0.5, mode='bilinear', align_corners=False)
        mf3 = F.interpolate(mf2,   scale_factor=0.5, mode='bilinear', align_corners=False)
        # 6-1) edge/motion
        e3  = self.edge_modules[2](e3, e3, ef3)
        offs3 = self.motion_modules[2](mf3, e3)
        e3  = self.dcn3(e3, offs3)

        # 7) decoder path (기존 FFTformer flow)
        d3 = self.decoder_level3(e3)
        u2 = self.up3_2(d3)
        u2 = self.reduce_chan_level2(u2)
        d2 = self.decoder_level2(u2 + e2)  # skip+fuse
        u1 = self.up2_1(d2)
        d1 = self.decoder_level1(u1 + e1)
        out = self.refinement(d1)
        out = self.output(out) + rgb

        return out
