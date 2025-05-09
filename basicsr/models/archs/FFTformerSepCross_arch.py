import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import DeformConv2d
import numbers
from basicsr.models.archs.NAFNetSepCross_util import (
    LayerNorm2d,
    EdgeAwareSharpening_ChannelAttentionTransformerBlock,
    MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock
)
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock

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


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

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


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d

        return output


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

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
## event UNetEncoder

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
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

# ─── FFTformer_cross with Edge/Motion modules ────────────────────────
class FFTformerSep(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 FB_heads = [1,2,4]
                 ):
        super().__init__()
        # 1) event UNetEncoder
        self.event_enc = UNetEncoder(in_channels=6, base_filters=dim)
        # 2) Image patch → transformer encoder
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1)
        self.enc1 = nn.Sequential(*[TransformerBlock(dim, ffn_expansion_factor, False) for _ in range(num_blocks[0])])
        self.attn1 = EventImage_ChannelAttentionTransformerBlock(dim, FB_heads[0], ffn_expansion_factor)
        self.down1 = Downsample(dim)
        self.enc2 = nn.Sequential(*[TransformerBlock(dim*2, ffn_expansion_factor, False) for _ in range(num_blocks[1])])
        self.attn2 = EventImage_ChannelAttentionTransformerBlock(dim*2, FB_heads[1], ffn_expansion_factor)
        self.down2 = Downsample(dim*2)
        self.enc3 = nn.Sequential(*[TransformerBlock(dim*4, ffn_expansion_factor, False) for _ in range(num_blocks[2])])
        self.attn3 = EventImage_ChannelAttentionTransformerBlock(dim*4, FB_heads[2], ffn_expansion_factor)
        # 3) Edge/Motion projections (인코더·디코더 모두 사용)
        self.edge_proj_enc = nn.ModuleList([
            nn.Conv2d(2, dim,    3, padding=1),
            nn.Conv2d(2, dim*2,  3, padding=1),
            nn.Conv2d(2, dim*4,  3, padding=1),
        ])
        self.motion_proj_enc = nn.ModuleList([
            nn.Conv2d(2, dim,    3, padding=1),
            nn.Conv2d(2, dim*2,  3, padding=1),
            nn.Conv2d(2, dim*4,  3, padding=1),
        ])
        self.edge_mod_enc   = nn.ModuleList([
            EdgeAwareSharpening_ChannelAttentionTransformerBlock(dim,  FB_heads[0]),
            EdgeAwareSharpening_ChannelAttentionTransformerBlock(dim*2,FB_heads[1]),
            EdgeAwareSharpening_ChannelAttentionTransformerBlock(dim*4,FB_heads[2]),
        ])
        self.mot_mod_enc    = nn.ModuleList([
            MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(dim,  FB_heads[0]),
            MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(dim*2,FB_heads[1]),
            MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(dim*4,FB_heads[2]),
        ])
        self.dcn_enc = nn.ModuleList([
            DeformConv2d(dim,   dim,   3, padding=1, bias=False),
            DeformConv2d(dim*2, dim*2, 3, padding=1, bias=False),
            DeformConv2d(dim*4, dim*4, 3, padding=1, bias=False),
        ])
 
        # 5) Decoder
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.dec3= nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.dec2=  nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.dec1= nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])
        
        # 6) Edge/Motion 모듈 (디코더)
        self.edge_proj_dec   = nn.ModuleList(self.edge_proj_enc[::-1])
        self.motion_proj_dec = nn.ModuleList(self.motion_proj_enc[::-1])
        self.edge_mod_dec    = nn.ModuleList(self.edge_mod_enc[::-1])
        self.mot_mod_dec     = nn.ModuleList(self.mot_mod_enc[::-1])
        self.dcn_dec         = nn.ModuleList(self.dcn_enc[::-1])
        # 7) 최종 출력
        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, y):
        B,C,H,W = y.shape
        img    = y[:,:3]
        evt    = y[:,3:]

        # SCER → edge/motion
        edge0 = evt[:,2:4]
        mot0  = torch.cat([evt[:,0:1], evt[:,-1:]], dim=1)

        # 1) event UNet 인코딩
        feats = self.event_enc(evt)  # [f1,f2,f3]

        # 초기 priors
        edge_feat, mot_feat = edge0, mot0

        # 2) image → level1 encoder
        x = self.patch_embed(img)
        x = self.enc1(x)
        x = self.attn1(x, feats[0])

        # --- 바로 Edge/Motion 모듈 적용 후 다운샘플 ---
        # scale=1
        ef = self.edge_proj_enc[0](edge_feat)
        mf = self.motion_proj_enc[0](mot_feat)
        x = self.edge_mod_enc[0](x, feats[0], ef)
        offs = self.mot_mod_enc[0](mf, feats[0])
        x = self.dcn_enc[0](x, offs)
        skips = [x]

        # priors 반해상도
        edge_feat = F.interpolate(edge_feat, scale_factor=0.5, mode='bilinear', align_corners=False)
        mot_feat  = F.interpolate(mot_feat,  scale_factor=0.5, mode='bilinear', align_corners=False)

        x = self.down1(x)  # → x2

        # 3) level2
        x = self.enc2(x)
        x = self.attn2(x, feats[1])
        ef = self.edge_proj_enc[1](edge_feat)
        mf = self.motion_proj_enc[1](mot_feat)
        x = self.edge_mod_enc[1](x, feats[1], ef)
        offs = self.mot_mod_enc[1](mf, feats[1])
        x = self.dcn_enc[1](x, offs)
        skips.append(x)

        edge_feat = F.interpolate(edge_feat,scale_factor=0.5, mode='bilinear', align_corners=False)
        mot_feat  = F.interpolate(mot_feat,  scale_factor=0.5, mode='bilinear', align_corners=False)

        x = self.down2(x)  # → x3

        # 4) level3
        x = self.enc3(x)
        x = self.attn3(x, feats[2])
        ef = self.edge_proj_enc[2](edge_feat)
        mf = self.motion_proj_enc[2](mot_feat)
        x = self.edge_mod_enc[2](x, feats[2], ef)
        offs = self.mot_mod_enc[2](mf, feats[2])
        x = self.dcn_enc[2](x, offs)
        skips.append(x)


        x_mid = skips[-1]

        # 6) decoder (원본 FFTformer_cross 방식)

        d3 = self.dec3(x_mid)
        d2 = self.up3_2(d3)
        d2 = self.fuse2(d2, skips[1])
        d2 = self.dec2(d2)
        d1 = self.up2_1(d2)
        d1 = self.fuse1(d1,skips[0])
        d1 = self.dec1(d1)

        out = self.refinement(d1)
        out = self.output(out) + img
        
        return out
    

# ─── 간단한 디버깅용 main ─────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FFTformerSep().to(device).eval()
    y = torch.randn(2, 9, 224, 224, device=device)
    out = model(y)
    print("out.shape:", out.shape)  # -> [2,3,256,256]
    # backward check
    out.mean().backward()
    print("OK")  
