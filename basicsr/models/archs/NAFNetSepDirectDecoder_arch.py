import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import deform_conv2d
import math

# arch_util 에 정의된 채널‐어텐션 블록들
from basicsr.models.archs.NAFNetSepCross_util import (
    LayerNorm2d,LayerNorm, Mutual_Attention, Mlp
)

# event encoder를 사용하지 않고 edge, motion을 blur feature와 cross attention
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class EMAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        # 1) SCER edge‐prior (2ch) → dim 차원으로 투사
        self.edge_proj = nn.Conv2d(2, dim, kernel_size=3, padding=1, bias=True)
        # 2) image_feat(=x) 와 edge_proj(e) concat → dim 차원으로 줄임
        self.merge_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # 이제 기존 블록과 동일
        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn        = Mutual_Attention(dim, num_heads, bias)

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden  = int(dim * ffn_expansion_factor)
        self.ffn   = Mlp(in_features=dim, hidden_features=hidden, act_layer=nn.GELU, drop=0.)

    def forward(self, image_feat, prior):

        x = image_feat + self.attn(self.norm1_image(image_feat), self.norm1_event(prior))

        B,C,H,W = x.shape
        flat = rearrange(x, 'b c h w -> b (h w) c')
        flat = flat + self.ffn(self.norm2(flat))
        fused = rearrange(flat, 'b (h w) c -> b c h w', h=H, w=W)

        return fused
    
class MotionDeformableTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1_motion = LayerNorm(dim, LayerNorm_type)
        self.norm1_image_feat  = LayerNorm(dim, LayerNorm_type)
        self.attn         = Mutual_Attention(dim, num_heads, bias)

        # offset 예측용 conv (2 * kH * kW 채널을 뽑아야 함)
        self.conv_offset  = nn.Conv2d(dim, 18, kernel_size=3, padding=1, bias=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * ffn_expansion_factor)
        self.ffn   = Mlp(dim, mlp_hidden, dim, act_layer=nn.GELU, drop=0.)

        self.deform_weight = nn.Parameter(torch.empty(dim, dim, 3, 3))
        nn.init.kaiming_uniform_(self.deform_weight, a=math.sqrt(5))

    def forward(self, image_feat, motion): 
        # motion, event: [B, C, H, W]

        b, c, h, w = motion.shape

        # cross‐attention
        feat = self.attn(self.norm1_image_feat(image_feat), self.norm1_motion(motion))  # [B, C, H, W]

        flat = to_3d(feat)                                  # [B, H*W, C]
        flat = flat + self.ffn(self.norm2(flat))            # Residual MLP
        feat = to_4d(flat, h, w)                            # [B, C, H, W]

        # offset 예측
        offset = self.conv_offset(feat)                     # [B, 2*kH*kW, H, W]

        out = deform_conv2d(
                input=image_feat,
                offset=offset,
                weight=self.deform_weight,
                bias=None,
                padding=1
                )
        
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ─── SimpleGate & 기본 NAF 블록 ────────────────────────────────────────
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

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

class EMNAFBlock(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.NAFBlock = NAFBlock(c)

        self.edge = EMAttentionTransformerBlock(c, num_heads)
        self.motion= MotionDeformableTransformerBlock(c, num_heads)

        # self.merge_conv = nn.Conv2d(c * 2, c, kernel_size=1, bias=True)


    def forward(self, inp, edge_f, motion_f):
        hi = self.NAFBlock(inp)

        edge_att = self.edge(hi,edge_f)
        motion_att = self.motion(edge_att,motion_f)

        # fused = torch.cat([edge_att, motion_att], dim=1)  
        # fused = self.merge_conv(fused)                   
        # 4) residual
        return motion_att + hi



class NAFNetSepEM_Direct_Decoder(nn.Module):
    def __init__(self,
                 img_channel=3,
                 width=64,
                 enc_blk_nums=[1,1,1,28],
                 dec_blk_nums=[1,1,1,1],
                 middle_blk_num=1,
                 num_heads=[1,2,4]):
        super().__init__()

        # 1) Input projections
        self.img_intro = nn.Conv2d(9, width, 3, padding=1)

        # 2) Per‐scale edge/motion projections
        #    scale 1     → width channels
        #    scale 1/2   → width*2 channels
        #    scale 1/4   → width*4 channels
        self.edge_projs = nn.ModuleList([
            nn.Conv2d(2,      width,   3, padding=1),
            nn.Conv2d(2,      width*2, 3, padding=1),
            nn.Conv2d(2,      width*4, 3, padding=1),
        ])
        self.motion_projs = nn.ModuleList([
            nn.Conv2d(2,      width,   3, padding=1),
            nn.Conv2d(2,      width*2, 3, padding=1),
            nn.Conv2d(2,      width*4, 3, padding=1),
        ])

        self.edge_projs_decoders = nn.ModuleList([
            nn.Conv2d(2,      width*4,   3, padding=1),
            nn.Conv2d(2,      width*2, 3, padding=1),
            nn.Conv2d(2,      width, 3, padding=1),
        ])
        self.motion_projs_decoders = nn.ModuleList([
            nn.Conv2d(2,      width*4,   3, padding=1),
            nn.Conv2d(2,      width*2, 3, padding=1),
            nn.Conv2d(2,      width, 3, padding=1),
        ])

        # 4) Encoder stages: edge‐aware, NAFBlock_event, motion‐driven deform
        self.edge_modules   = nn.ModuleList()
        self.motion_modules = nn.ModuleList()
        self.encoders       = nn.ModuleList()
        self.downs          = nn.ModuleList()
        self.dcns           = nn.ModuleList()

        chan = width
        j = 0
        for num in enc_blk_nums:
            if num != 1:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan) for _ in range(num)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[EMNAFBlock(chan, num_heads[j]) for _ in range(num)]
                    )
                )
                j += 1

            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # 5) Middle “bottleneck” blocks
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        # 6) Decoder stages
        self.ups  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_flag = 3
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if decoder_flag == 3:
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan) for _ in range(num)]
                    )
                )
                decoder_flag = 2
            else:
                self.decoders.append(
                    nn.Sequential(
                        *[EMNAFBlock(chan, num_heads[decoder_flag]) for _ in range(num)]
                    )
                )
                decoder_flag -= 1

        # 7) Final RGB projection
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

    def forward(self, y):
        B, C, H, W = y.shape
        img    = y[:, :3, :, :]
        evt    = y[:, 3:, :, :]
        # SCER: take channels 2&3 as edge, 0&5 as motion
        edge   = evt[:, 2:4, :, :]
        motion = torch.cat([evt[:, 0:1, :, :], evt[:, -1:, :, :]], dim=1)

        # 1) initial embeddings
        # x           = self.img_intro(img)
        x = self.img_intro(y)


        edge_feat   = edge
        motion_feat = motion

        edge_feats_decoder=[]
        motion_feats_decoder = []


        skips = []
        # 2) Encoder
        for i, enc in enumerate(self.encoders):
            if len(enc) == 1 and isinstance(enc[0], EMNAFBlock):
                ef = self.edge_projs[i](edge_feat)
                mf = self.motion_projs[i](motion_feat)
                edge_feats_decoder.append(edge_feat)
                motion_feats_decoder.append(motion_feat)
                x = enc[0](x, ef, mf)
            else:
                # 일반 NAFBlock sequence
                x = enc(x)

            skips.append(x)
            # 2.3) 다운샘플
            x = self.downs[i](x)
            # 2.4) prior 해상도 반으로
            edge_feat   = F.interpolate(edge_feat,   scale_factor=0.5, mode='bilinear', align_corners=False)
            motion_feat = F.interpolate(motion_feat, scale_factor=0.5, mode='bilinear', align_corners=False)


        # 3) Middle “bottleneck”
        x = self.middle(x)

        edge_feat   = edge
        motion_feat = motion
        i = 0
        # 4) Decoder
        for up, dec in zip(self.ups, self.decoders):
            x = up(x)
            skip = skips.pop()   # 마지막 skip 꺼내서
            x = x + skip
            if isinstance(dec[0], EMNAFBlock):
                edge_feat = edge_feats_decoder.pop()
                motion_feat = motion_feats_decoder.pop()

                ef = self.edge_projs_decoders[i](edge_feat)
                mf = self.motion_projs_decoders[i](motion_feat)
                x = dec[0](x,ef,mf)
                i+=1
            else:
                x = dec(x)


        # 5) Final projection + residual
        out = self.ending(x) + img
        return out[..., :H, :W]
# ─── 간단한 디버깅용 main ─────────────────────────────────────────────────
if __name__ == "__main__":
    device ='cuda'
    model = NAFNetSepEM_Direct().to(device).eval()
    y = torch.randn(8, 9, 256, 256, device=device)
    out = model(y)
    print("out.shape:", out.shape)  # -> [2,3,256,256]
    # backward check
    out.mean().backward()
    print("OK")  
