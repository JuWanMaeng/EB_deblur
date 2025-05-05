import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from einops import rearrange

# arch_util 에 정의된 채널‐어텐션 블록들
from basicsr.models.archs.arch_util import (
    LayerNorm2d,
    EdgeAwareSharpening_ChannelAttentionTransformerBlock,
    MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock
)
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock

# ─── 유틸리티 ──────────────────────────────────────────────────────────
def to_3d(x):
    # [B,C,H,W] → [B, H*W, C]
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    # [B, H*W, C] → [B, C, H, W]
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# ─── SimpleGate & 기본 NAF 블록 ────────────────────────────────────────
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """원본 NAFBlock (이미 정의돼 있으므로 파라미터만 간단히 표시합니다)"""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_rate=0.):
        super().__init__()
        dw = c * DW_Expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw, bias=True)
        self.sg    = SimpleGate()
        self.sca   = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(dw//2, dw//2, 1, bias=True))
        self.conv3 = nn.Conv2d(dw//2, c, 1, bias=True)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, FFN_Expand*c, 1, bias=True)
        self.conv5 = nn.Conv2d((FFN_Expand*c)//2, c, 1, bias=True)
        self.beta  = nn.Parameter(torch.zeros(1,c,1,1))
        self.gamma = nn.Parameter(torch.zeros(1,c,1,1))
        self.drop1 = nn.Dropout(drop_rate) if drop_rate>0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate>0 else nn.Identity()

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x); x = self.conv2(x); x = self.sg(x); x = x * self.sca(x)
        x = self.conv3(x); x = self.drop1(x)
        y = inp + x * self.beta

        x = self.norm2(y); x = self.conv4(x); x = self.sg(x)
        x = self.conv5(x); x = self.drop2(x)
        return y + x * self.gamma

# ─── 이벤트용 NAFBlock ─────────────────────────────────────────────────
class NAFBlock_event(nn.Module):
    """
    각 스케일에서 img_feat ←→ event_feat 를 fusion 하고
    일반 NAFBlock 처리를 수행합니다.
    """
    def __init__(self, c, dim, num_heads, num_layers,
                 DW_Expand=2, FFN_Expand=2, drop_rate=0.):
        super().__init__()
        # Cross‐attention + MLP 은 arch_util 에 이미 정의된 블록 사용
        
        self.cross = EventImage_ChannelAttentionTransformerBlock(
            dim=c, num_heads=num_heads, ffn_expansion_factor=2
        )
        self.nafbasic = NAFBlock(c, DW_Expand, FFN_Expand, drop_rate)

    def forward(self, img_feat, evt_feat):
        # img_feat, evt_feat: both [B, C, H, W]
        x = self.cross(img_feat, evt_feat)
        return self.nafbasic(x)

# ─── 이벤트 UNet 인코더 ─────────────────────────────────────────────────
class NAFEncoder(nn.Module):
    """evt_feat 만 받아서 3단계 피쳐 리턴"""
    def __init__(self, enc_blk_nums, width):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.downs     = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, 2*ch, 2, stride=2))
            ch *= 2

    def forward(self, x):
        feats = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            feats.append(x)
            x = down(x)
        return feats  # [f1, f2, f3]

# ─── 전체 NAFNet_cross ───────────────────────────────────────────────────
class NAFNet_cross(nn.Module):
    def __init__(self,
                 img_channel=3,
                 evt_channel=6,
                 width=64,
                 enc_blk_nums=[1,1,1],
                 dec_blk_nums=[1,1,1],
                 middle_blk_num=1,
                 dim=64,
                 num_heads=[1,2,4],
                 num_layers=[1,1,1]):
        super().__init__()
        # 1) 입력 프로젝션
        self.img_intro    = nn.Conv2d(img_channel, width, 3, padding=1)
        self.evt_intro    = nn.Conv2d(evt_channel, width, 3, padding=1)
        self.edge_proj    = nn.Conv2d(2, width, 3, padding=1)
        self.motion_proj = nn.Conv2d(2, width, 3, padding=1)
        self.ending       = nn.Conv2d(width, img_channel, 3, padding=1)

        # 2) 이벤트 UNet 인코더
        self.event_encoder = NAFEncoder(enc_blk_nums, width)

        # 3) 스케일별 모듈 정의
        self.edge_modules   = nn.ModuleList()
        self.motion_modules = nn.ModuleList()
        self.encoders       = nn.ModuleList()
        self.downs          = nn.ModuleList()
        self.dcns           = nn.ModuleList()
        ch = width
        for i, n in enumerate(enc_blk_nums):
            self.edge_modules.append(
                EdgeAwareSharpening_ChannelAttentionTransformerBlock(
                    dim=ch, num_heads=num_heads[i]
                )
            )
            self.motion_modules.append(
                MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(
                    dim=ch, num_heads=num_heads[i]
                )
            )
            # NAFBlock_event 를 n개 쌓기
            self.encoders.append(nn.ModuleList([
                NAFBlock_event(ch, dim, num_heads[i], num_layers[i])
                for _ in range(n)
            ]))
            # 다운샘플, DeformConv2d
            self.downs.append(nn.Conv2d(ch, 2*ch, 2, stride=2))
            self.dcns.append(DeformConv2d(ch, ch, 3, padding=1, bias=False))
            ch *= 2

        # 4) Middle + Decoder
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])
        self.ups    = nn.ModuleList()
        self.decs   = nn.ModuleList()
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch*2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            ch //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

    def forward(self, y):
        B,C,H,W = y.shape
        img    = y[:, :3]
        evt    = y[:, 3:]
        edge   = evt[:, 2:4]
        motion = torch.cat([evt[:,0:1], evt[:,-1:]], dim=1)

        # 임베딩
        x          = self.img_intro(img)
        evt_feat   = self.evt_intro(evt)
        edge_feat  = self.edge_proj(edge)
        motion_feat= self.motion_proj(motion)

        # multi‐scale 이벤트 피쳐
        e_feats = self.event_encoder(evt_feat) + [None]

        skips = []
        for i, (e_mod, m_mod, enc_list, down, dcn, e_f) in enumerate(zip(
            self.edge_modules,
            self.motion_modules,
            self.encoders,
            self.downs,
            self.dcns,
            e_feats
        )):
            # 1) Edge‐aware
            x = e_mod(x, e_f, edge_feat)

            # 2) NAFBlock_event
            for blk in enc_list:
                x = blk(x, e_f)

            # 3) Motion‐driven + DeformConv
            offset = m_mod(motion_feat, e_f)
            x = dcn(x, offset)

            skips.append(x)
            x = down(x)
            if e_f is not None:
                e_f = F.interpolate(e_f, 0.5, mode='bilinear', align_corners=False)
            edge_feat   = F.interpolate(edge_feat, 0.5, mode='bilinear', align_corners=False)
            motion_feat = F.interpolate(motion_feat, 0.5, mode='bilinear', align_corners=False)

        # middle
        x = self.middle(x)

        # decoder
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x) + skip
            x = dec(x)

        return self.ending(x) + img

# ─── 간단한 디버깅용 main ─────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NAFNet_cross().to(device).eval()
    y = torch.randn(2, 9, 256, 256, device=device)
    out = model(y)
    print("out.shape:", out.shape)  # -> [2,3,256,256]
    # backward check
    out.mean().backward()
    print("OK")  
