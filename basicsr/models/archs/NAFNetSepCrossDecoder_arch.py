import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from einops import rearrange

from basicsr.models.archs.NAFNetSepCross_util import (
    LayerNorm2d,
    EdgeAwareSharpening_ChannelAttentionTransformerBlock,
    MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock
)
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h,w=w)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1,x2 = x.chunk(2,dim=1)
        return x1*x2

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

class NAFEncoder(nn.Module):
    def __init__(self, enc_blk_nums, width):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch=width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch,2*ch,2,stride=2))
            ch*=2
    def forward(self,x):
        feats=[]
        for enc,down in zip(self.encoders,self.downs):
            x=enc(x); feats.append(x); x=down(x)
        return feats

class NAFNetSepCrossDecoder(nn.Module):
    def __init__(self, img_channel=3, evt_channel=6, width=64,
                 enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],
                 middle_blk_num=28, dim=64,
                 num_heads=[1,2,4], num_layers=[1,1,1]):
        super().__init__()

        # ─── 1) input proj ─────────────────────────────────
        self.img_intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.evt_intro = nn.Conv2d(evt_channel, width, 3, padding=1)

        # ─── 2) per-scale prior proj ────────────────────────
        self.enc_edge_projs = nn.ModuleList([
            nn.Conv2d(2,     width,   3, padding=1),
            nn.Conv2d(2,     width*2, 3, padding=1),
            nn.Conv2d(2,     width*4, 3, padding=1),
        ])
        self.enc_motion_projs = nn.ModuleList([
            nn.Conv2d(2,     width,   3, padding=1),
            nn.Conv2d(2,     width*2, 3, padding=1),
            nn.Conv2d(2,     width*4, 3, padding=1),
        ])

        # 3) 디코더 전용 edge/motion proj (역순)
        self.dec_edge_projs = nn.ModuleList([
            nn.Conv2d(2,     width*4, 3, padding=1),
            nn.Conv2d(2,     width*2, 3, padding=1),
            nn.Conv2d(2,     width,   3, padding=1),
        ])
        self.dec_motion_projs = nn.ModuleList([
            nn.Conv2d(2,     width*4, 3, padding=1),
            nn.Conv2d(2,     width*2, 3, padding=1),
            nn.Conv2d(2,     width,   3, padding=1),
        ])

        # ─── 3) event UNet encoder ─────────────────────────
        self.event_encoder = NAFEncoder(enc_blk_nums, width)

        # ─── 4) encoder stages ─────────────────────────────
        self.edge_modules   = nn.ModuleList()
        self.motion_modules = nn.ModuleList()
        self.encoders       = nn.ModuleList()
        self.downs          = nn.ModuleList()
        self.dcns           = nn.ModuleList()
        ch = width
        for i,n in enumerate(enc_blk_nums):
            self.edge_modules.append(
                EdgeAwareSharpening_ChannelAttentionTransformerBlock(ch, num_heads[i])
            )
            self.motion_modules.append(
                MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(ch, num_heads[i])
            )
            self.encoders.append(nn.ModuleList([
                NAFBlock_event(ch, dim, num_heads[i], num_layers[i])
                for _ in range(n)
            ]))
            self.downs.append(nn.Conv2d(ch,2*ch,2,stride=2))
            self.dcns.append(DeformConv2d(ch,ch,3,padding=1,bias=False))
            ch*=2

        # ─── 5) bottleneck ─────────────────────────────────
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        # ─── 6) decoder stages (with edge/motion) ──────────
        self.ups=[]
        self.decs=[]
        self.dec_edge_mod=[]
        self.dec_motion_mod=[]
        self.dec_dcns=[]
        for i,n in enumerate(dec_blk_nums[::-1]):
            # upsample
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch*2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            # 채널 반감
            prev_ch=ch//2
            # edge/motion proj at decoder
            self.dec_edge_mod.append(
                EdgeAwareSharpening_ChannelAttentionTransformerBlock(prev_ch, num_heads[-1-i])
            )
            self.dec_motion_mod.append(
                MotionDrivenScaleAdaptiveDeblurring_ChannelAttentionTransformerBlock(prev_ch, num_heads[-1-i])
            )
            # deform conv
            self.dec_dcns.append(DeformConv2d(prev_ch,prev_ch,3,padding=1,bias=False))
            # NAFBlocks
            self.decs.append(nn.Sequential(*[NAFBlock(prev_ch) for _ in range(n)]))
            ch=prev_ch

        self.ups = nn.ModuleList(self.ups)
        self.decs= nn.ModuleList(self.decs)
        self.dec_edge_mod   = nn.ModuleList(self.dec_edge_mod)
        self.dec_motion_mod = nn.ModuleList(self.dec_motion_mod)
        self.dec_dcns       = nn.ModuleList(self.dec_dcns)

        # ─── 7) final RGB ───────────────────────────────────
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

    def forward(self,y):
        B,C,H,W = y.shape
        img = y[:,:3]; evt=y[:,3:]
        edge   = evt[:,2:4]
        motion=torch.cat([evt[:,0:1],evt[:,-1:]],1)

        x       = self.img_intro(img)
        evt_feat= self.evt_intro(evt)
        e_feats = list(self.event_encoder(evt_feat))+[None]
        edge_feat=edge
        motion_feat=motion

        # ─── encoder w/ edge & motion ──────────────────────────────────────────
        skips = []
        for i, (e_mod, m_mod, enc_list, down, dcn, e_f) in enumerate(zip(
                self.edge_modules,         # Edge-aware block list
                self.motion_modules,       # Motion-driven block list
                self.encoders,             # NAFBlock_event lists
                self.downs,                # Downsample convs
                self.dcns,                 # DeformConv2d modules
                e_feats                    # multi-scale event features
            )):
            # 1) 현재 스케일에 맞춰 priors 투사
            ef = self.enc_edge_projs[i](edge_feat)      # [B, C_i, H_i, W_i]
            mf = self.enc_motion_projs[i](motion_feat)  # [B, C_i, H_i, W_i]

            # 2) NAFBlock_event (이미 event와 img를 fuse)
            for blk in enc_list:
                x = blk(x, e_f)

            # 3) Edge‐aware sharpening
            x = e_mod(x, e_f, ef)

            # 4) Motion‐driven offset 예측 + DeformConv
            offset = m_mod(mf, e_f)
            x      = dcn(x, offset)

            # 5) 이 단계 output을 skip에 저장
            skips.append(x)

            # 6) 다음 스케일로 다운샘플
            x = down(x)
            edge_feat   = F.interpolate(edge_feat,   scale_factor=0.5,
                                        mode='bilinear', align_corners=False)
            motion_feat = F.interpolate(motion_feat, scale_factor=0.5,
                                        mode='bilinear', align_corners=False)
        # ─── middle ───────────────────────────────────────
        x = self.middle(x)

        edge_feat   = F.interpolate(edge_feat,   scale_factor=2,
                                    mode='bilinear', align_corners=False)
        motion_feat = F.interpolate(motion_feat, scale_factor=2,
                                    mode='bilinear', align_corners=False)
        
        # 먼저 가장 낮은 해상도부터 시작할 때 사용할 proj index:
        for i, (up, dec, e_mod, m_mod, dcn, edge_proj, motion_proj) in enumerate(zip(
                self.ups,                 # upsampling 모듈
                self.decs,                # NAFBlock (decoder) 모듈
                self.dec_edge_mod,        # 디코더용 Edge‐aware 모듈
                self.dec_motion_mod,      # 디코더용 Motion‐driven 모듈
                self.dec_dcns,            # 디코더용 DeformConv2d
                self.dec_edge_projs,      # 디코더용 edge proj 리스트
                self.dec_motion_projs     # 디코더용 motion proj 리스트
            )):
            # 1) upsample + skip-connection
            x = up(x)
            x = x + skips.pop()

            # 2) 이 단계 해상도에 맞춰 priors 투사
            ef = edge_proj(edge_feat)       # [B, C_i, H_i, W_i]
            mf = motion_proj(motion_feat)   # [B, C_i, H_i, W_i]

            # 3) edge‐aware sharpening
            x = e_mod(x, x, ef)

            # 4) motion‐driven offset 예측 및 deform_conv
            offset = m_mod(mf, x)
            x      = dcn(x, offset)

            # 5) decoder NAFBlock
            x = dec(x)

            # 6) 다음 해상도를 위해 priors 업샘플
            edge_feat   = F.interpolate(edge_feat,   scale_factor=2,
                                        mode='bilinear', align_corners=False)
            motion_feat = F.interpolate(motion_feat, scale_factor=2,
                                        mode='bilinear', align_corners=False)
            
        # ─── 최종 projection + residual ─────────────────────────────────────────
        out = self.ending(x) + img
        return out[..., :H, :W]

if __name__=="__main__":
    m = NAFNetSepCrossDecoder().cuda().eval()
    y = torch.randn(2,9,256,256).cuda()
    o = m(y)
    print("OK", o.shape)
