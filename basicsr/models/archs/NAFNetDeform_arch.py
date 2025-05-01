import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock, LayerNorm2d, to_3d, to_4d
from mmcv.ops import DeformConv2d

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
    
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1-mask)*enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask*dec)
            out = out + out_enc + out_dec        
            
        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
            
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class EdgeSharpeningModule(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.edge_conv = nn.Conv2d(2, dim, 3, padding=1, bias=True)
        self.q_conv = nn.Conv2d(dim * 2, dim, 1, bias=True)
        self.cross_attn = EventImage_ChannelAttentionTransformerBlock(dim, num_heads, ffn_expansion_factor=2, bias=bias)

    def forward(self, img_feat, evt_feat, e_scer):
        e_edge = self.edge_conv(e_scer)
        q = self.q_conv(torch.cat([img_feat, e_edge], dim=1))
        return self.cross_attn(q, evt_feat)

class MotionDeblurModule(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.motion_conv = nn.Conv2d(2, dim, 3, padding=1, bias=True)
        self.cross_attn = EventImage_ChannelAttentionTransformerBlock(dim, num_heads, ffn_expansion_factor=2, bias=bias)
        self.offset_conv = nn.Conv2d(dim, 18, 3, padding=1, bias=True)
        self.dcn = DeformConv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, feat, evt_feat, e_scer):
        m_prior = self.motion_conv(e_scer)
        m_feat = self.cross_attn(m_prior, evt_feat)
        offsets = self.offset_conv(m_feat)
        return self.dcn(feat, offsets)

class EFNetExtended(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3,
                 fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super().__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        # initial convs
        self.conv_img1 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_img2 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_evt1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)
        # prepare channel dims per scale
        dims = [wf * (2**i) for i in range(depth)]
        prev_dims = [wf] + dims[:-1]
        # UNet blocks and modules
        self.down_img = nn.ModuleList()
        self.down_evt = nn.ModuleList()
        self.down_refine = nn.ModuleList()
        self.edge_modules = nn.ModuleList()
        self.motion_modules = nn.ModuleList()
        for i in range(depth):
            in_ch = prev_dims[i]
            out_ch = dims[i]
            self.down_img.append(UNetConvBlock(in_ch, out_ch, downsample=(i<depth-1), relu_slope=relu_slope, num_heads=num_heads[i]))
            self.down_evt.append(UNetEVConvBlock(in_ch, out_ch, downsample=(i<depth-1), relu_slope=relu_slope))
            self.down_refine.append(UNetConvBlock(in_ch, out_ch, downsample=(i<depth-1), relu_slope=relu_slope, use_emgc=(i<depth-1)))
            self.edge_modules.append(EdgeSharpeningModule(out_ch, num_heads[i]))
            self.motion_modules.append(MotionDeblurModule(out_ch, num_heads[i]))
        # decoder paths
        self.up1 = nn.ModuleList()
        self.up2 = nn.ModuleList()
        self.skip1 = nn.ModuleList()
        self.skip2 = nn.ModuleList()
        for i in reversed(range(depth-1)):
            in_ch = dims[i+1]
            out_ch = dims[i]
            self.up1.append(UNetUpBlock(in_ch, out_ch, relu_slope))
            self.up2.append(UNetUpBlock(in_ch, out_ch, relu_slope))
            self.skip1.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
            self.skip2.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.sam = SAM(dims[0])
        self.cat = nn.Conv2d(dims[0]*2, dims[0], 1, 1, 0)
        self.final = nn.Conv2d(dims[0], in_chn, 3, 1, 1)

    def forward(self, x, event, mask=None):
        # prepare SCER priors
        B,C,H,W = event.shape
        e_edge = event[:, C//2-2:C//2, :, :]
        e_motion = torch.cat([event[:, :1], event[:, -1:]], dim=1)
        # event encoding per scale
        ev_feats=[]
        e = self.conv_evt1(event)
        for i, block in enumerate(self.down_evt):
            if i<self.depth-1:
                e, e_up = block(e, self.fuse_before_downsample)
                ev_feats.append(e_up if self.fuse_before_downsample else e)
            else:
                e = block(e, self.fuse_before_downsample)
                ev_feats.append(e)
                # stage1 image + fusion
        x1 = self.conv_img1(x)
        img_feats=[]; dec1=[]
        for i, block in enumerate(self.down_img):
            if i<self.depth-1:
                # block returns (downsampled, skip) when downsample=True
                out_down, out_up = block(x1, event_filter=ev_feats[i], merge_before_downsample=self.fuse_before_downsample)
                # apply edge & motion modules on the skip feature
                fused_edge = self.edge_modules[i](out_up, ev_feats[i], e_edge)
                fused_motion = self.motion_modules[i](fused_edge, ev_feats[i], e_motion)
                img_feats.append(fused_motion)
                # use the downsampled output for next iteration
                x1 = out_down
            else:
                x1 = block(x1, event_filter=ev_feats[i], merge_before_downsample=self.fuse_before_downsample)
                img_feats.append(x1)
        # decode stage1
        for i, up in enumerate(self.up1):
            x1 = up(x1, self.skip1[i](img_feats[-i-1]))
            dec1.append(x1)

        for i, up in enumerate(self.up1):
            x1 = up(x1, self.skip1[i](img_feats[-i-1]))
            dec1.append(x1)
        sam_feat, out1 = self.sam(x1, x)
        # stage2 refine
        x2 = self.conv_img2(x)
        x2 = self.cat(torch.cat([x2, sam_feat], dim=1))
        refine_preds=[]
        for i, block in enumerate(self.down_refine):
            if i<self.depth-1:
                mask_i = None
                if mask is not None:
                    mask_i = F.interpolate(mask, scale_factor=0.5**i, mode='nearest')
                x2, x2_up = block(x2, img_feats[i], dec1[-i-1], mask=mask_i)
                refine_preds.append(x2_up)
                x2 = x2_up if self.fuse_before_downsample else x2
            else:
                x2 = block(x2)
        for i, up in enumerate(self.up2):
            x2 = up(x2, self.skip2[i](refine_preds[-i-1]))
        out2 = self.final(x2) + x
        return [out1, out2]

if __name__ == "__main__":
    model = EFNetExtended()
    b = torch.randn(1,3,256,256)
    e = torch.randn(1,6,256,256)
    o1,o2 = model(b, e)
    print(o1.shape, o2.shape)
