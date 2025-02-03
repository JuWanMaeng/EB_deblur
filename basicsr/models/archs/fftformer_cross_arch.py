import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.arch_util import LayerNorm, FlowImage_ChannelAttentionTransformerBlock
# from .VAE_local import VAE


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
## Flow UNetEncoder

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

##########################################################################
##---------- FFTformer -----------------------
class fftformer_cross(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 FB_num_heads = [1,2,4],
                 flow_dims = [128,256,512],
                 ):
        super(fftformer_cross, self).__init__()

        # Flow UNetEncoder
        self.FUNetEncoder = UNetEncoder(in_channels=6, base_filters=48)
        # self.vae = VAE.from_pretrained('/workspace/Marigold/checkpoint/stable-diffusion-2/vae')

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])
        
        self.FBCA_enc_level_1 = FlowImage_ChannelAttentionTransformerBlock(dim=dim, num_heads=FB_num_heads[0], ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        # self.FBCA_level1 = MultiLayerFBCA(dim=dim, bias=bias, num_layers=num_blocks[0])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])
        self.FBCA_enc_level_2 = FlowImage_ChannelAttentionTransformerBlock(dim=dim * 2 ** 1, num_heads=FB_num_heads[1], ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])
        self.FBCA_enc_level_3 = FlowImage_ChannelAttentionTransformerBlock(dim=dim * 2 ** 2, num_heads=FB_num_heads[2], ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, y):

        inp = y[:,0:3,:,:]
        inp_img = y[:,0:3,:,:]
        flow = y[:,3:,:,:]

        # flow_enc1, flow_enc2, flow_enc3 = self.FUNetEncoder(flow)
        with torch.no_grad():
            features = self.FUNetEncoder(flow)     # C: [48 -> 96 -> 192]
        flow_enc1, flow_enc2, flow_enc3= features

        inp_enc_level1 = self.patch_embed(inp)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # B, 48, 256, 256
        out_dec_level1 = self.FBCA_enc_level_1(out_enc_level1, flow_enc1)
        

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # B, 96, 128, 128
        out_enc_level2 = self.FBCA_enc_level_2(out_enc_level2, flow_enc2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # B, 192, 64, 64
        out_enc_level3 = self.FBCA_enc_level_3(out_enc_level3, flow_enc3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)  # B, 192, 64, 64

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # B, 96, 128, 128

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # B, 58, 256, 256

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

    

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define model parameters
    inp_channels = 3  # Number of input channels for image (RGB)
    flow_channels = 6  # Number of flow channels (u, v)
    out_channels = 3  # Output channels
    dim = 48  # Dimension of model
    num_blocks = [6, 6, 12]  # Number of blocks at each level
    num_refinement_blocks = 4  # Number of refinement blocks
    ffn_expansion_factor = 3  # Feed-forward expansion factor
    bias = False  # Bias flag

    # Instantiate model
    model = fftformer_cross(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias
    )

    # Dummy input: batch size 1, channels (3 for image + 2 for flow), 256x256 image resolution
    dummy_input = torch.randn(1, inp_channels + flow_channels, 224, 224)

    # Forward pass through the model
    output = model(dummy_input)

    # Output shape
    print("Output shape:", output.shape)
    print("Expected output shape:", (1, out_channels, 224, 224))
