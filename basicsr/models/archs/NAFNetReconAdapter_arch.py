import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
import os


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (g - y * mean_gy - mean_g) / torch.sqrt(var + eps)
        grad_w = (grad_output * y).sum(dim=(0, 2, 3))
        grad_b = grad_output.sum(dim=(0, 2, 3))
        return gx, grad_w, grad_b, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel//2, dw_channel//2, 1, bias=True))
        self.conv3 = nn.Conv2d(dw_channel//2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate>0 else nn.Identity()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel//2, c, 1, bias=True)
        self.norm2 = LayerNorm2d(c)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate>0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y); y = self.conv2(y); y = self.sg(y); y = y * self.sca(y)
        y = self.conv3(y); y = self.dropout1(y)
        x = x + y * self.beta
        y = self.conv4(self.norm2(x)); y = self.sg(y); y = self.conv5(y); y = self.dropout2(y)
        return x + y * self.gamma


class EnhancedDomainAdapter(nn.Module):
    """
    Adapter mapping event latent -> SDv2 latent with 
    normalization, GELU, and a residual skip (with 1×1 proj).
    """
    def __init__(self, in_channels: int, mid_channels: int = 64, out_channels: int = 8, eps: float = 1e-6):
        super().__init__()
        # normalize per-channel
        self.norm = nn.GroupNorm(1, in_channels, eps=eps)
        # two-stage 1×1 projection
        self.proj1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.act   = nn.GELU()
        self.proj2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        # projection for skip connection
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # x: [B, in_channels, H, W]
        y = self.norm(x)
        y = self.proj1(y)
        y = self.act(y)
        y = self.proj2(y)
        # add skip (projected if needed)
        return y + self.skip_proj(x)


class NAFNetReconAdapter(nn.Module):
    def __init__(self, img_channel=6, width=16, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None, latent_dim=128):
        super().__init__()
        enc_blk_nums = enc_blk_nums or []
        dec_blk_nums = dec_blk_nums or []
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan*2, 2, stride=2)); chan *= 2
        half_mid = middle_blk_num // 2
        self.middle_encoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_mid)])
        self.middle_decoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_mid)])
        self.latent_to_8 = nn.Conv2d(chan, latent_dim, 1)
        self.adapter = EnhancedDomainAdapter(latent_dim, mid_channels=latent_dim//2, out_channels=8)
        self.latent_from_8 = nn.Conv2d(latent_dim, chan, 1)
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan*2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

    def encode(self, inp):
        x = self.intro(inp)
        for enc, down in zip(self.encoders, self.downs): x = down(enc(x))
        x = self.middle_encoder(x)
        return self.latent_to_8(x)

    def map_to_sd_latent(self, z_evt):
        return self.adapter(z_evt)

    def decode(self, latent):
        if latent.shape[1] == 4:
            raise ValueError("Cannot decode SD latent through NAF decoder.")
        x = self.latent_from_8(latent)
        x = self.middle_decoder(x)
        for up, dec in zip(self.ups, self.decoders): x = dec(up(x))
        return self.ending(x)

    def forward(self, y):
        z_evt = self.encode(y)
        return self.decode(z_evt)


# Example usage:
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NAFNetReconAdapter(img_channel=6, width=64, middle_blk_num=28,
                                enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1], latent_dim=128)
    model.to(device)
    # load pretrained VAE
    ckpt = torch.load('/workspace/Marigold/checkpoint/NAF_VAE_128.pth', map_location=device)
    model.load_state_dict({k:v for k,v in ckpt['params'].items() if not k.startswith('adapter.')}, strict=False)
    # freeze event VAE parts
    for name, p in model.named_parameters():
        if not name.startswith('adapter.'):
            p.requires_grad = False
    # load SDv2 VAE
    sd_vae = AutoencoderKL.from_pretrained('/workspace/Marigold/checkpoint/stable-diffusion-2/vae').to(device).eval()
    for p in sd_vae.parameters(): p.requires_grad=False
    # setup adapter optimizer
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    # dummy training step example
    dummy_event = torch.randn(2,6,256,256,device=device)
    dummy_img   = torch.randn(2,3,256,256,device=device)
    with torch.no_grad():
        z_evt = model.encode(dummy_event)
        z_sd  = sd_vae.encode(dummy_img).latent_dist.sample()
    z_pred = model.map_to_sd_latent(z_evt)
    loss = loss_fn(z_pred, z_sd)
    print(f'Adapter loss: {loss.item():.6f}')
    optimizer.zero_grad(); loss.backward(); optimizer.step()
