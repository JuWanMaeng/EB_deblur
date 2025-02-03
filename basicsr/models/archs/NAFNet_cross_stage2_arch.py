
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numbers
import torchvision.transforms as transforms

from basicsr.models.archs.FB_Restormer_arch import FB_Restormer, OverlapPatchEmbed
from einops import rearrange
from PIL import Image
from copy import deepcopy

logger = logging.getLogger('basicsr')



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


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
        self.conv3 = nn.Conv2d(in_channels=dw_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel , out_channels=dw_channel , kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        # self.sg = SimpleGate()
        self.sg = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
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


##########################################################################
### Cross Attention

class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dq = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim,
                               bias=True)
        self.dk = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim,
                               bias=True)
        self.dv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim,
                               bias=True)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.dq(self.q(x))
        k = self.dk(self.k(y))
        v = self.dv(self.v(y)) 
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
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
  
##############################################   
### Q: Image, K,V: Flow
class IFCA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(IFCA, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_flow = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, flow):
        assert image.shape == flow.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_flow(flow)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

############################################## 
### Q: Flow, K,V: Image
class FICA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(FICA, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_flow = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, flow):
        assert flow.shape == image.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = flow.shape
        fused = flow + self.attn(self.norm1_image(flow), self.norm1_flow(image)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

class DoubleCrossAttentionBlock(nn.Module):
    def __init__(self,dim,num_heads,fb_dim):
        super(DoubleCrossAttentionBlock,self).__init__()

        self.IFCA = IFCA(dim = dim, num_heads= num_heads)
        self.FICA = FICA(dim = dim, num_heads= num_heads)

        self.flow_conv1x1 = nn.Conv2d(in_channels = fb_dim, out_channels=dim, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1)
    

    def forward(self, image, flow):
        flow = self.flow_conv1x1(flow)

        A = self.IFCA(image,flow)
        B = self.FICA(image,flow)

        # AB concat
        att = torch.cat([A,B], dim = 1)

        # # 1x1 conv로 차원 축소
        att = self.conv1(att)
        
        att = A
        return att

class MultiLayerDCAB(nn.Module):
    def __init__(self, dim, num_heads, num_layers,fb_dim):
        super(MultiLayerDCAB, self).__init__()
        # 여러 개의 Cross-Attention 레이어 쌓기
        self.layers = nn.ModuleList([DoubleCrossAttentionBlock(dim, num_heads,fb_dim) for _ in range(num_layers)])

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x, y)
        return x


class NAFBlock_flow(nn.Module):
    def __init__(self, c, dim, num_heads, num_layers, fb_dim,  DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel , out_channels=dw_channel , kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.sg = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel , out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = MultiLayerDCAB(dim=dim, num_heads=num_heads, num_layers=num_layers, fb_dim = fb_dim)

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
### F+B feature extractor
### stage 1 network
class FB_Encoder(FB_Restormer):
    def __init__(self,
                 inp_channels):
        super(FB_Encoder, self).__init__()

        self.inp_channels = inp_channels
        self.patch_embed = OverlapPatchEmbed(self.inp_channels, embed_dim=48)
        

    def forward(self, y):

        if y.shape[1] == 6:
            inp = y
            inp_img = y[:,0:3,:,:]
    

        inp_enc_level1 = self.patch_embed(inp)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) # [B, 192, H/4, W/4]

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        out_dec_level2 = self.decoder_level2(inp_dec_level2) # [B, 96, H/2, W/2]

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) # [B, 96, H, W]

        # out_dec_level1 = self.refinement(out_dec_level1)
        # output = self.output(out_dec_level1) + inp_img

        return [ out_dec_level1,out_dec_level2, out_dec_level3 ]



class NAFNet_cross_stage2(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dim=64, 
                 num_heads=[1,2,4], num_layers=[1,1,1], fb_dims = [96,96,192]):
        super().__init__()

        self.dim = dim
        # Flow UNetEncoder
        self.FB_Encoder = FB_Encoder(inp_channels=6)
        self.load_network(self.FB_Encoder, '/workspace/data/Restormer/experiments/FB_Restormer_A/models/A_best.pth')

        self.img_intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,groups=1,
                               bias=True)
        self.flow_intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
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

        for i,num in enumerate(enc_blk_nums):
            if num== 1:
                self.encoders.append(nn.Sequential(*[NAFBlock_flow(chan, dim=self.dim, num_heads=num_heads[i],num_layers=num_layers[i],
                                                                   fb_dim=fb_dims[i]) for _ in range(num)]))
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

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        # net = self.get_bare_model(net)
        logger.info(
            f'Loading Pretrained {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        # self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=False)
        

    def forward(self, y):
        B, C, H, W = y.shape

        inp = y[:,0:3,:,:]
        flow = y[:,3:,:,:]

        ############
        with torch.no_grad():
            guide_features = self.FB_Encoder(y)  #[ [B,96,H,W], [B, 96, H/2, W/2], [B, 192, H/4, W/4]]
        guide_features.append(None)

        x = self.img_intro(inp)  # original
        # inp = x_hat
        # x = self.img_intro(x_hat)  # iter restoration

        encs = []

        for encoder, down, flow in zip(self.encoders, self.downs, guide_features):
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



if __name__ == '__main__':
    # Define device to use (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    model = NAFNet_cross_stage2(enc_blk_nums=[1, 1, 1, 28], dec_blk_nums= [1, 1, 1, 1]).to(device)

    input_path = '/workspace/data/Gopro_my/train/000001/blur/000001.png'
    image = Image.open(input_path).convert('RGB')
    transform_to_tensor = transforms.Compose([
        transforms.RandomCrop((256, 256)),  # Randomly crop a 256x256 patch from the image
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    image_tensor = transform_to_tensor(image)

    image_tensor = torch.randn((4,6,256,256))
    image_tensor = image_tensor.to(device)
    # image_tensor = image_tensor.unsqueeze(0).to(device)

    # Forward pass through the model
    output = model(image_tensor)

    # Print the output shape to verify the forward pass
    print('Output shape:', output.shape)

    # Optionally, you can also print the output tensor to inspect its values
    # This can help verify the correctness of the computation
    print('Output Tensor:', output)