import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from diffusers import AutoencoderKL
from typing import Dict, Optional, Tuple, Union
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook

class VAE(AutoencoderKL):


    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode(x)

        sample = self.encoder.conv_in(x)

        features = []
        for down_block in self.encoder.down_blocks:
            for resblock in down_block.resnets:
                sample = resblock(sample, temb = None)
            features.append(sample)
            if down_block.downsamplers is not None:
                for down in down_block.downsamplers:
                    sample = down(sample)

            # sample = down_block.resnets(sample)
            # features.append(sample)



        # sample = self.encoder.mid_block(sample)

        return features

        # enc = self.encoder(x)
        # if self.quant_conv is not None:
        #     enc = self.quant_conv(enc)

        # return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:

        # if self.use_slicing and x.shape[0] > 1:
        #     encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
        #     h = torch.cat(encoded_slices)
        # else:
        h = self._encode(x)
        return h
        # posterior = DiagonalGaussianDistribution(h)

        # if not return_dict:
        #     return (posterior,)

        # return AutoencoderKLOutput(latent_dist=posterior)