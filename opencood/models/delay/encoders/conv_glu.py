import torch
import torch.nn as nn
import einops
from .blocks import Conv2DBlock, SpatioChannelMixer, DWConv, OverlapPatchEmbed, ConvolutionalGLU

class ConvGLUEncoder(torch.nn.Module):
    def __init__(self, args):
        super(ConvGLUEncoder, self).__init__()
        
        self.in_channels = args.get('input_channels', 1)
        self.hidden_dim = args.get('hidden_dim', 256)
        self.layers = args.get('layers', 3)

        self.patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=self.in_channels, embed_dim=self.hidden_dim)

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_features = self.hidden_dim 
            out_features = self.hidden_dim 
            size = 32 // (2 ** i)
            self.arch.append(
                SpatioChannelMixer(
                    hidden_dim=self.hidden_dim,
                    size=size,
                    spatial_mixer=Conv2DBlock(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1),
                    channel_mixer=ConvolutionalGLU(out_features, out_features),
                )
            )
            self.arch.append(einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'))
            self.arch.append(OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_features, embed_dim=out_features))
            self.arch.append(einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=5))

    def forward(self, x, H=None, W=None):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = einops.rearrange(x, '(b t) c hh hw -> b t c hh hw', b=B, t=T)

        hs = []
        for layer in self.arch:
            x = layer(x)
            hs.append(x.clone())

        return x, hs