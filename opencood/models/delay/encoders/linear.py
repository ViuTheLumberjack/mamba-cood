import torch
import einops
import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, args):
        super(LinearEncoder, self).__init__()
        self.input_channels = args.get('input_channels', 128)
        self.output_channels = args.get('hidden_dim', 128)
        self.linear = nn.Linear(self.input_channels, self.output_channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, 'b t c h w -> (b t) (c h w)')
        x = self.linear(x)
        x = einops.rearrange(x, '(b t) hd -> b t hd', b=B, t=T)
        return x, None
    
class LinearDecoder(nn.Module):
    def __init__(self, args):
        super(LinearDecoder, self).__init__()
        self.input_channels = args.get('input_channels', 512)
        self.output_channels = args.get('output_channels', 128)
        self.height = args.get('height', 64)
        self.width = args.get('width', 64)

        self.linear1 = nn.Linear(self.input_channels, self.input_channels)
        self.act = nn.Tanh()
        self.linear2 = nn.Linear(self.input_channels, self.output_channels*self.height*self.width)

    def forward(self, x, hs=None):
        B, T, C = x.shape
        x = einops.rearrange(x, 'b t hd -> (b t) hd')
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = einops.rearrange(x, '(b t) (c h w) -> b t c h w', b=B, t=T, c=self.output_channels, h=self.height, w=self.width)
        return x