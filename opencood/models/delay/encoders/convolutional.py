import torch
import torch.nn as nn
import einops
from .blocks import Conv2DBlock, Conv2DTransposeBlock

class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # A simple encoder architecture that takes in the input frames and produces a feature map for the predictor
        # using a few convolutional layers to compress the spatial dimensions and increase the channel dimension to hidden_dim 
        self.input_channels = args.get('input_channels', 1)
        self.hidden_dim = args.get('hidden_dim', 256)
        self.layers = args.get('layers', 3)

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_channels = self.input_channels if i == 0 else self.hidden_dim #// (2 ** (i - 1)) 
            out_channels = self.hidden_dim #self.hidden_dim // (2 ** i) 
            self.arch.append(Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1))  

    def forward(self, x):
        # Implement the forward pass for the encoder
        B, T, C, H, W = x.shape 
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        hs = []

        for layer in self.arch:
            x = layer(x)
            hs.append(x.clone())

        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B)
        return x, hs
    
class Decoder(torch.nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.out_channels = args.get('output_channels', 1)
        self.hidden_dim = args.get('input_channels', 256)
        self.layers = args.get('layers', 3)

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_channels = self.hidden_dim #in = self.hidden_dim // (2 ** (self.layers - i - 1)) 
            out_channels = self.out_channels if i == (self.layers - 1) else self.hidden_dim #// (2 ** (self.layers - i - 2)) 
            self.arch.append(Conv2DTransposeBlock(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1))     

    def forward(self, x, hidden_states=None):
        # Implement the forward pass for the decoder
        B, T, C, H, W = x.shape 
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        
        for layer in self.arch:
            x = layer(x)
            hs = hidden_states.pop() if hidden_states else torch.zeros_like(x)
            x = x + hs

        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B)
        return x