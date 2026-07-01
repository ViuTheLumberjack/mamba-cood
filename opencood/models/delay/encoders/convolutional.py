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
        self.layers = args.get('num_layers', 3)
        self.acts = args.get('activations', torch.nn.LeakyReLU)
        print(f"Encoder activations: {self.acts}, type {type(self.acts)}")
        if not isinstance(self.acts, list):
            self.acts = [getattr(nn, self.acts)] * self.layers 
        else:
            self.acts = [getattr(nn, act) if isinstance(act, str) else act for act in self.acts]
        assert len(self.acts) == self.layers, "Length of activations list must match number of layers"

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_channels = self.input_channels if i == 0 else self.hidden_dim #// (2 ** (i - 1)) 
            out_channels = self.hidden_dim #self.hidden_dim // (2 ** i) 
            self.arch.append(Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1, activation=self.acts[i]))  

    def forward(self, x):
        B, T, C, H, W = x.shape
        hs = []
        
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        for layer in self.arch:
            x = layer(x)
            # Store all so we can extract current frame later
            hs_all = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
            hs.append(hs_all[:, -1].clone())  # Only the current (last) frame
        
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x, hs
    
class Decoder(torch.nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.out_channels = args.get('output_channels', 1)
        self.hidden_dim = args.get('input_channels', 256)
        self.layers = args.get('num_layers', 3)
        self.acts = args.get('activations', torch.nn.LeakyReLU)
        if not isinstance(self.acts, list):
            self.acts = [getattr(nn, self.acts)] * self.layers 
        else:
            self.acts = [getattr(nn, act) if isinstance(act, str) else act for act in self.acts]
        assert len(self.acts) == self.layers, "Length of activations list must match number of layers"

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_channels = self.hidden_dim #in = self.hidden_dim // (2 ** (self.layers - i - 1)) 
            out_channels = self.out_channels if i == (self.layers - 1) else self.hidden_dim #// (2 ** (self.layers - i - 2)) 
            self.arch.append(Conv2DTransposeBlock(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1, activation=self.acts[i]))     

    def forward(self, x, hidden_states=None):
        T, B, C, H, W = x.shape  # T = num_preds (4)
        x = einops.rearrange(x, 't b c h w -> (t b) c h w')
        for i, layer in enumerate(self.arch):
            if hidden_states:
                hs = hidden_states.pop()  # [B, C, H, W] — current frame only
                hs = einops.repeat(hs, 'b c h w -> t b c h w', t=T)
                hs = einops.rearrange(hs, 't b c h w -> (t b) c h w')  # [B*T, C, H, W]
            else:
                hs = torch.zeros_like(x)
            
            x = x + hs
            x = layer(x)

        x = einops.rearrange(x, '(t b) c h w -> t b c h w', b=B)
        return x