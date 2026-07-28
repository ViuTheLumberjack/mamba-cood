import torch
import torch.nn as nn
import einops
from .blocks import Conv2DBlock, Conv2DTransposeBlock

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reps, activation=nn.LeakyReLU):
        super(DownsampleBlock, self).__init__()
        self.reps = reps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.forward_layers = nn.ModuleList()

        for i in range(reps):
            stride = self.stride if i == reps - 1 else 1  # Only apply stride on the last conv layer
            self.forward_layers.append(Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, activation=activation))
            in_channels = out_channels  

    def forward(self, x):
        for layer in self.forward_layers:
            x = layer(x)
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reps, activation=nn.LeakyReLU):
        super(UpsampleBlock, self).__init__()
        self.reps = reps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.forward_layers = nn.ModuleList()

        for i in range(reps):
            if i == reps - 1:
                stride = self.stride
                self.forward_layers.append(Conv2DTransposeBlock(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, activation=activation))
            else:
                stride = 1  # Only apply stride on the last conv layer
                self.forward_layers.append(Conv2DBlock(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, activation=activation))
            in_channels = out_channels  

    def forward(self, x):
        for layer in self.forward_layers:
            x = layer(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # A simple encoder architecture that takes in the input frames and produces a feature map for the predictor
        # using a few convolutional layers to compress the spatial dimensions and increase the channel dimension to hidden_dim 
        self.input_channels = args.get('input_channels', 1)
        self.hidden_dim = args.get('hidden_dim', 256)
        self.layers = args.get('num_layers', 3)
        self.acts = args.get('activations', torch.nn.LeakyReLU)
        self.reps = args.get('reps', 1)
        self.channels = args.get('channels', [self.hidden_dim] * self.layers)
        self.strides = args.get('strides', [2] * self.layers)
        
        if not isinstance(self.acts, list):
            self.acts = [getattr(nn, self.acts)] * self.layers 
        else:
            self.acts = [getattr(nn, act) if isinstance(act, str) else act for act in self.acts]
        assert len(self.acts) == self.layers, "Length of activations list must match number of layers"

        if not isinstance(self.reps, list):
            self.reps = [self.reps] * self.layers 
        assert len(self.reps) == self.layers, "Length of reps list must match number of layers"

        if not isinstance(self.channels, list):
            self.channels = [self.channels] * self.layers 
        assert len(self.channels) == self.layers, "Length of channels list must match number of layers"

        if not isinstance(self.strides, list):
            self.strides = [self.strides] * self.layers 
        assert len(self.strides) == self.layers, "Length of strides list must match number of layers"

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            self.arch.append(DownsampleBlock(
                in_channels=self.channels[i-1] if i > 0 else self.input_channels,
                out_channels=self.channels[i],
                stride=self.strides[i],
                reps=self.reps[i],
                activation=self.acts[i]
            ))

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
        self.input_channels = args.get('input_channels', 256)
        self.out_channels = args.get('output_channels', 1)
        self.layers = args.get('num_layers', 3)
        self.acts = args.get('activations', torch.nn.LeakyReLU)
        self.reps = args.get('reps', 1)
        self.channels = args.get('channels', [self.input_channels] * self.layers)
        self.strides = args.get('strides', [2] * self.layers)

        if not isinstance(self.acts, list):
            self.acts = [getattr(nn, self.acts)] * self.layers 
        else:
            self.acts = [getattr(nn, act) if isinstance(act, str) else act for act in self.acts]
        assert len(self.acts) == self.layers, "Length of activations list must match number of layers"
  
        if not isinstance(self.reps, list):
            self.reps = [self.reps] * self.layers 
        assert len(self.reps) == self.layers, "Length of reps list must match number of layers"

        if not isinstance(self.channels, list):
            self.channels = [self.channels] * self.layers 
        assert len(self.channels) == self.layers, "Length of channels list must match number of layers"

        if not isinstance(self.strides, list):
            self.strides = [self.strides] * self.layers 
        assert len(self.strides) == self.layers, "Length of strides list must match number of layers"
        
        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            self.arch.append(UpsampleBlock(
                in_channels=self.channels[i-1] if i > 0 else self.input_channels,
                out_channels=self.channels[i],
                stride=self.strides[i],
                reps=self.reps[i],
                activation=self.acts[i]
            ))
            
    def forward(self, x, hidden_states=None):
        B, T, C, H, W = x.shape  # T = num_preds (4)
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        for i, layer in enumerate(self.arch):
            if hidden_states:
                hs = hidden_states.pop()  # [B, C, H, W] — current frame only
                hs = einops.repeat(hs, 'b c h w -> b t c h w', t=T)
                hs = einops.rearrange(hs, 'b t c h w -> (b t) c h w')  # [B*T, C, H, W]
            else:
                hs = torch.zeros_like(x)
            
            x = x + hs
            x = layer(x)

        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B)
        return x