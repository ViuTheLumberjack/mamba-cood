import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

class FutureFramePredictor(nn.Module):
    def __init__(self, t=4, input_channels=8, height=48, width=176, projected_dim=256, num_layers=1):
        super().__init__()

        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.t = t
        self.projected_dim = projected_dim
        self.num_layers = num_layers

        self.linear_in = nn.Linear(t*height*width, projected_dim)
        self.mb = Mamba2(projected_dim, d_state=32, d_conv=t-1, expand=2)
        self.linear_out = nn.Linear(projected_dim, t*height*width)

    def forward(self, x):
        # x: [B, T, C, H, W]
        print(f"Input shape: {x.shape}")
        x = einops.rearrange(x, 'b t c h w -> b c (t h w)')
        #print(f"Input shape: {x.shape}")
        x = self.linear_in(x)
        #print(f"Input shape: {x.shape}")
        x = self.mb(x)
        #print(f"Input shape: {x.shape}")
        x = self.linear_out(x)
        #print(f"Input shape: {x.shape}")
        x = einops.rearrange(x, 'b c (t h w) -> b t c h w', t=self.t, h=self.height, w=self.width)
        #print(f"Output shape: {x.shape}")
        return x[:, -1, :, :, :]  # Return the last time step prediction
    
if __name__ == '__main__':
    B, T, C, H, W = 2, 5, 8, 48, 176 # Batch, Time, Channels, Height, Width
    x = torch.randn(B, T, C, H, W).to("cuda")
    x = einops.rearrange(x, 'b t c h w -> b t (c h w)')
    
    d_model = 768
    projection = torch.nn.Linear(C*H*W, d_model).to("cuda")
    x_projected = projection(x)

    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=d_model, # Model dimension d_model
        d_state=32,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x_projected)
    assert y.shape == x_projected.shape