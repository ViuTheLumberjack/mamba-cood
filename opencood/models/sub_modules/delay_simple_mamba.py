import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from einops.layers.torch import Rearrange 

class FutureFramePredictor(nn.Module):
    def __init__(self, t=4, input_channels=8, height=48, width=176, projected_dim=512, patch_size=8, num_layers=3):
        super().__init__()

        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.t = t
        self.projected_dim = projected_dim
        self.patch_size = patch_size
        self.num_layers = num_layers

        self.seqential = nn.Sequential(
            Rearrange('b t c (nh ph) (nw pw) -> b t nh nw (c ph pw)', ph=self.patch_size, pw=self.patch_size),
            Rearrange('b t nh nw d -> b (t nh nw) d'),
            nn.Linear(input_channels*patch_size*patch_size, projected_dim),
            nn.Sequential(*[
                Mamba2(
                    d_model=projected_dim,
                    d_state=32,
                    d_conv=t-1,
                    expand=2
                ) for _ in range(num_layers)
            ]),
            nn.Linear(projected_dim, input_channels*patch_size*patch_size),
            Rearrange('b (t nh nw) d -> b t nh nw d', nh=height//patch_size, nw=width//patch_size),
            Rearrange('b t nh nw (c ph pw) -> b t c (nh ph) (nw pw)', ph=self.patch_size, pw=self.patch_size)
        )

    def forward(self, x):
        ## 2 PHASES: SPATIAL PROJECTION and TEMPORAL PROJECTION
        # x: [B, T, C, H, W]
        ## TODO: Rearrange into patches and process with Mamba, then rearrange back

        y = self.seqential(x)
        return y[:, -1, :, :, :]  # Return the last time step prediction
    
if __name__ == '__main__':
    B, T, C, H, W = 2, 5, 8, 48, 176 # Batch, Time, Channels, Height, Width
    x = torch.randn(B, T, C, H, W).to("cuda")

    # Create patches of size 16x16 and collect row-major/column-major scans
    patch_size = 16
    patches_grid = einops.rearrange(x, 'b t c (nh ph) (nw pw) -> b t nh nw (c ph pw)', 
                         ph=patch_size, pw=patch_size)
    scan_row_major = einops.rearrange(patches_grid, 'b t nh nw d -> b (t nh nw) d')
    scan_col_major = einops.rearrange(patches_grid, 'b t nh nw d -> b (t nw nh) d')
    
    patch_tokens = torch.cat([scan_row_major, scan_col_major], dim=2)
    print(f"Patch token shape: {patch_tokens.shape}")  # [B, T, num_patches, 2*C*16*16]

    d_model = 768
    projection = torch.nn.Linear(2 * C * patch_size * patch_size, d_model).to("cuda")    
    x_projected = projection(patch_tokens)

    print(f"Projected patch token shape: {x_projected.shape}")  # [B, T, num_patches, d_model]

    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=d_model, # Model dimension d_model
        d_state=32,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x_projected)
    print(f"Output shape: {y.shape}")  # [B, T, num_patches, d_model]