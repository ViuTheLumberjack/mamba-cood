
import torch
import torch.nn as nn
import einops
from mamba_ssm import Mamba, Mamba2

class MambaBlock(nn.Module):
    """Mamba2 block with pre-norm and residual connection"""
    def __init__(self, hidden_dim, d_state=64, d_conv=8, expand=2, dropout=0.1, norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    
    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba2 block with pre-norm and residual connection"""
    def __init__(self, hidden_dim, d_state=64, d_conv=8, expand=2, dropout=0.1, norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.mamba_fwd = Mamba2(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_bwd = Mamba2(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x_norm = self.norm(x)
        fwd_out = self.mamba_fwd(x_norm)
        bwd_out = self.mamba_bwd(torch.flip(x_norm, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])
        return x + self.dropout(fwd_out + bwd_out)

class Conv2DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv2DTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv(torch.nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(torch.nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, H=None, W=None):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

class SpatioChannelMixer(torch.nn.Module):
    def __init__(self, spatial_mixer, channel_mixer, hidden_dim, norm=torch.nn.LayerNorm, drop=0., size=None):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm = norm([hidden_dim])
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = einops.rearrange(x, 'b t c h w -> b t h w c')
        y = self.norm(y)
        y = einops.rearrange(y, 'b t c h w -> (b t) c (h w)', b=B, h=H, w=W)
        y = self.spatial_mixer(y)
        y = einops.rearrange(y, '(b t) c (h w) -> b t c h w', b=B, t=T, h=H, w=W)
        y = self.drop(y)
        x = x + y

        z = einops.rearrange(x, 'b t c h w -> b t h w c')
        z = self.norm(z)
        z = einops.rearrange(z, 'b t c h w -> (b t) (h w) c')
        z = self.channel_mixer(z, H, W)
        z = self.drop(z)
        z = einops.rearrange(z, '(b t) (h w) c -> b t c h w', b=B, h=H, w=W)
        x = x + z
        
        return x

class ConvolutionalGLU(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = torch.nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x