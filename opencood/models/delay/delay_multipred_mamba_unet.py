import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from einops.layers.torch import Rearrange 

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
    def __init__(self, spatial_mixer, channel_mixer, hidden_dim, norm=torch.nn.LayerNorm, drop=0., size=16):
        super().__init__()
        self.spatial_mixer = spatial_mixer
        self.channel_mixer = channel_mixer
        self.norm = norm([hidden_dim, size, size])
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = self.norm(x)
        y = einops.rearrange(y, 'b t c h w -> (b t) (h w) c', b=B, h=H, w=W)
        y = self.spatial_mixer(y)
        y = einops.rearrange(y, '(b t) (h w) c -> b t c h w', b=B, t=T, h=H, w=W)
        y = self.drop(y)
        x = x + y

        z = self.norm(x)
        z = einops.rearrange(z, 'b t c h w -> (b t) (h w) c')
        z = self.channel_mixer(z, H, W)
        z = self.drop(z)
        z = einops.rearrange(z, '(b t) (h w) c -> b t c h w', b=B, h=H, w=W)
        x = x + z
        
        return x

class MambaGLUEncoder(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256, layers=3):
        super(MambaGLUEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=self.in_channels, embed_dim=self.hidden_dim)

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_features = self.hidden_dim 
            out_features = self.hidden_dim 
            self.arch.append(
                torch.nn.Sequential(
                    SpatioChannelMixer(
                        hidden_dim=hidden_dim,
                        size=64 // (2 ** (i+1)),
                        spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                        channel_mixer=ConvolutionalGLU(in_features, in_features),
                    ),
                einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'),
                OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_features, embed_dim=out_features),
                einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=5)
                ) if i < self.layers - 1 else
                torch.nn.Sequential(
                    SpatioChannelMixer(
                        hidden_dim=hidden_dim,
                        size=64 // (2 ** (i+1)),
                        spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                        channel_mixer=ConvolutionalGLU(in_features, in_features),
                    ),
                    einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'),
                    OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_features, embed_dim=out_features),
                    einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=5),
                    SpatioChannelMixer(
                        hidden_dim=hidden_dim,
                        size=64 // (2 ** (i+2)),
                        spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                        channel_mixer=ConvolutionalGLU(in_features, in_features),
                    ),
                )
            )

    def forward(self, x, H=None, W=None):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = einops.rearrange(x, '(b t) c hh hw -> b t c hh hw', b=B, t=T)

        hidden_states = [x.clone()]

        for layer in self.arch:
            #print(f"Encoder input shape: {x.shape}")
            x = layer(x)
            hidden_states.append(x.clone())

        return x, hidden_states

class MambaGLUDecoder(torch.nn.Module):
    def __init__(self, out_channels=1, hidden_dim=256, layers=3, past_frames=5):
        super(MambaGLUDecoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.past_frames = past_frames

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            in_features = self.hidden_dim 
            out_features = self.hidden_dim 
            self.arch.append(
                torch.nn.Sequential(
                    SpatioChannelMixer(
                        hidden_dim=hidden_dim,
                        size=64 // (2 ** (self.layers - i + 1)),
                        spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                        channel_mixer=ConvolutionalGLU(in_features, in_features),
                    ),
                    einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'),
                    torch.nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                    einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=self.past_frames)
                ) if i > 0 else
                torch.nn.Sequential(
                    SpatioChannelMixer(
                        hidden_dim=hidden_dim,
                        size=64 // (2 ** (self.layers+1)),
                        spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                        channel_mixer=ConvolutionalGLU(in_features, in_features),
                    ),
                    einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'),
                    torch.nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                    einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=self.past_frames),
                    SpatioChannelMixer(
                        hidden_dim=hidden_dim,
                        size=64 // (2 ** (self.layers)),
                        spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                        channel_mixer=ConvolutionalGLU(in_features, in_features),
                    )
                )
            )
        
        self.out_proj = torch.nn.Sequential(
            einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'),
            torch.nn.ConvTranspose2d(in_channels=self.hidden_dim, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
            einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=self.past_frames)
        )

    def forward(self, x, hidden_states):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        
        for layer in self.arch:
            #print(f"Decoder input shape: {x.shape}")
            x = layer(x) + hidden_states.pop()  # Add skip connection from encoder 

        x = self.out_proj(x)

        return x

class MambaUNet(nn.Module):
    def __init__(
        self, 
        args,
    ):
        super().__init__()

        self.input_channels = args.get('input_channels', 8)
        self.height = args.get('height', 48)
        self.width = args.get('width', 176)
        self.hidden_dim = args.get('hidden_dim', 512)
        self.patch_size = args.get('patch_size', 8)
        self.num_layers = args.get('num_layers', 2)
        self.d_state = args.get('d_state', 64) 
        self.d_conv = args.get('d_conv', 4)
        self.expand = args.get('expand', 2)
        self.use_bidirectional = args.get('use_bidirectional', False)
        self.dropout_rate = args.get('dropout', 0.0)
        self.residual_connection = args.get('residual', True) 

        self.prediction_horizon = args.get('future_delay', 0)
        self.prediction_horizon_list = args.get('future_delay_list', [0])
        self.num_future_preds = len(self.prediction_horizon_list)
        self.prediction_horizon_idx = self.prediction_horizon_list.index(self.prediction_horizon)

        self.past_k = args.get('past_k', 4) # Number of past frames to use for prediction (if using pred token)

        # Patch embedding
        self.encoder = MambaGLUEncoder(
            in_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            layers=self.num_layers
        )

        self.decoder = MambaGLUDecoder(
            out_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            layers=self.num_layers,
            past_frames=self.past_k
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] input frames
        
        Returns:
            [B, T_preds, C, H, W] predicted next frame
        """
        B, T, C, H, W = x.shape

        # Patchify and add positional encoding
        feat_enc, hidden_states = self.encoder(x)
        
        # TODO: fix me
        hidden_states.pop()

        preds = self.decoder(feat_enc, hidden_states)
        
        return preds[self.prediction_horizon_idx], preds


class AutoregressivePredictor(MambaUNet):
    """Variant that can predict multiple future frames autoregressively"""
    def __init__(self, args):
        super().__init__(args)
    
    def forward(self, x, num_future_frames=1):
        """
        Args:
            x: [B, T, C, H, W] input frames
            num_future_frames: number of frames to predict into future
        
        Returns:
            [B, num_future_frames, C, H, W] predicted future frames
        """
        B, T, C, H, W = x.shape
        predictions = []
        
        current_input = x
        
        for _ in range(num_future_frames):
            # Predict next frame
            next_frame: torch.Tensor = super().forward(current_input)  # [B, C, H, W]
            predictions.append(next_frame.unsqueeze(1))
            
            # Update input: remove oldest frame, add prediction
            current_input = torch.cat([
                current_input[:, 1:],  # Drop first frame
                next_frame.unsqueeze(1)  # Add prediction
            ], dim=1)
        
        return torch.cat(predictions, dim=1)  # [B, num_future_frames, C, H, W]


if __name__ == '__main__':
    B, T, C, H, W = 2, 5, 8, 48, 176
    x = torch.randn(B, T, C, H, W).to("cuda")

    args = {
        "input_channels":C,
        "height":H,
        "width":W,
        "hidden_dim":128,
        "patch_size":8,
        "num_layers":3,
        "d_state":64,
        "d_conv":4, 
        "use_bidirectional":True
    }

    # Single frame prediction
    model = MambaUNet(args).to("cuda")
    
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y[0].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    