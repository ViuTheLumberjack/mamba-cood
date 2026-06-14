import torch
import einops
import torch.nn as nn
from .blocks import MambaBlock, BiMambaBlock, OverlapPatchEmbed, ConvolutionalGLU, SpatioChannelMixer

class MambaGLUEncoder(torch.nn.Module):
    def __init__(self, args):
        super(MambaGLUEncoder, self).__init__()
        self.in_channels = args.get('in_channels', 1)
        self.layers = args.get('layers', 2)
        self.size = args.get('size', 64)

        if isinstance(reps, int):
            reps = [reps] * self.layers

        if isinstance(channels, int):
            channels = [channels] * self.layers

        if isinstance(strides, int):
            strides = [strides] * self.layers
        
        self.layers = self.layers
        self.reps = reps
        self.channels = channels
        self.strides = strides

        assert len(self.reps) == self.layers, "Length of reps list must match number of layers"
        assert len(self.channels) == self.layers, "Length of channels list must match number of layers"
        assert len(self.strides) == self.layers, "Length of strides list must match number of layers"

        self.patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=self.in_channels, embed_dim=self.channels[0])

        self.arch = torch.nn.ModuleList()
        self.downsamples = torch.nn.ModuleList()

        for i in range(self.layers):
            rep = self.reps[i] if i < len(self.reps) else self.reps[-1]
            stride = self.strides[i] if i < len(self.strides) else self.strides[-1]

            # 1. Build the identical spatial-channel blocks for this stage
            for r in range(rep):
                in_features = self.channels[i]
                
                self.arch.append(
                    torch.nn.Sequential(
                        SpatioChannelMixer(
                            hidden_dim=in_features,
                            size=self.size // (2 ** (i+1)),
                            spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                            channel_mixer=ConvolutionalGLU(in_features, in_features),
                        )
                    )
                )

            # 2. Build the downsampler separately
            if i < self.layers:
                in_features = self.channels[i]
                out_features = self.channels[i+1] if i < self.layers - 1 else self.channels[i]
                self.downsamples.append(
                    OverlapPatchEmbed(patch_size=3, stride=stride, in_chans=in_features, embed_dim=out_features)
                )

    def forward(self, x, H=None, W=None):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        
        # Initial patch embedding sequence
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

        hidden_states = []
        block_idx = 0

        for i in range(self.layers):
            rep = self.reps[i] if i < len(self.reps) else self.reps[-1]
            
            # Pass through the spatial-channel mixing blocks at the current resolution
            for r in range(rep):
                x = self.arch[block_idx](x)
                block_idx += 1
            
            # If not the final layer, capture residual *BEFORE* downscaling, then downscale
            hidden_states.append(x.clone())
            
            # Apply downscaling securely
            x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.downsamples[i](x)
            x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

        return x, hidden_states

class MambaGLUDecoder(torch.nn.Module):
    def __init__(self, args):
        super(MambaGLUDecoder, self).__init__()
        self.out_channels = args.get('out_channels', 1)
        self.layers = args.get('layers', 2)
        self.past_frames = args.get('past_frames', 5)
        self.size = args.get('size', 64)

        if isinstance(reps, int):
            reps = [reps] * self.layers

        if isinstance(channels, int):
            channels = [channels] * self.layers

        if isinstance(strides, int):
            strides = [strides] * self.layers

        self.reps = reps
        self.channels = channels
        self.strides = strides

        assert len(self.reps) == self.layers, "Length of reps list must match number of layers"
        assert len(self.channels) == self.layers, "Length of channels list must match number of layers"
        assert len(self.strides) == self.layers, "Length of strides list must match number of layers"

        self.arch = torch.nn.ModuleList()

        for i in range(self.layers):
            rep = self.reps[i]

            for r in range(rep): 
                in_features = self.channels[i]

                self.arch.append(
                    torch.nn.Sequential(
                        SpatioChannelMixer(
                            hidden_dim=in_features,
                            size=self.size // (2 ** (self.layers - i)),
                            spatial_mixer=MambaBlock(hidden_dim=in_features, d_state=64, d_conv=4, expand=2),
                            channel_mixer=ConvolutionalGLU(in_features, in_features),
                        )
                    ) 
                ) 
        
        # 2. Build the upsamplers separately
        self.upsamples = torch.nn.ModuleList()
        for i in range(self.layers):
            in_features = self.channels[i-1] if i > 0 else self.channels[0]
            out_features = self.channels[i] 
            self.upsamples.append(
                torch.nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=self.strides[i], padding=1, output_padding=1)
            )
        
        self.out_proj = torch.nn.Sequential(
            einops.layers.torch.Rearrange('b t c h w -> (b t) c h w'),
            torch.nn.ConvTranspose2d(in_channels=self.channels[-1], out_channels=self.channels[-1], kernel_size=3, stride=1, padding=1),
            torch.nn.ConvTranspose2d(in_channels=self.channels[-1], out_channels=self.channels[-1], kernel_size=3, stride=1, padding=1),
            torch.nn.ConvTranspose2d(in_channels=self.channels[-1], out_channels=self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), 
            einops.layers.torch.Rearrange('(b t) c h w -> b t c h w', t=self.past_frames)
        )

    def forward(self, x, hidden_states):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        
        block_idx = 0
        for i in range(self.layers):
            #print(f"Decoder layer {i}, input shape: {x.shape}")
            rep = self.reps[i] if i < len(self.reps) else self.reps[-1]

            # Apply upscaling securely
            x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.upsamples[i](x)
            x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

            # Add residual
            hs = hidden_states.pop() if hidden_states else torch.zeros_like(x)  # Use zeros if no hidden state is available
            x = x + hs
            
            # Pass through the spatial-channel mixing blocks at the current resolution
            for r in range(rep):
                x = self.arch[block_idx](x)
                block_idx += 1
            
        x = self.out_proj(x)
        #print(f"Max output value: {x.max().item()}")

        return x