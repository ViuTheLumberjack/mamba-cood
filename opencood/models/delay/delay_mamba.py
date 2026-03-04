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


class MambaFutureFramePredictor(nn.Module):
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
        self.dropout_rate = args.get('dropout', 0.1)
        self.scans = args.get("scans", "horizonatal")# "all", "horizontal", "vertical"

        self.num_patches = (self.height // self.patch_size) * (self.width // self.patch_size)
        self.input_dim = self.input_channels * self.patch_size * self.patch_size #* (1 if self.scans == "all" else 2)

        # Patch embedding
        self.flatten = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.embed_norm = nn.LayerNorm(self.hidden_dim)
        self.embed_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.pos_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Learnable positional encodings
        # Combined spatiotemporal encoding
        self.spatiotemporal_pos = nn.Parameter(
            torch.randn(1, 100, self.num_patches, self.hidden_dim) * 0.02
        )

        # Prediction token (alternative to using last frame patches)
        self.pred_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Mamba2 backbone with residual connections and normalization
        if self.use_bidirectional:
            block_cls = BiMambaBlock
        else:
            block_cls = MambaBlock
            
        self.blocks = nn.ModuleList([
            block_cls(self.hidden_dim, self.d_state, self.d_conv, self.expand) 
            for _ in range(self.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.hidden_dim)

        # Reconstruction head
        self.to_patch = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        self.fold = nn.Fold(
            output_size=(self.height, self.width), 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

    def patchify(self, x):
        """Convert frames to patches with embeddings"""
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        patches = self.flatten(x)  # [B*T, patch_dim, num_patches]
        #print("Patches:", patches.shape)
        patches = einops.rearrange(patches, 'bt pd np -> bt np pd')
        patches = self.embed(patches)  # [B*T, num_patches, hidden_dim]
        patches = self.embed_norm(patches)
        patches = self.embed_dropout(patches)
        patches = einops.rearrange(
            patches, '(b t) np hd -> b t np hd', b=B, t=T
        )
        return patches

    def add_positional_encoding(self, patches):
        """Add spatiotemporal positional encoding"""
        B, T, NP, HD = patches.shape
        patches = patches + self.spatiotemporal_pos[:, :T, :NP, :]
        patches = self.pos_dropout(patches)
        return patches

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] input frames
            return_intermediate: if True, return all frame predictions
        
        Returns:
            [B, C, H, W] predicted next frame (or [B, T_pred, C, H, W] if return_intermediate)
        """
        B, T, C, H, W = x.shape
        
        # Patchify and add positional encoding
        patches = self.patchify(x)  # [B, T, num_patches, hidden_dim]
        #print(patches.shape)
        patches = self.add_positional_encoding(patches)
        
        # Flatten spatiotemporal dimensions
        seq = einops.rearrange(patches, 'b t np hd -> b (t np) hd')
        #print("Seq shape:", seq.shape)
        
        # Add prediction token at the end
        # pred_tokens = self.pred_token.expand(B, self.num_patches, -1)
        # seq = torch.cat([seq, pred_tokens], dim=1)
        
        # Process through Mamba2 blocks
        for block in self.blocks:
            seq = block(seq)
        
        seq = self.final_norm(seq)
        #print("Final seq shape:", seq.shape)

        # Or use the last frame tokens directly:
        last_frame_tokens = seq[:, -self.num_patches:]  # [B, num_patches, hidden_dim]
        
        # Reconstruct patches
        recon_patches = self.to_patch(last_frame_tokens)  # [B, num_patches, patch_dim]
        recon_patches = einops.rearrange(recon_patches, 'b np pd -> b pd np')
        recon_frame = self.fold(recon_patches)  # [B, C, H, W]
        
        return recon_frame, recon_frame.unsqueeze(0)  # [B, C, H, W], [1, B, C, H, W]


class AutoregressivePredictor(MambaFutureFramePredictor):
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
            next_frame = super().forward(current_input)  # [B, C, H, W]
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

    # Single frame prediction
    args = {
        "input_channels":C,
        "height":H,
        "width":W,
        "hidden_dim":512,
        "patch_size":8,
        "num_layers":1,
        "d_state":64,
        "d_conv":4, 
        "espand": 2,
        "use_bidirectional":True
    }

    # Single frame prediction
    model = MambaFutureFramePredictor(args).to("cuda")
    
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Multi-frame autoregressive prediction
    auto_model = AutoregressivePredictor(
        t=T,
        input_channels=C,
        height=H,
        width=W,
        hidden_dim=512,
        patch_size=8,
        num_layers=6
    ).to("cuda")
    
    y_multi = auto_model(x, num_future_frames=3)
    print(f"\nAutoregressive output shape: {y_multi.shape}")