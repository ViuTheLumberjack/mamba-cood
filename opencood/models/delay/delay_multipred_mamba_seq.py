import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2 # TODO: add mamba 3 for RoPE
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


class MambaMultiPredictor(nn.Module):
    def __init__(
        self, 
        args,
    ):
        super().__init__()

        self.image_mode = args.get('image_mode', True)
        if self.image_mode:
            self.input_channels = args.get('input_channels', 8)
            self.height = args.get('height', 48)
            self.width = args.get('width', 176)
            self.patch_size = args.get('patch_size', 8)

            self.input_dim = self.input_channels * self.patch_size * self.patch_size
            self.num_patches = (self.height // self.patch_size) * (self.width // self.patch_size)
        else:
            self.input_dim = args.get('input_channels', 128)
            self.num_patches = 1

        self.hidden_dim = args.get('hidden_dim', 512)
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

        self.past_k = args.get('past_k', 4) + 1 # Number of past frames to use for prediction (if using pred token)

        # Patch embedding
        if self.image_mode:
            self.flatten = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            self.embed = nn.Linear(self.input_dim, self.hidden_dim)
            self.embed_norm = nn.RMSNorm(self.hidden_dim) if False else nn.Identity() 
            self.embed_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.pos_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        # Learnable positional encodings
        # Spatial and Temporal encodings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.num_patches, self.hidden_dim))
        self.temporal_pos = nn.Parameter(torch.randn(1, self.past_k + self.num_future_preds, 1, self.hidden_dim))

        # Prediction token (alternative to using last frame patches)
        self.pred_token = nn.Parameter(torch.randn(1, self.num_future_preds*self.num_patches, self.hidden_dim))
        self.pred_horizon_bias = nn.Parameter(torch.randn(self.num_future_preds, 1, 1, self.hidden_dim) * 0.5)

        # Mamba2 backbone with residual connections and normalization
        block_cls = BiMambaBlock if self.use_bidirectional else MambaBlock
            
        self.blocks = nn.ModuleList([
            block_cls(self.hidden_dim, self.d_state, self.d_conv, self.expand) 
            for _ in range(self.num_layers)
        ])

        self.final_norm = nn.RMSNorm(self.hidden_dim)

        self.reconstruction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
                    nn.Linear(self.hidden_dim, self.input_dim)
                ),
                Rearrange('b np pd -> b pd np') if self.image_mode else nn.Identity(),
                nn.Fold(
                    output_size=(self.height, self.width), 
                    kernel_size=self.patch_size, 
                    stride=self.patch_size
                ) if self.image_mode else nn.Identity()
            )
            for _ in range(len(args.get('future_delay_list', [0])))
        ])        

    def patchify(self, x):
        """Convert frames to patches with embeddings"""
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        patches = self.flatten(x)  # [B*T, patch_dim, num_patches]
        #print(f"Patch shape after flatten: {patches.shape}")
        patches = einops.rearrange(patches, 'bt pd np -> bt np pd')
        patches = self.embed(patches)  # [B*T, num_patches, hidden_dim]
        #print(f"Patch shape after embedding: {patches.shape}")
        patches = self.embed_norm(patches)
        patches = self.embed_dropout(patches)
        patches = einops.rearrange(
            patches, '(b t) np hd -> b t np hd', b=B, t=T
        )
        return patches

    def add_positional_encoding(self, patches):
        """Add spatiotemporal positional encoding"""
        B, T, NP, HD = patches.shape
        #patches = patches + self.spatiotemporal_pos[:, :TNP, :]
        
        patches = self.pos_dropout(patches)
        return patches

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] input frames
        
        Returns:
            [B, T_preds, C, H, W] predicted next frame
        """
        #print(f"Input shape: {x.shape}")

        # Patchify and add positional encoding
        if self.image_mode:
            B, T, C, H, W = x.shape
            patches = self.patchify(x)  # [B, T, num_patches, hidden_dim]
            seq = einops.rearrange(patches, 'b t np hd -> b (t np) hd')
        else:
            B, T, HD = x.shape
            seq = x
        
        # Add prediction token at the end
        num_pred_tokens = self.num_future_preds * self.num_patches
        pred_tokens = self.pred_token.expand(B, -1, -1)
        seq = torch.cat([seq, pred_tokens], dim=1)
        seq = self.add_positional_encoding(seq)
        
        # Process through Mamba2 blocks
        for block in self.blocks:
            seq = block(seq)
        
        seq = self.final_norm(seq)
        
        # Extract prediction tokens output
        pred_seq = seq[:, -num_pred_tokens:]  # [B, num_future_preds * num_patches, hidden_dim]
        if self.image_mode:
            pred_seq = einops.rearrange(
                pred_seq, 'b (f np) hd -> f b np hd', 
                f=self.num_future_preds, np=self.num_patches
            )
        else: 
            pred_seq = einops.rearrange(pred_seq, 'b f hd -> f b hd')
        
        # Add positional bias for prediction horizon
        pred_seq = pred_seq + self.pred_horizon_bias.expand(-1, B, self.num_patches, -1)  # [num_future_preds, B, num_patches, hidden_dim]

        # Reconstruct each future prediction with its own head
        preds = []
        #for i, head in enumerate(self.reconstruction_heads):
        for i in range(self.num_future_preds):
            head = self.reconstruction_heads[i]
            pred_frame = head(pred_seq[i])  # [B, num_patches, input_dim]

            preds.append(pred_frame)
        
        preds = torch.stack(preds, dim=0)  # [B, num_future_preds, C, H, W]
        #print(f"Predictions shape before residual: {preds.shape}")
        if self.residual_connection:
            # Add residual connection from last input frame
            last_frame = x[:, -1]  # [B, C, H, W]
            preds = preds + last_frame.unsqueeze(0)  # Broadcast to all predictions

        feat_enc = preds[self.prediction_horizon_idx]  # [B, C, H, W]
        
        return feat_enc, preds


class AutoregressivePredictor(MambaMultiPredictor):
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
    model = MambaMultiPredictor(args).to("cuda")
    
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y[0].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    