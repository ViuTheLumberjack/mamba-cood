import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from einops.layers.torch import Rearrange 
from opencood.models.delay.scan_patterns.row_major_scan import RowMajorScan2D
from opencood.models.delay.scan_patterns.column_major_scan import ColumnMajorScan2D  
from opencood.models.delay.scan_patterns.cross_scan2d import CrossScan2D

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

class MambaMultiPredictor4D(nn.Module):
    def __init__(
        self, 
        args,
    ):
        super().__init__()
        self.input_channels = args.get('input_channels', 8)
        self.height = args.get('height', 48)
        self.width = args.get('width', 176)
        self.patch_size = args.get('patch_size', 8)

        self.input_dim = self.input_channels #* self.patch_size * self.patch_size
        self.num_patches = self.width * self.height #(self.height // self.patch_size) * (self.width // self.patch_size)

        self.hidden_dim = args.get('hidden_dim', 512)
        self.num_layers = args.get('num_layers', 2)
        self.d_state = args.get('d_state', 64) 
        self.d_conv = args.get('d_conv', 4)
        self.expand = args.get('expand', 2)
        self.use_bidirectional = args.get('use_bidirectional', False)
        self.dropout_rate = args.get('dropout', 0.0)
        self.residual_connection = args.get('residual', True)
        self.factored_embeddings = args.get('factored_embeddings', False)
        self.factored_temporal_encoding = args.get('factored_temporal_encoding', False)

        self.prediction_horizon = args.get('future_delay', 0)
        self.prediction_horizon_list = args.get('future_delay_list', [0])
        self.num_future_preds = len(self.prediction_horizon_list)
        self.prediction_horizon_idx = self.prediction_horizon_list.index(self.prediction_horizon)

        self.past_k = args.get('past_k', 4) + 1 # Number of past frames to use for prediction (if using pred token)

        self.scan_type = args.get('scan_type', 'cross_scan')
        # Patch embedding
        match self.scan_type:
            case 'row_major':
                self.scan = RowMajorScan2D(bidirectional=self.use_bidirectional)
            case 'column_major':
                self.scan = ColumnMajorScan2D(bidirectional=self.use_bidirectional)
            case 'cross_scan':
                self.scan = CrossScan2D(bidirectional=self.use_bidirectional)
            case _:
                raise ValueError(f"Unknown scan type: {self.scan_type}")
            
        self.embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.embed_norm = nn.RMSNorm(self.hidden_dim) if False else nn.Identity() 
        self.embed_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.unembed = nn.Linear(self.hidden_dim, self.input_dim)

        self.pos_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        # Learnable positional encodings
        # Combined spatiotemporal encoding
        self.spatiotemporal_pos = nn.Parameter(
            torch.randn(1, self.scan.get_num_scans(), (self.num_future_preds + self.past_k)*self.num_patches, self.hidden_dim) * 0.02
        )

        # Prediction token (alternative to using last frame patches)
        self.pred_token = nn.Parameter(torch.randn(1, self.scan.get_num_scans(), self.num_future_preds*self.num_patches, self.hidden_dim) * 0.02)
        self.pred_horizon_bias = nn.Parameter(torch.randn(self.num_future_preds, 1, 1, self.hidden_dim) * 0.5)

        # Mamba2 backbone with residual connections and normalization
        block_cls = MambaBlock
            
        self.blocks = nn.ModuleList([nn.Sequential(*[
            block_cls(self.hidden_dim, self.d_state, self.d_conv, self.expand) 
            for _ in range(self.num_layers)
        ]) for _ in range(self.scan.get_num_scans())])  # 4 parallel scan patterns

        self.final_norm = nn.RMSNorm(self.hidden_dim)     

    def add_positional_encoding(self, patches):
        """Add spatiotemporal positional encoding"""
        B, S, TNP, HD = patches.shape
        patches = patches + self.spatiotemporal_pos[:, :S, :TNP, :]
        patches = self.pos_dropout(patches)
        return patches

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] input frames
        
        Returns:
            [B, T_preds, C, H, W] predicted next frame
        """
        print(f"Input shape: {x.shape}")

        # Patchify and add positional encoding
        B, T, C, H, W = x.shape
        
        p = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        p = self.scan(p)  # [B*T, 4, C, SEQ_LEN]
        p = einops.rearrange(p, '(b t) s c np -> b s (t np) c', b=B, t=T)  # [B*T*S, num_patches, C]
        
        if self.input_dim != self.hidden_dim:
            p = self.embed(p)
            p = self.embed_norm(p)
            p = self.embed_dropout(p)
        
        # Add prediction token at the end
        num_pred_tokens = self.num_future_preds * self.num_patches
        pred_tokens = self.pred_token.expand(B, -1, -1, -1)
        #print(f"Sequence shape before adding pred token: {p.shape}")
        #print(f"Prediction token shape: {pred_tokens.shape}")
        p = torch.cat([p, pred_tokens], dim=2)
        p = self.add_positional_encoding(p)
        
        p = einops.rearrange(p, 'b s np hd -> s b np hd')  # [B, S, T+T_pred, num_patches, hidden_dim]
        # Process through Mamba2 blocks
        processed_seqs = []
        for i in range(self.scan.get_num_scans()):
            sc_seq = p[i].contiguous()
            block = self.blocks[i]
            # Extract sequence i: [B*T, C, H*W]
            sc_seq = block(sc_seq)

            processed_seqs.append(sc_seq)
        
        p = torch.stack(processed_seqs, dim=0)  # [S, B, T+T_pred, num_patches, hidden_dim]
        p = self.final_norm(p)
        
        # Extract prediction tokens output
        p = p[:, :, -num_pred_tokens:]  # [B, num_future_preds * num_patches, hidden_dim]
        
        p = self.final_norm(p)

        p = einops.rearrange(
            p, 's b (f np) hd -> (f b) s hd np',
            f=self.num_future_preds, np=self.num_patches
        )
        #print(f"Pred sequence shape before unscan: {p.shape}")
        p = self.scan.unscan(p, self.height, self.width)  # [B, hidden_dim, H, W]
        p = einops.rearrange(p, '(f b) c h w -> f b c h w', f=self.num_future_preds)  # [B, num_patches, hidden_dim]
        
        if self.input_dim != self.hidden_dim:
            p = einops.rearrange(p, 'f b c h w -> f b h w c')
            p = self.unembed(p)
            p = einops.rearrange(p, 'f b h w c -> f b c h w')

        #print(f"Predictions shape before residual: {p.shape}")
        if self.residual_connection:
            # Add residual connection from last input frame
            last_frame = x[:, -1]  # [B, C, H, W]
            p = p + last_frame.unsqueeze(0)  # Broadcast to all predictions
            p = torch.nn.functional.relu(p)

        feat_enc = p[self.prediction_horizon_idx]  # [B, C, H, W]
        
        return feat_enc, p

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
    model = MambaMultiPredictor4D(args).to("cuda")
    
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y[0].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    