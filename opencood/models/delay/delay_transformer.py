import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_channels=8, hidden_dim=128, num_heads=4, num_layers=4, patch_size=4, height=48, width=176):
        super().__init__()
        self.patch_size = patch_size
        self.height, self.width = height, width
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.input_dim = input_channels * patch_size * patch_size

        self.flatten = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.embed = nn.Linear(self.input_dim, hidden_dim)

        self.temporal_pos = nn.Parameter(torch.randn(1, 100, hidden_dim))  # max T=100
        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_patch = nn.Linear(hidden_dim, self.input_dim)
        self.fold = nn.Fold(output_size=(height, width), kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        patches = self.flatten(x)  # [B*T, patch_dim, num_patches]
        patches = patches.transpose(1, 2)  # [B*T, num_patches, patch_dim]
        patches = self.embed(patches)  # [B*T, num_patches, hidden_dim]

        patches = patches.view(B, T, self.num_patches, -1)  # [B, T, num_patches, hidden_dim]
        patches += self.spatial_pos  # spatial positional encoding
        patches += self.temporal_pos[:, :T].unsqueeze(2)  # temporal positional encoding

        seq = patches.view(B, T * self.num_patches, -1)  # [B, T*num_patches, hidden_dim]
        out = self.transformer(seq)  # [B, T*num_patches, hidden_dim]

        last_frame_tokens = out[:, -self.num_patches:, :]  # Use last T slice for prediction
        recon_patches = self.to_patch(last_frame_tokens)  # [B, num_patches, patch_dim]
        recon_patches = recon_patches.transpose(1, 2)  # [B, patch_dim, num_patches]
        recon_frame = self.fold(recon_patches)  # [B, C, H, W]
        return recon_frame

# -------------------------------
# Full Model: Transformer Predictor Wrapper
# -------------------------------
class FutureFramePredictor(nn.Module):
    def __init__(self, input_channels=8, hidden_dim=128, patch_size=4, height=48, width=176):
        super().__init__()
        self.backbone = SpatioTemporalTransformer(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            height=height,
            width=width
        )

    def forward(self, x):
        return self.backbone(x)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    B, T, C, H, W = 2, 3, 8, 48, 176
    model = FutureFramePredictor(input_channels=C, height=H, width=W)
    input_tensor = torch.randn(B, T, C, H, W)

    with torch.no_grad():
        out = model(input_tensor)  # [B, C, H, W]
    
    print("Output shape:", out.shape)
