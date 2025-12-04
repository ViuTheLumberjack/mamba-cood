import torch
import torch.nn as nn

# --- Patch Embedding Module ---
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=8, patch_size=4, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.proj(x)  # [B*T, E, H', W']
        E, H_p, W_p = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, T, E, H_p * W_p).transpose(2, 3)  # [B, T, N, E]
        return x, H_p, W_p

# --- TimeSformer Forecast Model ---
class FutureFramePredictor(nn.Module):
    def __init__(self, in_channels=8, patch_size=4, embed_dim=512, num_heads=8, num_layers=4, height=48, width=176):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.embed_dim = embed_dim

        self.spatial_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.temporal_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width

    def forward(self, x):
        B, T, C, H, W = x.shape
        x, H_p, W_p = self.patch_embed(x)  # [B, T, N, E]

        # Spatial Attention
        for layer in self.spatial_blocks:
            x = x.view(B * T, self.num_patches, self.embed_dim)
            x = layer(x)
            x = x.view(B, T, self.num_patches, self.embed_dim)
        
        # Temporal Attention
        x = x.transpose(1, 2)  # [B, N, T, E]
        for layer in self.temporal_blocks:
            x = x.reshape(B * self.num_patches, T, self.embed_dim)
            x = layer(x)
            x = x.view(B, self.num_patches, T, self.embed_dim)

        # Predict next frame
        x = x[:, :, -1, :]  # [B, N, E]
        x = self.decoder(x)  # [B, N, patch_size^2 * C]
        x = x.view(B, H_p, W_p, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, H_p, p, W_p, p]
        x = x.view(B, C, H, W)
        return x

# --- Main Inference Code ---
def main():
    # Model setup
    B, T, C, H, W = 2, 5, 8, 48, 176
    model = FutureFramePredictor(in_channels=C, height=H, width=W)

    # Random input tensor simulating [Batch, Time, Channels, Height, Width]
    x = torch.randn(B, T, C, H, W)

    # Run inference
    model.eval()
    with torch.no_grad():
        y_pred = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_pred.shape}")  # Should be [B, C, H, W]

if __name__ == "__main__":
    main()