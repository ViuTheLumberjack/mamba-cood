import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)
    
class TransformerBottleneck(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=4, num_layers=1, height=24, width=88):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, height * width, dim))  # Learned 2D pos encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Add positional encoding
        x_seq = x_seq + self.pos_embed[:, :H*W, :]

        # Add condition token
        cond_token = self.cond_proj(cond).unsqueeze(1)  # [B, 1, C]
        x_input = torch.cat([cond_token, x_seq], dim=1)  # [B, 1+HW, C]

        # Pass through transformer
        x_out = self.transformer(x_input)[:, 1:, :]  # Remove cond token
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        return x_out

class UTransformerModifier(nn.Module):
    def __init__(self, in_channels=8, cond_dim=3, base_ch=32, h=48, w=176):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool = nn.MaxPool2d(2)

        # Pass reduced spatial size to transformer
        self.bottleneck = TransformerBottleneck(dim=base_ch * 2, cond_dim=cond_dim, height=h//2, width=w//2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x, cond):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))

        bottleneck = self.bottleneck(x2, cond)

        up = self.up1(bottleneck)
        merged = torch.cat([up, x1], dim=1)
        x_out = self.dec1(merged)

        delta = self.out_conv(x_out)
        return x + delta, delta
