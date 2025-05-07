import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# -------------------------------
# 3D CNN Block
# -------------------------------
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# -------------------------------
# 3D CNN Backbone
# -------------------------------
class CNN3D(nn.Module):
    def __init__(self, input_channels=8, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        in_channels = input_channels
        for _ in range(num_layers):
            layers.append(Conv3DBlock(in_channels, hidden_dim))
            in_channels = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.feature_extractor(x)
        return x

# -------------------------------
# Full Model: Predict Future Feature Map
# -------------------------------
class FutureFramePredictor(nn.Module):
    def __init__(self, input_channels=8, hidden_dim=64, kernel_size=(3, 3, 3), num_layers=3):
        super().__init__()
        self.encoder = CNN3D(input_channels, hidden_dim, num_layers)
        self.conv2d_decoder = nn.Conv2d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        #todo: choose only present  # -> [B, C, T, H, W]
        # x = x[:, :, -1, :, :].unsqueeze(2)         # [B, C, H, W]
        
        x = self.encoder(x)           # [B, hidden_dim, T, H, W]
        x = x[:, :, -1, :, :]         # [B, hidden_dim, H, W]
        out = self.conv2d_decoder(x)  # [B, C, H, W]
        return out

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    B, T, C, H, W = 2, 5, 8, 48, 176  # Batch, Time, Channels, Height, Width
    model = FutureFramePredictor(input_channels=C)
    
    x = torch.randn(B, T, C, H, W)  # Dummy input
    output = model(x)
    print("Output shape:", output.shape)  # Expected: [B, C, H, W]