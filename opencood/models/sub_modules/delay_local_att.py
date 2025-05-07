import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionBasedModifier(nn.Module):
    def __init__(self, in_channels=8, cond_dim=3):
        super(AttentionBasedModifier, self).__init__()
        self.in_channels = in_channels

        # Project condition vector into a spatial embedding
        self.cond_projector = nn.Sequential(
            nn.Linear(cond_dim, in_channels * 3),
            nn.ReLU()
        )

        # Attention generator: input = [x || cond_feature]
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # attention map in [0, 1]
        )

        # Modification network: predicts residual
        self.modifier = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=1)
        )

    def forward(self, x, cond):
        B, C, H, W = x.shape

        # Project condition to [B, C, 3] then expand spatially
        cond_feat = self.cond_projector(cond)  # [B, C*3]
        cond_feat = cond_feat.view(B, C, 3)  # [B, C, 3]
        cond_feat = cond_feat.mean(dim=2, keepdim=True)  # [B, C, 1]
        cond_feat = cond_feat.unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]

        # Concatenate input and cond embedding
        x_cond = torch.cat([x, cond_feat], dim=1)  # [B, 2C, H, W]

        # Get attention map
        attn = self.attention_net(x_cond)  # [B, 1, H, W]

        # Compute modification
        delta = self.modifier(x)  # [B, C, H, W]
        modulated = delta * attn  # apply attention

        return x + modulated, attn  # return both for visualization
