import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMModulation(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super(FiLMModulation, self).__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels * 2)  # For gamma and beta
        )

    def forward(self, x, cond):
        # x: [B, C, H, W], cond: [B, cond_dim]
        film_params = self.film_gen(cond)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=1)  # Each: [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class FeatureModifier(nn.Module):
    def __init__(self, in_channels=8, cond_dim=3):
        super(FeatureModifier, self).__init__()
        self.film = FiLMModulation(in_channels, cond_dim)

        self.modifier = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=1)  # Output residual
        )

    def forward(self, x, cond):
        # x: [B, 8, 48, 176], cond: [B, 3]
        x_film = self.film(x, cond)  # Apply FiLM modulation
        delta = self.modifier(x_film)  # Predict residual
        return x + delta  # Return modified feature