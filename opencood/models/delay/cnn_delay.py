import torch
import torch.nn as nn



class DelayModule(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_residual = self.encoder(x)
        return x_residual