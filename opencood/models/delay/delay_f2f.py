# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Individual architectures
class FutureFramePredictor(nn.Module):
    def __init__(self):
        super(FutureFramePredictor, self).__init__()
        # nCPI, nI = opt['n_channels_per_input'], opt['n_input_frames']
        # nCPT, nT = opt['n_channels_per_target'], opt['n_target_frames']

        nCPI, nI = 8, 3
        nCPT, nT = 8, 1
        ms = 1

        assert 32*ms >=   nCPT*nT, 'Too much compression before output !'

        self.conv1 = nn.Conv2d(nCPI*nI,  64*ms, 1, 1, 0)
        self.conv2 = nn.Conv2d(64*ms,    64*ms, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(64*ms,    64*ms, 3, 1, 2, 2)
        self.conv4 = nn.Conv2d(64*ms,    32*ms, 3, 1, 4, 4)
        self.conv5 = nn.Conv2d(32*ms,    32*ms, 3, 1, 4, 4)
        self.conv6 = nn.Conv2d(32*ms,    32*ms, 3, 1, 2, 2)
        self.conv7 = nn.Conv2d(32*ms,    nCPT*nT, 7, 1, 3)

        self._initialize_weights()

    def forward(self, x):
        # operations to perform
        B = x.shape[0]
        T = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]
        x = x.view(B, C*T, H, W)  # [B, C*T, H, W]

        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.conv7(F.relu(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    B, T, C, H, W = 2, 3, 8, 48, 176  # Batch, Time, Channels, Height, Width
    model = FutureFramePredictor()
    
    x = torch.randn(B, T, C, H, W)  # Dummy input
    output = model(x)
    print("Output shape:", output.shape)  # Expected: [B, C, H, W]