import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class RowMajorScan2D(nn.Module):
    """
    Takes a 2D spatial tensor and flattens it into 4 different 1D scan patterns.
    """
    def __init__(self, bidirectional=True):
        super().__init__()

        self.bidirectional = bidirectional

    def get_num_scans(self):
        return 2 if self.bidirectional else 1

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        B, C, H, W = x.shape
        
        # 1. Row-major forward (Normal flattening)
        # Traverses left-to-right, row by row.
        scan_1 = x.flatten(2)  # Shape: [B, C, H*W]
        
        if self.bidirectional:
            # 2. Row-major backward
            # Flips vertically and horizontally, then flattens.
            scan_2 = torch.flip(x, dims=[2, 3]).flatten(2)
            
            # Stack them together along a new dimension for easy processing
            # Shape: [Batch, 2, Channels, Sequence_Length]
            return torch.stack([scan_1, scan_2], dim=1)
        else:
            return scan_1.unsqueeze(1)  # Shape: [Batch, 1, Channels, Sequence_Length]

    def unscan(self, x, H, W):
        # x shape: [Batch, 2, Channels, H*W] 
        # H, W are the original spatial dimensions
        B = x.shape[0]
        C = x.shape[2]
        
        # Extract the 2 processed sequences

        # 1. Unscan row-major forward
        y1 = x[:, 0]
        out_1 = y1.view(B, C, H, W)
        
        if self.bidirectional:
            # 2. Unscan row-major backward (View first, then flip back)
            y2 = x[:, 1]
        
            out_2 = torch.flip(y2.view(B, C, H, W), dims=[2, 3])
        else:
            out_2 = torch.Tensor(0)  # Placeholder for unprocessed sequence

        # Merge the 2 representations back into a single feature map
        # Standard practice is simply summing them up.
        out = out_1 + out_2
        
        return out 