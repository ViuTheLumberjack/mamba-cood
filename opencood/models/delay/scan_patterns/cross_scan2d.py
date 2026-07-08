import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class CrossScan2D(nn.Module):
    """
    Takes a 2D spatial tensor and flattens it into 4 different 1D scan patterns.
    """
    def __init__(self, bidirectional=True):
        super().__init__()

        self.bidirectional = bidirectional

    def get_num_scans(self):
        return 4 if self.bidirectional else 2

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        B, C, H, W = x.shape
        
        # 1. Row-major forward (Normal flattening)
        # Traverses left-to-right, row by row.
        scan_1 = x.flatten(2)  # Shape: [B, C, H*W]
        
        # 3. Column-major forward
        # Transposes Height and Width, then flattens. Traverses top-to-bottom, col by col.
        scan_3 = x.transpose(2, 3).flatten(2)
        
        if self.bidirectional:
            # 2. Row-major backward
            # Flips vertically and horizontally, then flattens.
            scan_2 = torch.flip(x, dims=[2, 3]).flatten(2)
            
            # 4. Column-major backward
            # Flips vertically and horizontally, transposes, then flattens.
            scan_4 = torch.flip(x, dims=[2, 3]).transpose(2, 3).flatten(2)
            
            # Stack them together along a new dimension for easy processing
            # Shape: [Batch, 4, Channels, Sequence_Length]
            return torch.stack([scan_1, scan_2, scan_3, scan_4], dim=1)
        else:
            return torch.stack([scan_1, scan_3], dim=1)  # Shape: [Batch, 2, Channels, Sequence_Length]

    def unscan(self, x, H, W):
        # x shape: [Batch, 4, Channels, H*W] 
        # H, W are the original spatial dimensions
        B = x.shape[0]
        C = x.shape[2]
        
        # Extract the 4 processed sequences
        # 1. Unscan row-major forward
        y1 = x[:, 0]
        out_1 = y1.view(B, C, H, W)
        
        # 2. Unscan row-major backward (View first, then flip back)
        y2 = x[:, 1]
        out_2 = torch.flip(y2.view(B, C, H, W), dims=[2, 3])
        
        if self.bidirectional:
            # 3. Unscan column-major forward (View as W, H first, then transpose back)
            y3 = x[:, 2]
            out_3 = y3.view(B, C, W, H).transpose(2, 3)
            
            # 4. Unscan column-major backward
            y4 = x[:, 3]
            out_4 = torch.flip(y4.view(B, C, W, H).transpose(2, 3), dims=[2, 3])
        else:
            out_3 = torch.Tensor(0)  # Placeholder for unprocessed sequence
            out_4 = torch.Tensor(0)  # Placeholder for unprocessed sequence
        
        # Merge the 4 representations back into a single feature map
        # Standard practice is simply summing them up.
        out = out_1 + out_2 + out_3 + out_4
        
        return out 