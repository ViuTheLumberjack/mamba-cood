import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class ColumnMajorScan2D(nn.Module):
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
        
        # 3. Column-major forward
        # Transposes Height and Width, then flattens. Traverses top-to-bottom, col by col.
        scan_3 = x.transpose(2, 3).flatten(2)
        
        # 4. Column-major backward
        # Flips vertically and horizontally, transposes, then flattens.
        if self.bidirectional:
            scan_4 = torch.flip(x, dims=[2, 3]).transpose(2, 3).flatten(2)
            # Stack them together along a new dimension for easy processin
            
            # Stack them together along a new dimension for easy processing
            # Shape: [Batch, 2, Channels, Sequence_Length]
            return torch.stack([scan_3, scan_4], dim=1)
        else:
            return scan_3.unsqueeze(1)  # Shape: [Batch, 1, Channels, Sequence_Length]

    def unscan(self, x, H, W):
        # x shape: [Batch, 2/1, Channels, H*W] 
        # H, W are the original spatial dimensions
        B = x.shape[0]
        C = x.shape[2]
        
        # Extract the 2 processed sequences
        y3 = x[:, 0] 
        # 3. Unscan column-major forward (View as W, H first, then transpose back)
        out_3 = y3.view(B, C, W, H).transpose(2, 3)
        
        if self.bidirectional:
            y4 = x[:, 1]
            
            # 4. Unscan column-major backward
            out_4 = torch.flip(y4.view(B, C, W, H).transpose(2, 3), dims=[2, 3])
        else:
            out_4 = torch.zeros_like(out_3)  # Placeholder for unprocessed sequence

        # Merge the 2 representations back into a single feature map
        # Standard practice is simply summing them up.
        out = out_3 + out_4
        
        return out 