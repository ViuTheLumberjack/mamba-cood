import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
    
class PretrainedDinoEncoder(torch.nn.Module):
    def __init__(self, args):
        """
        dinov2_vits14 -> torch.Size([1, 384])
        dinov2_vitb14 -> torch.Size([1, 768])
        dinov2_vitl14 -> torch.Size([1, 1024])
        dinov2_vitg14 -> torch.Size([1, 1536])
        """
        super(PretrainedDinoEncoder, self).__init__()
        self.version = args.get('version', "dinov2_vitb14")
        self.encoder = torch.hub.load('facebookresearch/dinov2', self.version)

    def forward(self, x):
        with torch.no_grad():
            B, T, C, H, W = x.shape
            x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            # pad h and w to be divisible by 14
            
            pad_h = (14 - H % 14) % 14
            pad_w = (14 - W % 14) % 14
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            x = einops.repeat(x, 'b c h w -> b (c repeat_c) h w', repeat_c=3)  # Repeat channels to match expected input
            feat_enc = self.encoder(x)
            feat_enc = einops.rearrange(feat_enc, '(b t) hd -> b t hd', b=B, t=T)
        
        return feat_enc, None