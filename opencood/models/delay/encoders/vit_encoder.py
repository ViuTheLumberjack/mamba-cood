import torch
import torch.nn as nn
import timm
import einops

class VideoViTEncoderWrapper(nn.Module):
    def __init__(self, args):
        super(VideoViTEncoderWrapper, self).__init__()
        
        self.image_size = args.get('image_size', 64)
        self.patch_size = args.get('patch_size', 8)
        self.grid_size = self.image_size // self.patch_size
        self.use_pretrained = args.get('use_pretrained', False)
        self.features_only = args.get('features_only', True)  # We only need the feature maps, not the classification head
        self.out_indices = args.get('out_indices', [1])  # Extract features from the last stage of the ViT

        # Use a pre-trained ViT model with smaller patch size if available
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in22k', 
            pretrained=self.use_pretrained, 
            img_size=self.image_size,
            window_size=self.patch_size,
            features_only=self.features_only, 
            out_indices=self.out_indices
        )
        

    def forward(self, video_frames):
        """
        Args:
            video_frames: Tensor of shape (Batch, Time, Channels, Height, Width)
        Returns:
            spatial_features: Tensor of shape (Batch, Time, Hidden_Size, Grid_H, Grid_W)
        """
        B, T, C, H, W = video_frames.shape
        
        # 1. Fold time into the batch dimension to process frame-by-frame
        # Shape: (B * T, C, H, W)
        frames_folded = einops.rearrange(video_frames, 'b t c h w -> (b t) c h w')
        
        # 2. Pass through the ViT encoder
        # interpolate_pos_encoding=True allows the model to handle 64x64 if using 224x224 weights
        features = self.vit(frames_folded)
        
        # 3. Unfold the time dimension back out
        # Final Shape: (Batch, Time, Hidden_Size, Grid_H, Grid_W)
        spatial_features = einops.rearrange(features, 'b t h w c -> b t c h w', b=B, t=T)
        
        return spatial_features

if __name__ == "__main__":
    model = VideoViTEncoderWrapper(
        image_size=64, 
        patch_size=8, 
        num_channels=3, 
        hidden_size=256, 
        use_pretrained=True
    )

    # Dummy frame from your 64x64 video setup (Batch, Time, Channels, Height, Width)
    dummy_frame = torch.randn(1, 5, 3, 64, 64)

    # Extract spatial features
    features = model(dummy_frame)

    print(f"Input shape: {dummy_frame.shape}")
    print(f"Feature map shape: {features.shape}")