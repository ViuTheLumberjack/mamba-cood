import torch
import torch.nn as nn
import einops
import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, config):
        super(TemporalConsistencyLoss, self).__init__()
        self.loss_dict = {}


    def generate_grid(self, B, C, H, W, device):
        """
        Generates a normalized 3D base coordinate grid for grid_sample.
        Outputs coordinates in the range [-1, 1] for (x, y, z) = (W, H, C).
        """
        # Create unnormalized grid (0..W-1, 0..H-1, 0..C-1)
        x = torch.arange(W, device=device, dtype=torch.float32)
        y = torch.arange(H, device=device, dtype=torch.float32)
        z = torch.arange(C, device=device, dtype=torch.float32)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')

        # Normalize to [-1, 1] as required by F.grid_sample
        denom_w = max(W - 1, 1)
        denom_h = max(H - 1, 1)
        denom_c = max(C - 1, 1)
        grid_x = 2.0 * grid_x / denom_w - 1.0
        grid_y = 2.0 * grid_y / denom_h - 1.0
        grid_z = 2.0 * grid_z / denom_c - 1.0

        # Stack and expand to batch size: (B, C, H, W, 3)
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)

        return grid

    def warp_and_mask(self, pred_t, flow):
        """
        Warps the prediction at time t using the flow field.
        pred_t: (B, C, H, W) - BEV logits/probs at time t
        flow: (B, 2, C, H, W) - Flow field (pixel displacements dx, dy)
        """
        B, C, H, W = pred_t.shape
        device = pred_t.device

        # 1. Get base normalized grid
        base_grid = self.generate_grid(B, C, H, W, device)

        # 2. Normalize flow displacements to [-1, 1] scale
        # Flow x is divided by W/2, Flow y is divided by H/2.
        # No displacement is applied along C (z) by default.
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0, :, :, :] = flow[:, 0, :, :, :] / (W / 2.0)
        flow_norm[:, 1, :, :, :] = flow[:, 1, :, :, :] / (H / 2.0)

        # Permute flow to match grid shape: (B, C, H, W, 2)
        flow_norm = flow_norm.permute(0, 2, 3, 4, 1)
        flow_norm_3d = torch.zeros((B, C, H, W, 3), device=device, dtype=pred_t.dtype)
        flow_norm_3d[..., 0] = flow_norm[..., 0]
        flow_norm_3d[..., 1] = flow_norm[..., 1]

        # 3. Apply flow to base grid
        warped_grid = base_grid + flow_norm_3d

        # 4. Differentiable Warping
        # padding_mode='zeros' ensures out-of-bounds areas become 0
        pred_t_3d = pred_t.unsqueeze(1)  # (B, 1, C, H, W)
        warped_pred = F.grid_sample(
            pred_t_3d,
            warped_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1)

        # 5. Create the Visibility Mask (Occlusion/Edge Handling)
        # Any coordinate that fell outside [-1, 1] is invalid
        valid_mask = (warped_grid[..., 0] >= -1.0) & (warped_grid[..., 0] <= 1.0) & \
                     (warped_grid[..., 1] >= -1.0) & (warped_grid[..., 1] <= 1.0) & \
                     (warped_grid[..., 2] >= -1.0) & (warped_grid[..., 2] <= 1.0)

        # Shape mask to (B, C, H, W)
        valid_mask = valid_mask.float()

        return warped_pred, valid_mask

    def generate_flow_from_sequence(self, gt_sequence):
        """
        Extracts ground truth flow from a sequence of BEV masks.
        
        Args:
            gt_sequence: (B, T, C, H, W) binary/probability mask of vehicles.
        Returns:
            flow_maps: (B, 2, C, H, W) pixel displacement vectors.
        """
        B, T, C, H, W = gt_sequence.shape
        device = gt_sequence.device
        
        # 1. Create coordinate grids
        # y ranges from 0 to H-1, x ranges from 0 to W-1
        y_coords = torch.arange(H, device=device).view(1, 1, 1, H, 1).float()
        x_coords = torch.arange(W, device=device).view(1, 1, 1, 1, W).float()
        
        # 2. Calculate the Center of Mass (CoM) for each channel at each timestep
        # Sum of pixel values (mass)
        mass = gt_sequence.sum(dim=(3, 4), keepdim=True) + 1e-6 # Add epsilon to avoid div by zero
        
        # Sum of (coordinates * pixel values) / mass
        com_y = (gt_sequence * y_coords).sum(dim=(3, 4), keepdim=True) / mass
        com_x = (gt_sequence * x_coords).sum(dim=(3, 4), keepdim=True) / mass
        
        # Shape of com_x, com_y: (B, T, C, 1, 1)
        
        # 3. Calculate displacement (Flow Vector) between t and t+1
        # We want to know how much the CoM moved FROM t TO t+1
        dy = com_y[:, 1:, :, :, :] - com_y[:, :-1, :, :, :] # (B, T-1, C, 1, 1)
        dx = com_x[:, 1:, :, :, :] - com_x[:, :-1, :, :, :] # (B, T-1, C, 1, 1)
        
        # 4. Broadcast the flow vectors to the full HxW grid
        # We only want the flow to exist WHERE the channel was active at time t
        # (Background gets 0 flow)
        gt_t = gt_sequence[:, :-1, :, :, :] # Source frames (B, T-1, C, H, W)
        
        # Create empty flow maps
        flow_x = torch.zeros((B, T-1, C, H, W), device=device)
        flow_y = torch.zeros((B, T-1, C, H, W), device=device)
        
        # Apply the constant flow vector to the mask
        # If gt_t is binary (0 or 1), this works perfectly.
        # If it's probabilities, you might want to threshold it first: (gt_t > 0.5).float()
        flow_x = dx * gt_t
        flow_y = dy * gt_t
        
        # 5. Stack into final (B, T-1, 2, C, H, W) tensor
        flow_maps = torch.stack((flow_x, flow_y), dim=2)
        flow_maps = flow_maps.sum(dim=1) # (B, 2, C, H, W)

        return flow_maps

    def forward(self, feature_pred, predictions, target_dict):
        """
        Calculates the masked L1 temporal consistency loss.
        """
        target_dict_copy = target_dict.copy()
        target_dict = target_dict['ego']['label_dict']
        #added loss
        past = target_dict_copy['ego']['past_features']
        current = target_dict_copy['ego']['current_features']
        feature_gt = target_dict_copy['ego']['gt_features']
        feature_gt = einops.rearrange(feature_gt, 't b c h w -> b t c h w') 

        # print("Past shape:", past.shape)
        # print("Current shape:", current.shape)
        # print("GT shape:", feature_gt.shape)

        pred_t = feature_pred  # Prediction at time t

        flow = self.generate_flow_from_sequence(torch.cat([past, feature_gt], dim=1))

        # print("Flow shape:", flow.shape)
        # Warp pred_t to align with t+1
        warped_pred_t, valid_mask = self.warp_and_mask(current, flow)
        
        # Calculate L1 difference
        diff = torch.abs(pred_t - warped_pred_t)
        
        # Apply mask to ignore out-of-bounds/occluded regions
        masked_diff = diff * valid_mask
        
        # Compute mean over valid pixels only to prevent artificial loss deflation
        loss = masked_diff.sum() / (valid_mask.sum() + 1e-6)
        self.loss_dict.update({
                               'loss_feature': loss, 
                               'predictions_mean': predictions.mean(),
                               'predictions_std': predictions.std()
                               })

        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        pred_loss = self.loss_dict['loss_feature']
        mu = self.loss_dict['predictions_mean']
        std = self.loss_dict['predictions_std']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || %.4f - %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    pred_loss.item(), mu.item(), std.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || %.4f - %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    pred_loss.item(), mu.item(), std.item()))
        

if __name__ == '__main__':
    B, T, C, H, W = 2, 5, 8, 48, 176
    pred_t = torch.rand(B, C, H, W)
    gt_sequence = torch.rand(B, T, C, H, W)

    target_dict = {
        'ego': {
            'label_dict': {},
            'past_features': torch.rand(B, T - 1, C, H, W),
            'current_features': torch.rand(B, C, H, W),
            'gt_features': torch.rand(T - 1, B, C, H, W),
            'record_len': [T - 1],
        }
    }
    
    loss_module = TemporalConsistencyLoss(None)
    loss = loss_module(gt_sequence[:, -1], gt_sequence, target_dict)
    print("Temporal Consistency Loss:", loss.item())