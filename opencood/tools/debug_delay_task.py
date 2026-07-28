import torch
import torch.nn.functional as F
import numpy as np
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml

def analyze_task_difficulty():
    """Check if current→GT delta is actually meaningful and learnable."""
    
    hypes = load_yaml('opencood/hypes_yaml/point_pillar_v2xvit_delay_multpred_encoder.yaml')
    hypes['split_dataset'] = 'validate'
    hypes['len_past'] = 4  # ms
    hypes['mode'] = 'feature'
    dataset = build_dataset(hypes, visualize=False, train=False)
    
    deltas = []
    spatial_sparsities = []
    zeroes = []
    
    for idx in range(min(1000, len(dataset))):
        data = dataset[idx]
        
        current = data['ego']['current_features']      # (N, C, H, W)
        gt = data['ego']['gt_features']     # (B, T, C, H, W)
        past = data['ego']['past_features']             # (N, len_past, C, H, W)

        c = current.unsqueeze(1).repeat(1, gt.shape[1], 1, 1, 1)  # (B, T, C, H, W)
        g = gt       # (B, T, C, H, W)
        
        # 1. How different is GT from current?
        delta = (g - c).abs()
        relative_delta = delta.mean() / (c.abs().mean() + 1e-6)
        deltas.append(relative_delta.item())
        
        # 2. Is the delta spatially sparse or dense?
        delta_spatial = delta.mean(dim=0)  # (H, W)
        active_pixels = (delta_spatial > delta_spatial.mean()).float().mean()
        spatial_sparsities.append(active_pixels.item())

        # 3. Count the number of zeroes in the input data
        # (current + past) to see if the model has enough information to predict GT 
        input_data = torch.cat([past, current.unsqueeze(1)], dim=1)  # (B, len_past + 1, C, H, W)
        zero_count = (input_data == 0).sum().item()
        zero_ratio = zero_count / input_data.numel()
        zeroes.append(zero_ratio)

    deltas = np.array(deltas)
    sparsities = np.array(spatial_sparsities)
    zero_ratios = np.array(zeroes)

    print(f"=== TASK DIFFICULTY ANALYSIS ===")
    print(f"Samples analyzed: {len(deltas)}")
    print(f"")
    print(f"Relative delta (current→GT):")
    print(f"  Mean:   {deltas.mean():.4f}")
    print(f"  Median: {np.median(deltas):.4f}")
    print(f"  Std:    {deltas.std():.4f}")
    print(f"  Max:    {deltas.max():.4f}")
    print(f"")
    print(f"Spatial sparsity of delta:")
    print(f"  Mean:   {sparsities.mean():.4f}")
    print(f"  (0.0 = all change in one pixel, 1.0 = uniform change)")
    print(f"")
    print(f"Zero ratio in input data:")
    print(f"  Mean:   {zero_ratios.mean():.4f}")
    print(f"  Median: {np.median(zero_ratios):.4f}")
    print(f"  Std:    {zero_ratios.std():.4f}")
    print(f"  Max:    {zero_ratios.max():.4f}")
    print(f"")

    if deltas.mean() < 0.05:
        print("⚠️  CRITICAL: The delta is VERY SMALL (<5%).")
        print("   The current frame is already very close to GT.")
        print("   The model has almost nothing to learn.")
        print("   → Consider increasing delay (e.g., 800ms, 1000ms)")
        print("   → Or verify GT features are from the correct future timestamp")
    elif deltas.mean() < 0.15:
        print("⚠️  WARNING: Delta is small (5-15%).")
        print("   Learning signal exists but is weak.")
    else:
        print("✅ Delta is meaningful (>15%). Task should be learnable.")

if __name__ == '__main__':
    analyze_task_difficulty()