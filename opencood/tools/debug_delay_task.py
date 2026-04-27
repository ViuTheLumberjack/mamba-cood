import torch
import torch.nn.functional as F
import numpy as np
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml

def analyze_task_difficulty():
    """Check if current→GT delta is actually meaningful and learnable."""
    
    hypes = load_yaml('MODEL_v2xset/v2x-vit/config_training_100ms_noRTE.yaml')
    hypes['split_dataset'] = 'train'
    hypes['len_past'] = 4  # ms
    hypes['mode'] = 'feature'
    dataset = build_dataset(hypes, visualize=False, train=True)
    
    deltas = []
    spatial_sparsities = []
    
    for idx in range(min(200, len(dataset))):
        data = dataset[idx]
        
        current = data['ego']['current_features']      # (N, C, H, W)
        gt = data['ego']['gt_features'].squeeze(0)     # (N, C, H, W)
        past = data['ego']['past_features']             # (N, len_past, C, H, W)
        ego_flag = data['ego']['ego_list']

        print(current.shape, gt.shape, past.shape)
        
        for agent_idx in range(gt.shape[0]):
            if ego_flag[agent_idx] == 1:  # skip ego
                continue
            
            c = current[agent_idx]  # (C, H, W)
            g = gt[agent_idx]       # (C, H, W)
            
            # 1. How different is GT from current?
            delta = (g - c).abs()
            relative_delta = delta.mean() / (c.abs().mean() + 1e-6)
            deltas.append(relative_delta.item())
            
            # 2. Is the delta spatially sparse or dense?
            delta_spatial = delta.mean(dim=0)  # (H, W)
            active_pixels = (delta_spatial > delta_spatial.mean()).float().mean()
            spatial_sparsities.append(active_pixels.item())
    
    deltas = np.array(deltas)
    sparsities = np.array(spatial_sparsities)
    
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