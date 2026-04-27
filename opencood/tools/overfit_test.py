"""
Overfit the delay module on a SINGLE sample.
If it can't even memorize one sample, the architecture is broken.
"""
import torch
import torch.nn.functional as F
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml

def overfit_test():
    hypes = load_yaml('MODEL_v2xset/v2x-vit/config_training_400ms_noRTE.yaml')
    hypes['split_dataset'] = 'train'
    hypes['len_past'] = 4  # ms
    hypes['mode'] = 'feature'
    dataset = build_dataset(hypes, visualize=False, train=True)
    
    # Get one sample
    data = dataset[0]
    past = data['ego']['past_features'].cuda()       # input to delay module
    current = data['ego']['current_features'].cuda()  # current frame
    gt = data['ego']['gt_features'].squeeze(0).cuda()             # target

    data = torch.concat([past, current.unsqueeze(1)], dim=1)  # (N, len_past+1, C, H, W)
    
    # Build ONLY the delay module (not the full pipeline)
    from opencood.models.delay import build_delay_module  # or whichever you use
    model_config = hypes['model']['args']['delay']
    model = build_delay_module(model_config).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Target delta: {F.l1_loss(current, gt.squeeze(0)):.6f}")
    
    for step in range(2000):
        pred, predictions = model(data) 
        loss = F.l1_loss(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            with torch.no_grad():
                baseline = F.l1_loss(current, gt)
                improvement = (baseline - loss) / (baseline + 1e-6)
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            print(f"Step {step:4d} | Loss: {loss:.6f} | Baseline: {baseline:.6f} | "
                  f"Improvement: {improvement:.4f} | GradNorm: {grad_norm:.6f}")
    
    if improvement < 0.01:
        print("\n❌ FAILED: Model cannot even overfit one sample.")
        print("   → Architecture problem. The delay module cannot express the transformation.")
    else:
        print(f"\n✅ PASSED: Model achieved {improvement:.1%} improvement on single sample.")
        print("   → Architecture is fine. Problem is in training dynamics / data.")

if __name__ == '__main__':
    overfit_test()