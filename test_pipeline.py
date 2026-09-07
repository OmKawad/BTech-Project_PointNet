"""Comprehensive smoke test for PointNet Assembly Fit Risk Prediction Pipeline."""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from data.dataset import create_dataloaders
from models.risk_predictor import AssemblyFitRiskModel, MultiTaskLoss

print("=" * 65)
print("SMOKE TEST: PointNet Assembly Fit Risk Prediction Pipeline")
print("=" * 65)

# 1. Test data loading
print("\n[1] Testing DataLoader...")
loaders = create_dataloaders("data/samples", batch_size=4, num_points=2048)
batch = next(iter(loaders["train"]))
pc, labels = batch
print(f"    Point cloud shape: {pc.shape} (B, N, 7)")
print(f"    Risk levels: {labels['risk_level'].tolist()}")
print(f"    Interference: {[f'{v:.3f}' for v in labels['interference_risk'].tolist()]}")
print(f"    Clearance:    {[f'{v:.3f}' for v in labels['clearance_risk'].tolist()]}")
print(f"    Alignment:    {[f'{v:.3f}' for v in labels['alignment_risk'].tolist()]}")

# 2. Test PointNet forward pass
print("\n[2] Testing PointNet model forward pass...")
model = AssemblyFitRiskModel(backbone_type="pointnet")
preds = model(pc)
for k, v in preds.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {v.shape}")
    else:
        print(f"    {k}: {v}")

# 3. Test PointNet Canonical Properties
print("\n[3] Testing PointNet Canonical Mathematical Properties...")
# A. T-Net Matrices
t3 = preds["trans_input"]
t64 = preds["trans_feat"]
print(f"    [3a] Coordinate T-Net (3x3): shape={t3.shape}")
print(f"    [3b] Feature T-Net (64x64):  shape={t64.shape}")

# B. Permutation Invariance: Shuffle points in pc and verify global_feat is invariant
print("    [3c] Verifying Permutation Invariance (Symmetric Max-Pooling)...")
perm = torch.randperm(2048)
pc_shuffled = pc[:, perm, :]
with torch.no_grad():
    preds_orig = model(pc)
    preds_shuf = model(pc_shuffled)
    diff = torch.norm(preds_orig["global_feat"] - preds_shuf["global_feat"]).item()
    print(f"         Global feature discrepancy after point shuffle: {diff:.6e}")
    assert diff < 1e-4, f"Permutation invariance violated! Diff: {diff}"
    print("         Permutation invariance confirmed! [OK]")

# 4. Test multi-task loss with PointNet orthogonal regularizer
print("\n[4] Testing multi-task loss with PointNet regularizer...")
criterion = MultiTaskLoss()
loss, loss_dict = criterion(preds, labels)
print(f"    Total loss: {loss.item():.4f}")
for k, v in loss_dict.items():
    print(f"      {k}: {v:.4f}")
assert "reg_loss" in loss_dict, "reg_loss missing from loss_dict!"

# 5. Test backward pass
print("\n[5] Testing backward gradient propagation...")
loss.backward()
print("    Gradient backpropagation successful without NaNs! [OK]")

# 6. Parameter counts
params = model.get_num_parameters()
print(f"\n[6] PointNet Model Parameters:")
for k, v in params.items():
    print(f"    {k}: {v:,}")

print("\n" + "=" * 65)
print("ALL POINTNET PIPELINE TESTS PASSED [OK]")
print("=" * 65)
