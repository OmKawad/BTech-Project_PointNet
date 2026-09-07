"""
Benchmark Comparison
=====================
Compare our PointNet++ Assembly Fit Risk Model against baseline methods:
  1. Random Baseline
  2. MLP on hand-crafted geometric features (no point cloud learning)
  3. Standard PointNet (global only, no hierarchical SA)

Also visualises the comparison in a summary table and bar chart.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders


# ──────────────────────────────────────────────
# Baseline 1: Random Predictor
# ──────────────────────────────────────────────

def random_baseline(loader) -> Dict[str, float]:
    """Random predictions as a lower-bound baseline."""
    all_true_risk = []
    all_true_interf = []
    all_true_clear = []
    all_true_align = []
    all_true_fail = []

    for _, labels in loader:
        all_true_risk.extend(labels["risk_level"].numpy().tolist())
        all_true_interf.extend(labels["interference_risk"].numpy().tolist())
        all_true_clear.extend(labels["clearance_risk"].numpy().tolist())
        all_true_align.extend(labels["alignment_risk"].numpy().tolist())
        all_true_fail.extend(labels["failure_prob"].numpy().tolist())

    n = len(all_true_risk)
    pred_risk = np.random.randint(0, 3, n)
    pred_interf = np.random.uniform(0, 1, n)
    pred_clear = np.random.uniform(0, 1, n)
    pred_align = np.random.uniform(0, 1, n)
    pred_fail = np.random.uniform(0, 1, n)

    return {
        "method": "Random Baseline",
        "accuracy": accuracy_score(all_true_risk, pred_risk),
        "f1_macro": f1_score(all_true_risk, pred_risk, average="macro", zero_division=0),
        "interf_rmse": np.sqrt(mean_squared_error(all_true_interf, pred_interf)),
        "clear_rmse": np.sqrt(mean_squared_error(all_true_clear, pred_clear)),
        "align_rmse": np.sqrt(mean_squared_error(all_true_align, pred_align)),
        "fail_rmse": np.sqrt(mean_squared_error(all_true_fail, pred_fail)),
    }


# ──────────────────────────────────────────────
# Baseline 2: Simple MLP on hand-crafted features
# ──────────────────────────────────────────────

class HandcraftedMLPBaseline(nn.Module):
    """MLP baseline using hand-crafted geometric statistics instead of learned features."""

    def __init__(self):
        super().__init__()
        # Input: 12 hand-crafted features
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.interf_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.clear_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.align_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.fail_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(32, 3)

    def extract_features(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Extract hand-crafted features from raw point cloud."""
        B = point_cloud.shape[0]
        features = []
        for b in range(B):
            pc = point_cloud[b]
            xyz = pc[:, :3]
            normals = pc[:, 3:6]
            pids = pc[:, 6]
            mask_a = pids == 0
            mask_b = pids != 0

            # Stats
            centroid_a = xyz[mask_a].mean(0) if mask_a.any() else torch.zeros(3)
            centroid_b = xyz[mask_b].mean(0) if mask_b.any() else torch.zeros(3)
            dist = torch.norm(centroid_a - centroid_b)
            spread_a = xyz[mask_a].std() if mask_a.any() else torch.tensor(0.0)
            spread_b = xyz[mask_b].std() if mask_b.any() else torch.tensor(0.0)

            # Normal alignment
            mean_norm_a = normals[mask_a].mean(0) if mask_a.any() else torch.zeros(3)
            mean_norm_b = normals[mask_b].mean(0) if mask_b.any() else torch.zeros(3)
            norm_dot = (mean_norm_a * mean_norm_b).sum()

            feat = torch.stack([
                dist, spread_a, spread_b, norm_dot,
                centroid_a[0], centroid_a[1], centroid_a[2],
                centroid_b[0], centroid_b[1], centroid_b[2],
                mask_a.float().mean(), mask_b.float().mean(),
            ])
            features.append(feat)

        return torch.stack(features)

    def forward(self, point_cloud):
        feats = self.extract_features(point_cloud)
        feats = feats.to(point_cloud.device)
        h = self.mlp(feats)
        return {
            "interference_risk": self.interf_head(h).squeeze(-1),
            "clearance_risk": self.clear_head(h).squeeze(-1),
            "alignment_risk": self.align_head(h).squeeze(-1),
            "failure_prob": self.fail_head(h).squeeze(-1),
            "risk_logits": self.cls_head(h),
            "risk_level": self.cls_head(h).argmax(-1),
        }


# ──────────────────────────────────────────────
# Baseline 3: Vanilla PointNet (no hierarchy)
# ──────────────────────────────────────────────

class VanillaPointNet(nn.Module):
    """Standard PointNet without hierarchical set abstraction."""

    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.interf_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.clear_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.align_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.fail_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(128, 3)

    def forward(self, point_cloud):
        h = self.mlp1(point_cloud)         # (B, N, 512)
        h = h.max(dim=1)[0]               # (B, 512) global max pool
        h = self.head(h)
        return {
            "interference_risk": self.interf_head(h).squeeze(-1),
            "clearance_risk": self.clear_head(h).squeeze(-1),
            "alignment_risk": self.align_head(h).squeeze(-1),
            "failure_prob": self.fail_head(h).squeeze(-1),
            "risk_logits": self.cls_head(h),
            "risk_level": self.cls_head(h).argmax(-1),
        }


# ──────────────────────────────────────────────
# Training & evaluation helper
# ──────────────────────────────────────────────

def train_and_evaluate_baseline(model, train_loader, test_loader,
                                 device, epochs=30, method_name="Baseline"):
    """Quick train + eval for a baseline model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for pc, labels in train_loader:
            pc = pc.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            optimizer.zero_grad()
            preds = model(pc)
            loss = (mse(preds["interference_risk"], labels["interference_risk"]) +
                    mse(preds["clearance_risk"], labels["clearance_risk"]) +
                    mse(preds["alignment_risk"], labels["alignment_risk"]) +
                    ce(preds["risk_logits"], labels["risk_level"]))
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds = {k: [] for k in ["interference_risk", "clearance_risk",
                                   "alignment_risk", "failure_prob", "risk_level"]}
    all_targets = {k: [] for k in all_preds.keys()}

    with torch.no_grad():
        for pc, labels in test_loader:
            pc = pc.to(device)
            preds = model(pc)
            for k in all_preds:
                all_preds[k].extend(preds[k].cpu().numpy().tolist())
                all_targets[k].extend(labels[k].numpy().tolist())

    return {
        "method": method_name,
        "accuracy": accuracy_score(all_targets["risk_level"], all_preds["risk_level"]),
        "f1_macro": f1_score(all_targets["risk_level"], all_preds["risk_level"],
                             average="macro", zero_division=0),
        "interf_rmse": np.sqrt(mean_squared_error(all_targets["interference_risk"],
                                                    all_preds["interference_risk"])),
        "clear_rmse": np.sqrt(mean_squared_error(all_targets["clearance_risk"],
                                                   all_preds["clearance_risk"])),
        "align_rmse": np.sqrt(mean_squared_error(all_targets["alignment_risk"],
                                                   all_preds["alignment_risk"])),
        "fail_rmse": np.sqrt(mean_squared_error(all_targets["failure_prob"],
                                                  all_preds["failure_prob"])),
    }


# ──────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────

def run_benchmark(data_dir: str = "data/samples", output_dir: str = "evaluation_results"):
    """Run all baselines and compare with our model."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Device: {device}")

    loaders = create_dataloaders(data_dir, batch_size=16, num_points=2048)
    results = []

    # 1. Random baseline
    print("\n[1/3] Random Baseline...")
    results.append(random_baseline(loaders["test"]))

    # 2. Hand-crafted MLP
    print("[2/3] Hand-crafted MLP Baseline...")
    mlp_model = HandcraftedMLPBaseline()
    results.append(train_and_evaluate_baseline(
        mlp_model, loaders["train"], loaders["test"], device,
        epochs=30, method_name="MLP + Hand-crafted Features"))

    # 3. Vanilla PointNet
    print("[3/3] Vanilla PointNet Baseline...")
    pn_model = VanillaPointNet()
    results.append(train_and_evaluate_baseline(
        pn_model, loaders["train"], loaders["test"], device,
        epochs=30, method_name="Vanilla PointNet"))

    # 4. Our model (load from checkpoint if exists)
    ckpt_path = "checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        print("[+] Loading our PointNet Assembly Fit Risk Model...")
        from models.risk_predictor import AssemblyFitRiskModel
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model_cfg = config["model"]
        pn_cfg = model_cfg.get("pointnet", {})
        our_model = AssemblyFitRiskModel(
            backbone_type=model_cfg.get("backbone_type", "pointnet"),
            part_embed_dim=model_cfg.get("part_embed_dim", 16),
            max_parts=model_cfg.get("max_parts", 8),
            point_feat_dim=pn_cfg.get("point_feat_dim", 256),
            global_feat_dim=pn_cfg.get("global_feat_dim", 1024),
            use_tnet_input=pn_cfg.get("use_tnet_input", True),
            use_tnet_feature=pn_cfg.get("use_tnet_feature", True),
            num_heads=model_cfg["relation"]["num_heads"],
            relation_hidden_dim=model_cfg["relation"]["hidden_dim"],
            risk_hidden_dims=model_cfg["risk_head"]["hidden_dims"],
            dropout=0.0,
        ).to(device)
        our_model.load_state_dict(checkpoint["model_state_dict"])
        results.append(train_and_evaluate_baseline(
            our_model, loaders["train"], loaders["test"], device,
            epochs=0, method_name=f"Ours ({our_model.backbone_type.capitalize()} + Assembly Relation)"))

    # ── Print results table ──
    print(f"\n{'='*90}")
    print(f"{'Method':<40} {'Acc':>6} {'F1':>6} {'I_RMSE':>7} "
          f"{'C_RMSE':>7} {'A_RMSE':>7} {'F_RMSE':>7}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['method']:<40} {r['accuracy']:>6.3f} {r['f1_macro']:>6.3f} "
              f"{r['interf_rmse']:>7.4f} {r['clear_rmse']:>7.4f} "
              f"{r['align_rmse']:>7.4f} {r['fail_rmse']:>7.4f}")
    print(f"{'='*90}")

    # Save results
    save_path = os.path.join(output_dir, "benchmark_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
