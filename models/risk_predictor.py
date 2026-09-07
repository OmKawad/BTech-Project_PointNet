"""
Multi-Task Risk Prediction Network for Assembly Fit
=====================================================
Chains:
  PointNet Feature Backbone (with T-Nets & Permutation Invariance)
    → Assembly Relationship Analysis Module (Cross-Part Attention & Interface Reasoning)
    → Multi-Task Prediction Heads
    → Predictions:
        1. Interference Risk  ∈ [0, 1]   (continuous regression)
        2. Clearance Risk     ∈ [0, 1]   (continuous regression)
        3. Alignment Risk     ∈ [0, 1]   (continuous regression)
        4. Fit Failure Prob.  ∈ [0, 1]   (continuous regression)
        5. Risk Level ∈ {Low, Medium, High}  (3-class classification)
        6. T-Net Matrices: trans_input (3x3), trans_feat (64x64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.backbone import PointNetBackbone, PointNet2Backbone
from models.pointnet_utils import feature_transform_regularizer
from models.assembly_relation import AssemblyRelationModule


class RiskPredictionHead(nn.Module):
    """
    Multi-task prediction network with shared trunk and dedicated heads.
    """

    def __init__(self, input_dim: int = 512, hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Shared trunk
        trunk_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            trunk_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shared_trunk = nn.Sequential(*trunk_layers)

        last_dim = hidden_dims[-1]

        # Individual regression heads (each outputs a single scalar in [0, 1])
        self.interference_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.clearance_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.alignment_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.failure_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Classification head: Low (0) / Medium (1) / High (2)
        self.classification_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, input_dim) assembly relationship features

        Returns:
            dict containing all task predictions
        """
        shared = self.shared_trunk(features)  # (B, last_dim)

        interference = self.interference_head(shared).squeeze(-1)
        clearance = self.clearance_head(shared).squeeze(-1)
        alignment = self.alignment_head(shared).squeeze(-1)
        failure = self.failure_head(shared).squeeze(-1)

        risk_logits = self.classification_head(shared)  # (B, 3)
        risk_level = torch.argmax(risk_logits, dim=-1)   # (B,)

        return {
            "interference_risk": interference,
            "clearance_risk": clearance,
            "alignment_risk": alignment,
            "failure_prob": failure,
            "risk_logits": risk_logits,
            "risk_level": risk_level,
        }


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for assembly fit risk prediction:
      L = λ₁·MSE(interference) + λ₂·MSE(clearance) + λ₃·MSE(alignment)
        + λ₄·BCE(failure)      + λ₅·CE(risk_level)
        + λ_reg·Regularizer(trans_feat)
    """

    def __init__(self, weights: Dict[str, float] = None, reg_weight: float = 0.001):
        super().__init__()
        if weights is None:
            weights = {
                "interference": 1.0,
                "clearance": 1.0,
                "alignment": 1.0,
                "failure": 1.5,
                "classification": 2.0,
                "regularization": reg_weight,
            }
        self.weights = weights
        self.reg_weight = self.weights.get("regularization", reg_weight)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: dict from AssemblyFitRiskModel
            targets:     dict from DataLoader ground truth labels

        Returns:
            total_loss: scalar tensor
            loss_dict: breakdown dictionary for logging
        """
        l_interf = self.mse(predictions["interference_risk"], targets["interference_risk"])
        l_clear = self.mse(predictions["clearance_risk"], targets["clearance_risk"])
        l_align = self.mse(predictions["alignment_risk"], targets["alignment_risk"])
        l_fail = self.bce(predictions["failure_prob"], targets["failure_prob"])
        l_cls = self.ce(predictions["risk_logits"], targets["risk_level"])

        # PointNet Feature Transform Orthogonal Regularization Loss
        trans_feat = predictions.get("trans_feat")
        if trans_feat is not None:
            l_reg = feature_transform_regularizer(trans_feat)
        else:
            l_reg = torch.tensor(0.0, device=predictions["interference_risk"].device)

        total = (self.weights.get("interference", 1.0) * l_interf +
                 self.weights.get("clearance", 1.0) * l_clear +
                 self.weights.get("alignment", 1.0) * l_align +
                 self.weights.get("failure", 1.5) * l_fail +
                 self.weights.get("classification", 2.0) * l_cls +
                 self.reg_weight * l_reg)

        loss_dict = {
            "interference_loss": l_interf.item(),
            "clearance_loss": l_clear.item(),
            "alignment_loss": l_align.item(),
            "failure_loss": l_fail.item(),
            "classification_loss": l_cls.item(),
            "reg_loss": l_reg.item(),
            "total_loss": total.item(),
        }
        return total, loss_dict


# ──────────────────────────────────────────────
# Full End-to-End Model
# ──────────────────────────────────────────────

class AssemblyFitRiskModel(nn.Module):
    """
    Complete Assembly Fit Risk Prediction Model using PointNet.

    Pipeline:
      Point Cloud (B, N, 7)
        → PointNet Feature Backbone (with T-Net 3x3, T-Net 64x64, Symmetric Max Pooling)
        → Assembly Relationship Analysis Module (Cross-Part Attention & Mating Surface Analysis)
        → Multi-Task Risk Prediction Heads
        → {Interference, Clearance, Alignment, Failure, Risk Level, Transformation Matrices}
    """

    def __init__(self, backbone_type: str = "pointnet",
                 part_embed_dim: int = 16, max_parts: int = 8,
                 point_feat_dim: int = 256, global_feat_dim: int = 1024,
                 use_tnet_input: bool = True, use_tnet_feature: bool = True,
                 num_heads: int = 4, relation_hidden_dim: int = 256,
                 risk_hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        self.backbone_type = backbone_type

        # Backbone Selection (PointNet by default)
        if backbone_type.lower() == "pointnet":
            self.backbone = PointNetBackbone(
                part_embed_dim=part_embed_dim,
                max_parts=max_parts,
                global_feat_dim=global_feat_dim,
                point_feat_dim=point_feat_dim,
                use_tnet_input=use_tnet_input,
                use_tnet_feature=use_tnet_feature,
            )
            backbone_dim = point_feat_dim
        elif backbone_type.lower() in ("pointnet2", "pointnet++"):
            self.backbone = PointNet2Backbone(
                part_embed_dim=part_embed_dim,
                max_parts=max_parts,
            )
            backbone_dim = 640
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Choose 'pointnet' or 'pointnet++'")

        # Assembly Relationship Analysis
        self.relation_module = AssemblyRelationModule(
            backbone_feat_dim=backbone_dim,
            num_heads=num_heads,
            hidden_dim=relation_hidden_dim,
            dropout=dropout,
        )

        # Risk Prediction Heads
        self.risk_head = RiskPredictionHead(
            input_dim=self.relation_module.output_dim,
            hidden_dims=risk_hidden_dims or [256, 128],
            dropout=dropout,
        )

    def forward(self, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            point_cloud: (B, N, 7) – [x, y, z, nx, ny, nz, part_id]

        Returns:
            predictions dict with all risk values, classifications, and T-Net matrices
        """
        # Step 1: Feature extraction via PointNet
        backbone_out = self.backbone(point_cloud)

        # Step 2: Assembly relationship analysis
        relation_features = self.relation_module(backbone_out, point_cloud)

        # Step 3: Multi-task risk prediction
        predictions = self.risk_head(relation_features)

        # Attach T-Net transformation matrices and global feature for analysis & regularizer
        predictions["trans_input"] = backbone_out.get("trans_input")
        predictions["trans_feat"] = backbone_out.get("trans_feat")
        predictions["global_feat"] = backbone_out.get("global_feat")

        return predictions

    def get_num_parameters(self) -> Dict[str, int]:
        """Return parameter counts per component."""
        counts = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "relation_module": sum(p.numel() for p in self.relation_module.parameters()),
            "risk_head": sum(p.numel() for p in self.risk_head.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


if __name__ == "__main__":
    model = AssemblyFitRiskModel(backbone_type="pointnet")
    dummy = torch.randn(2, 2048, 7)
    dummy[:, :, 6] = torch.randint(0, 2, (2, 2048)).float()

    preds = model(dummy)
    print("=== Assembly Fit Risk Model (PointNet) Output ===")
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")

    params = model.get_num_parameters()
    print(f"\n=== Parameter Counts ===")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
