"""
PointNet & PointNet++ Backbones for 7D Assembly Point Clouds
============================================================
Input:  (B, N, 7)  →  [x, y, z, nx, ny, nz, part_id]

PointNet Architecture (Canonical with T-Net & Invariance Properties):
  1. Part ID → Learnable Categorical Embedding (part_embed_dim)
  2. Input T-Net (3x3): Canonical coordinate space alignment
  3. Shared MLP 1: (3 + 3 + E) → 64 → 64
  4. Feature T-Net (64x64): Canonical feature space alignment
  5. Shared MLP 2: 64 → 128 → 1024
  6. Symmetric Function: Global Max Pooling (Permutation Invariance)
  7. Local + Global Feature Concatenation: (64 + 1024 = 1088) → 512 → 256
  8. Orthogonal Regularization: ||I - A A^T||_F^2 on feature transform matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.pointnet_utils import (
    PointNetEncoder,
    TNet3d,
    TNetKd,
    feature_transform_regularizer,
)
from models.pointnet2_utils import (
    SetAbstractionMSG,
    GlobalSetAbstraction,
    farthest_point_sample,
    index_points,
)


class PartEmbedding(nn.Module):
    """Learnable categorical embedding for part_id values."""

    def __init__(self, max_parts: int = 8, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(max_parts, embed_dim)

    def forward(self, part_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            part_ids: (B, N) integer part identifiers
        Returns:
            embeddings: (B, N, embed_dim)
        """
        return self.embedding(part_ids.long())


class PointNetBackbone(nn.Module):
    """
    PointNet feature extraction backbone with full canonical properties:
      • Permutation Invariance via symmetric max-pooling
      • Input Spatial Alignment via T-Net (3x3)
      • Feature Space Alignment via T-Net (64x64)
      • Feature Transform Orthogonal Regularizer
      • Multi-scale / Local + Global point representation
    """

    def __init__(self, part_embed_dim: int = 16, max_parts: int = 8,
                 global_feat_dim: int = 1024, point_feat_dim: int = 256,
                 use_tnet_input: bool = True, use_tnet_feature: bool = True):
        super().__init__()

        self.part_embed_dim = part_embed_dim
        self.global_feat_dim = global_feat_dim
        self.point_feat_dim = point_feat_dim
        self.use_tnet_input = use_tnet_input
        self.use_tnet_feature = use_tnet_feature

        # Part ID learnable embedding
        self.part_embedding = PartEmbedding(max_parts, part_embed_dim)

        # Initial point input dimension: coordinates (3) + normals (3) + part_embed (E)
        in_channel = 3 + 3 + part_embed_dim

        # Canonical PointNet Encoder
        self.encoder = PointNetEncoder(
            in_channel=in_channel,
            global_feat_dim=global_feat_dim,
            use_tnet_input=use_tnet_input,
            use_tnet_feature=use_tnet_feature,
        )

        # Point feature refinement MLP (Local 64 + Global 1024 = 1088 -> 512 -> point_feat_dim)
        combined_in_dim = 64 + global_feat_dim
        self.point_mlp = nn.Sequential(
            nn.Conv1d(combined_in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, point_feat_dim, 1),
            nn.BatchNorm1d(point_feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            point_cloud: (B, N, 7) – [x, y, z, nx, ny, nz, part_id]

        Returns:
            dict containing:
                'global_feat':    (B, 1024) permutation-invariant global signature
                'point_features': (B, N, point_feat_dim) refined local+global point descriptors
                'point_feat_64':  (B, N, 64) local point features
                'trans_input':    (B, 3, 3) coordinate transformation matrix from input T-Net
                'trans_feat':     (B, 64, 64) feature transformation matrix from feature T-Net
                'part_ids':       (B, N) part IDs
                'xyz':            (B, N, 3) point coordinates
        """
        B, N, _ = point_cloud.shape

        xyz = point_cloud[:, :, :3].contiguous()         # (B, N, 3)
        normals = point_cloud[:, :, 3:6].contiguous()     # (B, N, 3)
        part_ids = point_cloud[:, :, 6].contiguous()      # (B, N)

        # Part embedding
        part_embeds = self.part_embedding(part_ids)       # (B, N, E)

        # Concatenate: [coords, normals, part_embeds] -> (B, N, 3 + 3 + E)
        feat_in = torch.cat([xyz, normals, part_embeds], dim=-1)  # (B, N, in_channel)
        feat_in_t = feat_in.transpose(1, 2)                       # (B, in_channel, N)
        xyz_t = xyz.transpose(1, 2)                               # (B, 3, N)

        # PointNet Encoder forward pass
        global_feat, pt_64, combined_feat, trans_input, trans_feat = self.encoder(
            feat_in_t, xyz=xyz_t
        )

        # Refine local-global point representations
        refined_pt_feats = self.point_mlp(combined_feat)          # (B, point_feat_dim, N)
        point_features = refined_pt_feats.transpose(1, 2)         # (B, N, point_feat_dim)
        pt_64_out = pt_64.transpose(1, 2)                         # (B, N, 64)

        return {
            "global_feat": global_feat,
            "point_features": point_features,
            "point_feat_64": pt_64_out,
            "trans_input": trans_input,
            "trans_feat": trans_feat,
            "part_ids": part_ids,
            "xyz": xyz,
        }

    def get_regularization_loss(self, trans_feat: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute PointNet orthogonal regularizer loss on feature transform matrix."""
        if trans_feat is None:
            return torch.tensor(0.0)
        return feature_transform_regularizer(trans_feat)


class PointNet2Backbone(nn.Module):
    """
    PointNet++ feature extraction backbone (Hierarchical Set Abstraction).
    Retained for architectural comparison and ablation studies.
    """

    def __init__(self, part_embed_dim: int = 16, max_parts: int = 8):
        super().__init__()

        self.part_embed_dim = part_embed_dim
        self.part_embedding = PartEmbedding(max_parts, part_embed_dim)

        in_features = 3 + part_embed_dim

        self.sa1 = SetAbstractionMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 64],
            in_channel=in_features,
            mlp_channels_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ]
        )

        self.sa2 = SetAbstractionMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            in_channel=320,
            mlp_channels_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ]
        )

        self.global_sa = GlobalSetAbstraction(
            in_channel=640 + 3,
            mlp_channels=[256, 512, 1024]
        )

        self.global_feat_dim = 1024

    def forward(self, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = point_cloud.shape

        xyz = point_cloud[:, :, :3].contiguous()
        normals = point_cloud[:, :, 3:6].contiguous()
        part_ids = point_cloud[:, :, 6].contiguous()

        part_embeds = self.part_embedding(part_ids)
        features = torch.cat([normals, part_embeds], dim=-1)

        sa1_xyz, sa1_features = self.sa1(xyz, features)
        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features)

        sa2_combined = torch.cat([sa2_xyz, sa2_features], dim=-1)
        global_feat = self.global_sa(sa2_xyz, sa2_combined)

        return {
            "global_feat": global_feat,
            "sa1_xyz": sa1_xyz,
            "sa1_features": sa1_features,
            "sa2_xyz": sa2_xyz,
            "sa2_features": sa2_features,
            "point_features": sa2_features,
            "trans_input": None,
            "trans_feat": None,
            "part_ids": part_ids,
            "xyz": xyz,
        }


if __name__ == "__main__":
    # Verify PointNet Backbone
    model = PointNetBackbone(part_embed_dim=16, max_parts=8)
    dummy = torch.randn(2, 2048, 7)
    dummy[:, :, 6] = torch.randint(0, 2, (2, 2048)).float()

    out = model(dummy)
    print("=== PointNet Backbone Output ===")
    print(f"Global feature: {out['global_feat'].shape}")
    print(f"Point features: {out['point_features'].shape}")
    print(f"T-Net Input (3x3): {out['trans_input'].shape}")
    print(f"T-Net Feature (64x64): {out['trans_feat'].shape}")
    reg_loss = model.get_regularization_loss(out['trans_feat'])
    print(f"Feature Transform Reg Loss: {reg_loss.item():.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
