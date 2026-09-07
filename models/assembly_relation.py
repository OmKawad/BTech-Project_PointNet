"""
Assembly Relationship Analysis Module
=======================================
This module models geometric interactions between distinct parts in a multi-component
assembly, operating on features extracted by PointNet (or PointNet++).

Key components:
  1. Part-Aware Feature Separation: Part A vs Part B partitioning
  2. Cross-Part Multi-Head Attention: Part A ↔ Part B mutual geometric dependency
  3. Mating Interface Extraction: Proximity analysis & normal compatibility dot products
  4. Relational Feature Fusion: Combines global shape signature, cross-attended part features,
     and interface statistics into a comprehensive 512-dim assembly relationship representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.pointnet2_utils import square_distance


class CrossPartAttention(nn.Module):
    """
    Multi-Head Cross-Attention between features of two distinct parts.

    Given queries from Part A and keys/values from Part B (and vice versa),
    captures mating surface correspondence and spatial alignment compatibility.
    """

    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert feat_dim % num_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_feats: torch.Tensor,
                context_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feats:   (B, N_q, D) features from one part
            context_feats: (B, N_c, D) features from the other part
        Returns:
            attended: (B, N_q, D)
        """
        B, N_q, D = query_feats.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(query_feats).view(B, N_q, H, d).transpose(1, 2)
        K = self.k_proj(context_feats).view(B, -1, H, d).transpose(1, 2)
        V = self.v_proj(context_feats).view(B, -1, H, d).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)

        x = self.norm1(query_feats + out)
        x = self.norm2(x + self.ffn(x))
        return x


class InterfaceExtractor(nn.Module):
    """
    Extract and analyse the mating interface between two parts.

    Computes:
      - Minimum clearance / interference distance
      - Interface proximity mask
      - Mean and std of interface distances
      - Normal compatibility (facing normals dot product)
    """

    def __init__(self, interface_threshold: float = 0.15):
        super().__init__()
        self.interface_threshold = interface_threshold

    def forward(self, xyz_a: torch.Tensor, xyz_b: torch.Tensor,
                normals_a: Optional[torch.Tensor] = None,
                normals_b: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            xyz_a: (B, N_a, 3) Part A coordinates
            xyz_b: (B, N_b, 3) Part B coordinates
            normals_a: (B, N_a, 3) Part A surface normals
            normals_b: (B, N_b, 3) Part B surface normals
        """
        B, N_a, _ = xyz_a.shape
        _, N_b, _ = xyz_b.shape

        dist_sq = square_distance(xyz_a, xyz_b)  # (B, N_a, N_b)
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8))

        min_dist_a, nn_idx_a = dist.min(dim=-1)  # (B, N_a)
        min_dist_b, _ = dist.min(dim=-2)        # (B, N_b)

        global_min_dist = min_dist_a.min(dim=-1)[0]  # (B,)

        interface_mask_a = min_dist_a < self.interface_threshold
        interface_mask_b = min_dist_b < self.interface_threshold

        results = {
            "min_dist": global_min_dist,
            "dist_a_to_b": min_dist_a,
            "dist_b_to_a": min_dist_b,
        }

        mean_dists = torch.zeros(B, device=xyz_a.device)
        std_dists = torch.zeros(B, device=xyz_a.device)
        for b in range(B):
            mask_a = interface_mask_a[b]
            if mask_a.any():
                d = min_dist_a[b][mask_a]
                mean_dists[b] = d.mean()
                std_dists[b] = d.std() if d.numel() > 1 else 0.0
            else:
                mean_dists[b] = min_dist_a[b].mean()
                std_dists[b] = min_dist_a[b].std()

        results["mean_interface_dist"] = mean_dists
        results["std_interface_dist"] = std_dists
        results["interface_ratio_a"] = interface_mask_a.float().mean(dim=-1)
        results["interface_ratio_b"] = interface_mask_b.float().mean(dim=-1)

        if normals_a is not None and normals_b is not None:
            batch_idx = torch.arange(B, device=xyz_a.device).unsqueeze(-1).expand_as(nn_idx_a)
            nn_normals_b = normals_b[batch_idx, nn_idx_a]
            dot_products = (normals_a * nn_normals_b).sum(dim=-1)

            normal_alignment = torch.zeros(B, device=xyz_a.device)
            for b in range(B):
                mask_a = interface_mask_a[b]
                if mask_a.any():
                    normal_alignment[b] = dot_products[b][mask_a].mean()
                else:
                    normal_alignment[b] = dot_products[b].mean()
            results["normal_alignment"] = normal_alignment
        else:
            results["normal_alignment"] = torch.zeros(B, device=xyz_a.device)

        results["interface_mask_a"] = interface_mask_a
        results["interface_mask_b"] = interface_mask_b
        return results


class AssemblyRelationModule(nn.Module):
    """
    Assembly Relationship Analysis Module compatible with PointNet (and PointNet++).
    """

    def __init__(self, backbone_feat_dim: int = 256,
                 num_heads: int = 4, hidden_dim: int = 256,
                 dropout: float = 0.1, max_attn_points: int = 128):
        super().__init__()
        self.max_attn_points = max_attn_points

        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(backbone_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-part attention (bidirectional)
        self.cross_attn_a_to_b = CrossPartAttention(hidden_dim, num_heads, dropout)
        self.cross_attn_b_to_a = CrossPartAttention(hidden_dim, num_heads, dropout)

        # Interface extractor
        self.interface_extractor = InterfaceExtractor(interface_threshold=0.15)

        # Fusion: [global_a (hidden_dim) + global_b (hidden_dim) + global_feat (1024) + interface_feats (6)]
        fusion_input_dim = hidden_dim * 2 + 1024 + 6
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.output_dim = 512

    def _separate_and_sample_parts(self, xyz: torch.Tensor, features: torch.Tensor,
                                  part_ids: torch.Tensor, target_n: int
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Separate point cloud features into Part A and Part B and regularize to target_n points.
        """
        B, N, D = features.shape

        feat_a_list = []
        feat_b_list = []
        xyz_a_list = []
        xyz_b_list = []

        for b in range(B):
            mask_a = (part_ids[b] == 0)
            mask_b = ~mask_a

            fa = features[b][mask_a]
            fb = features[b][mask_b]
            xa = xyz[b][mask_a]
            xb = xyz[b][mask_b]

            # Subsample or pad Part A
            if len(fa) >= target_n:
                # Subsample evenly
                step = max(1, len(fa) // target_n)
                idx = torch.arange(0, len(fa), step, device=fa.device)[:target_n]
                fa = fa[idx]
                xa = xa[idx]
            elif len(fa) > 0:
                pad = target_n - len(fa)
                fa = F.pad(fa, (0, 0, 0, pad))
                xa = F.pad(xa, (0, 0, 0, pad))
            else:
                fa = torch.zeros(target_n, D, device=features.device)
                xa = torch.zeros(target_n, 3, device=features.device)

            # Subsample or pad Part B
            if len(fb) >= target_n:
                step = max(1, len(fb) // target_n)
                idx = torch.arange(0, len(fb), step, device=fb.device)[:target_n]
                fb = fb[idx]
                xb = xb[idx]
            elif len(fb) > 0:
                pad = target_n - len(fb)
                fb = F.pad(fb, (0, 0, 0, pad))
                xb = F.pad(xb, (0, 0, 0, pad))
            else:
                fb = torch.zeros(target_n, D, device=features.device)
                xb = torch.zeros(target_n, 3, device=features.device)

            feat_a_list.append(fa)
            feat_b_list.append(fb)
            xyz_a_list.append(xa)
            xyz_b_list.append(xb)

        feat_a = torch.stack(feat_a_list)
        feat_b = torch.stack(feat_b_list)
        xyz_a = torch.stack(xyz_a_list)
        xyz_b = torch.stack(xyz_b_list)

        return feat_a, feat_b, xyz_a, xyz_b

    def forward(self, backbone_output: Dict[str, torch.Tensor],
                point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            backbone_output: dict from PointNetBackbone (or PointNet2Backbone)
            point_cloud: (B, N, 7) original [x, y, z, nx, ny, nz, part_id]

        Returns:
            relation_features: (B, 512)
        """
        global_feat = backbone_output["global_feat"]  # (B, 1024)

        # Extract features and coordinates depending on backbone type
        if "point_features" in backbone_output and backbone_output["point_features"] is not None:
            pt_features = backbone_output["point_features"]  # (B, N, backbone_feat_dim)
            xyz = point_cloud[:, :, :3]                     # (B, N, 3)
            part_ids = point_cloud[:, :, 6]                  # (B, N)
        elif "sa2_features" in backbone_output:
            pt_features = backbone_output["sa2_features"]    # (B, 128, 640)
            xyz = backbone_output["sa2_xyz"]                 # (B, 128, 3)
            original_xyz = point_cloud[:, :, :3]
            original_pids = point_cloud[:, :, 6]
            dists = square_distance(xyz, original_xyz)
            _, nn_idx = dists.min(dim=-1)
            B = xyz.shape[0]
            batch_idx = torch.arange(B, device=xyz.device).unsqueeze(-1).expand_as(nn_idx)
            part_ids = original_pids[batch_idx, nn_idx]
        else:
            raise KeyError("Backbone output must contain either 'point_features' or 'sa2_features'")

        # Project features to attention hidden dimension
        proj_features = self.feat_proj(pt_features)

        # Separate Part A and Part B points
        feat_a, feat_b, xyz_a, xyz_b = self._separate_and_sample_parts(
            xyz, proj_features, part_ids, target_n=self.max_attn_points
        )

        # Cross-part bidirectional attention
        attended_a = self.cross_attn_a_to_b(feat_a, feat_b)  # (B, target_n, hidden_dim)
        attended_b = self.cross_attn_b_to_a(feat_b, feat_a)  # (B, target_n, hidden_dim)

        # Global symmetric pooling per part (Permutation Invariance)
        global_a = attended_a.max(dim=1)[0]  # (B, hidden_dim)
        global_b = attended_b.max(dim=1)[0]  # (B, hidden_dim)

        # Interface analysis using original point clouds
        B = point_cloud.shape[0]
        orig_xyz = point_cloud[:, :, :3]
        orig_normals = point_cloud[:, :, 3:6]
        orig_pids = point_cloud[:, :, 6]

        xyz_a_orig_list, xyz_b_orig_list = [], []
        nrm_a_orig_list, nrm_b_orig_list = [], []
        target_half = min(512, point_cloud.shape[1] // 2)

        for b in range(B):
            mask_a = (orig_pids[b] == 0)
            xa, xb = orig_xyz[b][mask_a], orig_xyz[b][~mask_a]
            na, nb = orig_normals[b][mask_a], orig_normals[b][~mask_a]

            for arr_list, arr in [(xyz_a_orig_list, xa), (xyz_b_orig_list, xb),
                                  (nrm_a_orig_list, na), (nrm_b_orig_list, nb)]:
                if len(arr) >= target_half:
                    step = max(1, len(arr) // target_half)
                    idx = torch.arange(0, len(arr), step, device=arr.device)[:target_half]
                    arr_list.append(arr[idx])
                elif len(arr) > 0:
                    pad = target_half - len(arr)
                    arr_list.append(F.pad(arr, (0, 0, 0, pad)))
                else:
                    arr_list.append(torch.zeros(target_half, 3, device=orig_xyz.device))

        xyz_a_orig = torch.stack(xyz_a_orig_list)
        xyz_b_orig = torch.stack(xyz_b_orig_list)
        nrm_a_orig = torch.stack(nrm_a_orig_list)
        nrm_b_orig = torch.stack(nrm_b_orig_list)

        interface_info = self.interface_extractor(xyz_a_orig, xyz_b_orig, nrm_a_orig, nrm_b_orig)

        interface_feats = torch.stack([
            interface_info["min_dist"],
            interface_info["mean_interface_dist"],
            interface_info["std_interface_dist"],
            interface_info["normal_alignment"],
            interface_info["interface_ratio_a"],
            interface_info["interface_ratio_b"],
        ], dim=-1)  # (B, 6)

        # Fuse attended part representations, global shape signature, and interface metrics
        fused = torch.cat([global_a, global_b, global_feat, interface_feats], dim=-1)
        relation_features = self.fusion(fused)  # (B, 512)

        return relation_features
