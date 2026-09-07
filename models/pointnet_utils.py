"""
PointNet Core Utilities & Mathematical Building Blocks
======================================================
Canonical PointNet components (Qi et al., CVPR 2017):
  1. TNet3d: Spatial Transformer Network (STN) for 3D coordinate alignment
  2. TNetKd: Feature Transformer Network for intermediate feature alignment (e.g. 64x64)
  3. feature_transform_regularizer: Orthogonality constraint ||I - A A^T||_F^2
  4. PointNetEncoder: Shared MLPs, symmetric max pooling, and local+global fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TNet3d(nn.Module):
    """
    Spatial Transformer Network (T-Net) for 3D coordinate alignment.
    Predicts a 3x3 affine transformation matrix from raw point coordinates.

    Architecture:
      Input (B, 3, N)
        → Shared MLP (64, 128, 1024)
        → Symmetric Max Pooling → (B, 1024)
        → FC (512, 256)
        → FC (9) + Identity matrix initialization → (B, 3, 3)
    """

    def __init__(self, in_channel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize fc3 weights and bias to output the Identity Matrix
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) coordinate tensor

        Returns:
            trans: (B, 3, 3) coordinate transformation matrix
        """
        batch_size = x.size(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        # Symmetric max pooling across all points (Permutation Invariance)
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(batch_size, -1)

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)

        # Add 3x3 identity matrix
        iden = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        trans = out.view(batch_size, 3, 3) + iden
        return trans


class TNetKd(nn.Module):
    """
    Feature Transformer Network (T-Net) for K-dimensional feature alignment.
    Typically used for K=64 intermediate feature space.

    Architecture:
      Input (B, K, N)
        → Shared MLP (64, 128, 1024)
        → Symmetric Max Pooling → (B, 1024)
        → FC (512, 256)
        → FC (K*K) + Identity matrix initialization → (B, K, K)
    """

    def __init__(self, k: int = 64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, N) feature tensor

        Returns:
            trans: (B, K, K) feature transformation matrix
        """
        batch_size = x.size(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        # Symmetric max pooling across all points (Permutation Invariance)
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(batch_size, -1)

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)

        # Add KxK identity matrix
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        trans = out.view(batch_size, self.k, self.k) + iden
        return trans


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal Regularization Loss for PointNet Feature Transform matrix A:
      L_reg = ||I - A A^T||_F^2 / (K^2)

    Constrains the high-dimensional transformation matrix to be nearly orthogonal
    (rigid rotation/reflection), preventing arbitrary stretching or collapse of
    feature dimensions.

    Args:
        trans: (B, K, K) transformation matrix

    Returns:
        loss: scalar tensor
    """
    d = trans.size(1)
    batch_size = trans.size(0)
    iden = torch.eye(d, dtype=trans.dtype, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
    diff = torch.bmm(trans, trans.transpose(2, 1)) - iden
    loss = torch.mean(torch.norm(diff, dim=(1, 2)))
    return loss


class PointNetEncoder(nn.Module):
    """
    Canonical PointNet Feature Encoder.

    Steps:
      1. (Optional) Input Coordinate Transform via T-Net 3x3
      2. Initial Shared MLP: in_channel → 64 → 64
      3. (Optional) Feature Transform via T-Net 64x64
      4. Deep Shared MLP: 64 → 128 → 1024
      5. Symmetric Max Pooling → Global feature (1024)
      6. (Optional) Local-Global Feature Concatenation: [local_feat (64) | global_feat (1024)]
    """

    def __init__(self, in_channel: int = 3, global_feat_dim: int = 1024,
                 use_tnet_input: bool = True, use_tnet_feature: bool = True):
        super().__init__()
        self.use_tnet_input = use_tnet_input
        self.use_tnet_feature = use_tnet_feature
        self.global_feat_dim = global_feat_dim

        if use_tnet_input:
            self.stn = TNet3d(in_channel=3)
        else:
            self.stn = None

        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        if use_tnet_feature:
            self.fstn = TNetKd(k=64)
        else:
            self.fstn = None

        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, global_feat_dim, 1)
        self.bn4 = nn.BatchNorm1d(global_feat_dim)

    def forward(self, x: torch.Tensor, xyz: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, in_channel, N) input features (e.g. coordinates + normals + embeddings)
            xyz: (B, 3, N) optional raw coordinates for coordinate T-Net alignment

        Returns:
            global_feat: (B, global_feat_dim)
            point_feat_64: (B, 64, N) local point features
            combined_feat: (B, 64 + global_feat_dim, N) concatenated local+global features
            trans_input: (B, 3, 3) or None
            trans_feat: (B, 64, 64) or None
        """
        B, D, N = x.shape

        # Step 1: Input Transform on 3D coordinates
        trans_input = None
        if self.use_tnet_input:
            coords = xyz if xyz is not None else x[:, :3, :]
            trans_input = self.stn(coords)  # (B, 3, 3)
            # Apply affine transformation to coordinates
            coords_trans = torch.bmm(trans_input, coords)  # (B, 3, N)
            if xyz is not None:
                # Replace coordinate portion of features
                x = torch.cat([coords_trans, x[:, 3:, :]], dim=1)
            else:
                x = torch.cat([coords_trans, x[:, 3:, :]], dim=1) if D > 3 else coords_trans

        # Step 2: Shared MLP (in_channel → 64 → 64)
        x = F.relu(self.bn1(self.conv1(x)))
        point_feat_64 = F.relu(self.bn2(self.conv2(x)))  # (B, 64, N)

        # Step 3: Feature Transform on 64D space
        trans_feat = None
        if self.use_tnet_feature:
            trans_feat = self.fstn(point_feat_64)  # (B, 64, 64)
            point_feat_aligned = torch.bmm(trans_feat, point_feat_64)  # (B, 64, N)
        else:
            point_feat_aligned = point_feat_64

        # Step 4: Shared MLP (64 → 128 → 1024)
        x = F.relu(self.bn3(self.conv3(point_feat_aligned)))
        x = self.bn4(self.conv4(x))  # (B, global_feat_dim, N)

        # Step 5: Symmetric Function - Global Max Pooling (Permutation Invariance)
        global_feat = torch.max(x, 2, keepdim=False)[0]  # (B, global_feat_dim)

        # Step 6: Local + Global Feature Concatenation
        # Expand global feature across all N points
        global_feat_expanded = global_feat.unsqueeze(-1).repeat(1, 1, N)  # (B, global_feat_dim, N)
        combined_feat = torch.cat([point_feat_aligned, global_feat_expanded], dim=1)  # (B, 64 + 1024, N)

        return global_feat, point_feat_aligned, combined_feat, trans_input, trans_feat
