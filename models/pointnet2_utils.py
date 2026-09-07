"""
PointNet++ Building Blocks
===========================
Implements the fundamental operations for PointNet++:
  - Farthest Point Sampling (FPS)
  - Ball Query
  - K-Nearest Neighbors (KNN)
  - Set Abstraction (SA) Layer – Single-Scale Grouping (SSG) & Multi-Scale Grouping (MSG)
  - Feature Propagation (FP) Layer

All operations work on batched point clouds: (B, N, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────
# Core Point Cloud Operations
# ──────────────────────────────────────────────

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance between two point sets.

    Args:
        src: (B, N, C)
        dst: (B, M, C)

    Returns:
        dist: (B, N, M) pairwise squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)     # (B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).permute(0, 2, 1)  # (B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).

    Iteratively selects the point that is farthest from the already-selected set,
    producing a representative and well-distributed subset.

    Args:
        xyz: (B, N, 3) point cloud coordinates
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index into points tensor using the given indices.

    Args:
        points: (B, N, C)
        idx:    (B, S) or (B, S, K)

    Returns:
        indexed points with same batch dims and C features
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device) \
        .view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def ball_query(radius: float, nsample: int,
               xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball Query: find all points within a radius around each query point.

    Args:
        radius:  search radius
        nsample: maximum number of neighbors
        xyz:     (B, N, 3) all points
        new_xyz: (B, S, 3) query points (centroids)

    Returns:
        group_idx: (B, S, nsample) indices of neighbors
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Pairwise distances
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    # Mask points outside the ball
    group_idx = torch.arange(N, dtype=torch.long, device=device) \
        .view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N  # mark as invalid

    # Sort and take first nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Fill invalid entries with first valid index
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def knn_query(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    K-Nearest Neighbors query.

    Args:
        k:       number of neighbors
        xyz:     (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        group_idx: (B, S, k) indices of nearest neighbors
    """
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False)
    return group_idx


# ──────────────────────────────────────────────
# Shared MLP (1x1 Convolution over grouped points)
# ──────────────────────────────────────────────

class SharedMLP(nn.Module):
    """Stack of 1D convolutions applied per-point (shared weights)."""

    def __init__(self, channels: List[int], bn: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N) → (B, C_out, N)"""
        return self.mlp(x)


# ──────────────────────────────────────────────
# Set Abstraction Layer
# ──────────────────────────────────────────────

class SetAbstractionMSG(nn.Module):
    """
    PointNet++ Set Abstraction with Multi-Scale Grouping (MSG).

    For each centroid, groups points at multiple radii and concatenates
    the per-scale features.
    """

    def __init__(self, npoint: int, radii: List[float],
                 nsamples: List[int], in_channel: int,
                 mlp_channels_list: List[List[int]]):
        """
        Args:
            npoint:            Number of centroids to sample via FPS
            radii:             List of ball query radii (one per scale)
            nsamples:          List of max neighbors per scale
            in_channel:        Input feature dimension (excluding xyz)
            mlp_channels_list: List of MLP channel lists (one per scale)
        """
        super().__init__()
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i, mlp_channels in enumerate(mlp_channels_list):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # xyz coordinates are always included
            for out_channel in mlp_channels:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz: torch.Tensor,
                points: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz:    (B, N, 3) coordinates
            points: (B, N, D) features (or None)

        Returns:
            new_xyz:    (B, npoint, 3) sampled centroids
            new_points: (B, npoint, Σ(mlp[-1])) aggregated multi-scale features
        """
        B, N, _ = xyz.shape

        # Farthest Point Sampling
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        new_points_list = []
        for i, (radius, nsample) in enumerate(zip(self.radii, self.nsamples)):
            # Ball query
            group_idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
            grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, nsample, 3)
            # Center the group around the centroid
            grouped_xyz -= new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, group_idx)  # (B, npoint, nsample, D)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

            # (B, npoint, nsample, C) → (B, C, npoint, nsample) for Conv2d
            grouped_points = grouped_points.permute(0, 3, 1, 2)

            for j, (conv, bn) in enumerate(zip(self.conv_blocks[i], self.bn_blocks[i])):
                grouped_points = F.relu(bn(conv(grouped_points)))

            # Max pool over neighbors: (B, C_out, npoint, nsample) → (B, C_out, npoint)
            new_points = torch.max(grouped_points, dim=-1)[0]  # (B, C_out, npoint)
            new_points_list.append(new_points)

        # Concatenate multi-scale features
        new_points_concat = torch.cat(new_points_list, dim=1)  # (B, Σ, npoint)
        new_points_concat = new_points_concat.permute(0, 2, 1)  # (B, npoint, Σ)

        return new_xyz, new_points_concat


class SetAbstractionSSG(nn.Module):
    """
    PointNet++ Set Abstraction with Single-Scale Grouping (SSG).
    """

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp_channels: List[int]):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        last_channel = in_channel + 3
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor,
                points: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape

        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, group_idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


class GlobalSetAbstraction(nn.Module):
    """
    Global Set Abstraction: aggregate ALL points into a single global feature vector.
    Used as the final SA layer.
    """

    def __init__(self, in_channel: int, mlp_channels: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz:    (B, N, 3) – not used for grouping but kept for interface
            points: (B, N, D) features

        Returns:
            global_feature: (B, C_out) – single vector per sample
        """
        points = points.permute(0, 2, 1)  # (B, D, N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            points = F.relu(bn(conv(points)))
        global_feature = torch.max(points, dim=-1)[0]  # (B, C_out)
        return global_feature


# ──────────────────────────────────────────────
# Feature Propagation Layer (for optional segmentation / per-point tasks)
# ──────────────────────────────────────────────

class FeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation via distance-weighted interpolation.
    Used for upsampling features back to a denser point set.
    """

    def __init__(self, in_channel: int, mlp_channels: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor
                ) -> torch.Tensor:
        """
        Interpolate features from sparser xyz2 to denser xyz1.

        Args:
            xyz1:    (B, N, 3)  denser points
            xyz2:    (B, S, 3)  sparser points
            points1: (B, N, D1) features at xyz1 (skip connection)
            points2: (B, S, D2) features at xyz2

        Returns:
            new_points: (B, N, C_out)
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3-NN

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm  # (B, N, 3)

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, C, N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)  # (B, N, C_out)

        return new_points
