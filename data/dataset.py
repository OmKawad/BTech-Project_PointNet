"""
PyTorch Dataset and DataLoader for Assembly Fit Risk Prediction
================================================================
Loads synthetic assembly point clouds (N, 7) with labels,
handles train/val/test splits, augmentation, and batching.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


class AssemblyFitDataset(Dataset):
    """
    PyTorch Dataset for assembly fit risk prediction.

    Each sample returns:
        point_cloud : torch.Tensor of shape (N, 7) – [x, y, z, nx, ny, nz, part_id]
        labels      : dict of torch.Tensors:
            - interference_risk : scalar ∈ [0, 1]
            - clearance_risk    : scalar ∈ [0, 1]
            - alignment_risk    : scalar ∈ [0, 1]
            - failure_prob      : scalar ∈ [0, 1]
            - risk_level        : int ∈ {0, 1, 2}
    """

    def __init__(self, data_dir: str, split: str = "train",
                 num_points: int = 2048, augment: bool = True):
        """
        Args:
            data_dir:   Path to the directory containing .npy files and labels.json
            split:      One of 'train', 'val', 'test'
            num_points: Number of points to sample per cloud
            augment:    Whether to apply data augmentation (train only)
        """
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment and (split == "train")

        # Load labels metadata
        labels_path = os.path.join(data_dir, "labels.json")
        with open(labels_path, "r") as f:
            self.all_labels = json.load(f)

        # Determine split indices
        total = len(self.all_labels)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)

        if split == "train":
            self.indices = list(range(0, train_end))
        elif split == "val":
            self.indices = list(range(train_end, val_end))
        elif split == "test":
            self.indices = list(range(val_end, total))
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        print(f"[AssemblyFitDataset] Loaded {len(self.indices)} samples for '{split}' split")

    def __len__(self) -> int:
        return len(self.indices)

    def _augment_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation:
        - Random rotation around Z axis
        - Random jitter to coordinates
        - Random point dropout (5-10% of points replaced by duplicates)
        """
        pc = point_cloud.copy()
        xyz = pc[:, :3]
        normals = pc[:, 3:6]
        part_ids = pc[:, 6:7]

        # Random rotation around Z axis
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_z = np.array([[cos_a, -sin_a, 0],
                          [sin_a, cos_a, 0],
                          [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ rot_z.T
        normals = normals @ rot_z.T
        # Re-normalize normals
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.clip(norms, 1e-8, None)

        # Random jitter
        xyz += np.random.normal(0, 0.005, xyz.shape).astype(np.float32)

        # Random point dropout (replace 5-10% of points with copies of others)
        n = len(xyz)
        drop_ratio = np.random.uniform(0.0, 0.1)
        n_drop = int(n * drop_ratio)
        if n_drop > 0:
            drop_idx = np.random.choice(n, n_drop, replace=False)
            keep_idx = np.random.choice(n, n_drop, replace=True)
            xyz[drop_idx] = xyz[keep_idx]
            normals[drop_idx] = normals[keep_idx]
            part_ids[drop_idx] = part_ids[keep_idx]

        pc = np.concatenate([xyz, normals, part_ids], axis=-1)
        return pc

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        actual_idx = self.indices[idx]
        label_entry = self.all_labels[actual_idx]

        # Load point cloud
        pc_path = os.path.join(self.data_dir, f"assembly_{actual_idx:05d}.npy")
        point_cloud = np.load(pc_path).astype(np.float32)

        # Resample to exact num_points
        n = len(point_cloud)
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)
        point_cloud = point_cloud[choice]

        # Center the point cloud (translate centroid to origin)
        centroid = point_cloud[:, :3].mean(axis=0)
        point_cloud[:, :3] -= centroid

        # Normalize to unit sphere
        max_dist = np.max(np.linalg.norm(point_cloud[:, :3], axis=-1))
        if max_dist > 1e-8:
            point_cloud[:, :3] /= max_dist

        # Apply augmentation
        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)

        # Convert to tensors
        point_cloud_tensor = torch.from_numpy(point_cloud)  # (N, 7)

        labels = {
            "interference_risk": torch.tensor(label_entry["interference_risk"],
                                               dtype=torch.float32),
            "clearance_risk": torch.tensor(label_entry["clearance_risk"],
                                            dtype=torch.float32),
            "alignment_risk": torch.tensor(label_entry["alignment_risk"],
                                            dtype=torch.float32),
            "failure_prob": torch.tensor(label_entry["failure_prob"],
                                          dtype=torch.float32),
            "risk_level": torch.tensor(label_entry["risk_level"],
                                        dtype=torch.long),
        }

        return point_cloud_tensor, labels


def create_dataloaders(data_dir: str, batch_size: int = 16,
                       num_points: int = 2048, num_workers: int = 0
                       ) -> Dict[str, DataLoader]:
    """
    Create train, val, and test DataLoaders.

    Returns dict with keys: 'train', 'val', 'test'
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        augment = (split == "train")
        ds = AssemblyFitDataset(data_dir, split=split,
                                num_points=num_points, augment=augment)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


if __name__ == "__main__":
    # Quick smoke test
    loaders = create_dataloaders("data/samples", batch_size=4, num_points=2048)
    for split, loader in loaders.items():
        batch = next(iter(loader))
        pc, labels = batch
        print(f"[{split}] point_cloud: {pc.shape}, "
              f"risk_level: {labels['risk_level'].tolist()}")
