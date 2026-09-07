"""
Training Pipeline for Assembly Fit Risk Prediction
====================================================
End-to-end training loop with:
  - Multi-task loss (MSE + BCE + CrossEntropy)
  - Learning rate scheduling (cosine, step, plateau)
  - Warmup epochs
  - Gradient clipping
  - Early stopping
  - TensorBoard-style logging
  - Best model checkpointing
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_generator import generate_dataset
from data.dataset import create_dataloaders
from models.risk_predictor import AssemblyFitRiskModel, MultiTaskLoss


def load_config(config_path: str = "training/config.yaml") -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create AdamW optimizer with weight decay."""
    train_cfg = config["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict):
    """Create learning rate scheduler."""
    train_cfg = config["training"]
    sched_type = train_cfg.get("lr_scheduler", "cosine")

    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-6)
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("lr_step_size", 20),
            gamma=train_cfg.get("lr_gamma", 0.5))
    elif sched_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")

    return scheduler, sched_type


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: MultiTaskLoss, optimizer: optim.Optimizer,
                    device: torch.device, epoch: int,
                    grad_clip: float = 1.0) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {}
    num_batches = 0

    for batch_idx, (point_clouds, labels) in enumerate(loader):
        point_clouds = point_clouds.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        predictions = model(point_clouds)
        loss, loss_dict = criterion(predictions, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                  f"Loss: {loss_dict['total_loss']:.4f} "
                  f"(I:{loss_dict['interference_loss']:.3f} "
                  f"C:{loss_dict['clearance_loss']:.3f} "
                  f"A:{loss_dict['alignment_loss']:.3f} "
                  f"F:{loss_dict['failure_loss']:.3f} "
                  f"CLS:{loss_dict['classification_loss']:.3f})")

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: MultiTaskLoss, device: torch.device
             ) -> Dict[str, float]:
    """Run validation."""
    model.eval()
    total_losses = {}
    num_batches = 0
    correct = 0
    total = 0

    for point_clouds, labels in loader:
        point_clouds = point_clouds.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        predictions = model(point_clouds)
        _, loss_dict = criterion(predictions, labels)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1

        # Classification accuracy
        correct += (predictions["risk_level"] == labels["risk_level"]).sum().item()
        total += labels["risk_level"].shape[0]

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_losses["classification_accuracy"] = correct / max(total, 1)
    return avg_losses


def train(config_path: str = "training/config.yaml"):
    """Main training function."""
    config = load_config(config_path)
    train_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Step 1: Generate dataset if needed ──
    data_dir = data_cfg["data_dir"]
    labels_path = os.path.join(data_dir, "labels.json")
    total_samples = (data_cfg["num_assemblies_train"] +
                     data_cfg["num_assemblies_val"] +
                     data_cfg["num_assemblies_test"])

    if not os.path.exists(labels_path):
        print(f"[Train] Generating {total_samples} synthetic assemblies...")
        generate_dataset(
            num_samples=total_samples,
            n_points=data_cfg["num_points"],
            assembly_types=data_cfg.get("assembly_types"),
            save_dir=data_dir,
            noise_std=data_cfg.get("noise_std", 0.002),
            normal_noise_std=data_cfg.get("normal_noise_std", 0.01),
            seed=42,
        )
    else:
        print(f"[Train] Dataset already exists at {data_dir}")

    # ── Step 2: Create DataLoaders ──
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=train_cfg["batch_size"],
        num_points=data_cfg["num_points"],
        num_workers=0,
    )

    # ── Step 3: Create model ──
    pn_cfg = model_cfg.get("pointnet", {})
    model = AssemblyFitRiskModel(
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
        dropout=model_cfg["risk_head"]["dropout"],
    ).to(device)

    param_counts = model.get_num_parameters()
    print(f"\n[Train] Model parameters ({model.backbone_type.upper()} Backbone):")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # ── Step 4: Loss, optimizer, scheduler ──
    criterion = MultiTaskLoss(weights=train_cfg.get("loss_weights"))
    optimizer = create_optimizer(model, config)
    scheduler, sched_type = create_scheduler(optimizer, config)

    # ── Step 5: Training loop ──
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 15)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    history = {"train": [], "val": []}

    print(f"\n[Train] Starting training for {train_cfg['epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Scheduler: {sched_type}")
    print(f"  Early stopping patience: {patience}")
    print()

    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.time()

        # Warmup: linearly increase LR
        if epoch <= warmup_epochs:
            warmup_lr = train_cfg["learning_rate"] * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Train
        train_losses = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch,
            grad_clip=train_cfg.get("gradient_clip", 1.0))

        # Validate
        val_losses = validate(model, loaders["val"], criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{train_cfg['epochs']} "
              f"({epoch_time:.1f}s) lr={current_lr:.6f}")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"  Val   Loss: {val_losses['total_loss']:.4f}  "
              f"Accuracy: {val_losses['classification_accuracy']:.2%}")
        print(f"{'='*70}\n")

        history["train"].append(train_losses)
        history["val"].append(val_losses)

        # LR scheduling
        if epoch > warmup_epochs:
            if sched_type == "plateau":
                scheduler.step(val_losses["total_loss"])
            else:
                scheduler.step()

        # Checkpointing
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            patience_counter = 0
            if train_cfg.get("save_best", True):
                save_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                }, save_path)
                print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Train] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses["total_loss"],
        "config": config,
    }, final_path)

    # Save training history
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final model: {final_path}")
    print(f"  History: {history_path}")

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Assembly Fit Risk Model")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    args = parser.parse_args()
    train(args.config)
