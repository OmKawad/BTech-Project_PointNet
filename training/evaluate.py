"""
Evaluation Module for Assembly Fit Risk Prediction
====================================================
Quantitative evaluation on the test set:
  - Per-task RMSE (Interference, Clearance, Alignment, Failure)
  - ROC-AUC for each regression target (binarized)
  - Classification accuracy, F1-score (macro), confusion matrix
  - Detailed per-class precision/recall/F1
  - Visualisation of loss curves and confusion matrix
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.risk_predictor import AssemblyFitRiskModel, RISK_LABELS


@torch.no_grad()
def run_inference(model: nn.Module, loader, device: torch.device
                  ) -> Dict[str, np.ndarray]:
    """Run inference on an entire DataLoader and collect predictions + targets."""
    model.eval()
    all_preds = {
        "interference_risk": [], "clearance_risk": [],
        "alignment_risk": [], "failure_prob": [],
        "risk_logits": [], "risk_level": [],
    }
    all_targets = {
        "interference_risk": [], "clearance_risk": [],
        "alignment_risk": [], "failure_prob": [],
        "risk_level": [],
    }

    for point_clouds, labels in loader:
        point_clouds = point_clouds.to(device)
        preds = model(point_clouds)

        for key in ["interference_risk", "clearance_risk",
                     "alignment_risk", "failure_prob"]:
            all_preds[key].append(preds[key].cpu().numpy())
            all_targets[key].append(labels[key].numpy())

        all_preds["risk_logits"].append(preds["risk_logits"].cpu().numpy())
        all_preds["risk_level"].append(preds["risk_level"].cpu().numpy())
        all_targets["risk_level"].append(labels["risk_level"].numpy())

    # Concatenate all batches
    result_preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    result_targets = {k: np.concatenate(v) for k, v in all_targets.items()}

    return result_preds, result_targets


def compute_metrics(preds: Dict[str, np.ndarray],
                    targets: Dict[str, np.ndarray]) -> Dict:
    """Compute all evaluation metrics."""
    metrics = {}

    # ── Regression metrics (RMSE) ──
    for key in ["interference_risk", "clearance_risk",
                "alignment_risk", "failure_prob"]:
        rmse = np.sqrt(mean_squared_error(targets[key], preds[key]))
        metrics[f"{key}_rmse"] = rmse

    # ── Classification metrics ──
    y_true = targets["risk_level"]
    y_pred = preds["risk_level"]

    metrics["classification_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=["Low", "Medium", "High"],
        output_dict=True, zero_division=0,
    )
    metrics["per_class_report"] = report

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics["confusion_matrix"] = cm.tolist()

    # ── ROC-AUC for regression tasks (binarized at 0.5) ──
    for key in ["interference_risk", "clearance_risk",
                "alignment_risk", "failure_prob"]:
        binary_true = (targets[key] > 0.5).astype(int)
        if len(np.unique(binary_true)) > 1:
            auc = roc_auc_score(binary_true, preds[key])
        else:
            auc = float("nan")
        metrics[f"{key}_auc"] = auc

    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"],
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Risk Level", fontsize=12)
    ax.set_ylabel("True Risk Level", fontsize=12)
    ax.set_title("Assembly Fit Risk – Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def plot_training_history(history_path: str, save_dir: str):
    """Plot training and validation loss curves."""
    with open(history_path, "r") as f:
        history = json.load(f)

    train_losses = [e["total_loss"] for e in history["train"]]
    val_losses = [e["total_loss"] for e in history["val"]]
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    axes[0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-task losses (validation)
    task_keys = ["interference_loss", "clearance_loss",
                 "alignment_loss", "failure_loss", "classification_loss"]
    for key in task_keys:
        if key in history["val"][0]:
            vals = [e[key] for e in history["val"]]
            label = key.replace("_loss", "").capitalize()
            axes[1].plot(epochs, vals, label=label, linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Per-Task Validation Losses")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves: {save_path}")


def evaluate(checkpoint_path: str = "checkpoints/best_model.pth",
             data_dir: str = "data/samples",
             output_dir: str = "evaluation_results"):
    """Run full evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"[Evaluate] Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_cfg = config["model"]

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
        dropout=0.0,  # No dropout during evaluation
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded {model.backbone_type.upper()} model from epoch {checkpoint['epoch']}")

    # ── Load test data ──
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_points=config["data"]["num_points"],
    )

    # ── Run inference ──
    print("[Evaluate] Running inference on test set...")
    preds, targets = run_inference(model, loaders["test"], device)
    print(f"  Test samples: {len(targets['risk_level'])}")

    # ── Compute metrics ──
    metrics = compute_metrics(preds, targets)

    # ── Print results ──
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n  Classification Accuracy: {metrics['classification_accuracy']:.2%}")
    print(f"  F1-Score (Macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted):     {metrics['f1_weighted']:.4f}")
    print(f"\n  Regression RMSE:")
    for key in ["interference_risk", "clearance_risk",
                "alignment_risk", "failure_prob"]:
        print(f"    {key:25s}: RMSE={metrics[f'{key}_rmse']:.4f}  "
              f"AUC={metrics.get(f'{key}_auc', 'N/A')}")
    print(f"\n  Per-Class Report:")
    for cls_name in ["Low", "Medium", "High"]:
        r = metrics["per_class_report"].get(cls_name, {})
        print(f"    {cls_name:8s}: Precision={r.get('precision', 0):.3f}  "
              f"Recall={r.get('recall', 0):.3f}  "
              f"F1={r.get('f1-score', 0):.3f}")

    # ── Save metrics ──
    save_metrics = {k: v for k, v in metrics.items()
                    if k != "per_class_report"}
    save_metrics["per_class_report"] = metrics["per_class_report"]
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"\n  Saved metrics: {metrics_path}")

    # ── Plot confusion matrix ──
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.png"))

    # ── Plot training history if available ──
    history_path = os.path.join(os.path.dirname(checkpoint_path),
                                "training_history.json")
    if os.path.exists(history_path):
        plot_training_history(history_path, output_dir)

    print(f"\n[Evaluate] Evaluation complete! Results in {output_dir}/")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Assembly Fit Risk Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/samples")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir)
