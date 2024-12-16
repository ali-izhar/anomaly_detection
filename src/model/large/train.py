import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from .model_l import LargeGMAN


class WeightedFocalLoss(nn.Module):
    """Weighted focal loss with class imbalance handling."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Calculate class weights dynamically
        pos_weight = (target == 0).float().sum() / (target == 1).float().sum()

        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction="none")

        # Focal term
        pt = torch.exp(-bce)
        focal_term = (1 - pt) ** self.gamma

        # Weight positive examples more heavily
        weights = target * pos_weight + (1 - target)

        return (weights * focal_term * bce).mean()


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            features = batch["features"].to(device)
            edge_index = batch["edge_indices"][0][0].to(
                device
            )  # Use first timestep's graph
            edge_weight = (
                batch["edge_weights"][0][0].to(device)
                if batch["edge_weights"]
                else None
            )
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            output = model(features, edge_index, edge_weight)
            loss = criterion(output, targets)

            # Add regularization loss if available
            if hasattr(model, "regularization_loss"):
                loss += model.regularization_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Collect predictions and targets for metrics
            all_preds.append(output.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(train_loader)

    return metrics


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            edge_index = batch["edge_indices"][0][0].to(device)
            edge_weight = (
                batch["edge_weights"][0][0].to(device)
                if batch["edge_weights"]
                else None
            )
            targets = batch["targets"].to(device)

            output = model(features, edge_index, edge_weight)
            loss = criterion(output, targets)

            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(val_loader)

    return metrics


def calculate_metrics(predictions, targets):
    """Calculate various metrics for evaluation."""
    pred_binary = (predictions > 0.5).astype(float)

    # Basic metrics
    f1 = f1_score(targets.flatten(), pred_binary.flatten())

    # Try to calculate AUC, handle potential errors
    try:
        auc = roc_auc_score(targets.flatten(), predictions.flatten())
    except ValueError:
        auc = 0.0

    # Calculate average precision
    ap = average_precision_score(targets.flatten(), predictions.flatten())

    # Calculate precision at different thresholds
    precision, recall, _ = precision_recall_curve(
        targets.flatten(), predictions.flatten()
    )

    return {
        "f1_score": f1,
        "auc_score": auc,
        "avg_precision": ap,
        "max_precision": np.max(precision),
        "max_recall": np.max(recall),
    }


def train_model(train_loader, val_loader, model_config, training_config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = LargeGMAN(**model_config).to(device)

    # Setup training
    criterion = WeightedFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 1e-4),
    )

    # One Cycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_config["learning_rate"],
        epochs=training_config["epochs"],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Warm-up for 30% of training
        div_factor=25.0,  # Initial lr = max_lr/25
        final_div_factor=1e4,  # Final lr = max_lr/10000
    )

    best_val_metrics = {"loss": float("inf"), "f1_score": 0, "auc_score": 0}
    early_stopping_counter = 0

    # Training loop
    for epoch in range(training_config["epochs"]):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_metrics = validate(model, val_loader, criterion, device)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{training_config['epochs']}")
        print("Training Metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nValidation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

        # Early stopping based on validation loss
        if val_metrics["loss"] < best_val_metrics["loss"]:
            best_val_metrics = val_metrics
            early_stopping_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_metrics": best_val_metrics,
                },
                "best_model.pth",
            )
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= training_config["patience"]:
                print("\nEarly stopping triggered!")
                break

    return model
