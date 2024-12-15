# src/model/medium/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from .model_m import MediumASTGCN


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

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

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


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

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    pred_binary = (all_preds > 0.5).astype(float)
    f1 = f1_score(all_targets.flatten(), pred_binary.flatten())

    # Handle potential errors in AUC calculation (e.g., single class)
    try:
        auc = roc_auc_score(all_targets.flatten(), all_preds.flatten())
    except ValueError:
        auc = 0.0

    metrics = {
        "val_loss": total_loss / len(val_loader),
        "f1_score": f1,
        "auc_score": auc,
    }

    return metrics


def train_model(train_loader, val_loader, model_config, training_config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = MediumASTGCN(**model_config).to(device)

    # Setup training
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 1e-4),
    )

    # Cosine annealing scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=training_config.get("scheduler_t0", 5),
        T_mult=training_config.get("scheduler_t_mult", 2),
    )

    best_val_metrics = {"val_loss": float("inf"), "f1_score": 0, "auc_score": 0}
    early_stopping_counter = 0

    # Training loop
    for epoch in range(training_config["epochs"]):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}/{training_config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"AUC Score: {val_metrics['auc_score']:.4f}")

        # Early stopping based on validation loss
        if val_metrics["val_loss"] < best_val_metrics["val_loss"]:
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
