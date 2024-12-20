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


class StructureAwareLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred, target):
        # Dynamic weighting based on graph structure
        degree = target.sum(dim=-1, keepdim=True)
        importance_weight = (degree / degree.mean()).clamp(0.5, 2.0)

        # More aggressive class balancing
        neg_pos_ratio = (target == 0).float().sum() / (target == 1).float().sum()
        pos_weight = torch.tensor(neg_pos_ratio, device=pred.device).clamp(1.0, 20.0)

        # BCE loss with class weights and label smoothing
        smooth_target = torch.where(
            target > 0.5,
            target * 0.9 + 0.05,  # Positive smoothing
            target * 0.1,  # Negative smoothing
        )

        bce = F.binary_cross_entropy_with_logits(
            pred,
            smooth_target,
            pos_weight=pos_weight * torch.ones_like(target).to(pred.device),
            reduction="none",
        )

        # Stronger weighting for positive examples
        weights = torch.where(
            target > 0.5,
            importance_weight * 3.0,  # Increased positive weight
            importance_weight,
        )

        # Focal term with dynamic gamma
        pred_probs = torch.sigmoid(pred)
        pt = pred_probs * target + (1 - pred_probs) * (1 - target)
        focal_term = (1 - pt) ** (
            self.gamma * (1 + target)
        )  # Increase gamma for positives

        loss = weights * focal_term * bce

        # Add symmetry and sparsity constraints
        sym_loss = torch.abs(pred_probs - pred_probs.transpose(-2, -1)).mean()
        sparse_loss = torch.abs(pred_probs).mean()

        return loss.mean() + 0.1 * sym_loss + 0.01 * sparse_loss


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    scaler = torch.cuda.amp.GradScaler()

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

            # Use AMP
            with torch.cuda.amp.autocast():
                output = model(features, edge_index, edge_weight)
                loss = criterion(output, targets)
                if hasattr(model, "regularization_loss"):
                    loss += 0.1 * model.regularization_loss  # Reduced weight

            # Scale gradients
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            scaler.step(optimizer)
            scaler.update()

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


def calculate_metrics(predictions, targets, threshold=0.4):
    """Calculate various metrics for evaluation."""
    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(torch.tensor(predictions)).numpy()

    # Use lower threshold for better recall
    pred_binary = (pred_probs > threshold).astype(float)

    # Calculate metrics
    f1 = f1_score(targets.flatten(), pred_binary.flatten())
    auc = roc_auc_score(targets.flatten(), pred_probs.flatten())
    avg_precision = average_precision_score(targets.flatten(), pred_probs.flatten())

    # Calculate precision and recall at different thresholds
    precisions, recalls, _ = precision_recall_curve(
        targets.flatten(), pred_probs.flatten()
    )

    return {
        "f1_score": f1,
        "auc_score": auc,
        "avg_precision": avg_precision,
        "max_precision": np.max(precisions),
        "max_recall": np.max(recalls),
    }


def train_model(train_loader, val_loader, model_config, training_config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cudnn benchmarking for faster training
    torch.backends.cudnn.benchmark = True

    # Initialize model
    model = LargeGMAN(**model_config).to(device)

    # Use mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    # Reduced gradient accumulation for more frequent updates
    grad_accum_steps = 1

    criterion = StructureAwareLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # Use OneCycleLR instead
    total_steps = len(train_loader) * training_config["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.2,  # 20% of training for warmup
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy="cos",
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
