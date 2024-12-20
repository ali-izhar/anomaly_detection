# src/model/medium/train.py

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
    average_precision_score,
    precision_recall_curve,
)
from .model_m import MediumASTGCN


class GraphStructureLoss(nn.Module):
    def __init__(self, num_nodes=30, min_edges=2, max_edges=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.target_sparsity = 0.77  # From BA graph properties
        self.min_edges = min_edges
        self.max_edges = max_edges

    def forward(self, pred, target, epoch=0):
        # Clip predictions
        pred = torch.clamp(pred, min=-5.0, max=5.0)
        
        # Get predictions in probability space
        pred_probs = torch.sigmoid(pred)
        
        # Calculate metrics
        target_density = (target == 1).float().mean()
        pred_density = pred_probs.mean()
        
        # Dynamic positive weight with proper tensor handling
        base_weight = 5.0 * (1.0 - target_density)
        warmup_factor = min(1.0, epoch / 5.0)
        pos_weight = (base_weight * (1.0 - 0.5 * warmup_factor)).clone().detach().to(pred.device)
        
        # BCE with dynamic weighting
        bce = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=pos_weight,
            reduction='none'
        )
        
        # Density matching with target minimum
        min_density = 0.15  # Minimum target density (from BA properties)
        density_loss = F.relu(min_density - pred_density) * 50.0
        
        # Degree matching
        pred_degrees = pred_probs.sum(dim=-1)
        target_degrees = target.sum(dim=-1)
        degree_loss = F.smooth_l1_loss(pred_degrees, target_degrees) * 5.0
        
        total_loss = (
            bce.mean() +
            density_loss +
            degree_loss
        )
        
        return total_loss


def calculate_graph_metrics(predictions, targets, num_nodes=30):
    """Calculate graph-specific structural metrics for sparse graphs."""
    pred_adj = (predictions > 0.5).astype(float)
    true_adj = targets.astype(float)

    metrics = {}

    # 1. Sparsity metrics
    metrics.update(
        {
            "density_error": abs(pred_adj.mean() - true_adj.mean()),
            "sparsity_ratio": (pred_adj == 0).mean() / (true_adj == 0).mean(),
        }
    )

    # 2. Degree distribution metrics
    pred_degrees = pred_adj.sum(axis=-1)
    true_degrees = true_adj.sum(axis=-1)

    metrics.update(
        {
            "degree_corr": np.corrcoef(pred_degrees.flatten(), true_degrees.flatten())[
                0, 1
            ],
            "max_degree_ratio": pred_degrees.max() / (true_degrees.max() + 1e-6),
            "min_degree_ratio": (pred_degrees.min() + 1e-6)
            / (true_degrees.min() + 1e-6),
        }
    )

    # 3. Scale-free property metrics
    def power_law_fit(degrees):
        degrees = degrees[degrees > 0]
        if len(degrees) == 0:
            return 0
        log_degrees = np.log(degrees)
        return -np.polyfit(
            log_degrees, np.log(np.bincount(degrees.astype(int))[1:]), 1
        )[0]

    try:
        metrics["power_law_diff"] = abs(
            power_law_fit(pred_degrees.flatten())
            - power_law_fit(true_degrees.flatten())
        )
    except:
        metrics["power_law_diff"] = float("inf")

    # 4. Precision-Recall metrics for sparse data
    precision, recall, _ = precision_recall_curve(
        true_adj.flatten(), predictions.flatten()
    )
    metrics["auprc"] = average_precision_score(
        true_adj.flatten(), predictions.flatten()
    )

    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, epoch=0):
    """Train the model for one epoch with focus on sparse predictions."""
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            
            features = batch["features"].to(device)
            edge_index = batch["edge_indices"][0][0].to(device)
            edge_weight = batch["edge_weights"][0][0].to(device) if batch["edge_weights"] else None
            targets = batch["targets"].to(device)

            with torch.amp.autocast(device_type='cuda'):
                output = model(features, edge_index, edge_weight)
                loss = criterion(output, targets, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

            # Store predictions and targets for metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(output)
                all_preds.append(pred_probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # Calculate running metrics for progress bar
                pos_ratio = (pred_probs > 0.5).float().mean()
                max_prob = pred_probs.max().item()
                min_prob = pred_probs.min().item()
                sparsity = (targets == 0).float().mean()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pos_ratio": f"{pos_ratio.item():.3f}",
                "prob_range": f"[{min_prob:.2f}, {max_prob:.2f}]",
                "target_sparsity": f"{sparsity.item():.3f}"
            })

    # Calculate final metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_graph_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(train_loader)  # Average loss over batches

    return metrics


def train_model(train_loader, val_loader, model_config, training_config):
    """Main training function with sparse graph specific configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MediumASTGCN(**model_config).to(device)

    criterion = GraphStructureLoss(
        num_nodes=model_config["num_nodes"], min_edges=2, max_edges=8
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Cosine schedule with warm-up
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_config["learning_rate"],
        total_steps=len(train_loader) * training_config["epochs"],
        pct_start=0.1,  # Shorter warm-up for sparse data
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy="cos",
    )

    best_val_metrics = {
        "loss": float("inf"),
        "auprc": 0,  # Using AUPRC instead of F1 for sparse data
    }

    early_stopping_counter = 0

    for epoch in range(training_config["epochs"]):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}/{training_config['epochs']}")
        print("Training Metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nValidation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

        # Early stopping based on AUPRC
        if val_metrics["auprc"] > best_val_metrics["auprc"]:
            best_val_metrics = val_metrics
            early_stopping_counter = 0
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


def validate(model, val_loader, criterion, device):
    """Validate with focus on sparse graph metrics."""
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

            pred_probs = torch.sigmoid(output)
            all_preds.append(pred_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            total_loss += loss.item()

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_graph_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(val_loader)

    return metrics
