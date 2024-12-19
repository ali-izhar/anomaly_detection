# src/model/medium/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from .model_m import MediumASTGCN


class GraphStructureLoss(nn.Module):
    def __init__(self, num_nodes=30, min_edges=2, max_edges=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.min_edges = min_edges
        self.max_edges = max_edges
        
    def forward(self, pred, target):
        # Very aggressive positive class weighting
        pos_weight = ((target == 0).float().sum() / (target == 1).float().sum()).clamp(5.0, 20.0)
        
        # BCE loss with strong class weights
        bce = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=pos_weight * torch.ones_like(target).to(pred.device),
            reduction='none'
        )
        
        # Get predicted probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Strong penalty for uniform predictions
        diversity_loss = -torch.std(pred_probs, dim=-1).mean()
        
        # Encourage target degree distribution
        pred_degrees = pred_probs.sum(dim=-1)
        target_degrees = target.sum(dim=-1)
        degree_loss = F.mse_loss(pred_degrees, target_degrees)
        
        # Strongly encourage some positive predictions
        min_positive_ratio = 0.2  # At least 20% positive predictions
        sparsity_loss = F.relu(min_positive_ratio - pred_probs.mean()) * 10.0
        
        # Combine losses with stronger weights on auxiliary terms
        total_loss = bce.mean() + diversity_loss + degree_loss + sparsity_loss
        
        return total_loss


def calculate_graph_metrics(predictions, targets, num_nodes=30):
    """Calculate graph-specific structural metrics."""
    pred_adj = (predictions > 0.5).astype(float)
    true_adj = targets.astype(float)
    
    metrics = {}
    
    # 1. Degree Statistics
    pred_degrees = pred_adj.sum(axis=-1)  # [batch, nodes]
    true_degrees = true_adj.sum(axis=-1)  # [batch, nodes]
    
    metrics.update({
        "avg_degree_error": np.abs(pred_degrees.mean() - true_degrees.mean()),
        "degree_std_error": np.abs(pred_degrees.std() - true_degrees.std()),
        "max_degree_error": np.abs(pred_degrees.max() - true_degrees.max()),
    })
    
    # 2. Clustering Coefficient
    def calc_clustering(adj):
        tri = np.matmul(np.matmul(adj, adj), adj)
        degrees = adj.sum(axis=-1)
        possible_tri = degrees * (degrees - 1) / 2
        clustering = np.zeros_like(degrees)
        mask = possible_tri > 0
        clustering[mask] = tri[mask].diagonal() / (possible_tri[mask] * 6)
        return clustering.mean()
    
    for i in range(min(10, len(pred_adj))):  # Calculate for first 10 graphs
        metrics[f"clustering_error_{i}"] = abs(
            calc_clustering(pred_adj[i]) - calc_clustering(true_adj[i])
        )
    
    # 3. Path Length Distribution
    def calc_path_lengths(adj):
        dist = np.zeros_like(adj)
        dist[adj > 0] = 1
        dist[adj == 0] = float('inf')
        np.fill_diagonal(dist, 0)
        
        for k in range(num_nodes):
            dist = np.minimum(
                dist,
                dist[:, np.newaxis, k] + dist[k, np.newaxis, :]
            )
        
        finite_mask = dist != float('inf')
        if finite_mask.sum() > 0:
            return dist[finite_mask].mean()
        return 0
    
    metrics["path_length_error"] = np.mean([
        abs(calc_path_lengths(pred_adj[i]) - calc_path_lengths(true_adj[i]))
        for i in range(min(10, len(pred_adj)))
    ])
    
    # 4. Edge Distribution
    metrics.update({
        "edge_density_error": abs(pred_adj.mean() - true_adj.mean()),
        "edge_variance_error": abs(pred_adj.var() - true_adj.var()),
    })
    
    return metrics


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate all metrics for evaluation."""
    pred_binary = (predictions > threshold).astype(float)
    
    # Basic metrics
    basic_metrics = {
        "f1_score": f1_score(targets.flatten(), pred_binary.flatten()),
        "auc_score": roc_auc_score(targets.flatten(), predictions.flatten()),
        "avg_precision": average_precision_score(targets.flatten(), predictions.flatten()),
    }
    
    # Graph structure metrics
    graph_metrics = calculate_graph_metrics(predictions, targets)
    
    # Combine metrics
    return {**basic_metrics, **graph_metrics}


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            features = batch["features"].to(device)
            edge_index = batch["edge_indices"][0][0].to(device)
            edge_weight = batch["edge_weights"][0][0].to(device) if batch["edge_weights"] else None
            targets = batch["targets"].to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(features, edge_index, edge_weight)
                loss = criterion(output, targets)
            
            # Use gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
            
            # Store predictions and targets
            pred_probs = torch.sigmoid(output)
            all_preds.append(pred_probs.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update progress bar with more metrics
            current_metrics = calculate_metrics(
                pred_probs.detach().cpu().numpy(),
                targets.cpu().numpy()
            )
            with torch.no_grad():
                pos_ratio = (pred_probs > 0.5).float().mean()
                max_prob = pred_probs.max()
                min_prob = pred_probs.min()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "f1": f"{current_metrics['f1_score']:.4f}",
                "pos_ratio": f"{pos_ratio.item():.3f}",
                "prob_range": f"[{min_prob.item():.2f}, {max_prob.item():.2f}]"
            })
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(train_loader)
    
    return metrics


def train_model(train_loader, val_loader, model_config, training_config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = MediumASTGCN(**model_config).to(device)
    
    # Loss and optimizer
    criterion = GraphStructureLoss(
        num_nodes=model_config["num_nodes"],
        min_edges=2,  # From BA config
        max_edges=8   # From BA config
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * training_config["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    best_val_metrics = {"loss": float("inf"), "f1_score": 0}
    early_stopping_counter = 0
    
    for epoch in range(training_config["epochs"]):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{training_config['epochs']}")
        print("Training Metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nValidation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Early stopping check
        if val_metrics["f1_score"] > best_val_metrics["f1_score"]:
            best_val_metrics = val_metrics
            early_stopping_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_metrics": best_val_metrics,
            }, "best_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= training_config["patience"]:
                print("\nEarly stopping triggered!")
                break
    
    return model


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
            edge_weight = batch["edge_weights"][0][0].to(device) if batch["edge_weights"] else None
            targets = batch["targets"].to(device)
            
            # Forward pass
            output = model(features, edge_index, edge_weight)
            loss = criterion(output, targets)
            
            # Store predictions and targets
            pred_probs = torch.sigmoid(output)
            all_preds.append(pred_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(val_loader)
    
    return metrics
