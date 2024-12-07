# src/models/train.py

import os
import sys
import logging
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.spatiotemporal import SpatioTemporalPredictor, STModelConfig
from src.models.graph_dataset import (
    GraphDataConfig,
    GraphSequenceDataset,
    create_dataloader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for different feature types."""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        total_loss = 0.0
        for feat_type in pred:
            if feat_type in self.weights:
                loss = self.mse(pred[feat_type], target[feat_type])
                total_loss += self.weights[feat_type] * loss.mean()
        return total_loss


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        x = {k: v.to(device) for k, v in batch["x"].items()}
        y = {k: v.to(device) for k, v in batch["y"].items()}
        adj = batch["adj"].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(x, adj)

        # Calculate loss for each feature type
        loss = 0.0
        for feat_name in predictions:
            if feat_name in y:
                feat_loss = criterion(predictions[feat_name], y[feat_name])
                loss += feat_loss

        # Backward pass
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = {k: v.to(device) for k, v in batch["x"].items()}
            y = {k: v.to(device) for k, v in batch["y"].items()}
            adj = batch["adj"].to(device)

            predictions = model(x, adj)
            loss = criterion(predictions, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(config_path: str):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert numerical parameters to proper types
    model_config = config["model"]
    model_config["learning_rate"] = float(model_config["learning_rate"])
    model_config["weight_decay"] = float(model_config["weight_decay"])
    model_config["clip_grad_norm"] = float(model_config["clip_grad_norm"])
    
    training_config = config["training"]
    training_config["lr_schedule"]["factor"] = float(training_config["lr_schedule"]["factor"])
    training_config["lr_schedule"]["min_lr"] = float(training_config["lr_schedule"]["min_lr"])
    
    for key in training_config["loss_weights"]:
        training_config["loss_weights"][key] = float(training_config["loss_weights"][key])
    
    # Setup paths
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    if config["training"]["tensorboard"]:
        writer = SummaryWriter(output_dir / "runs")

    # Create data config
    data_config = GraphDataConfig(
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        forecast_horizon=config["data"]["forecast_horizon"],
        batch_size=config["data"]["batch_size"],
        use_centrality=config["data"]["use_centrality"],
        use_spectral=config["data"]["use_spectral"],
        enable_augmentation=config["data"]["enable_augmentation"],
        noise_level=config["data"]["noise_level"],
        num_nodes=config["data"]["num_nodes"],
        max_seq_length=config["data"]["max_seq_length"],
        min_seq_length=config["data"]["min_seq_length"],
        num_change_points=config["data"]["num_change_points"],
        svd_dim=config["data"]["svd_dim"],
        lsvd_dim=config["data"]["lsvd_dim"],
    )

    # Create datasets and dataloaders
    train_dataset = GraphSequenceDataset(
        config["paths"]["data_dir"], "train", data_config
    )
    val_dataset = GraphSequenceDataset(config["paths"]["data_dir"], "val", data_config)
    test_dataset = GraphSequenceDataset(
        config["paths"]["data_dir"], "test", data_config
    )

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)
    test_loader = create_dataloader(test_dataset, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model config
    st_model_config = STModelConfig(
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=float(model_config["dropout"]),  # Ensure float
        gnn_type=model_config["gnn_type"],
        attention_heads=model_config["attention_heads"],
        lstm_layers=model_config["lstm_layers"],
        bidirectional=model_config["bidirectional"],
        num_nodes=config["data"]["num_nodes"],
        window_size=config["data"]["window_size"],
        forecast_horizon=config["data"]["forecast_horizon"],
        num_features=22
    )
    
    model = SpatioTemporalPredictor(st_model_config).to(device)

    # Setup training with explicit float conversion
    criterion = WeightedMSELoss(training_config["loss_weights"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["lr_schedule"]["factor"]),
        patience=training_config["lr_schedule"]["patience"],
        min_lr=float(training_config["lr_schedule"]["min_lr"]),
    )

    # Memory optimization
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Optional: Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        with torch.cuda.amp.autocast():
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                clip_grad_norm=model_config["clip_grad_norm"],
            )

        # Validate
        if (epoch + 1) % config["training"]["val_interval"] == 0:
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            logger.info(
                f"Epoch [{epoch+1}/{config['training']['epochs']}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if config["training"]["tensorboard"]:
                writer.add_scalars(
                    "Loss", {"train": train_loss, "val": val_loss}, epoch
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config["training"]["patience"]:
                logger.info("Early stopping triggered")
                break

        # Save checkpoint
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": best_val_loss,
                },
                output_dir / f"checkpoint_epoch_{epoch+1}.pth",
            )

    # Test best model
    model.load_state_dict(torch.load(output_dir / "best_model.pth"))
    test_loss = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}")

    if config["training"]["tensorboard"]:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
