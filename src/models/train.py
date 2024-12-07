"""Training script for spatio-temporal graph prediction model."""

import sys
import logging
from pathlib import Path
import yaml
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.spatiotemporal import SpatioTemporalPredictor, STModelConfig
from models.graph_dataset import (
    GraphSequenceDataset,
    GraphDataConfig,
    create_dataloader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for different feature types."""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = 0.0
        for feat_name in pred:
            if feat_name in self.weights:
                feat_loss = F.mse_loss(pred[feat_name], target[feat_name])
                loss += self.weights[feat_name] * feat_loss
        return loss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float,
    log_interval: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        x = {k: v.to(device) for k, v in batch["x"].items()}
        y = {k: v.to(device) for k, v in batch["y"].items()}
        adj = batch["adj"].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(x, adj)

        # Compute loss
        loss = criterion(predictions, y)

        # Backward pass
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        # Log progress
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            logger.info(
                f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


def main():
    # Load configuration
    config = load_config("train_config.yaml")

    # Setup paths
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data config
    data_config = GraphDataConfig(
        window_size=int(config["data"]["window_size"]),
        stride=int(config["data"]["stride"]),
        forecast_horizon=int(config["data"]["forecast_horizon"]),
        batch_size=int(config["data"]["batch_size"]),
        use_centrality=bool(config["data"]["use_centrality"]),
        use_spectral=bool(config["data"]["use_spectral"]),
        enable_augmentation=bool(config["data"]["enable_augmentation"]),
        noise_level=float(config["data"]["noise_level"]),
    )

    # Create model config
    model_config = STModelConfig(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        gnn_type=str(config["model"]["gnn_type"]),
        attention_heads=int(config["model"]["attention_heads"]),
        lstm_layers=int(config["model"]["lstm_layers"]),
        bidirectional=bool(config["model"]["bidirectional"]),
    )

    # Create datasets and dataloaders
    train_dataset = GraphSequenceDataset(data_dir, "train", data_config)
    train_loader = create_dataloader(
        train_dataset, batch_size=data_config.batch_size, shuffle=True
    )

    # Create model
    model = SpatioTemporalPredictor(model_config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = WeightedMSELoss(config["training"]["loss_weights"])
    optimizer = Adam(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )

    # Setup tensorboard
    if config["training"]["tensorboard"]:
        writer = SummaryWriter(output_dir / "runs")

    # Train for one epoch
    logger.info("Starting training...")
    train_loss = train_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=config["model"]["clip_grad_norm"],
        log_interval=config["training"]["log_interval"],
    )

    logger.info(f"Training completed. Average loss: {train_loss:.4f}")

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_loss": train_loss,
        },
        output_dir / "model_checkpoint.pt",
    )

    if config["training"]["tensorboard"]:
        writer.close()


if __name__ == "__main__":
    main()
