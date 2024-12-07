"""Mini training script for testing spatio-temporal model."""

import sys
from pathlib import Path
import logging
import yaml
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from contextlib import nullcontext

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.spatiotemporal import SpatioTemporalPredictor, STModelConfig
from models.graph_dataset import (
    GraphSequenceDataset,
    GraphDataConfig,
    create_dataloader,
)
from visualization import analyze_predictions, plot_loss_breakdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for different feature types."""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = 0.0
        feature_losses = {}
        for feat_name in pred:
            if feat_name in self.weights:
                feat_loss = F.mse_loss(pred[feat_name], target[feat_name])
                feature_losses[feat_name] = feat_loss.item()
                loss += self.weights[feat_name] * feat_loss
        return loss, feature_losses


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
) -> Tuple[float, Dict[str, float]]:
    """Single training step."""
    # Move data to device
    x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
    y = {k: v.to(device, non_blocking=True) for k, v in batch["y"].items()}
    adj = batch["adj"].to(device, non_blocking=True)

    # Zero gradients
    for param in model.parameters():
        param.grad = None

    # Forward pass with optional mixed precision
    amp_context = (
        autocast(device_type=str(device)) if scaler is not None else nullcontext()
    )
    with amp_context:
        predictions = model(x, adj)
        loss, feature_losses = criterion(predictions, y)

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item(), feature_losses, predictions


def validate_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, torch.Tensor]]:
    """Single validation step."""
    model.eval()
    with torch.no_grad():
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = {k: v.to(device, non_blocking=True) for k, v in batch["y"].items()}
        adj = batch["adj"].to(device, non_blocking=True)

        predictions = model(x, adj)
        loss, feature_losses = criterion(predictions, y)

    return loss.item(), feature_losses, predictions


def plot_training_curves(train_losses: list, val_losses: list, save_path: Path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(save_path / "loss_curves.png")
    plt.close()


def plot_predictions(
    batch: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    save_path: Path,
    batch_idx: int,
):
    """Plot actual vs predicted values for a few features."""
    features_to_plot = ["degree", "closeness", "svd"]  # Example features

    plt.figure(figsize=(15, 5))
    for i, feat in enumerate(features_to_plot):
        plt.subplot(1, 3, i + 1)
        actual = batch["y"][feat][0, 0].cpu().numpy()  # First node, first timestep
        pred = predictions[feat][0, 0].detach().cpu().numpy()
        plt.scatter(actual, pred, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{feat} Predictions")

    plt.tight_layout()
    plt.savefig(save_path / f"predictions_batch_{batch_idx}.png")
    plt.close()


def main():
    # Load configuration
    config = yaml.safe_load(open("train_config.yaml"))

    # Setup
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and loaders (with smaller batch size for testing)
    data_config = GraphDataConfig(
        window_size=20, stride=5, forecast_horizon=10, batch_size=16
    )

    train_dataset = GraphSequenceDataset(
        config["paths"]["data_dir"], "train", data_config
    )
    val_dataset = GraphSequenceDataset(config["paths"]["data_dir"], "val", data_config)

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)

    # Create model and optimizer with matching dimensions
    model_config = STModelConfig(
        hidden_dim=64,  # Increased to match LSTM input size
        num_layers=2,
        dropout=0.2,
        bidirectional=False,  # Make sure this matches decoder expectations
        lstm_layers=2,
        window_size=20,
        forecast_horizon=10,
    )

    model = SpatioTemporalPredictor(model_config).to(device)

    criterion = WeightedMSELoss(config["training"]["loss_weights"])
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Training loop
    train_losses = []
    val_losses = []
    feature_losses_dict = {feat: [] for feat in config["training"]["loss_weights"]}
    max_batches = 50

    # Store final predictions for visualization
    final_val_batch = None
    final_predictions = None

    logger.info("Starting mini training...")

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        # Training step
        model.train()
        loss, feat_losses, predictions = train_step(
            model, batch, criterion, optimizer, device, scaler
        )
        train_losses.append(loss)

        # Record feature-specific losses
        for feat, feat_loss in feat_losses.items():
            feature_losses_dict[feat].append(feat_loss)

        # Validation step (every 10 batches)
        if batch_idx % 10 == 0:
            val_batch = next(iter(val_loader))
            val_loss, val_feat_losses, val_predictions = validate_step(
                model, val_batch, criterion, device
            )
            val_losses.append(val_loss)

            # Store final validation batch and predictions
            final_val_batch = val_batch
            final_predictions = val_predictions

            # Log progress
            logger.info(
                f"Batch {batch_idx}/{max_batches}, "
                f"Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Create visualization directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, vis_dir)
    plot_loss_breakdown(feature_losses_dict, vis_dir)

    # Generate final prediction visualizations
    if final_val_batch is not None and final_predictions is not None:
        logger.info("Generating final visualizations...")
        analyze_predictions(final_val_batch, final_predictions, vis_dir)

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "feature_losses": feature_losses_dict,
        },
        output_dir / "mini_checkpoint.pt",
    )

    logger.info("Mini training completed!")


if __name__ == "__main__":
    main()
