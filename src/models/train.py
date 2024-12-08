"""Training script for spatio-temporal graph prediction model."""

import sys
import logging
from pathlib import Path
import yaml
from typing import Dict, Optional, Tuple
from collections import defaultdict
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        feature_losses = {}

        for feat_name in pred:
            if feat_name in self.weights:
                feat_loss = F.mse_loss(pred[feat_name], target[feat_name])
                weighted_loss = self.weights[feat_name] * feat_loss
                feature_losses[feat_name] = feat_loss.item()
                total_loss += weighted_loss

        return total_loss, feature_losses


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
    scaler: Optional[GradScaler],
    clip_grad_norm: float,
    log_interval: int,
) -> float:
    """Optimized training loop for better GPU utilization."""
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = {k: v.to(device, non_blocking=True) for k, v in batch["y"].items()}
        adj = batch["adj"].to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        amp_context = autocast(device_type=str(device)) if use_amp else nullcontext()
        with amp_context:
            predictions = model(x, adj)
            loss, _ = criterion(predictions, y)

        # Backward pass with scaling
        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        # Logging
        if batch_idx % log_interval == 0:
            logger.info(
                f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, "
                f"GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB"
            )

        total_loss += loss.item()

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Run validation loop."""
    model.eval()
    total_loss = 0.0
    feature_losses_dict = defaultdict(float)
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
            y = {k: v.to(device, non_blocking=True) for k, v in batch["y"].items()}
            adj = batch["adj"].to(device, non_blocking=True)

            predictions = model(x, adj)
            batch_loss, batch_feat_losses = criterion(predictions, y)

            total_loss += batch_loss.item()
            for feat, feat_loss in batch_feat_losses.items():
                feature_losses_dict[feat] += feat_loss

    # Average losses
    avg_loss = total_loss / num_batches
    avg_feature_losses = {k: v / num_batches for k, v in feature_losses_dict.items()}

    return avg_loss, avg_feature_losses


def main():
    # Load configuration
    config = load_config("train_config.yaml")

    # Setup paths and logging
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "training.log"
    setup_logging(log_file)
    logger.info(f"Config: {config}")

    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    set_random_seeds(seed)

    # Setup device and data parallel if multiple GPUs
    device = setup_device()
    logger.info(f"Using device: {device}")

    # Create datasets
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

    # Create train/val/test datasets
    train_dataset = GraphSequenceDataset(
        config["paths"]["data_dir"], "train", data_config
    )
    val_dataset = GraphSequenceDataset(config["paths"]["data_dir"], "val", data_config)
    test_dataset = GraphSequenceDataset(
        config["paths"]["data_dir"], "test", data_config
    )

    # Get hardware configuration
    hw_config = config.get("hardware", {})
    num_workers = min(
        hw_config.get("num_workers", 8),
        os.cpu_count() or 1,  # Fallback to 1 if CPU count can't be determined
    )
    pin_memory = hw_config.get("pin_memory", True) and torch.cuda.is_available()
    prefetch_factor = hw_config.get("prefetch_factor", 2)
    persistent_workers = hw_config.get("persistent_workers", True)

    # Create dataloaders with hardware optimizations
    train_loader = create_dataloader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    # Create model
    model_config = STModelConfig(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        gnn_type=str(config["model"]["gnn_type"]),
        attention_heads=int(config["model"]["attention_heads"]),
        lstm_layers=int(config["model"]["lstm_layers"]),
        bidirectional=bool(config["model"]["bidirectional"]),
    )

    model = SpatioTemporalPredictor(model_config)

    # Multi-GPU support if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup training
    criterion = WeightedMSELoss(config["training"]["loss_weights"])
    optimizer = Adam(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["lr_schedule"]["factor"],
        patience=config["training"]["lr_schedule"]["patience"],
        min_lr=float(config["training"]["lr_schedule"]["min_lr"]),
        verbose=True,
    )

    # Initialize mixed precision training
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Setup tensorboard
    writer = (
        SummaryWriter(output_dir / "runs")
        if config["training"]["tensorboard"]
        else None
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = config["training"]["patience"]
    patience_counter = 0
    start_epoch = 0

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Training
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            clip_grad_norm=config["model"]["clip_grad_norm"],
            log_interval=config["training"]["log_interval"],
        )

        # Validation
        val_loss, val_feature_losses = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            for feat, loss in val_feature_losses.items():
                writer.add_scalar(f"Feature_Loss/{feat}", loss, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                config=config,
                path=output_dir / "best_model",
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation on test set
    logger.info("Loading best model for testing...")
    load_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        path=output_dir / "best_model",
    )
    test_loss, test_feature_losses = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Cleanup
    if writer:
        writer.close()


def setup_logging(log_file: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Only set deterministic mode if not using benchmark
    if not torch.backends.cudnn.benchmark:
        torch.backends.cudnn.deterministic = True


def setup_device() -> torch.device:
    """Setup compute device with optimized settings for RTX 4090."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        # Enable memory pinning and optimize CUDA settings
        torch.cuda.init()

        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmarking and deterministic algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set memory allocation to optimal for RTX 4090
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory

        logger.info(f"GPU Device: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(os.cpu_count())

    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    config: dict,
    path: Path,
):
    """Save model checkpoint."""
    # Save model state dict separately with weights_only=True
    torch.save(model.state_dict(), path.with_suffix(".model"), weights_only=True)

    # Save other training state
    metadata = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config,
    }
    torch.save(metadata, path.with_suffix(".meta"))


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    path: Path,
) -> Tuple[int, float]:
    """Load model checkpoint and training state."""
    # Load model weights safely
    model.load_state_dict(torch.load(path.with_suffix(".model"), weights_only=True))

    # Load training state if optimizer and scheduler are provided
    if optimizer is not None and scheduler is not None:
        metadata = torch.load(path.with_suffix(".meta"))
        optimizer.load_state_dict(metadata["optimizer_state_dict"])
        scheduler.load_state_dict(metadata["scheduler_state_dict"])
        return metadata["epoch"], metadata["loss"]

    return 0, float("inf")


if __name__ == "__main__":
    main()
