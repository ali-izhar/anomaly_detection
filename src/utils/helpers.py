# src/utils/helpers.py

import torch
import logging
from pathlib import Path
from typing import Dict, Optional


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: torch.nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        path: Path to save the model
        optimizer: Optional optimizer to save state
        epoch: Optional current epoch number
        best_metric: Optional best validation metric
    """
    # Create checkpoint directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }

    try:
        torch.save(checkpoint, path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model to {path}: {str(e)}")


def load_model(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load model checkpoint.

    Args:
        model: Model to load weights into
        path: Path to the checkpoint file
        device: Device to load the model on
        optimizer: Optional optimizer to load state

    Returns:
        Dictionary containing checkpoint info (epoch, best_metric)
    """
    try:
        checkpoint = torch.load(path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Load optimizer state if provided
        if optimizer and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logging.info(f"Model loaded from {path}")

        return {
            "epoch": checkpoint.get("epoch"),
            "best_metric": checkpoint.get("best_metric"),
        }

    except Exception as e:
        logging.error(f"Error loading model from {path}: {str(e)}")
        raise


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, lr: float, epoch: int, warmup_epochs: int = 5
) -> None:
    """Adjust learning rate with warmup.

    Args:
        optimizer: Optimizer to adjust
        lr: Target learning rate
        epoch: Current epoch number
        warmup_epochs: Number of warmup epochs
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
        lr_adj = lr * lr_scale
    else:
        # Cosine decay
        lr_adj = (
            lr
            * 0.5
            * (
                1.0
                + torch.cos(
                    torch.tensor(
                        (epoch - warmup_epochs) / (warmup_epochs * 2) * torch.pi
                    )
                ).item()
            )
        )

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_adj
