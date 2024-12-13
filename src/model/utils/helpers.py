# src/model/utils/helpers.py

import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_optimizer(config: Dict[str, Any], model_params) -> optim.Optimizer:
    """Create optimizer based on config."""
    opt_config = config["optimizer"]
    opt_type = opt_config["type"].lower()

    if opt_type == "adam":
        return Adam(
            model_params,
            lr=opt_config["learning_rate"],
            weight_decay=opt_config["weight_decay"],
            betas=(opt_config["beta1"], opt_config["beta2"]),
            amsgrad=opt_config["amsgrad"],
        )
    elif opt_type == "sgd":
        return SGD(
            model_params,
            lr=opt_config["learning_rate"],
            weight_decay=opt_config["weight_decay"],
            momentum=opt_config.get("momentum", 0.9),
        )
    elif opt_type == "adamw":
        return AdamW(
            model_params,
            lr=opt_config["learning_rate"],
            weight_decay=opt_config["weight_decay"],
            betas=(opt_config["beta1"], opt_config["beta2"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def get_scheduler(config: Dict[str, Any], optimizer: optim.Optimizer):
    """Create learning rate scheduler based on config."""
    sched_config = config["scheduler"]
    sched_type = sched_config["type"].lower()

    if sched_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=sched_config["mode"],
            factor=sched_config["factor"],
            patience=sched_config["patience"],
            min_lr=sched_config["min_lr"],
            threshold=sched_config["threshold"],
        )
    elif sched_type == "step":
        return StepLR(
            optimizer, step_size=sched_config["step_size"], gamma=sched_config["gamma"]
        )
    elif sched_type == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=sched_config["T_max"], eta_min=sched_config["min_lr"]
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Create loss function based on config."""
    loss_config = config["loss"]
    if loss_config["type"].lower() == "bce":
        return nn.BCELoss(
            weight=(
                torch.tensor(loss_config["weights"]) if loss_config["weights"] else None
            ),
            reduction=loss_config["reduction"],
        )
    raise ValueError(f"Unsupported loss type: {loss_config['type']}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    history: Dict,
    config: Dict,
    checkpoint_dir: str = "checkpoints",
    is_best: bool = False,
) -> str:
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "history": history,
        "config": config,
    }

    if is_best:
        checkpoint_path = checkpoint_dir / "model_best.pt"
    else:
        checkpoint_path = checkpoint_dir / config["checkpoint"][
            "filename_template"
        ].format(epoch=epoch, loss=loss)

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Clean up old checkpoints
    if not is_best and config["checkpoint"]["keep_last"] > 0:
        checkpoints = sorted(
            checkpoint_dir.glob("model_epoch_*.pt"),
            key=lambda x: int(x.stem.split("_")[2]),
        )
        while len(checkpoints) > config["checkpoint"]["keep_last"]:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
