# src/utils/metrics.py

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss.
    
    Args:
        pred: Predictions [batch_size, forecast_horizon, num_features]
        target: Ground truth [batch_size, forecast_horizon, num_features]
        
    Returns:
        MSE loss value
    """
    return nn.MSELoss()(pred, target)


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error loss.
    
    Args:
        pred: Predictions [batch_size, forecast_horizon, num_features]
        target: Ground truth [batch_size, forecast_horizon, num_features]
        
    Returns:
        MAE loss value
    """
    return nn.L1Loss()(pred, target)


def rmse_loss(mse: Tensor) -> Tensor:
    """Root mean squared error loss.
    
    Args:
        mse: Mean squared error value
        
    Returns:
        RMSE value
    """
    return torch.sqrt(mse)


def compute_metrics(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None
) -> Dict[str, float]:
    """Compute multiple metrics.
    
    Args:
        pred: Predictions [batch_size, forecast_horizon, num_features]
        target: Ground truth [batch_size, forecast_horizon, num_features]
        mask: Optional mask for valid timesteps [batch_size, forecast_horizon]
        
    Returns:
        Dictionary of metric values
    """
    if mask is not None:
        # Apply mask
        mask = mask.unsqueeze(-1)  # [batch_size, forecast_horizon, 1]
        pred = pred * mask
        target = target * mask
    
    mse = mse_loss(pred, target)
    mae = mae_loss(pred, target)
    rmse = rmse_loss(mse)
    
    return {
        "MSE": mse.item(),
        "MAE": mae.item(),
        "RMSE": rmse.item()
    }


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: callable,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to run on
        loss_fn: Loss function
        
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    total_metrics = None
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_seq, target_seq = batch
            
            # Move data to device
            input_data = {
                "adj_matrices": input_seq["adj_matrices"].to(device, non_blocking=True),
                "features": input_seq["features"].to(device, non_blocking=True)
            }
            targets = target_seq.to(device, non_blocking=True)

            # Get predictions
            outputs = model(input_data)
            predictions = outputs["predictions"]

            # Compute loss and metrics
            loss = loss_fn(predictions, targets)
            metrics = compute_metrics(predictions, targets)
            
            # Update totals
            batch_size = len(predictions)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Accumulate metrics
            if total_metrics is None:
                total_metrics = {k: v * batch_size for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    total_metrics[k] += v * batch_size

    # Calculate averages
    avg_loss = total_loss / num_samples
    avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}

    return avg_loss, avg_metrics
