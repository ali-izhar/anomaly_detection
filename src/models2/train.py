"""Training script for the graph temporal forecasting model."""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models2.datasets import get_dataloaders
from src.models2.forecast import GraphTemporalForecaster
from src.utils.metrics import mse_loss, evaluate
from src.utils.helpers import save_model, count_parameters


def setup_gpu():
    """Setup GPU device."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # Get GPU device
    device = torch.device("cuda:0")  # Use first GPU
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # Convert to GB
    logging.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Set GPU memory usage
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    return device


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        # Move data to GPU
        input_data = {
            "adj_matrices": input_seq["adj_matrices"].to(device, non_blocking=True),
            "features": input_seq["features"].to(device, non_blocking=True)
        }
        targets = target_seq.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # Slightly more efficient than zero_grad()
        outputs = model(input_data)
        predictions = outputs["predictions"]
        
        # Compute loss and backward pass
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        # Update total loss
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logging.info(
                f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}"
            )
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / len(train_loader.dataset)


def main(config_path: str) -> None:
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create checkpoints directory
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    # Set device and GPU settings
    device = setup_gpu()

    # Prepare data loaders with GPU optimizations
    data_dir = Path("dataset")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, 
        config,
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,  # Enable faster data transfer to GPU
    )

    # Initialize model
    model = GraphTemporalForecaster(
        config=config,
        num_nodes=config["data"]["n_nodes"],
        node_feat_dim=config["data"]["n_features"],
        device=device
    ).to(device)
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    logging.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Get optimizers and schedulers
    optim_config = model.configure_optimizers(
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Training setup
    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    epochs_no_improve = 0
    
    try:
        # Training loop
        for epoch in range(1, config["training"]["epochs"] + 1):
            logging.info(f"\nEpoch {epoch}/{config['training']['epochs']}")
            
            # Train one epoch
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optim_config["optimizers"]["gnn"],
                loss_fn=mse_loss,
                device=device,
                grad_clip=config["training"].get("grad_clip", 1.0)
            )
            
            # Evaluate
            val_loss = evaluate(model, val_loader, device, mse_loss)
            
            # Step schedulers
            for scheduler in optim_config["schedulers"].values():
                scheduler.step()
                
            # Log progress
            logging.info(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_model(model, checkpoints_dir / "best_model.pth")
                logging.info("Saved new best model")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info("Early stopping triggered")
                    break

            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
    finally:
        # Save final model state
        save_model(model, checkpoints_dir / "final_model.pth")
        logging.info("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()
    main(args.config)
