# src/model/run.py

from pathlib import Path
import torch
from torch.utils.data import Subset
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import gc
import GPUtil

from dataset import DynamicGraphDataset
from small.train import train_model as train_small
from medium.train import train_model as train_medium
from utils.logger import setup_logging

logger = setup_logging()


def print_gpu_utilization():
    """Print GPU utilization stats."""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(f'GPU {gpu.id} - Memory Used: {gpu.memoryUsed/gpu.memoryTotal*100:.1f}% ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)')


def setup_model_config(dataset, model_type: str = "small") -> dict:
    """Setup model configuration based on dataset properties and model type."""
    base_config = {
        "in_channels": dataset.num_features,
        "num_nodes": dataset.num_nodes,
        "window_size": dataset.config["processing"]["temporal_window"],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 10,
        "patience": 5,
        "batch_size": 128,
    }
    
    if model_type == "small":
        base_config.update({
            "hidden_channels": 32,
            "out_channels": 32,
        })
    elif model_type == "medium":
        base_config.update({
            "hidden_channels": 64,
            "out_channels": 64,
            "num_heads": 4,
            "num_layers": 2,
            "weight_decay": 1e-4,
            "scheduler_t0": 5,
            "scheduler_t_mult": 2,
        })
    
    return base_config


def create_experiment_dir(base_dir: str = "experiments", model_type: str = "small") -> Path:
    """Create a directory for the current experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / model_type / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def main(args):
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("Initial GPU Memory Usage:")
    print_gpu_utilization()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = DynamicGraphDataset(
        variant=args.variant, data_dir=args.data_dir, graph_type=args.graph_type
    )

    # Create train/val/test splits
    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(seed=args.seed)

    # Create data loaders with pin_memory=True for faster GPU transfer
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dataset.config["training"]["num_workers"],
        collate_fn=dataset._collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataset.config["training"]["num_workers"],
        collate_fn=dataset._collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    # Setup model configuration
    model_config = setup_model_config(dataset, args.model_type)
    model_config.update({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    })

    # Create experiment directory
    exp_dir = create_experiment_dir(args.exp_dir, args.model_type)
    logger.info(f"Experiment directory: {exp_dir}")

    # Save configurations
    with open(exp_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)

    # Train model
    logger.info(f"Starting training for {args.model_type} model...")
    logger.info("GPU Memory Usage before training:")
    print_gpu_utilization()
    
    if args.model_type == "small":
        model = train_small(train_loader, val_loader, model_config)
    else:  # medium
        model = train_medium(train_loader, val_loader, model_config)

    logger.info("GPU Memory Usage after training:")
    print_gpu_utilization()

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config,
        },
        exp_dir / "final_model.pth",
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on dynamic graph data")

    # Dataset parameters
    parser.add_argument(
        "--variant",
        type=str,
        default="node_level",
        choices=["node_level", "global", "combined"],
        help="Dataset variant to use",
    )
    parser.add_argument(
        "--data-dir", type=str, default="datasets", help="Path to dataset directory"
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default=None,
        choices=["BA", "ER", "NW"],
        help="Specific graph type to use (optional)",
    )

    # Model parameters
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="Type of model to train"
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Output parameters
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments",
        help="Directory to save experiment results",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # Set cuda to be deterministic for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)
