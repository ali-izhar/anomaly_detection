# src/model/run.py

from pathlib import Path
import torch
from torch.utils.data import Subset
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

from dataset import DynamicGraphDataset
from small.train import train_model
from utils.logger import setup_logging

logger = setup_logging()


def setup_model_config(dataset) -> dict:
    """Setup model configuration based on dataset properties."""
    return {
        "in_channels": dataset.num_features,
        "hidden_channels": 32,
        "out_channels": 32,
        "num_nodes": dataset.num_nodes,
        "window_size": dataset.config["processing"]["temporal_window"],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 100,
        "patience": 10,
        "batch_size": 32,
    }


def create_experiment_dir(base_dir: str = "experiments") -> Path:
    """Create a directory for the current experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def main(args):
    # Load dataset
    logger.info("Loading dataset...")
    dataset = DynamicGraphDataset(
        variant=args.variant, data_dir=args.data_dir, graph_type=args.graph_type
    )

    # Create train/val/test splits
    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(seed=args.seed)

    # Create data loaders
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dataset.config["training"]["num_workers"],
        collate_fn=dataset._collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataset.config["training"]["num_workers"],
        collate_fn=dataset._collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataset.config["training"]["num_workers"],
        collate_fn=dataset._collate_fn,
    )

    # Setup model configuration
    model_config = setup_model_config(dataset)
    model_config.update(
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        }
    )

    # Create experiment directory
    exp_dir = create_experiment_dir(args.exp_dir)
    logger.info(f"Experiment directory: {exp_dir}")

    # Save configurations
    with open(exp_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)

    # Train model
    logger.info("Starting training...")
    model = train_model(train_loader, val_loader, model_config)

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
    parser = argparse.ArgumentParser(
        description="Train STGCN model on dynamic graph data"
    )

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

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
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

    main(args)
