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
from torchinfo import summary
import os

from dataset import DynamicGraphDataset
from small.train import train_model as train_small
from medium.train import train_model as train_medium
from large.train import train_model as train_large
from utils.logger import setup_logging

logger = setup_logging()


def print_gpu_utilization():
    """Print GPU utilization stats."""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(
            f"GPU {gpu.id} - Memory Used: {gpu.memoryUsed/gpu.memoryTotal*100:.1f}% ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)"
        )


def setup_model_config(dataset, model_type: str = "small") -> dict:
    """Setup model configuration based on dataset properties and model type."""
    base_config = {
        "in_channels": dataset.num_features,
        "num_nodes": dataset.num_nodes,
        "window_size": dataset.config["processing"]["temporal_window"],
    }

    # Model configurations
    model_configs = {
        "small": {
            "hidden_channels": 32,
            "out_channels": 32,
            "dropout": 0.1,
        },
        "medium": {
            "hidden_channels": 128,
            "out_channels": 128,
            "num_heads": 4,
            "num_layers": 3,
            "dropout": 0.3,
        },
        "large": {
            "hidden_channels": 256,
            "out_channels": 256,
            "spatial_heads": 8,
            "temporal_heads": 8,
            "num_layers": 3,
            "dropout": 0.2,
            "l1_lambda": 0.005,
        },
    }

    # Training configurations
    training_configs = {
        "small": {
            "batch_size": 128,
            "learning_rate": 0.001,
            "epochs": 15,
            "patience": 5,
            "weight_decay": 1e-4,
        },
        "medium": {
            "batch_size": 128,
            "learning_rate": 0.00005,
            "epochs": 1, # test
            "patience": 10,
            "weight_decay": 1e-4,
            "scheduler_t0": 5,
            "scheduler_t_mult": 2,
        },
        "large": {
            "batch_size": 128,
            "learning_rate": 0.0003,
            "epochs": 5,
            "patience": 10,
            "weight_decay": 1e-4,
            "early_stopping_metric": "f1_score",
            "threshold": 0.3,
            "gradient_clip": 0.5,
            "warmup_epochs": 2,
        },
    }

    # Merge base config with model-specific config
    model_config = {**base_config, **model_configs[model_type]}
    training_config = training_configs[model_type]

    return model_config, training_config


def create_experiment_dir(
    base_dir: str = "experiments", model_type: str = "small"
) -> Path:
    """Create a directory for the current experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / model_type / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_model_summary(model, model_config):
    """Get model summary string using torchinfo."""
    try:
        # Create sample inputs
        batch_size = 2
        sample_features = torch.randn(
            batch_size,
            model_config["window_size"],
            model_config["num_nodes"],
            model_config["in_channels"],
        )
        sample_edge_index = torch.randint(0, model_config["num_nodes"], (2, 100))

        # Get model summary with more detailed configuration
        model_summary = summary(
            model=model,
            input_size=[sample_features.shape, sample_edge_index.shape],
            input_data=None,  # Don't pass actual data
            batch_dim=0,
            col_names=["input_size", "output_size", "num_params"],
            col_width=20,
            row_settings=["var_names"],
            device=torch.device("cpu"),  # Keep on CPU for summary
            mode="eval",
            verbose=0,
        )
        return str(model_summary)
    except Exception as e:
        # Fallback to manual parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        summary_str = (
            f"Failed to generate detailed summary: {str(e)}\n\n"
            f"Model Structure:\n{str(model)}\n\n"
            f"Parameter Summary:\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Input shape: [batch_size, {model_config['window_size']}, "
            f"{model_config['num_nodes']}, {model_config['in_channels']}]\n"
            f"Edge index shape: [2, num_edges]"
        )
        return summary_str


def main(args):
    # Memory optimization settings
    torch.cuda.empty_cache()
    gc.collect()

    # Set memory allocation settings
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # Enable deterministic training for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info("Initial GPU Memory Usage:")
    print_gpu_utilization()

    # Load dataset
    if args.model_type == "medium":
        logger.info("Loading BA graph dataset...")
        dataset = DynamicGraphDataset(
            variant=args.variant,
            data_dir=args.data_dir,
            graph_type="BA",  # Force BA graphs for medium model
        )
    else:
        dataset = DynamicGraphDataset(
            variant=args.variant, data_dir=args.data_dir, graph_type=args.graph_type
        )

    # Setup model and training configurations
    model_config, training_config = setup_model_config(dataset, args.model_type)

    # Create train/val/test splits
    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(seed=args.seed)

    # Log dataset split information
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Training samples: {len(train_idx)}")
    logger.info(f"Validation samples: {len(val_idx)}")
    logger.info(f"Test samples: {len(test_idx)}")

    # Use batch_size from training_config instead of args
    batch_size = training_config["batch_size"]

    # Calculate and log number of batches
    num_training_batches = len(train_idx) // batch_size + (
        1 if len(train_idx) % batch_size != 0 else 0
    )
    logger.info(
        f"Number of training batches: {num_training_batches} (training_samples={len(train_idx)} / batch_size={batch_size})"
    )

    # Create data loaders with configuration
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    if args.model_type == "medium":
        num_workers = min(8, os.cpu_count())  # Adjust based on system
        persistent_workers = True
        pin_memory = True
        prefetch_factor = 2
    else:
        num_workers = dataset.config["training"]["num_workers"]
        persistent_workers = True
        pin_memory = True
        prefetch_factor = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset._collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset._collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Create experiment directory
    exp_dir = create_experiment_dir(args.exp_dir, args.model_type)
    logger.info(f"Experiment directory: {exp_dir}")

    # Initialize model based on type
    logger.info(f"\nInitializing {args.model_type} model...")
    if args.model_type == "small":
        from small.model_s import SmallSTGCN

        model = SmallSTGCN(**model_config)
    elif args.model_type == "medium":
        from medium.model_m import MediumASTGCN

        model = MediumASTGCN(**model_config)
    else:  # large
        from large.model_l import LargeGMAN

        model = LargeGMAN(**model_config)

    # Get model summary
    summary_str = get_model_summary(model, model_config)

    # Save model architecture and configurations
    with open(exp_dir / "model_architecture.txt", "w") as f:
        f.write("Model Configuration:\n")
        f.write(yaml.dump(model_config))
        f.write("\nModel Architecture:\n")
        f.write(str(model))
        f.write("\n\nModel Summary:\n")
        f.write(summary_str)

    # Log only the model structure
    logger.info("\nModel Architecture:")
    logger.info(str(model))
    logger.info("\nParameter Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train model
    logger.info(f"Starting training for {args.model_type} model...")
    logger.info("GPU Memory Usage before training:")
    print_gpu_utilization()

    if args.model_type == "small":
        model = train_small(train_loader, val_loader, model_config, training_config)
    elif args.model_type == "medium":
        model = train_medium(train_loader, val_loader, model_config, training_config)
    else:  # large
        model = train_large(train_loader, val_loader, model_config, training_config)

    logger.info("GPU Memory Usage after training:")
    print_gpu_utilization()

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": training_config,
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
        default="BA",
        choices=["BA", "ER", "NW"],
        help="Specific graph type to use (default: BA)",
    )

    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Type of model to train",
    )

    # Training parameters - note these are fallback values
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size from config"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs from config"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config",
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

    # Force BA graphs for medium model
    if args.model_type == "medium":
        args.graph_type = "BA"

    main(args)
