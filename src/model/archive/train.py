import sys
import yaml
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import torch
import torch.nn.utils as torch_utils
import torch.cuda as cuda
from torch.cuda import amp  # For mixed precision training

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dataset import DynamicGraphDataset
from link_predictor import DynamicLinkPredictor
from utils.visualize import SequenceVisualizer
from utils.helpers import (
    set_seed,
    get_optimizer,
    get_scheduler,
    get_loss_function,
    save_checkpoint,
)
from evaluate import ModelEvaluator
from utils.logger import setup_logging


def load_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def monitor_graph_types(metrics, graph_type):
    """
    Log and aggregate metrics for distinct graph types.

    Args:
        metrics (dict): Metrics dictionary.
        graph_type (str): Graph type identifier (e.g., "BA", "ER", "NW").
    """
    logger.info(f"Metrics for graph type {graph_type}:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")


def get_sequence_indices(
    dataset: DynamicGraphDataset, config: Dict[str, Any], strategy: str = "balanced"
) -> np.ndarray:
    """
    Get sequence indices based on different selection strategies.

    Strategies:
    - balanced: Equal number of sequences from each graph type
    - proportional: Number of sequences proportional to available sequences
    - random: Random selection across all sequences
    - change_points: Select sequences based on number of change points
    - difficulty: Select sequences based on graph complexity
    """
    available_sequences = {
        graph_type: dataset.get_graph_type_indices(graph_type)
        for graph_type in config["data"]["graph_types"]
    }

    # Log available sequences
    for graph_type, indices in available_sequences.items():
        logger.info(f"Available {graph_type} sequences: {len(indices)}")

    total_sequences = config["data"]["total_sequences"]
    if config["debug"]["enabled"]:
        total_sequences = min(total_sequences, config["debug"]["subset_size"])

    selected_indices = []

    if strategy == "balanced":
        # Equal number from each type
        sequences_per_type = total_sequences // len(config["data"]["graph_types"])
        for graph_type in config["data"]["graph_types"]:
            indices = available_sequences[graph_type]
            selected = indices[:sequences_per_type]
            selected_indices.extend(selected)

    elif strategy == "proportional":
        # Proportional to available sequences
        total_available = sum(len(indices) for indices in available_sequences.values())
        for graph_type in config["data"]["graph_types"]:
            indices = available_sequences[graph_type]
            num_sequences = int((len(indices) / total_available) * total_sequences)
            selected = indices[:num_sequences]
            selected_indices.extend(selected)

    elif strategy == "random":
        # Random selection across all types
        all_indices = np.concatenate(
            [indices for indices in available_sequences.values()]
        )
        np.random.shuffle(all_indices)
        selected_indices = all_indices[:total_sequences]

    elif strategy == "change_points":
        # Select based on number of change points
        all_sequences = []
        for graph_type, indices in available_sequences.items():
            for idx in indices:
                change_points = dataset.get_change_points(idx)
                all_sequences.append((idx, len(change_points)))

        # Sort by number of change points
        all_sequences.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [seq[0] for seq in all_sequences[:total_sequences]]

    elif strategy == "difficulty":
        # Select based on graph complexity (e.g., edge density)
        all_sequences = []
        for graph_type, indices in available_sequences.items():
            for idx in indices:
                # Calculate average edge density
                adj_matrices = dataset.adjacency_matrices[idx]
                density = np.mean(
                    [
                        (adj > 0).sum() / (adj.shape[0] * adj.shape[1])
                        for adj in adj_matrices
                    ]
                )
                all_sequences.append((idx, density))

        # Sort by complexity
        all_sequences.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [seq[0] for seq in all_sequences[:total_sequences]]

    else:
        raise ValueError(f"Unknown sequence selection strategy: {strategy}")

    # Log selection statistics
    selected_types = {
        graph_type: sum(
            1
            for idx in selected_indices
            if dataset.metadata[idx]["graph_type"] == graph_type
        )
        for graph_type in config["data"]["graph_types"]
    }
    logger.info(f"Selection strategy: {strategy}")
    logger.info("Selected sequences per type:")
    for graph_type, count in selected_types.items():
        logger.info(f"  {graph_type}: {count} sequences")

    return np.array(selected_indices)


def train_model(
    model,
    dataset,
    sequence_indices,
    config: Dict[str, Any],
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the model with temporal sequences."""
    # GPU setup and logging
    if device == "cuda":
        gpu_info = cuda.get_device_properties(0)
        logger.info(f"Using GPU: {gpu_info.name}")
        logger.info(f"Memory: {gpu_info.total_memory / 1024**2:.0f} MB")

    model = model.to(device)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)
    criterion = get_loss_function(config).to(device)
    scaler = amp.GradScaler()

    logger.info("Starting training")

    history = {"train_loss": [], "val_loss": [], "test_metrics": {}}

    evaluator = ModelEvaluator(model, device=device)

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0

        for seq_idx in tqdm(sequence_indices, desc=f"Epoch {epoch+1}"):
            x, edge_index, edge_weight, y = dataset.get_temporal_batch(
                seq_idx, temporal_periods=config["model"]["temporal_periods"]
            )
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=True):
                adj_pred, _ = model(x, edge_index, edge_weight)
                loss = criterion(adj_pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(sequence_indices)
        logger.info(f"Epoch {epoch+1}: Avg Train Loss = {avg_loss:.4f}")
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        val_metrics = evaluator.evaluate_sequence(
            dataset, config["data"]["val_indices"], config
        )
        logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
        history["val_loss"].append(val_metrics["loss"])

        # Monitor metrics for graph types
        for graph_type in config["data"]["graph_types"]:
            graph_type_indices = dataset.get_graph_type_indices(graph_type)
            graph_type_metrics = evaluator.evaluate_sequence(
                dataset, graph_type_indices, config
            )
            monitor_graph_types(graph_type_metrics, graph_type)
            history["test_metrics"].setdefault(graph_type, []).append(
                graph_type_metrics
            )

        scheduler.step(val_metrics["loss"])

        # Save checkpoint
        if epoch % config["training"]["save_interval"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"], history, config
            )

    return model, history


if __name__ == "__main__":
    config = load_config()
    logger = setup_logging(config["logging"])
    dataset = DynamicGraphDataset(config["data"]["variant"])

    train_indices = get_sequence_indices(dataset, config, strategy="balanced")

    model = DynamicLinkPredictor(
        num_nodes=dataset.num_nodes,
        num_features=dataset.num_features,
        hidden_channels=config["model"]["hidden_channels"],
        num_layers=config["model"]["num_layers"],
    )

    trained_model, history = train_model(model, dataset, train_indices, config)

    # Save final metrics
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig("training_metrics.png")
