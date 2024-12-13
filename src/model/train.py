# src/model/train.py

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
    # Set random seed
    if config.get("seed"):
        set_seed(config["seed"])
        logger.info(f"Random seed set to {config['seed']}")

    # Debug mode
    if config["debug"]["enabled"]:
        logger.info("Debug mode enabled")
        sequence_indices = sequence_indices[: config["debug"]["subset_size"]]
        config["training"]["num_epochs"] = 2
        logger.info(f"Using {len(sequence_indices)} sequences for debugging")

    logger.info(f"Starting training with {len(sequence_indices)} sequences")
    logger.info(
        f"Model parameters: lr={config['optimizer']['learning_rate']}, "
        f"batch_size={config['training']['batch_size']}, "
        f"temporal_periods={config['training']['temporal_periods']}"
    )

    model = model.to(device)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)
    criterion = get_loss_function(config)

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Prepare data for all sequences
    logger.info("Preparing data splits...")
    train_data = []
    val_data = []
    test_data = []

    for seq_idx in tqdm(sequence_indices, desc="Processing sequences"):
        x, edge_indices, edge_weights, y = dataset.get_temporal_batch(
            seq_idx, temporal_periods=config["training"]["temporal_periods"]
        )

        num_windows = len(x)
        num_val = int(num_windows * config["training"]["val_ratio"])
        num_test = int(num_windows * 0.2)

        if num_val > 0 and num_test > 0:
            test_data.append(
                (
                    x[-num_test:],
                    edge_indices[-num_test:],
                    edge_weights[-num_test:],
                    y[-num_test:],
                )
            )
            val_data.append(
                (
                    x[-num_test - num_val : -num_test],
                    edge_indices[-num_test - num_val : -num_test],
                    edge_weights[-num_test - num_val : -num_test],
                    y[-num_test - num_val : -num_test],
                )
            )
            train_data.append(
                (
                    x[: -num_test - num_val],
                    edge_indices[: -num_test - num_val],
                    edge_weights[: -num_test - num_val],
                    y[: -num_test - num_val],
                )
            )
        else:
            train_data.append((x, edge_indices, edge_weights, y))

    logger.info(
        f"Data split complete. Train sequences: {len(train_data)}, "
        f"Val sequences: {len(val_data)}, Test sequences: {len(test_data)}"
    )

    # Create evaluator
    evaluator = ModelEvaluator(
        model, device=device, visualization_dir=config["visualization"]["save_dir"]
    )

    # Add more debug logging
    logger.debug(f"Configuration loaded: {config}")
    logger.debug(f"Device: {device}")
    logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        logger.debug(f"\nStarting epoch {epoch+1}")
        model.train()
        total_loss = 0
        batch_count = 0
        epoch_start_time = datetime.now()

        # Training
        train_pbar = tqdm(
            train_data,
            desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Train]",
            disable=not config["logging"]["progress_bar"],
        )

        for seq_idx, seq_data in enumerate(train_pbar):
            logger.debug(f"Processing sequence {seq_idx+1}/{len(train_data)}")
            x, edge_indices, edge_weights, y = seq_data
            sequence_losses = []

            # Process in batches
            num_batches = (len(x) + config["training"]["batch_size"] - 1) // config[
                "training"
            ]["batch_size"]

            for b in range(num_batches):
                start_idx = b * config["training"]["batch_size"]
                end_idx = min((b + 1) * config["training"]["batch_size"], len(x))

                batch_x = x[start_idx:end_idx].to(device)
                batch_y = y[start_idx:end_idx].to(device)
                batch_edge_index = edge_indices[end_idx - 1].to(device)
                batch_edge_weight = edge_weights[end_idx - 1].to(device)

                optimizer.zero_grad()
                adj_pred, _ = model(batch_x, batch_edge_index, batch_edge_weight)
                loss = criterion(adj_pred, batch_y)
                loss.backward()

                if config["gradient_clipping"]["enabled"]:
                    torch_utils.clip_grad_norm_(
                        model.parameters(),
                        config["gradient_clipping"]["max_norm"],
                        norm_type=config["gradient_clipping"]["norm_type"],
                    )

                optimizer.step()

                sequence_losses.append(loss.item())
                total_loss += loss.item()
                batch_count += 1

                # Enhanced debug logging
                if batch_count % config["logging"]["log_interval"] == 0:
                    avg_loss = sum(
                        sequence_losses[-config["logging"]["log_interval"] :]
                    ) / len(sequence_losses[-config["logging"]["log_interval"] :])
                    logger.debug(
                        f"Epoch {epoch+1}, Sequence {seq_idx+1}/{len(train_data)}, "
                        f"Batch {b+1}/{num_batches}: Loss = {avg_loss:.4f}, "
                        f"LR = {optimizer.param_groups[0]['lr']:.6f}"
                    )

            # Update progress bar less frequently
            if seq_idx % max(1, len(train_data) // 10) == 0:
                train_pbar.set_postfix(
                    {
                        "loss": f"{sum(sequence_losses) / len(sequence_losses):.4f}",
                        "avg_loss": f"{total_loss/batch_count:.4f}",
                    }
                )

        # Log epoch summary
        epoch_time = datetime.now() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} completed in {epoch_time}. "
            f"Average loss: {total_loss/batch_count:.4f}"
        )

        # Validation
        val_metrics = evaluator.evaluate_predictions(
            val_data,
            phase="val",
            metrics=config["metrics"]["validation"],
            visualize=(epoch % config["visualization"]["plot_interval"] == 0),
        )

        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]

        # Log validation results
        logger.info(f"Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(
            total_loss / batch_count if batch_count > 0 else float("inf")
        )
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Save checkpoint
        if config["checkpoint"]["save_best"] and val_loss < best_val_loss:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                history,
                config,
                config["checkpoint"]["dir"],
                is_best=True,
            )

        if (
            config["checkpoint"]["save_interval"] > 0
            and epoch % config["checkpoint"]["save_interval"] == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                history,
                config,
                config["checkpoint"]["dir"],
            )

    # Final evaluation
    logger.info("Performing final evaluation...")
    test_metrics = {}

    # Evaluate on all test sequences
    for seq_idx, seq_data in enumerate(test_data):
        logger.info(f"Evaluating test sequence {seq_idx+1}/{len(test_data)}")
        seq_metrics = evaluator.evaluate_predictions(
            seq_data,
            phase=f"test_seq_{seq_idx}",
            metrics=config["metrics"]["test"],
            visualize=True,
        )

        # Aggregate metrics
        for metric, value in seq_metrics.items():
            if metric not in test_metrics:
                test_metrics[metric] = []
            test_metrics[metric].append(value)

    # Average metrics across sequences
    avg_test_metrics = {
        metric: np.mean(values)
        for metric, values in test_metrics.items()
        if metric != "confusion_matrix"
    }

    logger.info("\nAverage Test Results:")
    for metric, value in avg_test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Load best model
    model.load_state_dict(best_model)
    logger.info("Training completed!")
    return model, history, test_metrics


if __name__ == "__main__":
    config = load_config()
    logger = setup_logging(config["logging"])

    try:
        dataset = DynamicGraphDataset(variant=config["data"]["variant"])
        logger.info(f"Dataset loaded: {len(dataset.metadata)} sequences")

        # Get sequence indices using the specified strategy
        train_sequences = get_sequence_indices(
            dataset, config, strategy=config["data"]["selection_strategy"]
        )
        logger.info(f"Selected {len(train_sequences)} sequences for training")

        # Visualize sequence properties
        visualizer = SequenceVisualizer(save_dir=config["visualization"]["save_dir"])
        visualizer.plot_sequence_properties(
            dataset, train_sequences, config, save_prefix="training_sequences"
        )

        # Create model
        model = DynamicLinkPredictor(
            num_nodes=dataset.num_nodes,
            num_features=dataset.num_features,
            hidden_channels=config["model"]["hidden_channels"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            K=config["model"]["K"],
            use_edge_weights=config["model"]["use_edge_weights"],
            attention_heads=config["model"]["attention_heads"],
            attention_dropout=config["model"]["attention_dropout"],
            temporal_periods=config["training"]["temporal_periods"],
            batch_size=config["training"]["batch_size"],
        )
        logger.info(
            f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Train model
        trained_model, history, test_metrics = train_model(
            model,
            dataset,
            train_sequences,
            config,
            device=config["device"],
        )

        # Plot and save training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["val_auc"], label="Validation AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()

        logger.info("Training history plot saved")

    except Exception as e:
        logger.exception("An error occurred during training")
        raise e
