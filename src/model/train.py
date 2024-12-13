import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, Any, Optional

from dataset import DynamicGraphDataset
from link_predictor import DynamicLinkPredictor

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Set up logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also print to console
        ],
    )


def visualize_graphs(original_adj, predicted_adj, timestep, save_path=None):
    """
    Visualize original and predicted graphs side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create NetworkX graphs
    G_orig = nx.from_numpy_array(original_adj.cpu().numpy())
    G_pred = nx.from_numpy_array((predicted_adj > 0.5).cpu().numpy().astype(float))

    # Draw original graph
    pos = nx.spring_layout(G_orig)
    nx.draw(
        G_orig, pos, ax=ax1, node_color="lightblue", node_size=500, with_labels=True
    )
    ax1.set_title(f"Original Graph (t={timestep})")

    # Draw predicted graph
    nx.draw(
        G_pred, pos, ax=ax2, node_color="lightgreen", node_size=500, with_labels=True
    )
    ax2.set_title(f"Predicted Graph (t={timestep})")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def evaluate_predictions(
    model, dataset, device, phase="test", visualize=False, full_metrics=False
):
    """
    Evaluate model predictions and compute metrics.
    visualize: bool, whether to generate graph visualizations
    full_metrics: bool, whether to compute all metrics or just loss and AUC
    """
    model.eval()
    all_preds = []
    all_targets = []
    metrics = {}
    total_loss = 0
    batch_count = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for time, snapshot in enumerate(dataset):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = (
                snapshot.edge_attr.to(device)
                if snapshot.edge_attr is not None
                else None
            )
            y = snapshot.y.to(device)

            adj_pred = model(x, edge_index, edge_weight)
            loss = criterion(adj_pred, (y > 0).float())
            total_loss += loss.item()
            batch_count += 1

            all_preds.append(adj_pred.cpu())
            all_targets.append(y.cpu())

            if visualize and time in [0, len(all_preds) // 2, len(all_preds) - 1]:
                visualize_graphs(y, adj_pred, time, f"graphs_{phase}_t{time}.png")

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    all_preds_np = all_preds.numpy().ravel()
    all_targets_np = all_targets.numpy().ravel()
    preds_binary = (all_preds > 0.5).float().numpy().ravel()

    # Basic metrics (always computed)
    metrics["loss"] = total_loss / batch_count if batch_count > 0 else float("inf")
    metrics["auc"] = roc_auc_score(all_targets_np, all_preds_np)

    # Full metrics (only if requested)
    if full_metrics:
        metrics["precision"] = precision_score(
            all_targets_np, preds_binary, zero_division=0
        )
        metrics["recall"] = recall_score(all_targets_np, preds_binary, zero_division=0)
        metrics["f1"] = f1_score(all_targets_np, preds_binary, zero_division=0)

    return metrics


def load_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


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

    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "history": history,
        "config": config,
    }

    # Save checkpoint
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


def train_model(
    model,
    dataset,
    sequence_indices,
    config: Dict[str, Any],
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the model with temporal sequences."""
    logger.info(f"Starting training with {len(sequence_indices)} sequences")
    logger.info(
        f"Model parameters: lr={config['training']['learning_rate']}, "
        f"batch_size={config['training']['batch_size']}, "
        f"temporal_periods={config['training']['temporal_periods']}"
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=config["training"]["patience"] // 2
    )
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Prepare data for all sequences
    logger.info("Preparing data splits...")
    train_data = []
    val_data = []

    for seq_idx in tqdm(sequence_indices, desc="Processing sequences"):
        x, edge_indices, edge_weights, y = dataset.get_temporal_batch(
            seq_idx, temporal_periods=config["training"]["temporal_periods"]
        )

        num_windows = len(x)
        num_val = int(num_windows * config["training"]["val_ratio"])

        if num_val > 0:
            val_data.append(
                (
                    x[-num_val:],
                    edge_indices[-num_val:],
                    edge_weights[-num_val:],
                    y[-num_val:],
                )
            )
            train_data.append(
                (
                    x[:-num_val],
                    edge_indices[:-num_val],
                    edge_weights[:-num_val],
                    y[:-num_val],
                )
            )
        else:
            train_data.append((x, edge_indices, edge_weights, y))

    logger.info(
        f"Data split complete. Train sequences: {len(train_data)}, "
        f"Val sequences: {len(val_data)}"
    )

    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        batch_count = 0

        # Training
        train_pbar = tqdm(
            train_data,
            desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Train]",
        )
        for seq_idx, seq_data in enumerate(train_pbar):
            x, edge_indices, edge_weights, y = seq_data

            # Process in batches
            num_batches = (len(x) + config["training"]["batch_size"] - 1) // config[
                "training"
            ]["batch_size"]
            batch_losses = []

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
                optimizer.step()

                batch_losses.append(loss.item())
                total_loss += loss.item()
                batch_count += 1

                # Update progress bar
                if batch_count % config["logging"]["log_interval"] == 0:
                    avg_loss = (
                        sum(batch_losses[-config["logging"]["log_interval"] :])
                        / config["logging"]["log_interval"]
                    )
                    train_pbar.set_postfix(
                        {
                            "batch_loss": f"{avg_loss:.4f}",
                            "avg_loss": f"{total_loss/batch_count:.4f}",
                        }
                    )

        avg_train_loss = total_loss / batch_count if batch_count > 0 else float("inf")

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            val_pbar = tqdm(
                val_data,
                desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Val]",
            )
            for seq_data in val_pbar:
                x, edge_indices, edge_weights, y = seq_data

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

                    adj_pred, _ = model(batch_x, batch_edge_index, batch_edge_weight)
                    batch_loss = criterion(adj_pred, batch_y).item()

                    val_loss += batch_loss
                    val_preds.append(adj_pred.cpu())
                    val_targets.append(batch_y.cpu())

                    val_pbar.set_postfix({"val_loss": f"{batch_loss:.4f}"})

        val_loss /= len(val_data)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_auc = roc_auc_score(val_targets.numpy().ravel(), val_preds.numpy().ravel())

        # Log epoch results
        logger.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val AUC: {val_auc:.4f}"
        )

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
        history["train_loss"].append(avg_train_loss)
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

    # Load best model
    model.load_state_dict(best_model)
    logger.info("Training completed!")
    return model, history


if __name__ == "__main__":
    # Load config
    config = load_config()

    # Set up logging
    setup_logging(log_dir=config["logging"]["dir"], level=config["logging"]["level"])
    logger.info("Starting script")

    try:
        # Create dataset
        dataset = DynamicGraphDataset(variant=config["data"]["variant"])
        logger.info(f"Dataset loaded: {len(dataset.metadata)} sequences")

        # Get sequence indices
        train_sequences = []
        for graph_type in config["data"]["graph_types"]:
            indices = dataset.get_graph_type_indices(graph_type)
            train_sequences.extend(indices[: config["data"]["sequences_per_type"]])
        train_sequences = np.array(train_sequences)
        logger.info(f"Training with {len(train_sequences)} sequences")

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
        trained_model, history = train_model(
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
