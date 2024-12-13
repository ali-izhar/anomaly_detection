# src/model/evaluate.py

import logging

from typing import Dict, List, Optional
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating link prediction models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        visualization_dir: str = "visualizations/evaluation",
    ):
        self.model = model
        self.device = device
        self.vis_dir = Path(visualization_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self,
        data,
        phase: str = "test",
        metrics: Optional[List[str]] = None,
        visualize: bool = False,
    ) -> Dict:
        """Evaluate model predictions and compute metrics."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        batch_count = 0
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            if isinstance(data, list):
                # Create progress bar for sequences
                seq_pbar = tqdm(
                    enumerate(data),
                    total=len(data),
                    desc=f"Evaluating {phase}",
                    leave=False,
                )

                for seq_idx, seq_data in seq_pbar:
                    if not isinstance(seq_data, tuple) or len(seq_data) != 4:
                        raise ValueError(
                            f"Invalid sequence data format at index {seq_idx}"
                        )

                    x, edge_indices, edge_weights, y = seq_data
                    num_batches = (len(x) + 32 - 1) // 32
                    sequence_losses = []  # Track losses for this sequence

                    # Create progress bar for batches
                    batch_pbar = tqdm(
                        range(num_batches),
                        desc=f"Sequence {seq_idx+1}/{len(data)}",
                        leave=False,
                    )

                    for b in batch_pbar:
                        start_idx = b * 32
                        end_idx = min((b + 1) * 32, len(x))

                        batch_x = x[start_idx:end_idx].to(self.device)
                        batch_y = y[start_idx:end_idx].to(self.device)
                        batch_edge_index = edge_indices[end_idx - 1].to(self.device)
                        batch_edge_weight = edge_weights[end_idx - 1].to(self.device)

                        # Get logits (not probabilities)
                        logits, _ = self.model(
                            batch_x, batch_edge_index, batch_edge_weight
                        )

                        # Compute loss with logits
                        loss = criterion(logits, batch_y)
                        total_loss += loss.item()
                        sequence_losses.append(loss.item())
                        batch_count += 1

                        # Convert to probabilities for metrics
                        probs = torch.sigmoid(logits)
                        all_preds.append(probs.cpu())
                        all_targets.append(batch_y.cpu())

                        # Calculate edge statistics
                        pred_edges = (probs > 0.5).float().mean().item()
                        true_edges = batch_y.float().mean().item()

                        if visualize and b in [0, num_batches // 2, num_batches - 1]:
                            try:
                                self.visualize_predictions(
                                    batch_y, probs, b, f"{phase}_seq_{seq_idx}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Visualization failed at sequence {seq_idx}, batch {b}: {str(e)}"
                                )

                        # Update batch progress
                        batch_pbar.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "pred_edges": f"{pred_edges:.3f}",
                                "true_edges": f"{true_edges:.3f}",
                            }
                        )

                    # Update sequence progress
                    avg_sequence_loss = sum(sequence_losses) / len(sequence_losses)
                    seq_pbar.set_postfix(
                        {
                            "avg_loss": f"{avg_sequence_loss:.4f}",
                            "batch_count": batch_count,
                        }
                    )

            # Handle single tuple
            elif isinstance(data, tuple) and len(data) == 4:
                x, edge_indices, edge_weights, y = data
                sequence_losses = []  # Track losses for this sequence

                # Process in batches
                num_batches = (len(x) + 32 - 1) // 32

                # Create progress bar for batches
                batch_pbar = tqdm(
                    range(num_batches), desc=f"Processing {phase}", leave=False
                )

                for b in batch_pbar:
                    start_idx = b * 32
                    end_idx = min((b + 1) * 32, len(x))

                    batch_x = x[start_idx:end_idx].to(self.device)
                    batch_y = y[start_idx:end_idx].to(self.device)
                    batch_edge_index = edge_indices[end_idx - 1].to(self.device)
                    batch_edge_weight = edge_weights[end_idx - 1].to(self.device)

                    # Get logits and compute loss
                    logits, _ = self.model(batch_x, batch_edge_index, batch_edge_weight)
                    loss = criterion(logits, batch_y)
                    total_loss += loss.item()
                    sequence_losses.append(loss.item())
                    batch_count += 1

                    # Convert to probabilities for metrics
                    probs = torch.sigmoid(logits)
                    all_preds.append(probs.cpu())
                    all_targets.append(batch_y.cpu())

                    # Calculate edge statistics
                    pred_edges = (probs > 0.5).float().mean().item()
                    true_edges = batch_y.float().mean().item()

                    # Update progress bar
                    batch_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "pred_edges": f"{pred_edges:.3f}",
                            "true_edges": f"{true_edges:.3f}",
                        }
                    )

                    if visualize and b in [0, num_batches // 2, num_batches - 1]:
                        self.visualize_predictions(batch_y, probs, b, phase)

            # Handle DynamicGraphTemporalSignalBatch
            elif hasattr(data, "features"):
                sequence_losses = []

                # Create progress bar
                time_pbar = tqdm(
                    enumerate(
                        zip(
                            data.features,
                            data.edge_indices,
                            data.edge_weights,
                            data.targets,
                        )
                    ),
                    total=len(data.features),
                    desc=f"Processing {phase}",
                    leave=False,
                )

                for time, (features, edge_index, edge_weight, target) in time_pbar:
                    x = torch.FloatTensor(features).to(self.device)
                    edge_index = torch.LongTensor(edge_index).to(self.device)
                    edge_weight = torch.FloatTensor(edge_weight).to(self.device)
                    y = torch.FloatTensor(target).to(self.device)

                    # Get predictions and compute loss
                    logits = self.model(x, edge_index, edge_weight)
                    loss = criterion(logits, (y > 0).float())
                    total_loss += loss.item()
                    sequence_losses.append(loss.item())
                    batch_count += 1

                    # Convert to probabilities for metrics
                    probs = torch.sigmoid(logits)
                    all_preds.append(probs.cpu())
                    all_targets.append(y.cpu())

                    # Calculate edge statistics
                    pred_edges = (probs > 0.5).float().mean().item()
                    true_edges = y.float().mean().item()

                    # Update progress bar
                    time_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "pred_edges": f"{pred_edges:.3f}",
                            "true_edges": f"{true_edges:.3f}",
                        }
                    )

                    if visualize and time in [
                        0,
                        len(data.features) // 2,
                        len(data.features) - 1,
                    ]:
                        self.visualize_predictions(y, probs, time, phase)

            else:
                raise ValueError(f"Unsupported data format: {type(data)}")

            # Stack predictions and targets
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Convert to numpy for metric computation
            preds_np = all_preds.numpy().ravel()
            targets_np = all_targets.numpy().ravel()
            preds_binary = (all_preds > 0.5).float().numpy().ravel()

            # Compute metrics
            results = {
                "loss": total_loss / batch_count if batch_count > 0 else float("inf")
            }

            if metrics is None:
                metrics = ["auc", "precision", "recall", "f1"]

            for metric in metrics:
                if metric == "auc":
                    results["auc"] = roc_auc_score(targets_np, preds_np)
                elif metric == "ap":
                    results["ap"] = average_precision_score(targets_np, preds_np)
                elif metric == "precision":
                    results["precision"] = precision_score(
                        targets_np, preds_binary, zero_division=0
                    )
                elif metric == "recall":
                    results["recall"] = recall_score(
                        targets_np, preds_binary, zero_division=0
                    )
                elif metric == "f1":
                    results["f1"] = f1_score(targets_np, preds_binary, zero_division=0)
                elif metric == "confusion_matrix":
                    results["confusion_matrix"] = confusion_matrix(
                        targets_np, preds_binary
                    )

            # Log results
            logger.info(f"\n{phase.capitalize()} Results:")
            for metric, value in results.items():
                if metric != "confusion_matrix":
                    logger.info(f"{metric}: {value:.4f}")

            return results

    def visualize_predictions(
        self,
        true_adj: torch.Tensor,
        pred_adj: torch.Tensor,
        timestep: int,
        phase: str = "test",
    ):
        """
        Visualize original and predicted graphs side by side.

        Args:
            true_adj: True adjacency matrix (batch_size, num_nodes, num_nodes)
            pred_adj: Predicted adjacency matrix (batch_size, num_nodes, num_nodes)
            timestep: Current timestep
            phase: Phase name (train/val/test)
        """
        # Take first graph from batch for visualization
        if true_adj.dim() == 3:
            true_adj = true_adj[0]  # Take first graph from batch
        if pred_adj.dim() == 3:
            pred_adj = pred_adj[0]  # Take first graph from batch

        # Ensure values are in [0,1] range
        true_adj = true_adj.clamp(0, 1)
        pred_adj = pred_adj.clamp(0, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Create NetworkX graphs
        G_true = nx.from_numpy_array(true_adj.cpu().numpy())
        G_pred = nx.from_numpy_array((pred_adj > 0.5).cpu().numpy().astype(float))

        # Use same layout for both graphs
        pos = nx.spring_layout(G_true)

        # Draw true graph
        nx.draw(
            G_true,
            pos,
            ax=ax1,
            node_color="lightblue",
            node_size=500,
            with_labels=True,
            width=1.5,
            edge_color="gray",
        )
        ax1.set_title(f"True Graph (t={timestep})")

        # Draw predicted graph
        nx.draw(
            G_pred,
            pos,
            ax=ax2,
            node_color="lightgreen",
            node_size=500,
            with_labels=True,
            width=1.5,
            edge_color="gray",
        )
        ax2.set_title(f"Predicted Graph (t={timestep})")

        # Add metrics to the plot
        true_edges = true_adj.sum().item()
        pred_edges = (pred_adj > 0.5).sum().item()
        plt.figtext(
            0.02,
            0.02,
            f"True edges: {true_edges}\nPredicted edges: {pred_edges}",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Save figure
        plt.savefig(
            self.vis_dir / f"{phase}_comparison_t{timestep}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Log visualization info
        logger.debug(
            f"Visualization saved for {phase} at timestep {timestep}. "
            f"True edges: {true_edges}, Predicted edges: {pred_edges}"
        )

    def evaluate_sequence(
        self,
        dataset,
        sequence_idx: int,
        config: Dict,
        save_prefix: Optional[str] = None,
    ) -> Dict:
        """Evaluate model on a specific sequence."""
        # Get train/val/test splits
        train_data, val_data, test_data = dataset.get_train_val_test_split(sequence_idx)

        results = {}
        # Evaluate each split
        for phase, data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            phase_results = self.evaluate_predictions(
                data,
                phase=phase,
                metrics=config["metrics"][phase],
                visualize=True,
            )
            results[phase] = phase_results

            # Save visualizations if requested
            if save_prefix:
                self.plot_metrics(results, save_prefix=save_prefix)

        return results

    def plot_metrics(self, results: Dict, save_prefix: str):
        """Plot evaluation metrics."""
        phases = list(results.keys())
        metrics = [m for m in results[phases[0]].keys() if m != "confusion_matrix"]

        for metric in metrics:
            plt.figure(figsize=(8, 6))
            for phase in phases:
                if metric in results[phase]:
                    plt.plot([results[phase][metric]], label=f"{phase}")

            plt.title(f"{metric.upper()} by Phase")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.vis_dir / f"{save_prefix}_{metric}.png")
            plt.close()
