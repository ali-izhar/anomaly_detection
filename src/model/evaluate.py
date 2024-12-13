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
        """
        Evaluate model predictions and compute metrics.

        Args:
            data: Can be:
                - tuple (x, edge_indices, edge_weights, y)
                - list of such tuples
                - DynamicGraphTemporalSignalBatch
            phase: Phase name (train/val/test)
            metrics: List of metrics to compute
            visualize: Whether to generate visualizations
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        batch_count = 0
        criterion = nn.BCELoss()

        with torch.no_grad():
            # Handle list of tuples
            if isinstance(data, list):
                for seq_idx, seq_data in enumerate(data):
                    if not isinstance(seq_data, tuple) or len(seq_data) != 4:
                        raise ValueError(
                            f"Invalid sequence data format at index {seq_idx}"
                        )

                    x, edge_indices, edge_weights, y = seq_data

                    # Process in batches
                    num_batches = (
                        len(x) + 32 - 1
                    ) // 32  # Using default batch size of 32

                    # Only visualize first sequence at specific intervals
                    vis_timesteps = (
                        [0, num_batches // 2, num_batches - 1]
                        if visualize and seq_idx == 0
                        else []
                    )

                    for b in range(num_batches):
                        start_idx = b * 32
                        end_idx = min((b + 1) * 32, len(x))

                        batch_x = x[start_idx:end_idx].to(self.device)
                        batch_y = y[start_idx:end_idx].to(self.device)
                        batch_edge_index = edge_indices[end_idx - 1].to(self.device)
                        batch_edge_weight = edge_weights[end_idx - 1].to(self.device)

                        adj_pred, _ = self.model(
                            batch_x, batch_edge_index, batch_edge_weight
                        )

                        loss = criterion(adj_pred, batch_y)
                        total_loss += loss.item()
                        batch_count += 1

                        all_preds.append(adj_pred.cpu())
                        all_targets.append(batch_y.cpu())

                        if visualize and b in vis_timesteps:
                            try:
                                self.visualize_predictions(
                                    batch_y, adj_pred, b, f"{phase}_seq_{seq_idx}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Visualization failed at sequence {seq_idx}, batch {b}: {str(e)}"
                                )

            # Handle single tuple
            elif isinstance(data, tuple) and len(data) == 4:
                x, edge_indices, edge_weights, y = data

                # Process in batches
                num_batches = (len(x) + 32 - 1) // 32  # Using default batch size of 32

                # Only visualize at specific intervals if requested
                vis_timesteps = (
                    [0, num_batches // 2, num_batches - 1] if visualize else []
                )

                for b in range(num_batches):
                    start_idx = b * 32
                    end_idx = min((b + 1) * 32, len(x))

                    batch_x = x[start_idx:end_idx].to(self.device)
                    batch_y = y[start_idx:end_idx].to(self.device)
                    batch_edge_index = edge_indices[end_idx - 1].to(self.device)
                    batch_edge_weight = edge_weights[end_idx - 1].to(self.device)

                    adj_pred, _ = self.model(
                        batch_x, batch_edge_index, batch_edge_weight
                    )

                    loss = criterion(adj_pred, batch_y)
                    total_loss += loss.item()
                    batch_count += 1

                    all_preds.append(adj_pred.cpu())
                    all_targets.append(batch_y.cpu())

                    if visualize and b in vis_timesteps:
                        try:
                            self.visualize_predictions(batch_y, adj_pred, b, phase)
                        except Exception as e:
                            logger.warning(
                                f"Visualization failed at batch {b}: {str(e)}"
                            )

            # Handle DynamicGraphTemporalSignalBatch
            elif hasattr(data, "features"):
                for time, (features, edge_index, edge_weight, target) in enumerate(
                    zip(
                        data.features,
                        data.edge_indices,
                        data.edge_weights,
                        data.targets,
                    )
                ):
                    x = torch.FloatTensor(features).to(self.device)
                    edge_index = torch.LongTensor(edge_index).to(self.device)
                    edge_weight = torch.FloatTensor(edge_weight).to(self.device)
                    y = torch.FloatTensor(target).to(self.device)

                    adj_pred = self.model(x, edge_index, edge_weight)
                    loss = criterion(adj_pred, (y > 0).float())
                    total_loss += loss.item()
                    batch_count += 1

                    all_preds.append(adj_pred.cpu())
                    all_targets.append(y.cpu())

                    if visualize and time in [
                        0,
                        len(data.features) // 2,
                        len(data.features) - 1,
                    ]:
                        self.visualize_predictions(y, adj_pred, time, phase)

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
                results["confusion_matrix"] = confusion_matrix(targets_np, preds_binary)

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
