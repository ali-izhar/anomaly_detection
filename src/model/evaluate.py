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
import torch.nn.functional as F
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
        visualize: int = -1,  # -1 for no viz, n for first n predictions
    ) -> Dict:
        """Evaluate model predictions and compute metrics."""
        self.model.eval()
        all_preds = []
        all_targets = []
        sequence_losses = []

        logger.info(f"\nEvaluating {phase} phase:")

        with torch.no_grad():
            if isinstance(data, tuple):
                x, edge_indices, edge_weights, y = data
                num_batches = (len(x) + 32 - 1) // 32  # Using default batch size of 32

                seq_pbar = tqdm(
                    range(num_batches),
                    desc=f"Sequence evaluation",
                    leave=False,
                )

                for b in seq_pbar:
                    start_idx = b * 32
                    end_idx = min((b + 1) * 32, len(x))

                    # Get batch data
                    batch_x = x[start_idx:end_idx].to(self.model.device)
                    batch_y = y[start_idx:end_idx].to(self.model.device)
                    batch_edge_index = edge_indices[end_idx - 1].to(self.model.device)
                    batch_edge_weight = edge_weights[end_idx - 1].to(self.model.device)

                    # Get predictions
                    probs, _ = self.model(batch_x, batch_edge_index, batch_edge_weight)

                    # Compute loss
                    loss = F.binary_cross_entropy_with_logits(probs, batch_y)
                    sequence_losses.append(loss.item())

                    # Store predictions and targets
                    all_preds.append(torch.sigmoid(probs))
                    all_targets.append(batch_y)

                    # Visualize if requested
                    if visualize >= 0 and b < visualize:
                        self.visualize_prediction(
                            batch_y[0],
                            probs[0],
                            b,
                            phase,
                            edge_index=batch_edge_index,
                            edge_weight=batch_edge_weight,
                        )

                    # Update progress bar with stats
                    pred_edges = (torch.sigmoid(probs) > 0.5).float().mean().item()
                    true_edges = batch_y.float().mean().item()
                    seq_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "pred_density": f"{pred_edges:.3f}",
                            "true_density": f"{true_edges:.3f}",
                        }
                    )

            else:
                # Handle dataset case
                for seq_idx, seq_data in enumerate(
                    tqdm(data, desc=f"Evaluating {phase}")
                ):
                    # ... (rest of dataset evaluation code)
                    pass

        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        results = self.compute_metrics(all_preds, all_targets, sequence_losses)

        # Log detailed results
        logger.info("\nEvaluation Results:")
        for metric, value in results.items():
            if metric != "confusion_matrix":
                logger.info(f"{metric}: {value:.4f}")

        if "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            logger.info("\nConfusion Matrix:")
            logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

            # Calculate additional statistics
            total = cm.sum()
            accuracy = (cm[0, 0] + cm[1, 1]) / total
            precision = (
                cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            )
            recall = (
                cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            )

            logger.info("\nDetailed Statistics:")
            logger.info(f"Total Edges: {total}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(
                f"F1 Score: {2 * precision * recall / (precision + recall):.4f}"
            )

        return results

    def visualize_prediction(
        self, y_true, y_pred, index, phase, edge_index=None, edge_weight=None
    ):
        """Visualize true vs predicted adjacency matrices."""
        plt.figure(figsize=(15, 5))

        # True Graph
        plt.subplot(131)
        plt.imshow(y_true.cpu().numpy(), cmap="RdBu")
        plt.title(f"True Graph\nEdge Density: {y_true.float().mean().item():.3f}")
        plt.colorbar()

        # Predicted Probabilities
        plt.subplot(132)
        pred_probs = torch.sigmoid(y_pred).cpu().numpy()
        plt.imshow(pred_probs, cmap="RdBu")
        plt.title(f"Predicted Probabilities\nMean Prob: {pred_probs.mean():.3f}")
        plt.colorbar()

        # Thresholded Predictions
        plt.subplot(133)
        pred_binary = (pred_probs > 0.5).astype(float)
        plt.imshow(pred_binary, cmap="RdBu")
        plt.title(f"Thresholded Predictions\nEdge Density: {pred_binary.mean():.3f}")
        plt.colorbar()

        plt.suptitle(f"{phase} - Sample {index}")
        plt.tight_layout()

        # Save the figure
        save_path = self.vis_dir / f"{phase}_prediction_{index}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved visualization to {save_path}")

    def compute_metrics(self, all_preds, all_targets, sequence_losses):
        # Convert to numpy for metric computation
        preds_np = all_preds.numpy().ravel()
        targets_np = all_targets.numpy().ravel()
        preds_binary = (all_preds > 0.5).float().numpy().ravel()

        # Compute metrics
        results = {
            "loss": (
                sum(sequence_losses) / len(sequence_losses)
                if sequence_losses
                else float("inf")
            )
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
