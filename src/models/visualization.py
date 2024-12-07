"""Visualization utilities for analyzing model performance."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_loss_breakdown(
    feature_losses: Dict[str, List[float]],
    save_path: Path,
    title: str = "Feature-wise Loss Evolution",
):
    """Plot individual feature losses over time."""
    plt.figure(figsize=(12, 6))
    for feature, losses in feature_losses.items():
        plt.plot(losses, label=feature)

    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path / "feature_losses.png")
    plt.close()


def plot_prediction_heatmap(
    actual: torch.Tensor,
    predicted: torch.Tensor,
    feature_name: str,
    timestep: int,
    save_path: Path,
):
    """Plot heatmap comparing actual vs predicted values for a specific feature.

    Args:
        actual: Tensor of shape [num_nodes] or [num_nodes, feature_dim]
        predicted: Tensor of shape [num_nodes] or [num_nodes, feature_dim]
        feature_name: Name of the feature being plotted
        timestep: Current timestep
        save_path: Directory to save plots
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Handle different feature dimensions
    if feature_name == "svd":
        # For SVD features, plot first dimension
        actual_np = actual[..., 0].cpu().numpy()
        pred_np = predicted[..., 0].cpu().numpy()
    elif feature_name == "lsvd":
        # For LSVD features, plot first dimension
        actual_np = actual[..., 0].cpu().numpy()
        pred_np = predicted[..., 0].cpu().numpy()
    else:
        # For centrality features
        actual_np = actual.cpu().numpy()
        pred_np = predicted.cpu().numpy()

    # Reshape to 10x10 grid
    actual_np = actual_np.reshape(10, 10)
    pred_np = pred_np.reshape(10, 10)
    diff = actual_np - pred_np

    # Plot actual values
    sns.heatmap(actual_np, ax=ax1, cmap="viridis")
    ax1.set_title(f"Actual {feature_name}\nTimestep {timestep}")

    # Plot predicted values
    sns.heatmap(pred_np, ax=ax2, cmap="viridis")
    ax2.set_title(f"Predicted {feature_name}\nTimestep {timestep}")

    # Plot difference
    sns.heatmap(diff, ax=ax3, cmap="RdBu", center=0)
    ax3.set_title("Difference\n(Actual - Predicted)")

    plt.tight_layout()
    plt.savefig(save_path / f"{feature_name}_heatmap_t{timestep}.png")
    plt.close()


def plot_temporal_predictions(
    actual: torch.Tensor,
    predicted: torch.Tensor,
    feature_name: str,
    node_idx: int,
    save_path: Path,
):
    """Plot temporal evolution of predictions for a specific node and feature.

    Args:
        actual: Tensor of shape [seq_len] or [seq_len, feature_dim]
        predicted: Tensor of shape [seq_len] or [seq_len, feature_dim]
        feature_name: Name of the feature being plotted
        node_idx: Index of the node being plotted
        save_path: Directory to save plots
    """
    plt.figure(figsize=(10, 6))

    # Handle different feature dimensions
    if feature_name in ["svd", "lsvd"]:
        # Plot first dimension for embedding features
        actual_np = actual[..., 0].cpu().numpy()
        pred_np = predicted[..., 0].cpu().numpy()
    else:
        actual_np = actual.cpu().numpy()
        pred_np = predicted.cpu().numpy()

    plt.plot(actual_np, label="Actual", marker="o")
    plt.plot(pred_np, label="Predicted", marker="s")
    plt.fill_between(
        range(len(actual_np)), actual_np, pred_np, alpha=0.2, label="Difference"
    )

    plt.xlabel("Time Step")
    plt.ylabel(
        f"{feature_name} (dim 0)" if feature_name in ["svd", "lsvd"] else feature_name
    )
    plt.title(f"Temporal Evolution - Node {node_idx}")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path / f"{feature_name}_temporal_node{node_idx}.png")
    plt.close()


def plot_feature_correlations(
    predictions: Dict[str, torch.Tensor],
    actual: Dict[str, torch.Tensor],
    save_path: Path,
):
    """Plot correlation matrix between different features."""
    # Get first timestep and first batch for correlation analysis
    pred_features = []
    actual_features = []
    feature_names = []

    for feat_name in predictions:
        if feat_name == "svd":
            # Handle 2D SVD features
            pred_feat = (
                predictions[feat_name][0, 0, :, 0].cpu().numpy()
            )  # First dimension only
            actual_feat = actual[feat_name][0, 0, :, 0].cpu().numpy()
        elif feat_name == "lsvd":
            # Handle 16D LSVD features
            pred_feat = (
                predictions[feat_name][0, 0, :, 0].cpu().numpy()
            )  # First dimension only
            actual_feat = actual[feat_name][0, 0, :, 0].cpu().numpy()
        else:
            # Handle centrality features
            pred_feat = predictions[feat_name][0, 0].cpu().numpy()
            actual_feat = actual[feat_name][0, 0].cpu().numpy()

        pred_features.append(pred_feat)
        actual_features.append(actual_feat)
        feature_names.append(f"{feat_name}_pred")
        feature_names.append(f"{feat_name}_actual")

    # Create correlation matrix
    all_features = np.column_stack(pred_features + actual_features)
    corr_matrix = np.corrcoef(all_features.T)

    # Plot correlation matrix
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="RdBu",
        center=0,
        annot=True,
        fmt=".2f",
    )
    plt.title("Feature Correlations (Predicted vs Actual)\nFirst Timestep")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path / "feature_correlations.png", bbox_inches="tight")
    plt.close()


def analyze_predictions(
    batch: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    save_path: Path,
    timestep: int = 0,
):
    """Comprehensive analysis of model predictions."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Analyze each feature type
    for feat_name in predictions:
        try:
            # Plot heatmap for first timestep
            plot_prediction_heatmap(
                batch["y"][feat_name][0, timestep],  # First batch, specified timestep
                predictions[feat_name][0, timestep],  # First batch, specified timestep
                feat_name,
                timestep,
                save_path,
            )

            # Plot temporal evolution for first node
            plot_temporal_predictions(
                batch["y"][feat_name][
                    0, :, 0
                ],  # First batch, all timesteps, first node
                predictions[feat_name][
                    0, :, 0
                ],  # First batch, all timesteps, first node
                feat_name,
                node_idx=0,
                save_path=save_path,
            )
        except Exception as e:
            logger.warning(f"Error plotting {feat_name}: {str(e)}")

    try:
        # Plot feature correlations
        plot_feature_correlations(predictions, batch["y"], save_path)
    except Exception as e:
        logger.warning(f"Error plotting correlations: {str(e)}")
