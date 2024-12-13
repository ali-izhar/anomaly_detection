# src/model/utils/visualize.py

import logging

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


logger = logging.getLogger(__name__)


class SequenceVisualizer:
    """Utility class for visualizing sequence distributions and properties."""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_sequence_distribution(
        self,
        dataset,
        selected_indices: np.ndarray,
        config: Dict[str, Any],
        save_name: str = "sequence_distribution.png",
    ):
        """Plot distribution of selected sequences across graph types."""
        # Get graph type counts
        all_counts = defaultdict(lambda: {"total": 0, "selected": 0})

        # Count total sequences per type
        for idx in range(len(dataset.metadata)):
            graph_type = dataset.metadata[idx]["graph_type"]
            all_counts[graph_type]["total"] += 1

        # Count selected sequences per type
        for idx in selected_indices:
            graph_type = dataset.metadata[idx]["graph_type"]
            all_counts[graph_type]["selected"] += 1

        # Prepare data for plotting
        graph_types = list(all_counts.keys())
        total_counts = [all_counts[gt]["total"] for gt in graph_types]
        selected_counts = [all_counts[gt]["selected"] for gt in graph_types]

        # Create bar plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(graph_types))
        width = 0.35

        plt.bar(x - width / 2, total_counts, width, label="Total Available")
        plt.bar(x + width / 2, selected_counts, width, label="Selected")

        plt.xlabel("Graph Type")
        plt.ylabel("Number of Sequences")
        plt.title(
            f'Sequence Distribution ({config["data"]["selection_strategy"]} strategy)'
        )
        plt.xticks(x, graph_types)
        plt.legend()

        # Add value labels
        for i, v in enumerate(total_counts):
            plt.text(i - width / 2, v, str(v), ha="center", va="bottom")
        for i, v in enumerate(selected_counts):
            plt.text(i + width / 2, v, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()

        logger.info(f"Sequence distribution plot saved to {save_name}")

    def plot_change_point_distribution(
        self,
        dataset,
        selected_indices: np.ndarray,
        config: Dict[str, Any],
        save_name: str = "change_point_distribution.png",
    ):
        """Plot distribution of change points in selected sequences."""
        change_points_selected = [
            len(dataset.get_change_points(idx)) for idx in selected_indices
        ]

        plt.figure(figsize=(10, 6))

        # Plot histogram
        plt.hist(
            change_points_selected,
            bins=range(max(change_points_selected) + 2),
            alpha=0.7,
            rwidth=0.8,
        )

        plt.xlabel("Number of Change Points")
        plt.ylabel("Number of Sequences")
        plt.title("Distribution of Change Points in Selected Sequences")

        # Add mean line
        mean_changes = np.mean(change_points_selected)
        plt.axvline(
            mean_changes,
            color="r",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_changes:.2f}",
        )

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()

        logger.info(f"Change point distribution plot saved to {save_name}")

    def plot_edge_density_distribution(
        self,
        dataset,
        selected_indices: np.ndarray,
        config: Dict[str, Any],
        save_name: str = "edge_density_distribution.png",
    ):
        """Plot distribution of edge densities in selected sequences."""
        densities = []
        for idx in selected_indices:
            adj_matrices = dataset.adjacency_matrices[idx]
            seq_densities = [
                (adj > 0).sum() / (adj.shape[0] * adj.shape[1]) for adj in adj_matrices
            ]
            densities.append(np.mean(seq_densities))

        plt.figure(figsize=(10, 6))

        # Plot density distribution
        sns.histplot(densities, bins=20, kde=True)

        plt.xlabel("Average Edge Density")
        plt.ylabel("Number of Sequences")
        plt.title("Distribution of Edge Densities in Selected Sequences")

        # Add mean and std lines
        mean_density = np.mean(densities)
        std_density = np.std(densities)
        plt.axvline(
            mean_density,
            color="r",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_density:.3f}",
        )
        plt.axvline(
            mean_density + std_density,
            color="g",
            linestyle=":",
            linewidth=2,
            label=f"±1 STD: {std_density:.3f}",
        )
        plt.axvline(mean_density - std_density, color="g", linestyle=":", linewidth=2)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()

        logger.info(f"Edge density distribution plot saved to {save_name}")

    def plot_sequence_properties(
        self,
        dataset,
        selected_indices: np.ndarray,
        config: Dict[str, Any],
        save_prefix: str = "sequence_analysis",
    ):
        """Generate all sequence analysis plots."""
        self.plot_sequence_distribution(
            dataset, selected_indices, config, f"{save_prefix}_distribution.png"
        )
        self.plot_change_point_distribution(
            dataset, selected_indices, config, f"{save_prefix}_change_points.png"
        )
        self.plot_edge_density_distribution(
            dataset, selected_indices, config, f"{save_prefix}_edge_density.png"
        )
