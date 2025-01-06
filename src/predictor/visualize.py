# src/predictor/visualize.py

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Any, Tuple
import networkx as nx
import numpy as np

from graph.features import NetworkFeatureExtractor

feature_extractor = NetworkFeatureExtractor()


class PlotStyle:
    """Style configuration for all visualizations."""

    # Font sizes
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    LARGE_SIZE = 14

    # Colors
    TEXT_COLOR = "#2f3640"
    BACKGROUND_COLOR = "#f5f6fa"
    GRID_COLOR = "#dcdde1"

    # Line styles
    LINE_STYLES = {
        "actual": dict(
            color="#0984e3", linestyle="-", linewidth=1.5, alpha=0.7
        ),  # Blue
        "predicted": dict(
            color="#d63031", linestyle="--", linewidth=1.5, alpha=0.7
        ),  # Red
        "change_point": dict(
            color="#00b894", linestyle="--", linewidth=1, alpha=0.3
        ),  # Green
    }

    # Node and edge styles
    NODE_STYLES = {
        "actual": dict(node_color="#0984e3", node_size=50, alpha=0.7),
        "predicted": dict(node_color="#d63031", node_size=50, alpha=0.7),
    }

    EDGE_STYLES = {
        "actual": dict(edge_color="#0984e3", alpha=0.4, width=0.5),
        "predicted": dict(edge_color="#d63031", alpha=0.4, width=0.5),
    }

    @classmethod
    def apply_style(cls):
        """Apply the style settings to matplotlib."""
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams.update(
            {
                "font.size": cls.SMALL_SIZE,
                "axes.titlesize": cls.MEDIUM_SIZE,
                "axes.labelsize": cls.SMALL_SIZE,
                "xtick.labelsize": cls.SMALL_SIZE,
                "ytick.labelsize": cls.SMALL_SIZE,
                "legend.fontsize": cls.SMALL_SIZE,
                "figure.titlesize": cls.LARGE_SIZE,
                "axes.grid": True,
                "grid.alpha": 0.2,
                "grid.color": cls.GRID_COLOR,
                "axes.facecolor": cls.BACKGROUND_COLOR,
                "figure.facecolor": "white",
                "text.color": cls.TEXT_COLOR,
            }
        )


class Visualizer:
    """Simplified visualization utilities for network forecasting."""

    def __init__(self):
        """Initialize visualizer with consistent style."""
        PlotStyle.apply_style()

    def plot_metric_evolution(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        min_history: int,
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Plot the evolution of key network metrics over time."""
        # Calculate metrics
        actual_metrics = [
            feature_extractor.get_all_metrics(net["graph"]).__dict__
            for net in actual_series
        ]
        pred_metrics = [
            feature_extractor.get_all_metrics(p["graph"]).__dict__ for p in predictions
        ]
        pred_times = [p["time"] for p in predictions]

        # Get change points
        change_points = [
            i
            for i, state in enumerate(actual_series)
            if state.get("is_change_point", False)
        ]

        # Create figure
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")

        # Create title
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )

        # Create GridSpec with space for title
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, top=0.85)

        fig.suptitle(
            f"Evolution of Network Metrics - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=0.95,
            weight="bold",
        )

        # Define metrics to plot with shorter names
        metrics = [
            ("avg_degree", "Average Degree"),
            ("clustering", "Clustering Coefficient"),
            ("avg_betweenness", "Betweenness Centrality"),
            ("spectral_gap", "Spectral Gap"),
            ("algebraic_connectivity", "Algebraic Connectivity"),
            ("density", "Network Density"),
        ]

        times = list(range(len(actual_series)))

        # Plot each metric
        for idx, (metric_name, title) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # Plot change points
            for cp in change_points:
                ax.axvline(
                    x=cp,
                    label="Change Point" if cp == change_points[0] else "",
                    **PlotStyle.LINE_STYLES["change_point"],
                )

            # Plot actual and predicted values
            ax.plot(
                times,
                [m[metric_name] for m in actual_metrics],
                label="Actual",
                **PlotStyle.LINE_STYLES["actual"],
            )
            ax.plot(
                pred_times,
                [m[metric_name] for m in pred_metrics],
                label="Predicted",
                **PlotStyle.LINE_STYLES["predicted"],
            )

            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.2)

            if idx == 0:  # Only add legend to first plot
                ax.legend()

    def plot_network_comparison(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        time_points: List[int],
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (15, 5),
    ) -> None:
        """Plot network structure comparison at specified time points."""
        n_points = len(time_points)

        # Create figure with space for title
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, n_points, hspace=0.4, wspace=0.3, top=0.85)

        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Network Structure Comparison - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=0.95,
            weight="bold",
        )

        for i, t in enumerate(time_points):
            # Handle negative indexing
            if t < 0:
                t = len(predictions) + t

            # Plot actual network
            ax_actual = fig.add_subplot(gs[0, i])
            G_actual = actual_series[t]["graph"]
            pos = nx.spring_layout(G_actual, seed=42)

            nx.draw_networkx_edges(
                G_actual, pos, ax=ax_actual, **PlotStyle.EDGE_STYLES["actual"]
            )
            nx.draw_networkx_nodes(
                G_actual, pos, ax=ax_actual, **PlotStyle.NODE_STYLES["actual"]
            )

            ax_actual.set_title(f"Actual (t={t})")
            ax_actual.axis("off")

            # Plot predicted network
            ax_pred = fig.add_subplot(gs[1, i])
            pred_idx = t - len(actual_series) + len(predictions)
            G_pred = predictions[pred_idx]["graph"]

            nx.draw_networkx_edges(
                G_pred, pos, ax=ax_pred, **PlotStyle.EDGE_STYLES["predicted"]
            )
            nx.draw_networkx_nodes(
                G_pred, pos, ax=ax_pred, **PlotStyle.NODE_STYLES["predicted"]
            )

            ax_pred.set_title(f"Predicted (t={t})")
            ax_pred.axis("off")

    def plot_adjacency_matrices(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        time_points: List[int],
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Plot adjacency matrix comparison at specified time points.

        Color coding for predicted matrix:
        - Green: Correctly predicted links (true positives)
        - Red: Incorrectly predicted links (false positives)
        - White: No link predicted
        """
        n_points = len(time_points)

        # Create figure with space for title and legend
        fig = plt.figure(figsize=figsize)

        # Create GridSpec with minimal spacing
        gs = fig.add_gridspec(
            n_points,
            2,  # n_points rows, 2 columns (actual & predicted side by side)
            height_ratios=[1] * n_points,
            width_ratios=[1, 1],
            left=0.1,
            right=0.85,
            bottom=0.05,
            top=0.9,
            hspace=0.15,  # Reduced vertical spacing
            wspace=0.05,  # Minimal horizontal spacing
        )

        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Adjacency Matrix Comparison - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=0.98,
            weight="bold",
        )

        # Create legend axes on the right
        legend_ax = fig.add_axes([0.87, 0.4, 0.1, 0.2])  # Reduced legend size
        legend_ax.axis("off")
        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Correct Prediction"),
            Patch(facecolor="red", alpha=0.7, label="False Positive"),
        ]
        legend = legend_ax.legend(
            handles=legend_elements, loc="center", title="Prediction"
        )
        legend.get_title().set_fontweight("bold")

        # Add column headers with less space
        fig.text(
            0.3,
            0.93,
            "Actual",
            ha="center",
            va="bottom",
            fontsize=PlotStyle.MEDIUM_SIZE,
            fontweight="bold",
        )
        fig.text(
            0.6,
            0.93,
            "Predicted",
            ha="center",
            va="bottom",
            fontsize=PlotStyle.MEDIUM_SIZE,
            fontweight="bold",
        )

        for i, t in enumerate(time_points):
            if t < 0:
                t = len(predictions) + t

            # Plot actual adjacency
            ax_actual = fig.add_subplot(gs[i, 0])
            adj_actual = actual_series[t]["adjacency"]
            im_actual = ax_actual.imshow(adj_actual, cmap="Blues")
            ax_actual.set_title(f"t={t}", pad=2, fontsize=PlotStyle.SMALL_SIZE)

            # Remove ticks but keep frame
            ax_actual.set_xticks([])
            ax_actual.set_yticks([])

            # Get predicted adjacency
            pred_idx = t - len(actual_series) + len(predictions)
            adj_pred = predictions[pred_idx]["adjacency"]

            # Calculate metrics
            true_links = np.sum(adj_actual)  # Total actual links
            correct_pred = np.sum((adj_actual == 1) & (adj_pred == 1))  # True positives
            false_pos = np.sum((adj_actual == 0) & (adj_pred == 1))  # False positives
            total_pred = np.sum(adj_pred)  # Total predicted links

            link_coverage = correct_pred / true_links if true_links > 0 else 0
            false_pos_rate = false_pos / total_pred if total_pred > 0 else 0

            # Create color-coded prediction matrix
            colored_pred = np.zeros((*adj_pred.shape, 3))  # RGB array

            # Color coding
            correct_mask = (adj_actual == 1) & (adj_pred == 1)
            colored_pred[correct_mask] = [0, 1, 0]  # Green

            false_pos_mask = (adj_actual == 0) & (adj_pred == 1)
            colored_pred[false_pos_mask] = [1, 0, 0]  # Red

            # Plot predicted adjacency with color coding
            ax_pred = fig.add_subplot(gs[i, 1])
            ax_pred.imshow(colored_pred)

            # Add metrics as text with shorter format
            metrics_text = f"Coverage: {link_coverage:.1%} | FPR: {false_pos_rate:.1%}"
            ax_pred.set_title(metrics_text, pad=2, fontsize=PlotStyle.SMALL_SIZE)

            # Remove ticks but keep frame
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
