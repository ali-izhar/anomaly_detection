# src/predictor/visualize.py

import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import numpy as np

from graph.features import NetworkFeatureExtractor

feature_extractor = NetworkFeatureExtractor()


class PlotStyle:
    """Style configuration for all visualizations."""

    # Font sizes
    TINY_SIZE = 6  # For very small annotations
    SMALL_SIZE = 8  # Basic text size (tick labels, stats)
    MEDIUM_SIZE = 10  # For important labels
    LARGE_SIZE = 14  # For main titles only

    # Colors
    TEXT_COLOR = "#2f3640"  # Dark gray for text
    BACKGROUND_COLOR = "#f5f6fa"  # Light gray background
    GRID_COLOR = "#dcdde1"  # Light gray for grids

    # Line styles
    LINE_STYLES = {
        "actual": dict(color="#0984e3", linestyle="-", linewidth=1, alpha=0.7),  # Blue
        "predicted": dict(
            color="#d63031", linestyle="--", linewidth=1, alpha=0.7
        ),  # Red
        "change_point": dict(
            color="#00b894", linestyle="--", linewidth=1, alpha=0.3
        ),  # Green
        "history": dict(
            color="#fdcb6e", linestyle="-", linewidth=1, alpha=0.7
        ),  # Orange
    }

    # Node and edge styles
    NODE_STYLES = {
        "actual": dict(
            node_color="#0984e3", node_size=30, linewidths=0.5, edgecolors="white"
        ),
        "predicted": dict(
            node_color="#d63031", node_size=30, linewidths=0.5, edgecolors="white"
        ),
    }

    EDGE_STYLES = {
        "actual": dict(edge_color="#74b9ff", alpha=0.4, width=0.5),
        "predicted": dict(edge_color="#fab1a0", alpha=0.4, width=0.5),
    }

    # Stats box style
    STATS_BOX_STYLE = dict(
        facecolor="white",
        alpha=0.9,
        edgecolor="none",
        boxstyle="round,pad=0.4",
    )

    # Colormap settings
    CMAP = "Blues"
    ERROR_COLORS = {"fp": plt.cm.Reds, "fn": plt.cm.YlOrBr}

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
                "axes.labelcolor": cls.TEXT_COLOR,
                "xtick.color": cls.TEXT_COLOR,
                "ytick.color": cls.TEXT_COLOR,
            }
        )


def calculate_prediction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate custom prediction metrics that better reflect our over-prediction strategy.

    Parameters
    ----------
    y_true : np.ndarray
        True adjacency matrix (upper triangle)
    y_pred : np.ndarray
        Predicted adjacency matrix (upper triangle)

    Returns
    -------
    Dict[str, float]
        Dictionary containing the following metrics:
        - edge_coverage: Percentage of actual edges we successfully predicted
        - prediction_efficiency: Percentage of our predictions that were correct
        - over_prediction_ratio: Ratio of predicted edges to actual edges
    """
    true_edges = np.sum(y_true)
    pred_edges = np.sum(y_pred)

    # True positives (correctly predicted edges)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))

    # Calculate metrics
    edge_coverage = true_positives / true_edges if true_edges > 0 else 0.0
    prediction_efficiency = true_positives / pred_edges if pred_edges > 0 else 0.0
    over_prediction_ratio = pred_edges / true_edges if true_edges > 0 else float("inf")

    return {
        "edge_coverage": edge_coverage,
        "prediction_efficiency": prediction_efficiency,
        "over_prediction_ratio": over_prediction_ratio,
    }


class Visualizer:
    """Visualization utilities for network forecasting."""

    def __init__(self):
        """Initialize visualizer with consistent style."""
        PlotStyle.apply_style()

    def plot_network_snapshots(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        time_points: List[int],
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (15, 6),
    ) -> None:
        """Plot network structure snapshots at specified time points."""
        n_points = len(time_points)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, n_points, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Add more padding between title and plots
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Network Structure Evolution - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=1.02,  # Increased from 0.95
            color=PlotStyle.TEXT_COLOR,
            weight="bold",
        )

        for i, t in enumerate(time_points):
            if t < 0:  # Handle negative indexing
                t = len(predictions) + t

            # Plot actual network
            ax_actual = fig.add_subplot(gs[0, i])
            ax_actual.set_facecolor(PlotStyle.BACKGROUND_COLOR)
            G_actual = actual_series[t]["graph"]
            pos = nx.spring_layout(
                G_actual, k=1 / np.sqrt(G_actual.number_of_nodes()), seed=42
            )

            # Draw edges first
            nx.draw_networkx_edges(
                G_actual, pos, ax=ax_actual, **PlotStyle.EDGE_STYLES["actual"]
            )
            # Then draw nodes
            nx.draw_networkx_nodes(
                G_actual, pos, ax=ax_actual, **PlotStyle.NODE_STYLES["actual"]
            )

            # Add network stats with consistent style
            stats_actual = (
                f"Nodes: {G_actual.number_of_nodes():,d}\n"
                f"Edges: {G_actual.number_of_edges():,d}\n"
                f"Density: {nx.density(G_actual):.3f}"
            )
            ax_actual.text(
                0.02,
                0.98,
                stats_actual,
                transform=ax_actual.transAxes,
                verticalalignment="top",
                fontsize=PlotStyle.SMALL_SIZE,
                color=PlotStyle.TEXT_COLOR,
                bbox=PlotStyle.STATS_BOX_STYLE,
            )
            ax_actual.set_title(
                f"Actual Network (t={t})",
                fontsize=PlotStyle.MEDIUM_SIZE,
                pad=10,
                color=PlotStyle.TEXT_COLOR,
            )

            # Plot predicted network
            ax_pred = fig.add_subplot(gs[1, i])
            ax_pred.set_facecolor(PlotStyle.BACKGROUND_COLOR)
            pred_idx = t - len(actual_series) + len(predictions)
            G_pred = predictions[pred_idx]["graph"]

            # Draw edges first
            nx.draw_networkx_edges(
                G_pred, pos, ax=ax_pred, **PlotStyle.EDGE_STYLES["predicted"]
            )
            # Then draw nodes
            nx.draw_networkx_nodes(
                G_pred, pos, ax=ax_pred, **PlotStyle.NODE_STYLES["predicted"]
            )

            # Calculate metrics
            actual_adj = nx.to_numpy_array(G_actual)
            pred_adj = nx.to_numpy_array(G_pred)
            triu_indices = np.triu_indices_from(actual_adj, k=1)
            y_true = actual_adj[triu_indices]
            y_pred = pred_adj[triu_indices]

            # Calculate custom metrics
            metrics = calculate_prediction_metrics(y_true, y_pred)

            # Add network stats with consistent style
            stats_pred = (
                f"Nodes: {G_pred.number_of_nodes():,d}\n"
                f"Edges: {G_pred.number_of_edges():,d}\n"
                f"Density: {nx.density(G_pred):.3f}\n"
                f"Edge Coverage: {metrics['edge_coverage']:.1%}\n"  # Percentage of actual edges found
                f"Prediction Efficiency: {metrics['prediction_efficiency']:.1%}\n"  # Percentage of predictions that were correct
                f"Over-prediction: {metrics['over_prediction_ratio']:.1f}x"  # How many times more edges we predicted
            )
            ax_pred.text(
                0.02,
                0.98,
                stats_pred,
                transform=ax_pred.transAxes,
                verticalalignment="top",
                fontsize=PlotStyle.SMALL_SIZE,
                color=PlotStyle.TEXT_COLOR,
                bbox=PlotStyle.STATS_BOX_STYLE,
            )
            ax_pred.set_title(
                f"Predicted Network (t={t})",
                fontsize=PlotStyle.MEDIUM_SIZE,
                pad=10,
                color=PlotStyle.TEXT_COLOR,
            )

            # Set equal aspect ratio and remove axes
            ax_actual.set_aspect("equal")
            ax_pred.set_aspect("equal")
            ax_actual.axis("off")
            ax_pred.axis("off")

    def plot_metric_evolution(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        min_history: int,
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Plot the evolution of multiple network metrics over time."""
        # Calculate metrics
        actual_metrics = [
            feature_extractor.get_all_metrics(net["graph"]).__dict__
            for net in actual_series
        ]
        pred_metrics = [
            feature_extractor.get_all_metrics(p["graph"]).__dict__ for p in predictions
        ]
        pred_times = [p["time"] for p in predictions]
        history_sizes = [p["history_size"] for p in predictions]

        # Get change points
        change_points = self._get_change_points(actual_series)

        # Create figure with white background
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")

        # Create main title with more padding
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Evolution of Network Metrics Over Time - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            color=PlotStyle.TEXT_COLOR,
            weight="bold",
            y=1.02,
        )

        # Create GridSpec for plots with reduced spacing
        gs = fig.add_gridspec(
            4,
            2,
            top=0.90,  # Reduced from 0.92 to accommodate title padding
            bottom=0.08,
            left=0.1,
            right=0.85,  # Make room for legend
            hspace=0.4,
            wspace=0.3,
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

        # Create a single legend for all plots
        legend_elements = [
            plt.Line2D([0], [0], **PlotStyle.LINE_STYLES["actual"], label="Actual"),
            plt.Line2D(
                [0], [0], **PlotStyle.LINE_STYLES["predicted"], label="Predicted"
            ),
            plt.Line2D(
                [0], [0], **PlotStyle.LINE_STYLES["change_point"], label="Change Point"
            ),
        ]

        # Plot each metric
        for idx, (metric_name, title) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            ax.set_facecolor(PlotStyle.BACKGROUND_COLOR)

            # Plot change points in background
            ymin, ymax = self._get_metric_range(
                actual_metrics, pred_metrics, metric_name
            )
            for cp in change_points:
                ax.axvline(x=cp, **PlotStyle.LINE_STYLES["change_point"])

            # Plot actual and predicted values
            ax.plot(
                times,
                [m[metric_name] for m in actual_metrics],
                **PlotStyle.LINE_STYLES["actual"],
            )
            ax.plot(
                pred_times,
                [m[metric_name] for m in pred_metrics],
                **PlotStyle.LINE_STYLES["predicted"],
            )

            # Style the subplot
            ax.set_title(title, fontsize=PlotStyle.MEDIUM_SIZE, pad=5)
            ax.set_xlabel("Time", fontsize=PlotStyle.SMALL_SIZE)
            ax.tick_params(labelsize=PlotStyle.TINY_SIZE)
            ax.grid(True, alpha=0.2, color=PlotStyle.GRID_COLOR)

        # Add a single legend for all subplots
        fig.legend(
            handles=legend_elements,
            loc="center right",
            bbox_to_anchor=(0.98, 0.5),
            fontsize=PlotStyle.SMALL_SIZE,
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )

        plt.tight_layout()

    def _get_metric_range(
        self,
        actual_metrics: List[Dict[str, float]],
        pred_metrics: List[Dict[str, float]],
        metric_name: str,
    ) -> Tuple[float, float]:
        """Calculate the y-axis range for a metric with padding."""
        actual_values = [m[metric_name] for m in actual_metrics]
        pred_values = [m[metric_name] for m in pred_metrics]

        ymin = min(min(actual_values), min(pred_values))
        ymax = max(max(actual_values), max(pred_values))

        padding = (ymax - ymin) * 0.1
        return ymin - padding, ymax + padding

    def _get_change_points(self, actual_series: List[Dict[str, Any]]) -> List[int]:
        """Get list of change points from the actual series.

        Parameters
        ----------
        actual_series : List[Dict[str, Any]]
            The actual network time series

        Returns
        -------
        List[int]
            List of change point indices
        """
        return [
            i
            for i, state in enumerate(actual_series)
            if state.get("is_change_point", False)
        ]

    def _get_single_change_point(
        self, actual_series: List[Dict[str, Any]], change_point: Optional[int] = None
    ) -> int:
        """Get a single change point, either specified or the first detected one.

        Parameters
        ----------
        actual_series : List[Dict[str, Any]]
            The actual network time series
        change_point : Optional[int], optional
            Specific change point to use, by default None

        Returns
        -------
        int
            The selected change point index

        Raises
        ------
        ValueError
            If no change points are found in the series
        TypeError
            If change_point is provided but not an integer
        """
        if change_point is not None:
            if not isinstance(change_point, (int, np.integer)):
                raise TypeError("change_point must be an integer")
            return change_point

        change_points = self._get_change_points(actual_series)
        if not change_points:
            raise ValueError("No change points found in the series")
        return change_points[0]

    def plot_adjacency_comparison(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        change_point: Optional[int] = None,
        window: int = 1,
        figsize: Tuple[int, int] = (15, 5),
        model_type: str = "Unknown",
    ) -> None:
        """Plot comparison of actual vs predicted adjacency matrices around a change point."""
        # Get single change point
        cp = self._get_single_change_point(actual_series, change_point)

        # Ensure we have valid time points
        max_t = len(actual_series) - 1
        time_points = [
            max(0, cp - window),  # Before (ensure non-negative)
            cp,  # At change point
            min(max_t, cp + window),  # After (ensure within bounds)
        ]

        # Create figure with white background
        plt.style.use("seaborn-v0_8-white")
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.patch.set_facecolor(PlotStyle.BACKGROUND_COLOR)

        # Main title with larger font and more padding
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Network Structure Around Change Point (t={cp}) - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=1.08,
            color=PlotStyle.TEXT_COLOR,
            weight="bold",
        )

        # Row titles with consistent style
        fig.text(
            0.02,
            0.75,
            "Actual",
            fontsize=PlotStyle.MEDIUM_SIZE,
            rotation=90,
            color=PlotStyle.TEXT_COLOR,
            weight="medium",
        )
        fig.text(
            0.02,
            0.25,
            "Predicted",
            fontsize=PlotStyle.MEDIUM_SIZE,
            rotation=90,
            color=PlotStyle.TEXT_COLOR,
            weight="medium",
        )

        # Column titles with consistent style
        titles = ["Before Change", "At Change Point", "After Change"]
        for i, title in enumerate(titles):
            fig.text(
                0.25 + i * 0.25,
                0.95,
                title,
                fontsize=PlotStyle.MEDIUM_SIZE,
                ha="center",
                color=PlotStyle.TEXT_COLOR,
            )

        # Stats box style
        stats_box_style = dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="none",
            boxstyle="round,pad=0.4",
        )

        for i, t in enumerate(time_points):
            # Plot actual adjacency matrix
            ax_actual = axes[0, i]
            ax_actual.set_facecolor(PlotStyle.BACKGROUND_COLOR)
            adj_actual = actual_series[t]["adjacency"]

            # Calculate difference matrix for error visualization
            pred_idx = t - len(actual_series) + len(predictions)
            adj_pred = predictions[pred_idx]["adjacency"]

            # Calculate errors (only upper triangle to avoid counting twice)
            triu_indices = np.triu_indices_from(adj_actual, k=1)
            y_true = adj_actual[triu_indices]
            y_pred = adj_pred[triu_indices]

            # Calculate custom metrics
            metrics = calculate_prediction_metrics(y_true, y_pred)

            error_mask = np.zeros_like(adj_actual, dtype=bool)
            error_mask[triu_indices] = y_true != y_pred
            error_mask = error_mask | error_mask.T

            # Count edges
            actual_edges = int(np.sum(y_true))
            pred_edges = int(np.sum(y_pred))

            # Plot matrices with error overlay
            im_actual = ax_actual.imshow(adj_actual, cmap=PlotStyle.CMAP)
            if error_mask.any():
                fp_mask = error_mask & (adj_pred > adj_actual)
                fn_mask = error_mask & (adj_actual > adj_pred)

                if fp_mask.any():
                    ax_actual.imshow(
                        np.ma.masked_where(~fp_mask, np.ones_like(adj_actual)),
                        cmap=PlotStyle.ERROR_COLORS["fp"],
                        alpha=0.3,
                    )
                if fn_mask.any():
                    ax_actual.imshow(
                        np.ma.masked_where(~fn_mask, np.ones_like(adj_actual)),
                        cmap=PlotStyle.ERROR_COLORS["fn"],
                        alpha=0.3,
                    )

            # Add stats for actual with consistent style
            stats_actual = f"t = {t}\nEdges = {actual_edges:,d}"
            ax_actual.text(
                0.02,
                0.98,
                stats_actual,
                transform=ax_actual.transAxes,
                verticalalignment="top",
                fontsize=PlotStyle.SMALL_SIZE,
                color=PlotStyle.TEXT_COLOR,
                bbox=stats_box_style,
            )

            # Plot predicted matrix
            ax_pred = axes[1, i]
            ax_pred.set_facecolor(PlotStyle.BACKGROUND_COLOR)
            im_pred = ax_pred.imshow(adj_pred, cmap=PlotStyle.CMAP)
            if error_mask.any():
                if fp_mask.any():
                    ax_pred.imshow(
                        np.ma.masked_where(~fp_mask, np.ones_like(adj_actual)),
                        cmap=PlotStyle.ERROR_COLORS["fp"],
                        alpha=0.3,
                    )
                if fn_mask.any():
                    ax_pred.imshow(
                        np.ma.masked_where(~fn_mask, np.ones_like(adj_actual)),
                        cmap=PlotStyle.ERROR_COLORS["fn"],
                        alpha=0.3,
                    )

            # Add metrics for predicted with consistent style
            stats_pred = (
                f"t = {t}\n"
                f"Edges = {pred_edges:,d}\n"
                f"Edge Coverage: {metrics['edge_coverage']:.1%}\n"
                f"Prediction Efficiency: {metrics['prediction_efficiency']:.1%}\n"
                f"Over-prediction: {metrics['over_prediction_ratio']:.1f}x"
            )
            ax_pred.text(
                0.02,
                0.98,
                stats_pred,
                transform=ax_pred.transAxes,
                verticalalignment="top",
                fontsize=PlotStyle.SMALL_SIZE,
                color=PlotStyle.TEXT_COLOR,
                bbox=stats_box_style,
            )

            # Remove ticks and add colorbar with consistent style
            ax_actual.set_xticks([])
            ax_actual.set_yticks([])
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # Add colorbars with consistent style
            cbar_actual = plt.colorbar(
                im_actual, ax=ax_actual, fraction=0.046, pad=0.04
            )
            cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

            # Style colorbar ticks
            cbar_actual.ax.tick_params(
                labelsize=PlotStyle.SMALL_SIZE, labelcolor=PlotStyle.TEXT_COLOR
            )
            cbar_pred.ax.tick_params(
                labelsize=PlotStyle.SMALL_SIZE, labelcolor=PlotStyle.TEXT_COLOR
            )

        plt.tight_layout()

    def plot_all_adjacency_matrices(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        model_type: str = "Unknown",
        max_matrices: int = 10,
        figsize: Tuple[int, int] = (15, 8),
    ) -> None:
        """Plot all adjacency matrices for comparison.

        Parameters
        ----------
        actual_series : List[Dict[str, Any]]
            The actual network time series
        predictions : List[Dict[str, Any]]
            The predicted network time series
        model_type : str, optional
            The type of network model used, by default "Unknown"
        max_matrices : int, optional
            Maximum number of matrices to plot, by default 10
        figsize : Tuple[int, int], optional
            Figure size, by default (15, 8)
        """
        # Sample time points if there are too many
        total_points = len(actual_series)
        if total_points > max_matrices:
            step = total_points // max_matrices
            time_points = list(range(0, total_points, step))
            if time_points[-1] != total_points - 1:
                time_points.append(total_points - 1)
        else:
            time_points = list(range(total_points))

        n_points = len(time_points)
        n_cols = min(5, n_points)
        n_rows = (n_points + n_cols - 1) // n_cols * 2  # *2 for actual and predicted

        # Create figure
        fig = plt.figure(figsize=figsize)
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Network Evolution Over Time - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=1.02,
            color=PlotStyle.TEXT_COLOR,
            weight="bold",
        )

        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        for idx, t in enumerate(time_points):
            row = (idx // n_cols) * 2
            col = idx % n_cols

            # Plot actual
            ax_actual = fig.add_subplot(gs[row, col])
            adj_actual = actual_series[t]["adjacency"]
            im_actual = ax_actual.imshow(adj_actual, cmap=PlotStyle.CMAP)
            ax_actual.set_title(f"Actual t={t}", fontsize=PlotStyle.SMALL_SIZE)
            ax_actual.set_xticks([])
            ax_actual.set_yticks([])

            # Plot predicted
            ax_pred = fig.add_subplot(gs[row + 1, col])
            pred_idx = t - len(actual_series) + len(predictions)
            adj_pred = predictions[pred_idx]["adjacency"]
            im_pred = ax_pred.imshow(adj_pred, cmap=PlotStyle.CMAP)

            # Calculate metrics
            triu_indices = np.triu_indices_from(adj_actual, k=1)
            y_true = adj_actual[triu_indices]
            y_pred = adj_pred[triu_indices]
            metrics = calculate_prediction_metrics(y_true, y_pred)

            ax_pred.set_title(
                f"Predicted t={t}\n"
                f"Coverage={metrics['edge_coverage']:.1%}\n"
                f"Efficiency={metrics['prediction_efficiency']:.1%}",
                fontsize=PlotStyle.SMALL_SIZE,
            )
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # Add colorbar
            plt.colorbar(im_actual, ax=ax_actual, fraction=0.046, pad=0.04)
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        plt.tight_layout()

    def plot_adjacency_analysis(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """Create an advanced analysis plot of adjacency predictions.

        This visualization combines multiple views to show:
        1. Edge prediction frequency heatmap
        2. Edge stability analysis
        3. Prediction success patterns
        4. Time-based edge evolution

        Inspired by the Les Misérables co-occurrence matrix visualization technique.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Network Prediction Analysis - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE,
            y=0.95,
            color=PlotStyle.TEXT_COLOR,
            weight="bold",
        )

        # Create GridSpec for flexible subplot layout
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3
        )

        # 1. Edge Prediction Success Heatmap (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_prediction_success_heatmap(actual_series, predictions, ax1)

        # 2. Edge Evolution Timeline (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_edge_evolution_timeline(actual_series, predictions, ax2)

        # 3. Prediction Statistics (top right)
        ax3 = fig.add_subplot(gs[0, 1])
        self._plot_prediction_statistics(actual_series, predictions, ax3)

        # 4. Edge Stability Analysis (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_edge_stability_analysis(actual_series, predictions, ax4)

        plt.tight_layout()

    def _plot_prediction_success_heatmap(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        ax: plt.Axes,
    ) -> None:
        """Plot heatmap showing prediction success patterns."""
        # Get dimensions
        n_nodes = actual_series[0]["adjacency"].shape[0]
        n_time = len(predictions)

        # Create success rate matrix
        success_matrix = np.zeros((n_nodes, n_nodes))
        total_predictions = np.zeros((n_nodes, n_nodes))

        for t, (actual, pred) in enumerate(
            zip(actual_series[len(actual_series) - len(predictions) :], predictions)
        ):
            actual_adj = actual["adjacency"]
            pred_adj = pred["adjacency"]
            success_matrix += actual_adj == pred_adj
            total_predictions += 1

        success_rate = success_matrix / total_predictions

        # Plot heatmap
        im = ax.imshow(success_rate, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title("Edge Prediction Success Rate", pad=10)
        plt.colorbar(im, ax=ax, label="Success Rate")
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")

    def _plot_edge_evolution_timeline(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        ax: plt.Axes,
    ) -> None:
        """Plot timeline of edge count and prediction metrics."""
        times = range(len(predictions))

        # Calculate metrics over time
        edge_coverage = []
        pred_efficiency = []
        over_pred_ratio = []

        for actual, pred in zip(
            actual_series[len(actual_series) - len(predictions) :], predictions
        ):
            triu_indices = np.triu_indices_from(actual["adjacency"], k=1)
            y_true = actual["adjacency"][triu_indices]
            y_pred = pred["adjacency"][triu_indices]
            metrics = calculate_prediction_metrics(y_true, y_pred)

            edge_coverage.append(metrics["edge_coverage"])
            pred_efficiency.append(metrics["prediction_efficiency"])
            over_pred_ratio.append(metrics["over_prediction_ratio"])

        # Plot metrics
        ax.plot(
            times,
            edge_coverage,
            label="Edge Coverage",
            color=PlotStyle.LINE_STYLES["actual"]["color"],
        )
        ax.plot(
            times,
            pred_efficiency,
            label="Prediction Efficiency",
            color=PlotStyle.LINE_STYLES["predicted"]["color"],
        )
        ax.plot(
            times,
            over_pred_ratio,
            label="Over-prediction Ratio",
            color=PlotStyle.LINE_STYLES["change_point"]["color"],
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Metric Value")
        ax.set_title("Edge Prediction Metrics Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_prediction_statistics(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        ax: plt.Axes,
    ) -> None:
        """Plot statistical analysis of prediction patterns."""
        # Calculate overall metrics
        all_edge_coverage = []
        all_pred_efficiency = []
        all_over_pred = []

        for actual, pred in zip(
            actual_series[len(actual_series) - len(predictions) :], predictions
        ):
            triu_indices = np.triu_indices_from(actual["adjacency"], k=1)
            y_true = actual["adjacency"][triu_indices]
            y_pred = pred["adjacency"][triu_indices]
            metrics = calculate_prediction_metrics(y_true, y_pred)

            all_edge_coverage.append(metrics["edge_coverage"])
            all_pred_efficiency.append(metrics["prediction_efficiency"])
            all_over_pred.append(metrics["over_prediction_ratio"])

        # Create box plots
        data = [all_edge_coverage, all_pred_efficiency, all_over_pred]
        labels = ["Edge\nCoverage", "Prediction\nEfficiency", "Over-prediction\nRatio"]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Style boxes
        colors = [
            PlotStyle.LINE_STYLES[style]["color"]
            for style in ["actual", "predicted", "change_point"]
        ]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title("Distribution of Prediction Metrics")
        ax.grid(True, alpha=0.3)

    def _plot_edge_stability_analysis(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        ax: plt.Axes,
    ) -> None:
        """Plot analysis of edge prediction stability."""
        # Calculate edge stability
        n_nodes = actual_series[0]["adjacency"].shape[0]
        edge_changes = np.zeros((n_nodes, n_nodes))

        for t in range(1, len(predictions)):
            prev_pred = predictions[t - 1]["adjacency"]
            curr_pred = predictions[t]["adjacency"]
            edge_changes += np.abs(curr_pred - prev_pred)

        # Normalize by number of time steps
        edge_stability = 1 - (edge_changes / (len(predictions) - 1))

        # Plot heatmap
        im = ax.imshow(edge_stability, cmap="viridis")
        ax.set_title("Edge Prediction Stability")
        plt.colorbar(im, ax=ax, label="Stability (higher is more stable)")
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")
