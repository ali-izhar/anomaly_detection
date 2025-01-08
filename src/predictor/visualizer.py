# src/predictor/visualize.py

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Any, Tuple, Union
import networkx as nx
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from pathlib import Path

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

    def _create_figure_with_suptitle(
        self,
        title: str,
        model_type: str = None,
        figsize: Tuple[int, int] = (20, 15),
    ) -> Tuple[plt.Figure, str]:
        """Create a figure with a formatted super title."""
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")

        model_name = None
        if model_type:
            model_name = {
                "ba": "Barabási-Albert",
                "er": "Erdős-Rényi",
                "ws": "Watts-Strogatz",
                "sbm": "Stochastic Block Model",
                "rcp": "Random Core-Periphery",
                "lfr": "LFR Benchmark",
            }.get(model_type.lower(), model_type)

            title = f"{title} - {model_name} Model"

        fig.suptitle(title, fontsize=PlotStyle.LARGE_SIZE + 2, y=0.95, weight="bold")

        return fig, model_name

    def _add_legend_to_figure(
        self,
        fig: plt.Figure,
        elements: List[Tuple[str, str, float]],
        title: str,
        position: List[float] = [0.88, 0.4, 0.12, 0.25],
    ) -> None:
        """Add a legend to the figure."""
        legend_ax = fig.add_axes(position)
        legend_ax.axis("off")

        legend_patches = [
            Patch(facecolor=color, alpha=alpha, label=label)
            for color, label, alpha in elements
        ]

        legend = legend_ax.legend(
            handles=legend_patches,
            loc="center",
            title=title,
            fontsize=PlotStyle.MEDIUM_SIZE,
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(PlotStyle.MEDIUM_SIZE)

    def _add_prediction_legend(
        self,
        fig: plt.Figure,
        position: List[float] = [0.87, 0.4, 0.12, 0.2],
        title: str = "Prediction Types",
    ) -> None:
        """Add prediction legend with consistent styling."""
        elements = [
            ("green", "Correct Prediction", 0.7),
            ("red", "False Positive", 0.7),
            ("gray", "Missed Edge", 0.3),
        ]
        self._add_legend_to_figure(fig, elements, title, position)

    def _add_metric_textbox(
        self,
        ax: plt.Axes,
        text: str,
        position: Tuple[float, float],
        va: str = "bottom",
        fontsize: int = None,
    ) -> None:
        """Add a metric textbox to an axis."""
        if fontsize is None:
            fontsize = PlotStyle.SMALL_SIZE

        ax.text(
            position[0],
            position[1],
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            ha="right",
            va=va,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
        )

    def _setup_axis_with_grid(
        self,
        ax: plt.Axes,
        title: str,
        xlabel: str = None,
        ylabel: str = None,
    ) -> None:
        """Set up an axis with consistent styling."""
        ax.grid(True, alpha=0.2, linestyle="--", color=PlotStyle.GRID_COLOR)
        ax.set_facecolor(PlotStyle.BACKGROUND_COLOR)

        ax.set_title(title, pad=20, fontsize=PlotStyle.MEDIUM_SIZE + 2, weight="bold")

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=PlotStyle.MEDIUM_SIZE)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=PlotStyle.MEDIUM_SIZE)

        ax.tick_params(labelsize=PlotStyle.SMALL_SIZE)

    def _calculate_network_metrics(
        self,
        G_actual: nx.Graph,
        G_pred: nx.Graph,
    ) -> Tuple[List, List, List, Dict[str, Union[int, float]]]:
        """Calculate network comparison metrics."""
        correct_edges = []
        false_positive_edges = []
        missed_edges = []

        for e in G_pred.edges():
            if G_actual.has_edge(*e):
                correct_edges.append(e)
            else:
                false_positive_edges.append(e)

        for e in G_actual.edges():
            if not G_pred.has_edge(*e):
                missed_edges.append(e)

        total_actual = len(G_actual.edges())
        total_predicted = len(G_pred.edges())
        correct_count = len(correct_edges)
        false_positive_count = len(false_positive_edges)
        missed_count = len(missed_edges)

        coverage = correct_count / total_actual if total_actual > 0 else 0
        fpr = false_positive_count / total_predicted if total_predicted > 0 else 0

        metrics = {
            "total_actual": total_actual,
            "total_predicted": total_predicted,
            "correct_count": correct_count,
            "false_positive_count": false_positive_count,
            "missed_count": missed_count,
            "coverage": coverage,
            "fpr": fpr,
        }

        return correct_edges, false_positive_edges, missed_edges, metrics

    def _calculate_adjacency_metrics(
        self, adj_actual: np.ndarray, adj_pred: np.ndarray
    ) -> Dict[str, Union[int, float]]:
        """Calculate coverage, FPR, and related counts for adjacency matrices."""
        true_links = np.sum(adj_actual)
        correct_pred = np.sum((adj_actual == 1) & (adj_pred == 1))  # True positives
        false_pos = np.sum((adj_actual == 0) & (adj_pred == 1))  # False positives
        missed = np.sum((adj_actual == 1) & (adj_pred == 0))  # Missed links
        total_pred = np.sum(adj_pred)

        coverage = correct_pred / true_links if true_links > 0 else 0
        fpr = false_pos / total_pred if total_pred > 0 else 0

        return {
            "coverage": coverage,
            "fpr": fpr,
            "correct_count": int(correct_pred),
            "false_positive_count": int(false_pos),
            "missed_count": int(missed),
            "total_actual": int(true_links),
            "total_predicted": int(total_pred),
        }

    def _create_colored_pred_matrix(
        self, adj_actual: np.ndarray, adj_pred: np.ndarray
    ) -> np.ndarray:
        """Create an RGB array highlighting correct predictions (green), false positives (red), and missed edges (gray)."""
        colored_pred = np.zeros((*adj_pred.shape, 3))

        # Correct predictions (green)
        correct_mask = (adj_actual == 1) & (adj_pred == 1)
        colored_pred[correct_mask] = [0, 1, 0]  # Green

        # False positives (red)
        false_pos_mask = (adj_actual == 0) & (adj_pred == 1)
        colored_pred[false_pos_mask] = [1, 0, 0]  # Red

        # Missed edges (gray)
        missed_mask = (adj_actual == 1) & (adj_pred == 0)
        colored_pred[missed_mask] = [0.7, 0.7, 0.7]  # Gray

        return colored_pred

    def _draw_color_coded_edges(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        ax: plt.Axes,
        correct_edges: List,
        false_positive_edges: List,
        missed_edges: List,
    ) -> None:
        """Draw edges with different colors for correct, false positives, and missed edges."""
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=correct_edges,
            edge_color="green",
            alpha=0.7,
            width=0.5,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=false_positive_edges,
            edge_color="red",
            alpha=0.7,
            width=0.5,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=missed_edges,
            edge_color="gray",
            alpha=0.3,
            width=0.5,
            style="dashed",
        )

    def plot_metric_evolution(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        min_history: int,
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (20, 15),
    ) -> None:
        """Create an enhanced visualization of network metric evolution over time."""
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

        # Create figure with GridSpec for better layout
        fig, model_name = self._create_figure_with_suptitle(
            "Evolution of Network Metrics", model_type, figsize
        )

        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[1, 1, 1],
            width_ratios=[1.2, 1],
            hspace=0.25,
            wspace=0.25,
            top=0.85,
        )

        # Define metrics to plot with descriptions
        metrics = [
            ("avg_degree", "Average Degree", "Mean number of connections per node"),
            ("clustering", "Clustering Coefficient", "Local density of connections"),
            (
                "avg_betweenness",
                "Betweenness Centrality",
                "Node importance in information flow",
            ),
            ("spectral_gap", "Spectral Gap", "Network connectivity robustness"),
            (
                "algebraic_connectivity",
                "Algebraic Connectivity",
                "Overall network connectivity",
            ),
            ("density", "Network Density", "Ratio of actual to possible edges"),
        ]

        times = list(range(len(actual_series)))

        # Create legend axes
        actual_style = PlotStyle.LINE_STYLES["actual"]
        pred_style = PlotStyle.LINE_STYLES["predicted"]
        change_style = PlotStyle.LINE_STYLES["change_point"]
        legend_elements = [
            (actual_style["color"], "Actual", actual_style["alpha"]),
            (pred_style["color"], "Predicted", pred_style["alpha"]),
            (change_style["color"], "Change Point", change_style["alpha"]),
            ("#95a5a6", "Prediction Start", 0.5),
        ]
        self._add_legend_to_figure(
            fig, legend_elements, "Series Types", position=[0.87, 0.4, 0.12, 0.2]
        )

        # Plot each metric
        for idx, (metric_name, title, description) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            self._setup_axis_with_grid(ax, title)

            # Plot change points
            for cp in change_points:
                ax.axvline(
                    x=cp,
                    label="Change Point" if cp == change_points[0] and idx == 0 else "",
                    **PlotStyle.LINE_STYLES["change_point"],
                )

            actual_values = [m[metric_name] for m in actual_metrics]
            pred_values = [m[metric_name] for m in pred_metrics]

            ax.plot(
                times,
                actual_values,
                label="Actual" if idx == 0 else "",
                **PlotStyle.LINE_STYLES["actual"],
            )
            ax.plot(
                pred_times,
                pred_values,
                label="Predicted" if idx == 0 else "",
                **PlotStyle.LINE_STYLES["predicted"],
            )

            # Mark start of prediction
            ax.axvline(
                x=min_history,
                color="#95a5a6",
                linestyle=":",
                alpha=0.5,
                label="Prediction Start" if idx == 0 else "",
            )

            # Add metric description
            self._add_metric_textbox(ax, description, (0.98, 0.02))

            # Calculate and display error metrics
            if len(pred_values) > 0:
                actual_pred_range = actual_values[min_history:][: len(pred_values)]
                if len(actual_pred_range) > 0:
                    mae = np.mean(
                        np.abs(np.array(pred_values) - np.array(actual_pred_range))
                    )

                    rmse = np.sqrt(
                        np.mean(
                            (np.array(pred_values) - np.array(actual_pred_range)) ** 2
                        )
                    )

                    error_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}"
                    self._add_metric_textbox(ax, error_text, (0.98, 0.98))

        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

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

        # Create figure
        fig, model_name = self._create_figure_with_suptitle(
            "Network Structure Comparison", model_type, figsize
        )

        gs = fig.add_gridspec(2, n_points, hspace=0.4, wspace=0.3, top=0.85)

        # Create legend for edges
        legend_ax = fig.add_axes([0.87, 0.4, 0.1, 0.2])
        legend_ax.axis("off")
        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Correct Prediction"),
            Patch(facecolor="red", alpha=0.7, label="False Positive"),
            Patch(facecolor="gray", alpha=0.3, label="Missed Edge"),
        ]
        legend = legend_ax.legend(
            handles=legend_elements, loc="center", title="Edge Types"
        )
        legend.get_title().set_fontweight("bold")

        for i, t in enumerate(time_points):
            # Handle negative indexing
            if t < 0:
                t = len(predictions) + t

            # Actual network
            ax_actual = fig.add_subplot(gs[0, i])
            G_actual = actual_series[t]["graph"]
            pos = nx.spring_layout(G_actual, seed=42)

            nx.draw_networkx_nodes(
                G_actual, pos, ax=ax_actual, **PlotStyle.NODE_STYLES["actual"]
            )
            nx.draw_networkx_edges(
                G_actual, pos, ax=ax_actual, edge_color="gray", alpha=0.4, width=0.5
            )
            ax_actual.set_title(f"Actual (t={t})")
            ax_actual.axis("off")

            # Predicted network
            ax_pred = fig.add_subplot(gs[1, i])
            pred_idx = t - len(actual_series) + len(predictions)
            G_pred = predictions[pred_idx]["graph"]

            nx.draw_networkx_nodes(
                G_pred, pos, ax=ax_pred, **PlotStyle.NODE_STYLES["predicted"]
            )

            # Color-coded edges
            correct_edges, false_positive_edges, missed_edges, metrics = (
                self._calculate_network_metrics(G_actual, G_pred)
            )
            self._draw_color_coded_edges(
                G_pred, pos, ax_pred, correct_edges, false_positive_edges, missed_edges
            )

            metrics_text = (
                f"Predicted (t={t})\n"
                f"Coverage: {metrics['coverage']:.1%}\n"
                f"FPR: {metrics['fpr']:.1%}\n"
                f"Correct: {metrics['correct_count']}\n"
                f"FP: {metrics['false_positive_count']}\n"
                f"Missed: {metrics['missed_count']}"
            )
            ax_pred.set_title(metrics_text, pad=2, fontsize=PlotStyle.SMALL_SIZE)
            ax_pred.axis("off")

    def plot_adjacency_matrices(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        time_points: List[int],
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Plot adjacency matrix comparison at specified time points."""
        n_points = len(time_points)

        # Create figure
        fig, model_name = self._create_figure_with_suptitle(
            "Adjacency Matrix Comparison", model_type, figsize
        )

        gs = fig.add_gridspec(
            n_points,
            2,
            height_ratios=[1] * n_points,
            width_ratios=[1, 1],
            left=0.1,
            right=0.85,
            bottom=0.05,
            top=0.9,
            hspace=0.15,
            wspace=0.05,
        )

        # Create legend for predictions
        legend_ax = fig.add_axes([0.87, 0.4, 0.1, 0.2])
        legend_ax.axis("off")
        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Correct Prediction"),
            Patch(facecolor="red", alpha=0.7, label="False Positive"),
        ]
        legend = legend_ax.legend(
            handles=legend_elements, loc="center", title="Prediction"
        )
        legend.get_title().set_fontweight("bold")

        # Column headers
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

            # Actual adjacency
            ax_actual = fig.add_subplot(gs[i, 0])
            adj_actual = actual_series[t]["adjacency"]
            ax_actual.imshow(adj_actual, cmap="Blues")
            ax_actual.set_title(f"t={t}", pad=2, fontsize=PlotStyle.SMALL_SIZE)
            ax_actual.set_xticks([])
            ax_actual.set_yticks([])

            # Predicted adjacency
            ax_pred = fig.add_subplot(gs[i, 1])
            pred_idx = t - len(actual_series) + len(predictions)
            adj_pred = predictions[pred_idx]["adjacency"]

            # Metrics
            metrics = self._calculate_adjacency_metrics(adj_actual, adj_pred)
            link_coverage = metrics["coverage"]
            false_pos_rate = metrics["fpr"]

            # Color-coded adjacency
            colored_pred = self._create_colored_pred_matrix(adj_actual, adj_pred)
            ax_pred.imshow(colored_pred)

            metrics_text = f"Coverage: {link_coverage:.1%} | FPR: {false_pos_rate:.1%}"
            ax_pred.set_title(metrics_text, pad=2, fontsize=PlotStyle.SMALL_SIZE)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

    def plot_node_degree_evolution(
        self,
        network_series: List[Dict[str, Any]],
        output_path: str = None,
        figsize: Tuple[int, int] = (20, 15),
    ) -> None:
        """Create a comprehensive dashboard for node degree evolution analysis."""
        n_nodes = len(network_series[0]["graph"].nodes())
        n_timesteps = len(network_series)
        degree_matrix = np.zeros((n_timesteps, n_nodes))

        for t, state in enumerate(network_series):
            degrees = dict(state["graph"].degree())
            for node in range(n_nodes):
                degree_matrix[t, node] = degrees[node]

        # Create figure
        fig, model_name = self._create_figure_with_suptitle(
            "Network Degree Evolution Analysis", model_type="Unknown", figsize=figsize
        )

        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.2, 1],
            width_ratios=[1.2, 1],
            hspace=0.25,
            wspace=0.25,
        )

        # Main Evolution Plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        degree_variances = np.var(degree_matrix, axis=0)
        normalized_variances = (degree_variances - np.min(degree_variances)) / (
            np.max(degree_variances) - np.min(degree_variances)
        )

        for node in range(n_nodes):
            alpha = 0.1 + 0.6 * normalized_variances[node]
            ax1.plot(
                range(n_timesteps),
                degree_matrix[:, node],
                alpha=alpha,
                linewidth=0.8,
                color="gray",
            )

        mean_degree = np.mean(degree_matrix, axis=1)
        q1 = np.percentile(degree_matrix, 25, axis=1)
        q3 = np.percentile(degree_matrix, 75, axis=1)

        ax1.plot(
            range(n_timesteps),
            mean_degree,
            color="#e74c3c",
            linewidth=2.5,
            label="Mean Degree",
            zorder=5,
        )
        ax1.fill_between(
            range(n_timesteps),
            q1,
            q3,
            color="#e74c3c",
            alpha=0.2,
            label="25-75 Percentile",
            zorder=4,
        )

        change_points = [
            i
            for i, state in enumerate(network_series)
            if state.get("is_change_point", False)
        ]
        for cp in change_points:
            ax1.axvline(
                x=cp,
                color="#2ecc71",
                linestyle="--",
                alpha=0.5,
                label="Change Point" if cp == change_points[0] else None,
                zorder=3,
            )

        ax1.set_xlabel("Time Step", fontsize=PlotStyle.MEDIUM_SIZE)
        ax1.set_ylabel("Node Degree", fontsize=PlotStyle.MEDIUM_SIZE)
        ax1.set_title(
            "Node Degree Trajectories",
            pad=20,
            fontsize=PlotStyle.MEDIUM_SIZE + 2,
            weight="bold",
        )
        ax1.legend(fontsize=PlotStyle.SMALL_SIZE + 2)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Degree Distribution Evolution (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        max_degree = int(np.max(degree_matrix))
        degree_counts = np.zeros((n_timesteps, max_degree + 1))

        for t in range(n_timesteps):
            degrees = degree_matrix[t]
            for d in range(max_degree + 1):
                degree_counts[t, d] = np.sum(degrees == d)

        degree_counts = degree_counts / n_nodes
        degree_counts = np.log1p(degree_counts)

        im = ax2.imshow(
            degree_counts.T,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            interpolation="nearest",
        )

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Log(1 + Fraction of Nodes)", fontsize=PlotStyle.SMALL_SIZE + 2)

        for cp in change_points:
            ax2.axvline(x=cp, color="#2ecc71", linestyle="--", alpha=0.5)

        ax2.set_xlabel("Time Step", fontsize=PlotStyle.MEDIUM_SIZE)
        ax2.set_ylabel("Degree", fontsize=PlotStyle.MEDIUM_SIZE)
        ax2.set_title(
            "Degree Distribution Evolution",
            pad=20,
            fontsize=PlotStyle.MEDIUM_SIZE + 2,
            weight="bold",
        )

        # Degree Statistics (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        max_degrees = np.max(degree_matrix, axis=1)
        min_degrees = np.min(degree_matrix, axis=1)
        median_degrees = np.median(degree_matrix, axis=1)
        std_degrees = np.std(degree_matrix, axis=1)

        ax3.fill_between(
            range(n_timesteps),
            min_degrees,
            max_degrees,
            alpha=0.2,
            color="#3498db",
            label="Degree Range",
        )
        ax3.plot(
            range(n_timesteps),
            median_degrees,
            color="#e67e22",
            label="Median",
            linewidth=2,
        )
        ax3.plot(
            range(n_timesteps),
            std_degrees,
            color="#9b59b6",
            label="Standard Deviation",
            linewidth=2,
        )

        for cp in change_points:
            ax3.axvline(x=cp, color="#2ecc71", linestyle="--", alpha=0.5)

        ax3.set_xlabel("Time Step", fontsize=PlotStyle.MEDIUM_SIZE)
        ax3.set_ylabel("Value", fontsize=PlotStyle.MEDIUM_SIZE)
        ax3.set_title(
            "Degree Statistics Over Time",
            pad=20,
            fontsize=PlotStyle.MEDIUM_SIZE + 2,
            weight="bold",
        )
        ax3.legend(fontsize=PlotStyle.SMALL_SIZE + 2)
        ax3.grid(True, alpha=0.3, linestyle="--")

        # Top Nodes Analysis (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        avg_degrees = np.mean(degree_matrix, axis=0)
        top_by_avg = np.argsort(avg_degrees)[-3:]
        top_by_var = np.argsort(degree_variances)[-2:]

        for node in top_by_avg:
            ax4.plot(
                range(n_timesteps),
                degree_matrix[:, node],
                alpha=0.8,
                linewidth=2,
                label=f"Node {node} (Avg: {avg_degrees[node]:.1f})",
            )

        for node in top_by_var:
            if node not in top_by_avg:
                ax4.plot(
                    range(n_timesteps),
                    degree_matrix[:, node],
                    alpha=0.8,
                    linewidth=2,
                    label=f"Node {node} (Var: {degree_variances[node]:.1f})",
                )

        for cp in change_points:
            ax4.axvline(x=cp, color="#2ecc71", linestyle="--", alpha=0.5)

        ax4.set_xlabel("Time Step", fontsize=PlotStyle.MEDIUM_SIZE)
        ax4.set_ylabel("Degree", fontsize=PlotStyle.MEDIUM_SIZE)
        ax4.set_title(
            "Notable Nodes Analysis",
            pad=20,
            fontsize=PlotStyle.MEDIUM_SIZE + 2,
            weight="bold",
        )
        ax4.legend(
            fontsize=PlotStyle.SMALL_SIZE + 2,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        ax4.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

    def plot_prediction_dashboard(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        time_points: List[int],
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (20, 15),
    ) -> None:
        """Create a comprehensive dashboard combining network structure and adjacency matrix visualizations."""
        n_points = len(time_points)

        # Create figure
        fig, _ = self._create_figure_with_suptitle(
            "Network Prediction Dashboard", model_type, figsize
        )

        gs = fig.add_gridspec(
            n_points,
            2,
            width_ratios=[1, 1],
            height_ratios=[1] * n_points,
            hspace=0.3,
            wspace=0.15,
            top=0.85,
            bottom=0.05,
            left=0.05,
            right=0.85,
        )

        # Add prediction legend
        self._add_prediction_legend(fig)

        for idx, t in enumerate(time_points):
            if t < 0:
                t = len(predictions) + t

            G_actual = actual_series[t]["graph"]
            pred_idx = t - len(actual_series) + len(predictions)
            G_pred = predictions[pred_idx]["graph"]

            # Nested GridSpec for each row
            gs_row = gs[idx, :].subgridspec(
                2, 2, height_ratios=[1, 1], hspace=0.1, wspace=0.2
            )

            # Calculate edge metrics
            correct_edges, false_positive_edges, missed_edges, metrics = (
                self._calculate_network_metrics(G_actual, G_pred)
            )
            edge_colors = {
                "green": correct_edges,
                "red": false_positive_edges,
                "gray": missed_edges,
            }
            pos = nx.spring_layout(G_actual, seed=42)

            # 1. Network Structure (top row)
            ax_net_actual = fig.add_subplot(gs_row[0, 0])
            ax_net_pred = fig.add_subplot(gs_row[0, 1])

            self._draw_network(
                G_actual, ax_net_actual, pos, f"Network Structure (t={t})"
            )
            self._draw_network(
                G_pred,
                ax_net_pred,
                pos,
                self._format_metrics_text(metrics, t),
                is_predicted=True,
                edge_colors=edge_colors,
            )

            # 2. Adjacency Matrices (bottom row)
            ax_adj_actual = fig.add_subplot(gs_row[1, 0])
            ax_adj_pred = fig.add_subplot(gs_row[1, 1])

            adj_actual = actual_series[t]["adjacency"]
            adj_pred = predictions[pred_idx]["adjacency"]

            self._plot_adjacency_matrix(
                adj_actual, ax=ax_adj_actual, title="Adjacency Matrix"
            )
            self._plot_adjacency_matrix(
                adj_actual, adj_pred, ax=ax_adj_pred, title="Predicted Matrix"
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _draw_network(
        self,
        G: nx.Graph,
        ax: plt.Axes,
        pos: Dict[int, Tuple[float, float]],
        title: str,
        is_predicted: bool = False,
        edge_colors: Dict[str, List] = None,
    ) -> None:
        """Draw a network with consistent styling."""
        node_style = PlotStyle.NODE_STYLES["predicted" if is_predicted else "actual"]
        nx.draw_networkx_nodes(G, pos, ax=ax, **node_style)

        if edge_colors:
            # Use color-coded edges
            for color, edges in edge_colors.items():
                alpha = 0.7 if color in ["green", "red"] else 0.3
                style = "dashed" if color == "gray" else "solid"
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edgelist=edges,
                    edge_color=color,
                    alpha=alpha,
                    width=0.5,
                    style=style,
                )
        else:
            # Default edges
            nx.draw_networkx_edges(
                G, pos, ax=ax, edge_color="gray", alpha=0.4, width=0.5
            )

        ax.set_title(title, pad=10, fontsize=PlotStyle.MEDIUM_SIZE)
        ax.axis("off")

    def _plot_adjacency_matrix(
        self,
        adj_actual: np.ndarray,
        adj_pred: np.ndarray = None,
        ax: plt.Axes = None,
        title: str = "",
    ) -> None:
        """Plot adjacency matrix with consistent styling."""
        if adj_pred is None:
            ax.imshow(adj_actual, cmap="Blues")
        else:
            colored_pred = self._create_colored_pred_matrix(adj_actual, adj_pred)
            ax.imshow(colored_pred)

        ax.set_title(title, pad=10, fontsize=PlotStyle.MEDIUM_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_change_points(
        self,
        ax: plt.Axes,
        change_points: List[int],
        show_label: bool = True,
    ) -> None:
        """Plot change points with consistent styling."""
        for cp in change_points:
            ax.axvline(
                x=cp,
                color="#2ecc71",
                linestyle="--",
                alpha=0.5,
                label="Change Point" if show_label and cp == change_points[0] else None,
                zorder=3,
            )

    def _format_metrics_text(
        self, metrics: Dict[str, Union[int, float]], t: int, compact: bool = False
    ) -> str:
        """Format metrics text with consistent styling."""
        if compact:
            return (
                f"Predicted (t={t})\n"
                f"Coverage: {metrics['coverage']:.1%} | FPR: {metrics['fpr']:.1%}"
            )

        return (
            f"Prediction Metrics:\n"
            f"Coverage: {metrics['coverage']:.1%} | FPR: {metrics['fpr']:.1%}\n"
            f"Correct: {metrics['correct_count']} | "
            f"FP: {metrics['false_positive_count']} | "
            f"Missed: {metrics['missed_count']}"
        )

    def plot_performance_extremes(
        self,
        actual_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        min_history: int,
        model_type: str = "Unknown",
        figsize: Tuple[int, int] = (20, 20),
    ) -> None:
        """Plot network comparisons at time steps with best, worst, and average performance."""
        # Calculate scores for all predictions
        scores = []
        for t in range(len(predictions)):
            true_adj = actual_series[t]["adjacency"]
            pred_adj = predictions[t]["adjacency"]
            metrics = self._calculate_adjacency_metrics(true_adj, pred_adj)
            coverage = metrics["coverage"]
            fpr = metrics["fpr"]
            score = (
                2 * coverage * (1 - fpr) / (coverage + (1 - fpr))
                if coverage + (1 - fpr) > 0
                else 0
            )
            scores.append(
                {
                    "time": t + min_history,
                    "score": score,
                    "coverage": coverage,
                    "fpr": fpr,
                    "index": t,  # Store original index for correct data access
                }
            )

        # Find best, worst, and average time points
        score_values = [s["score"] for s in scores]
        best_idx = np.argmax(score_values)
        worst_idx = np.argmin(score_values)

        # Find point closest to mean score
        mean_score = np.mean(score_values)
        avg_idx = min(
            range(len(scores)), key=lambda i: abs(scores[i]["score"] - mean_score)
        )

        best_point = scores[best_idx]
        worst_point = scores[worst_idx]
        avg_point = scores[avg_idx]

        time_points = [best_point["time"], avg_point["time"], worst_point["time"]]
        point_labels = ["Best", "Average", "Worst"]
        point_indices = [
            best_point["index"],
            avg_point["index"],
            worst_point["index"],
        ]  # For accessing data

        # Create figure
        fig, model_name = self._create_figure_with_suptitle(
            "Network Prediction Performance Analysis", model_type, figsize
        )

        # Create two grid specs: one for networks and one for metric plots
        gs_main = fig.add_gridspec(
            4,
            1,  # 4 rows: 3 for networks, 1 for metrics
            height_ratios=[1, 1, 1, 0.8],
            hspace=0.4,
            top=0.85,
            bottom=0.08,
            right=0.85,  # Leave space for legend
        )

        # Grid spec for network visualizations (top 3 rows)
        gs_nets = gs_main[0:3, 0].subgridspec(
            3,
            4,  # 3 rows (best/avg/worst), 4 cols (actual graph, pred graph, actual adj, pred adj)
            height_ratios=[1, 1, 1],
            width_ratios=[1, 1, 1, 1],
            hspace=0.4,
            wspace=0.3,
        )

        # Grid spec for metric plots (bottom row)
        gs_metrics = gs_main[3, 0].subgridspec(
            1,
            3,  # 1 row, 3 cols for Score, Coverage, FPR
            width_ratios=[1, 1, 1],
            wspace=0.3,
        )

        # Plot networks (reusing existing code)
        for idx, (t, label, orig_idx) in enumerate(
            zip(time_points, point_labels, point_indices)
        ):
            # Get actual and predicted networks using original index
            G_actual = actual_series[orig_idx]["graph"]
            G_pred = predictions[orig_idx]["graph"]
            pos = nx.spring_layout(G_actual, seed=42)

            # Calculate metrics
            correct_edges, false_positive_edges, missed_edges, metrics = (
                self._calculate_network_metrics(G_actual, G_pred)
            )

            # Calculate score for display
            coverage = metrics["coverage"]
            fpr = metrics["fpr"]
            score = (
                2 * coverage * (1 - fpr) / (coverage + (1 - fpr))
                if coverage + (1 - fpr) > 0
                else 0
            )
            metrics["score"] = score

            # Plot actual network
            ax_actual = fig.add_subplot(gs_nets[idx, 0])
            nx.draw_networkx_nodes(
                G_actual, pos, ax=ax_actual, **PlotStyle.NODE_STYLES["actual"]
            )
            nx.draw_networkx_edges(
                G_actual, pos, ax=ax_actual, edge_color="black", alpha=0.6
            )
            ax_actual.set_title(
                f"{label} Actual Network (t={t})",  # Use adjusted time
                pad=10,
                fontsize=PlotStyle.MEDIUM_SIZE,
            )
            ax_actual.axis("off")

            # Plot predicted network with color-coded edges
            ax_pred = fig.add_subplot(gs_nets[idx, 1])
            nx.draw_networkx_nodes(
                G_actual, pos, ax=ax_pred, **PlotStyle.NODE_STYLES["actual"]
            )
            self._draw_color_coded_edges(
                G_pred, pos, ax_pred, correct_edges, false_positive_edges, missed_edges
            )

            # Add metrics text as a vertical list with semi-transparent background
            metrics_text = (
                f"Coverage: {metrics['coverage']:.3f}\n"
                f"FPR: {metrics['fpr']:.3f}\n"
                f"Score: {metrics['score']:.3f}"
            )
            ax_pred.set_title(
                f"{label} Predicted Network", pad=10, fontsize=PlotStyle.MEDIUM_SIZE
            )
            # Add metrics text box with semi-transparent background
            ax_pred.text(
                0.02,
                0.98,  # Position in top-left
                metrics_text,
                transform=ax_pred.transAxes,
                fontsize=PlotStyle.SMALL_SIZE + 1,
                verticalalignment="top",
                bbox=dict(
                    facecolor="whitesmoke",
                    alpha=0.7,
                    edgecolor="none",
                    pad=5,
                    boxstyle="round,pad=0.5",
                ),
            )
            ax_pred.axis("off")

            # Plot actual adjacency matrix
            ax_adj_actual = fig.add_subplot(gs_nets[idx, 2])
            ax_adj_actual.imshow(actual_series[orig_idx]["adjacency"], cmap="Blues")
            ax_adj_actual.set_title(
                f"{label} Actual Adjacency", pad=10, fontsize=PlotStyle.MEDIUM_SIZE
            )
            ax_adj_actual.set_xticks([])
            ax_adj_actual.set_yticks([])

            # Plot predicted adjacency matrix
            ax_adj_pred = fig.add_subplot(gs_nets[idx, 3])
            colored_adj = self._create_colored_adjacency(
                actual_series[orig_idx]["adjacency"], predictions[orig_idx]["adjacency"]
            )
            ax_adj_pred.imshow(colored_adj)
            ax_adj_pred.set_title(
                f"{label} Predicted Adjacency", pad=10, fontsize=PlotStyle.MEDIUM_SIZE
            )
            ax_adj_pred.set_xticks([])
            ax_adj_pred.set_yticks([])

        # Plot metric evolution
        times = [s["time"] for s in scores]
        metrics_data = {
            "Score": [s["score"] for s in scores],
            "Coverage": [s["coverage"] for s in scores],
            "FPR": [s["fpr"] for s in scores],
        }

        # Get change points
        change_points = [
            i + min_history
            for i, state in enumerate(actual_series)
            if state.get("is_change_point", False)
        ]

        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = fig.add_subplot(gs_metrics[0, idx])

            # Calculate and plot average performance band
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axhline(y=mean_val, color="gray", linestyle="--", alpha=0.5)
            ax.fill_between(
                times, mean_val - std_val, mean_val + std_val, color="gray", alpha=0.2
            )

            # Plot metric values
            ax.plot(times, values, "-o", markersize=3, zorder=3, color="blue")

            # Plot change points
            for cp in change_points:
                ax.axvline(x=cp, color="purple", linestyle=":", alpha=0.5)

            # Highlight best, average, and worst points
            highlight_points = [
                (best_point["time"], best_point["index"], "Best", "green"),
                (avg_point["time"], avg_point["index"], "Average", "blue"),
                (worst_point["time"], worst_point["index"], "Worst", "red"),
            ]

            for t, orig_idx, label, color in highlight_points:
                if metric_name == "Score":
                    metric_value = scores[orig_idx]["score"]
                elif metric_name == "Coverage":
                    metric_value = scores[orig_idx]["coverage"]
                else:  # FPR
                    metric_value = scores[orig_idx]["fpr"]
                ax.plot(t, metric_value, "o", color=color, markersize=8, zorder=4)

            ax.set_xlabel("Time Step")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} Evolution")
            ax.grid(True, alpha=0.3)

            # Add mean ± std text
            text_y = 0.02 if metric_name == "FPR" else 0.98
            va = "bottom" if metric_name == "FPR" else "top"
            ax.text(
                0.98,
                text_y,
                f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}",
                transform=ax.transAxes,
                ha="right",
                va=va,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
                fontsize=PlotStyle.SMALL_SIZE,
            )

        # Add prediction legend with adjusted position (centered in middle row)
        self._add_prediction_legend(
            fig,
            position=[0.87, 0.45, 0.12, 0.2],  # Centered in middle row
            title="Prediction Types\n\nScore: harmonic mean of\ncoverage and (1-FPR)",
        )

        # Add metrics evolution legend (aligned with metric plots)
        metrics_legend_ax = fig.add_axes(
            [0.87, 0.08, 0.12, 0.2]
        )  # Aligned with bottom row
        metrics_legend_ax.axis("off")

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="blue",
                marker="o",
                markersize=8,
                label="Time Series",
                linestyle="-",
                linewidth=1,
            ),
            plt.Line2D(
                [0],
                [0],
                color="green",
                marker="o",
                markersize=8,
                label="Best",
                linestyle="",
            ),
            plt.Line2D(
                [0],
                [0],
                color="blue",
                marker="o",
                markersize=8,
                label="Average",
                linestyle="",
            ),
            plt.Line2D(
                [0],
                [0],
                color="red",
                marker="o",
                markersize=8,
                label="Worst",
                linestyle="",
            ),
            plt.Line2D([0], [0], color="gray", linestyle="--", label="Mean"),
            plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.2, label="±1 std dev"),
            plt.Line2D([0], [0], color="purple", linestyle=":", label="Change Point"),
        ]

        metrics_legend = metrics_legend_ax.legend(
            handles=legend_elements,
            loc="center",
            title="Metrics Evolution",
            fontsize=PlotStyle.SMALL_SIZE,
        )
        metrics_legend.get_title().set_fontweight("bold")
        metrics_legend.get_title().set_fontsize(PlotStyle.MEDIUM_SIZE)

        plt.tight_layout(rect=[0, 0.08, 0.85, 0.95])

    def create_martingale_comparison_dashboard(
        self,
        network_series: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        min_history: int,
        actual_martingales: Dict[str, Dict],
        pred_martingales: Dict[str, Dict],
        actual_shap: np.ndarray,
        pred_shap: np.ndarray,
        output_path: Path,
        threshold: float = 30.0,
    ) -> None:
        """Create a simplified dashboard comparing actual and predicted martingales with SHAP values."""
        # Get feature names from network metrics
        feature_names = ["degree", "clustering", "betweenness", "closeness"]

        # Get actual change points from network series
        change_points = [
            i
            for i, state in enumerate(network_series)
            if state.get("is_change_point", False)
        ]

        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3
        )

        # Color palette for features
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))

        # 1. Reset Martingales (Top Left)
        ax_actual_mart = fig.add_subplot(gs[0, 0])
        for idx, feature in enumerate(feature_names):
            values = actual_martingales["reset"][feature]["martingales"]
            ax_actual_mart.plot(
                values,
                label=feature.capitalize(),
                color=colors[idx],
                linewidth=1.5,
                alpha=0.6,
            )

        # Plot threshold line
        ax_actual_mart.axhline(
            y=threshold, color="k", linestyle="--", alpha=0.5, label="Threshold"
        )

        # Plot actual change points
        for cp in change_points:
            ax_actual_mart.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

        # Plot combined martingales
        martingale_arrays = []
        for feature in feature_names:
            values = actual_martingales["reset"][feature]["martingales"]
            martingale_arrays.append(values)

        M_sum = np.sum(martingale_arrays, axis=0)
        M_avg = M_sum / len(feature_names)

        ax_actual_mart.plot(
            M_avg, color="#FF4B4B", label="Average", linewidth=2.5, alpha=0.9
        )
        ax_actual_mart.plot(
            M_sum,
            color="#2F2F2F",
            label="Sum",
            linewidth=2.5,
            linestyle="-.",
            alpha=0.8,
        )

        # Customize reset martingales plot
        ax_actual_mart.grid(True, linestyle="--", alpha=0.3)
        ax_actual_mart.yaxis.set_minor_locator(AutoMinorLocator())
        ax_actual_mart.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
        ax_actual_mart.set_xlabel("Time Steps", fontsize=12, labelpad=5)
        ax_actual_mart.set_ylabel("Martingale Values", fontsize=12, labelpad=10)
        ax_actual_mart.set_title("Reset Martingale Measures", fontsize=12, pad=15)
        legend = ax_actual_mart.legend(
            fontsize=10,
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(1, 1.02),
            frameon=True,
            facecolor="none",
            edgecolor="none",
        )
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_alpha(0)

        # 2. SHAP Values Over Time (Top Right)
        ax_actual_shap = fig.add_subplot(gs[0, 1])
        for idx, feature in enumerate(feature_names):
            ax_actual_shap.plot(
                actual_shap[:, idx],
                label=feature.capitalize(),
                color=colors[idx],
                linewidth=1.5,
                alpha=0.7,
            )

        # Add actual change point indicators
        for cp in change_points:
            ax_actual_shap.axvline(x=cp, color="red", linestyle="--", alpha=0.3)

        ax_actual_shap.set_title("SHAP Values Over Time", fontsize=12, pad=20)
        ax_actual_shap.set_xlabel("Time Steps", fontsize=10)
        ax_actual_shap.set_ylabel("Feature Importance", fontsize=10)
        ax_actual_shap.legend(fontsize=8, title="Centrality Measures")
        ax_actual_shap.grid(True, alpha=0.3)

        # 3. Predicted Martingales (Bottom Left)
        ax_pred_mart = fig.add_subplot(gs[1, 0])
        for idx, feature in enumerate(feature_names):
            values = pred_martingales["reset"][feature]["martingales"]
            ax_pred_mart.plot(
                values,
                label=feature.capitalize(),
                color=colors[idx],
                linewidth=1.5,
                alpha=0.6,
            )

        # Plot threshold line
        ax_pred_mart.axhline(
            y=threshold, color="k", linestyle="--", alpha=0.5, label="Threshold"
        )

        # Plot actual change points
        for cp in change_points:
            ax_pred_mart.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

        # Plot combined predicted martingales
        pred_martingale_arrays = []
        for feature in feature_names:
            values = pred_martingales["reset"][feature]["martingales"]
            pred_martingale_arrays.append(values)

        M_sum_pred = np.sum(pred_martingale_arrays, axis=0)
        M_avg_pred = M_sum_pred / len(feature_names)

        ax_pred_mart.plot(
            M_avg_pred, color="#FF4B4B", label="Average", linewidth=2.5, alpha=0.9
        )
        ax_pred_mart.plot(
            M_sum_pred,
            color="#2F2F2F",
            label="Sum",
            linewidth=2.5,
            linestyle="-.",
            alpha=0.8,
        )

        # Customize predicted martingales plot
        ax_pred_mart.grid(True, linestyle="--", alpha=0.3)
        ax_pred_mart.yaxis.set_minor_locator(AutoMinorLocator())
        ax_pred_mart.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
        ax_pred_mart.set_xlabel("Time Steps", fontsize=12, labelpad=5)
        ax_pred_mart.set_ylabel("Martingale Values", fontsize=12, labelpad=10)
        ax_pred_mart.set_title(
            "Predicted Reset Martingale Measures", fontsize=12, pad=15
        )
        legend = ax_pred_mart.legend(
            fontsize=10,
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(1, 1.02),
            frameon=True,
            facecolor="none",
            edgecolor="none",
        )
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_alpha(0)

        # 4. Predicted SHAP Values (Bottom Right)
        ax_pred_shap = fig.add_subplot(gs[1, 1])
        for idx, feature in enumerate(feature_names):
            ax_pred_shap.plot(
                pred_shap[:, idx],
                label=feature.capitalize(),
                color=colors[idx],
                linewidth=1.5,
                alpha=0.7,
            )

        # Add actual change point indicators
        for cp in change_points:
            ax_pred_shap.axvline(x=cp, color="red", linestyle="--", alpha=0.3)

        ax_pred_shap.set_title("Predicted SHAP Values Over Time", fontsize=12, pad=20)
        ax_pred_shap.set_xlabel("Time Steps", fontsize=10)
        ax_pred_shap.set_ylabel("Feature Importance", fontsize=10)
        ax_pred_shap.legend(fontsize=8, title="Centrality Measures")
        ax_pred_shap.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

    def _compute_edge_coverage(self, G_actual: nx.Graph, G_pred: nx.Graph) -> float:
        """Compute edge coverage (true positive rate) between actual and predicted graphs."""
        actual_edges = set(G_actual.edges())
        pred_edges = set(G_pred.edges())
        if not actual_edges:
            return 1.0  # If no actual edges, consider perfect coverage
        return len(actual_edges.intersection(pred_edges)) / len(actual_edges)

    def _compute_false_positive_rate(
        self, G_actual: nx.Graph, G_pred: nx.Graph
    ) -> float:
        """Compute false positive rate between actual and predicted graphs."""
        actual_edges = set(G_actual.edges())
        pred_edges = set(G_pred.edges())
        false_positives = len(pred_edges - actual_edges)
        true_negatives = (
            len(G_actual.nodes()) * (len(G_actual.nodes()) - 1)
        ) // 2 - len(actual_edges)
        if true_negatives == 0:
            return 0.0  # If no possible true negatives, consider perfect FPR
        return false_positives / true_negatives

    def _plot_network_comparison(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        actual: Dict[str, Any],
        pred: Dict[str, Any],
        label: str,
        metrics: Dict[str, float],
    ) -> None:
        """Plot network comparison with actual and predicted graphs."""
        # Create a subgridspec for this row
        subgs = gs.subgridspec(1, 4, wspace=0.3)

        # Get graphs
        G_actual = actual["graph"]
        G_pred = pred["graph"]

        # Use same layout for both graphs
        pos = nx.spring_layout(G_actual, seed=42)

        # Plot actual network
        ax_actual = fig.add_subplot(subgs[0])
        nx.draw(
            G_actual,
            pos,
            ax=ax_actual,
            node_color="lightblue",
            node_size=100,
            edge_color="black",
            alpha=0.7,
            with_labels=False,
        )
        ax_actual.set_title(f"{label} Actual Network (t={actual['time']})")

        # Plot predicted network
        ax_pred = fig.add_subplot(subgs[1])
        nx.draw(
            G_pred,
            pos,
            ax=ax_pred,
            node_color="lightblue",
            node_size=100,
            edge_color="black",
            alpha=0.7,
            with_labels=False,
        )
        ax_pred.set_title(
            f"{label} Predicted Network\nCoverage: {metrics['coverage']:.3f}, FPR: {metrics['fpr']:.3f}"
        )

        # Plot actual adjacency matrix
        ax_adj_actual = fig.add_subplot(subgs[2])
        ax_adj_actual.imshow(actual["adjacency"], cmap="Blues")
        ax_adj_actual.set_title(f"{label} Actual Adjacency")
        ax_adj_actual.axis("off")

        # Plot predicted adjacency matrix with color-coded differences
        ax_adj_pred = fig.add_subplot(subgs[3])
        colored_adj = self._create_colored_adjacency(
            actual["adjacency"], pred["adjacency"]
        )
        ax_adj_pred.imshow(colored_adj)
        ax_adj_pred.set_title(f"{label} Predicted Adjacency")
        ax_adj_pred.axis("off")

    def _create_colored_adjacency(
        self, actual_adj: np.ndarray, pred_adj: np.ndarray
    ) -> np.ndarray:
        """Create a colored adjacency matrix highlighting differences."""
        # Create RGB array (green for correct, red for false positive, gray for missed)
        colored = np.zeros((*actual_adj.shape, 3))

        # Correct predictions (green)
        correct = actual_adj == pred_adj
        colored[correct & (actual_adj == 1)] = [0, 1, 0]  # Green for correct edges

        # False positives (red)
        colored[(actual_adj == 0) & (pred_adj == 1)] = [1, 0, 0]

        # Missed edges (gray)
        colored[(actual_adj == 1) & (pred_adj == 0)] = [
            0.7,
            0.7,
            0.7,
        ]  # Changed from blue to gray

        return colored
