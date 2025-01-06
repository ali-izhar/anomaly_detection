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
        figsize: Tuple[int, int] = (20, 15),
    ) -> None:
        """Create an enhanced visualization of network metric evolution over time.

        Parameters
        ----------
        actual_series : List[Dict]
            List of actual network states
        predictions : List[Dict]
            List of predicted network states
        min_history : int
            Minimum history length before predictions start
        model_type : str
            Type of network model being analyzed
        figsize : Tuple[int, int]
            Figure size (width, height) in inches
        """
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
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")

        # Create title with model name formatting
        model_name = {
            "ba": "Barabási-Albert",
            "er": "Erdős-Rényi",
            "ws": "Watts-Strogatz",
            "sbm": "Stochastic Block Model",
            "rcp": "Random Core-Periphery",
            "lfr": "LFR Benchmark",
        }.get(model_type.lower(), model_type)

        # Create GridSpec with space for title and legend
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[1, 1, 1],
            width_ratios=[1.2, 1],
            hspace=0.25,
            wspace=0.25,
            top=0.85,
        )

        fig.suptitle(
            f"Evolution of Network Metrics - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE + 2,
            y=0.95,
            weight="bold",
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

        # Create legend axes on the right
        legend_ax = fig.add_axes([0.88, 0.4, 0.12, 0.25])  # Moved back to the right
        legend_ax.axis("off")
        legend_elements = [
            Patch(
                facecolor=PlotStyle.LINE_STYLES["actual"]["color"],
                alpha=PlotStyle.LINE_STYLES["actual"]["alpha"],
                label="Actual",
            ),
            Patch(
                facecolor=PlotStyle.LINE_STYLES["predicted"]["color"],
                alpha=PlotStyle.LINE_STYLES["predicted"]["alpha"],
                label="Predicted",
            ),
            Patch(
                facecolor=PlotStyle.LINE_STYLES["change_point"]["color"],
                alpha=PlotStyle.LINE_STYLES["change_point"]["alpha"],
                label="Change Point",
            ),
            Patch(facecolor="#95a5a6", alpha=0.5, label="Prediction Start"),
        ]
        legend = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            title="Series Types",
            fontsize=PlotStyle.MEDIUM_SIZE,
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(PlotStyle.MEDIUM_SIZE)

        # Plot each metric with enhanced styling
        for idx, (metric_name, title, description) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # Add subtle background grid
            ax.grid(True, alpha=0.2, linestyle="--", color=PlotStyle.GRID_COLOR)
            ax.set_facecolor(PlotStyle.BACKGROUND_COLOR)

            # Plot change points with enhanced visibility
            for cp in change_points:
                ax.axvline(
                    x=cp,
                    label="Change Point" if cp == change_points[0] and idx == 0 else "",
                    **PlotStyle.LINE_STYLES["change_point"],
                )

            # Extract metric values
            actual_values = [m[metric_name] for m in actual_metrics]
            pred_values = [m[metric_name] for m in pred_metrics]

            # Plot actual values with confidence interval
            ax.plot(
                times,
                actual_values,
                label="Actual" if idx == 0 else "",
                **PlotStyle.LINE_STYLES["actual"],
            )

            # Plot predicted values
            ax.plot(
                pred_times,
                pred_values,
                label="Predicted" if idx == 0 else "",
                **PlotStyle.LINE_STYLES["predicted"],
            )

            # Add min history marker
            ax.axvline(
                x=min_history,
                color="#95a5a6",
                linestyle=":",
                alpha=0.5,
                label="Prediction Start" if idx == 0 else "",
            )

            # Enhanced title and labels
            ax.set_title(
                title, pad=10, fontsize=PlotStyle.MEDIUM_SIZE + 2, weight="bold"
            )
            ax.set_xlabel("Time Step", fontsize=PlotStyle.MEDIUM_SIZE)
            ax.set_ylabel("Value", fontsize=PlotStyle.MEDIUM_SIZE)

            # Add metric description as text box
            ax.text(
                0.98,
                0.02,
                description,
                transform=ax.transAxes,
                fontsize=PlotStyle.SMALL_SIZE,
                style="italic",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
            )

            # Calculate and display error metrics for predictions
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
                    ax.text(
                        0.98,
                        0.98,
                        error_text,
                        transform=ax.transAxes,
                        fontsize=PlotStyle.SMALL_SIZE,
                        ha="right",
                        va="top",
                        bbox=dict(
                            facecolor="white", alpha=0.8, edgecolor="none", pad=3
                        ),
                    )

            # Customize tick labels
            ax.tick_params(labelsize=PlotStyle.SMALL_SIZE)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

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

        # Create legend axes on the right
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

            # Plot actual network
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

            # Plot predicted network with color-coded edges
            ax_pred = fig.add_subplot(gs[1, i])
            pred_idx = t - len(actual_series) + len(predictions)
            G_pred = predictions[pred_idx]["graph"]

            # Draw nodes
            nx.draw_networkx_nodes(
                G_pred, pos, ax=ax_pred, **PlotStyle.NODE_STYLES["predicted"]
            )

            # Color-code edges based on prediction accuracy
            correct_edges = []
            false_positive_edges = []
            missed_edges = []

            # Find correct and false positive edges
            for e in G_pred.edges():
                if G_actual.has_edge(*e):
                    correct_edges.append(e)
                else:
                    false_positive_edges.append(e)

            # Find missed edges
            for e in G_actual.edges():
                if not G_pred.has_edge(*e):
                    missed_edges.append(e)

            # Draw edges with different colors
            nx.draw_networkx_edges(
                G_pred,
                pos,
                ax=ax_pred,
                edgelist=correct_edges,
                edge_color="green",
                alpha=0.7,
                width=0.5,
            )
            nx.draw_networkx_edges(
                G_pred,
                pos,
                ax=ax_pred,
                edgelist=false_positive_edges,
                edge_color="red",
                alpha=0.7,
                width=0.5,
            )
            nx.draw_networkx_edges(
                G_pred,
                pos,
                ax=ax_pred,
                edgelist=missed_edges,
                edge_color="gray",
                alpha=0.3,
                width=0.5,
                style="dashed",
            )

            # Calculate metrics
            total_actual = len(G_actual.edges())
            total_predicted = len(G_pred.edges())
            correct_count = len(correct_edges)
            false_positive_count = len(false_positive_edges)
            missed_count = len(missed_edges)

            coverage = correct_count / total_actual if total_actual > 0 else 0
            fpr = false_positive_count / total_predicted if total_predicted > 0 else 0

            # Add metrics as text
            metrics_text = (
                f"Predicted (t={t})\n"
                f"Coverage: {coverage:.1%}\n"
                f"FPR: {fpr:.1%}\n"
                f"Correct: {correct_count}\n"
                f"False+: {false_positive_count}\n"
                f"Missed: {missed_count}"
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

    def plot_node_degree_evolution(
        self,
        network_series: List[Dict[str, Any]],
        output_path: str = None,
        figsize: Tuple[int, int] = (20, 15),
    ) -> None:
        """Create a comprehensive dashboard for node degree evolution analysis.

        Parameters
        ----------
        network_series : List[Dict]
            List of network states over time
        output_path : str, optional
            Path to save the plot. If None, display only.
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches
        """
        # Extract data
        n_nodes = len(network_series[0]["graph"].nodes())
        n_timesteps = len(network_series)
        degree_matrix = np.zeros((n_timesteps, n_nodes))

        for t, state in enumerate(network_series):
            degrees = dict(state["graph"].degree())
            for node in range(n_nodes):
                degree_matrix[t, node] = degrees[node]

        # Create figure with GridSpec for layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.2, 1],
            width_ratios=[1.2, 1],
            hspace=0.25,
            wspace=0.25,
        )

        # Add title
        fig.suptitle(
            "Network Degree Evolution Analysis",
            fontsize=PlotStyle.LARGE_SIZE + 2,
            y=0.95,
            weight="bold",
        )

        # 1. Main Evolution Plot (top left) - Enhanced version
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot individual node trajectories with gradient alpha based on degree variance
        degree_variances = np.var(degree_matrix, axis=0)
        normalized_variances = (degree_variances - np.min(degree_variances)) / (
            np.max(degree_variances) - np.min(degree_variances)
        )

        for node in range(n_nodes):
            alpha = (
                0.1 + 0.6 * normalized_variances[node]
            )  # More variable nodes are more visible
            ax1.plot(
                range(n_timesteps),
                degree_matrix[:, node],
                alpha=alpha,
                linewidth=0.8,
                color="gray",
            )

        # Plot mean and quartiles
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

        # Add change points if they exist
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

        # 2. Degree Distribution Evolution (top right) - Enhanced heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        max_degree = int(np.max(degree_matrix))
        degree_counts = np.zeros((n_timesteps, max_degree + 1))

        for t in range(n_timesteps):
            degrees = degree_matrix[t]
            for d in range(max_degree + 1):
                degree_counts[t, d] = np.sum(degrees == d)

        # Normalize and apply log scale for better visualization
        degree_counts = degree_counts / n_nodes
        degree_counts = np.log1p(degree_counts)  # log1p to handle zeros

        im = ax2.imshow(
            degree_counts.T,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            interpolation="nearest",
        )

        # Add colorbar with scientific notation
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Log(1 + Fraction of Nodes)", fontsize=PlotStyle.SMALL_SIZE + 2)

        # Add change points to heatmap
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

        # 3. Degree Statistics (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])

        # Calculate various statistics
        max_degrees = np.max(degree_matrix, axis=1)
        min_degrees = np.min(degree_matrix, axis=1)
        median_degrees = np.median(degree_matrix, axis=1)
        std_degrees = np.std(degree_matrix, axis=1)

        # Plot with enhanced styling
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

        # Add change points
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

        # 4. Top Nodes Analysis (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])

        # Find nodes with highest average and variance
        avg_degrees = np.mean(degree_matrix, axis=0)
        top_by_avg = np.argsort(avg_degrees)[-3:]  # Top 3 by average
        top_by_var = np.argsort(degree_variances)[-2:]  # Top 2 by variance

        # Plot top nodes
        for node in top_by_avg:
            ax4.plot(
                range(n_timesteps),
                degree_matrix[:, node],
                alpha=0.8,
                linewidth=2,
                label=f"Node {node} (Avg: {avg_degrees[node]:.1f})",
            )

        for node in top_by_var:
            if node not in top_by_avg:  # Avoid duplicates
                ax4.plot(
                    range(n_timesteps),
                    degree_matrix[:, node],
                    alpha=0.8,
                    linewidth=2,
                    label=f"Node {node} (Var: {degree_variances[node]:.1f})",
                )

        # Add change points
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

        # Adjust layout and save
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
        """Create a comprehensive dashboard combining network structure and adjacency matrix visualizations.

        For each time point, shows:
        - Network structure (actual vs predicted)
        - Adjacency matrix (actual vs predicted)
        - Prediction metrics
        - Color-coded edges and matrix cells
        """
        n_points = len(time_points)

        # Create figure with GridSpec for layout
        fig = plt.figure(figsize=figsize)

        # Create main title
        model_name = (
            "Barabási-Albert"
            if model_type.lower() == "ba"
            else "Erdős-Rényi" if model_type.lower() == "er" else model_type
        )
        fig.suptitle(
            f"Network Prediction Dashboard - {model_name} Model",
            fontsize=PlotStyle.LARGE_SIZE + 2,
            y=0.98,
            weight="bold",
        )

        # Create a grid with n_points rows and space for legend
        gs = fig.add_gridspec(
            n_points,
            2,
            width_ratios=[1, 1],
            height_ratios=[1] * n_points,
            hspace=0.3,
            wspace=0.15,
            top=0.92,
            bottom=0.05,
            left=0.05,
            right=0.85,
        )

        # Create legend axes on the right
        legend_ax = fig.add_axes([0.87, 0.4, 0.12, 0.2])
        legend_ax.axis("off")
        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Correct Prediction"),
            Patch(facecolor="red", alpha=0.7, label="False Positive"),
            Patch(facecolor="gray", alpha=0.3, label="Missed Edge"),
        ]
        legend = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            title="Prediction Types",
            fontsize=PlotStyle.MEDIUM_SIZE,
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(PlotStyle.MEDIUM_SIZE)

        for idx, t in enumerate(time_points):
            if t < 0:
                t = len(predictions) + t

            # Get actual and predicted networks
            G_actual = actual_series[t]["graph"]
            pred_idx = t - len(actual_series) + len(predictions)
            G_pred = predictions[pred_idx]["graph"]

            # Create nested GridSpec for each row
            gs_row = gs[idx, :].subgridspec(
                2, 2, height_ratios=[1, 1], hspace=0.1, wspace=0.2
            )

            # 1. Network Structure (top row)
            ax_net_actual = fig.add_subplot(gs_row[0, 0])
            ax_net_pred = fig.add_subplot(gs_row[0, 1])

            # Common node positions
            pos = nx.spring_layout(G_actual, seed=42)

            # Plot actual network
            nx.draw_networkx_nodes(
                G_actual, pos, ax=ax_net_actual, **PlotStyle.NODE_STYLES["actual"]
            )
            nx.draw_networkx_edges(
                G_actual, pos, ax=ax_net_actual, edge_color="gray", alpha=0.4, width=0.5
            )
            ax_net_actual.set_title(
                f"Network Structure (t={t})", pad=10, fontsize=PlotStyle.MEDIUM_SIZE
            )
            ax_net_actual.axis("off")

            # Plot predicted network with color-coded edges
            nx.draw_networkx_nodes(
                G_pred, pos, ax=ax_net_pred, **PlotStyle.NODE_STYLES["predicted"]
            )

            # Color-code edges
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

            # Draw edges with different colors
            nx.draw_networkx_edges(
                G_pred,
                pos,
                ax=ax_net_pred,
                edgelist=correct_edges,
                edge_color="green",
                alpha=0.7,
                width=0.5,
            )
            nx.draw_networkx_edges(
                G_pred,
                pos,
                ax=ax_net_pred,
                edgelist=false_positive_edges,
                edge_color="red",
                alpha=0.7,
                width=0.5,
            )
            nx.draw_networkx_edges(
                G_pred,
                pos,
                ax=ax_net_pred,
                edgelist=missed_edges,
                edge_color="gray",
                alpha=0.3,
                width=0.5,
                style="dashed",
            )

            # Calculate metrics
            total_actual = len(G_actual.edges())
            total_predicted = len(G_pred.edges())
            correct_count = len(correct_edges)
            false_positive_count = len(false_positive_edges)
            missed_count = len(missed_edges)
            coverage = correct_count / total_actual if total_actual > 0 else 0
            fpr = false_positive_count / total_predicted if total_predicted > 0 else 0

            metrics_text = (
                f"Prediction Metrics:\n"
                f"Coverage: {coverage:.1%} | FPR: {fpr:.1%}\n"
                f"Correct: {correct_count} | False+: {false_positive_count} | Missed: {missed_count}"
            )
            ax_net_pred.set_title(metrics_text, pad=10, fontsize=PlotStyle.MEDIUM_SIZE)
            ax_net_pred.axis("off")

            # 2. Adjacency Matrices (bottom row)
            ax_adj_actual = fig.add_subplot(gs_row[1, 0])
            ax_adj_pred = fig.add_subplot(gs_row[1, 1])

            # Plot actual adjacency
            adj_actual = actual_series[t]["adjacency"]
            im_actual = ax_adj_actual.imshow(adj_actual, cmap="Blues")
            ax_adj_actual.set_title(
                "Adjacency Matrix", pad=10, fontsize=PlotStyle.MEDIUM_SIZE
            )

            # Create color-coded prediction matrix
            adj_pred = predictions[pred_idx]["adjacency"]
            colored_pred = np.zeros((*adj_pred.shape, 3))

            # Color coding
            correct_mask = (adj_actual == 1) & (adj_pred == 1)
            colored_pred[correct_mask] = [0, 1, 0]  # Green

            false_pos_mask = (adj_actual == 0) & (adj_pred == 1)
            colored_pred[false_pos_mask] = [1, 0, 0]  # Red

            ax_adj_pred.imshow(colored_pred)
            ax_adj_pred.set_title(
                "Predicted Matrix", pad=10, fontsize=PlotStyle.MEDIUM_SIZE
            )

            # Remove ticks but keep frame
            ax_adj_actual.set_xticks([])
            ax_adj_actual.set_yticks([])
            ax_adj_pred.set_xticks([])
            ax_adj_pred.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
