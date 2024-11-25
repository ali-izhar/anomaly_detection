# tests/visualize_martingales.py

"""
Martingale Visualization Module

This module provides visualization utilities for martingale-based change point detection
results across different graph types (BA, ER, NW).
"""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from src.graph.features import adjacency_to_graph
from src.models.threshold_model import CustomThresholdModel


class MartingaleVisualizer:
    """Visualization class for martingale analysis results."""

    def __init__(
        self,
        graphs: List[np.ndarray],
        change_points: List[int],
        martingales: Dict[str, Dict[str, Any]],
        graph_type: str = "BA",
        threshold: float = 30,
        epsilon: float = 0.8,
        output_dir: str = "martingale_outputs",
    ):
        """Initialize visualizer with analysis results.

        Args:
            graphs: List of adjacency matrices
            change_points: True change point indices
            martingales: Dictionary containing reset and cumulative martingales
            graph_type: Type of graph (BA, ER, NW)
            output_dir: Directory to save outputs
        """
        self.graphs = graphs
        self.change_points = change_points
        self.martingales = martingales
        self.graph_type = graph_type
        self.threshold = threshold
        self.epsilon = epsilon
        self.output_dir = Path(output_dir)
        self.shap_values = self._compute_shap_values()

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        self.colors = sns.color_palette(
            [
                "#FF1F5B",  # bright red
                "#009ADE",  # bright blue
                "#F28522",  # orange
                "#58B272",  # green
                "#AF58BA",  # purple
                "#00CD6C",  # emerald
                "#FFC61E",  # yellow
                "#1D2B5E",  # navy
                "#E6302C",  # coral
                "#562883",  # deep purple
            ],
            n_colors=len(martingales["reset"]),
        )

    def _compute_shap_values(self) -> np.ndarray:
        """Compute SHAP values using CustomThresholdModel on martingale values."""
        # Convert martingale values to feature matrix
        feature_matrix = []

        for name, results in self.martingales["reset"].items():
            # Convert array of arrays to flat array
            martingale_values = np.array(
                [
                    x.item() if isinstance(x, np.ndarray) else x
                    for x in results["martingales"]
                ]
            )
            feature_matrix.append(martingale_values)

        X = np.vstack(feature_matrix).T  # [n_timesteps x n_features]

        model = CustomThresholdModel(threshold=self.threshold)
        return model.compute_shap_values(
            X=X,
            change_points=self.change_points,
            sequence_length=len(self.graphs),
            window_size=5,  # Can be made configurable if needed
        )

    def create_dashboard(self) -> None:
        """Generate comprehensive dashboard with all visualizations."""
        time_points = [
            0,  # Start
            self.change_points[0],  # Change 1
            self.change_points[1],  # Change 2
            self.change_points[2],  # Change 3
            len(self.graphs) - 1,  # End
        ]

        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[
                0.6,
                1.2,
                1.2,
            ],
            hspace=0.4,
        )

        # Row 1: Graph Evolution (5 columns)
        gs_top = gs[0].subgridspec(1, 5, wspace=0.15)
        for i, t in enumerate(time_points):
            ax = fig.add_subplot(gs_top[0, i])
            G = adjacency_to_graph(self.graphs[t])

            degrees = dict(G.degree())
            node_sizes = [
                1000 * (v + 1) / max(degrees.values()) for v in degrees.values()
            ]
            node_colors = list(degrees.values())

            pos = nx.spring_layout(G, k=0.8, iterations=50)
            nx.draw(
                G,
                pos,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                edge_color="gray",
                alpha=0.7,
                ax=ax,
                cmap=plt.cm.viridis,
            )

            stats = f"t={t} | N={G.number_of_nodes()} | E={G.number_of_edges()}"
            ax.set_title(stats, fontsize=10, pad=5)

        # Row 2: Martingales (2 columns)
        gs_middle = gs[1].subgridspec(1, 2, wspace=0.15)
        self._plot_reset_martingales(fig, gs_middle[0, 0])
        self._plot_cumulative_martingales(fig, gs_middle[0, 1])

        # Row 3: SHAP Analysis (2 columns)
        gs_bottom = gs[2].subgridspec(1, 2, wspace=0.15)
        self._plot_shap_values(fig, gs_bottom[0, 0])
        self._plot_feature_importance(fig, gs_bottom[0, 1])

        fig.suptitle(
            f"{self.graph_type} Graph Change Point Analysis", fontsize=16, y=0.99
        )

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / f"{self.graph_type.lower()}_martingale_analysis.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

    def save_results(self) -> None:
        """Save detailed martingale analysis results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(
            self.output_dir / f"{self.graph_type.lower()}_martingale_results.txt", "w"
        ) as f:
            f.write(f"=== {self.graph_type} Graph Martingale Analysis ===\n\n")
            f.write(f"True Change Points: {self.change_points}\n\n")

            for mart_type, results in self.martingales.items():
                f.write(f"\n=== {mart_type.upper()} MARTINGALES ===\n")

                for name, data in results.items():
                    f.write(f"\nCentrality Measure: {name.upper()}\n")
                    f.write("-" * 30 + "\n")

                    # Detection points
                    detected = data["change_detected_instant"]
                    f.write(f"Detected changes at: {detected}\n")

                    # Statistics
                    mart_values = np.array(
                        [
                            x.item() if isinstance(x, np.ndarray) else x
                            for x in data["martingales"]
                        ]
                    )
                    f.write(f"Maximum value: {float(np.max(mart_values)):.3f}\n")
                    f.write(f"Average value: {float(np.mean(mart_values)):.3f}\n")
                    f.write(f"Standard deviation: {float(np.std(mart_values)):.3f}\n\n")

    def _plot_graph_evolution(
        self, fig: plt.Figure, gs: plt.GridSpec, time_points: List[int]
    ) -> None:
        """Plot graph structure evolution."""
        for i, t in enumerate(time_points):
            ax = fig.add_subplot(gs[0, i])
            G = adjacency_to_graph(self.graphs[t])

            degrees = dict(G.degree())
            node_sizes = [
                1000 * (v + 1) / max(degrees.values()) for v in degrees.values()
            ]
            node_colors = list(degrees.values())

            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw(
                G,
                pos,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                font_size=5,
                edge_color="gray",
                alpha=0.7,
                ax=ax,
                cmap=plt.cm.viridis,
            )

            stats = (
                f"t={t}\nN={G.number_of_nodes()}, E={G.number_of_edges()}\n"
                f"Avg deg={np.mean(list(degrees.values())):.1f}"
            )
            ax.set_title(stats, fontsize=8)

    def _plot_reset_martingales(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Plot reset martingales."""
        ax = fig.add_subplot(gs)
        self._plot_martingale_sequence(
            ax, self.martingales["reset"], "Reset Martingale Measures"
        )

    def _plot_cumulative_martingales(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Plot cumulative martingales."""
        ax = fig.add_subplot(gs)
        self._plot_martingale_sequence(
            ax,
            self.martingales["cumulative"],
            "Cumulative Martingale Measures",
            cumulative=True,
        )

    def _plot_martingale_sequence(
        self,
        ax: plt.Axes,
        martingales: Dict[str, Dict[str, Any]],
        title: str,
        cumulative: bool = False,
    ) -> None:
        """Plot martingale sequences with change points."""
        for cp in self.change_points:
            ax.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

        for (name, results), color in zip(martingales.items(), self.colors):
            # Convert array of arrays to flat array
            martingale_values = np.array(
                [
                    x.item() if isinstance(x, np.ndarray) else x
                    for x in results["martingales"]
                ]
            )

            if cumulative:
                ax.semilogy(
                    martingale_values,
                    color=color,
                    label=name.capitalize(),
                    linewidth=1.5,
                    alpha=0.6,
                )
            else:
                ax.plot(
                    martingale_values,
                    color=color,
                    label=name.capitalize(),
                    linewidth=1.5,
                    alpha=0.6,
                )

        # Plot combined martingales
        martingale_arrays = []
        for m in martingales.values():
            values = np.array(
                [x.item() if isinstance(x, np.ndarray) else x for x in m["martingales"]]
            )
            martingale_arrays.append(values)

        M_sum = np.sum(martingale_arrays, axis=0)
        M_avg = M_sum / len(martingales)

        if cumulative:
            ax.semilogy(
                M_avg, color="#FF4B4B", label="Average", linewidth=2.5, alpha=0.9
            )
            ax.semilogy(
                M_sum,
                color="#2F2F2F",
                label="Sum",
                linewidth=2.5,
                linestyle="-.",
                alpha=0.8,
            )
        else:
            ax.plot(M_avg, color="#FF4B4B", label="Average", linewidth=2.5, alpha=0.9)
            ax.plot(
                M_sum,
                color="#2F2F2F",
                label="Sum",
                linewidth=2.5,
                linestyle="-.",
                alpha=0.8,
            )

        # Customize plot
        self._customize_martingale_plot(ax, title, cumulative)

    def _customize_martingale_plot(
        self, ax: plt.Axes, title: str, cumulative: bool = False
    ) -> None:
        """Customize martingale plot appearance."""
        ax.grid(True, linestyle="--", alpha=0.3)

        # Only add minor locator for non-log scale
        if not cumulative:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
        else:
            # Use 10^x notation for log scale
            def log_format(x, p):
                exponent = int(np.log10(x))
                return f"$10^{{{exponent}}}$"

            ax.yaxis.set_major_formatter(FuncFormatter(log_format))

        ax.set_xlabel("Time Steps", fontsize=12, labelpad=5)
        ax.set_ylabel(
            "Martingale Values" + (" (log scale)" if cumulative else ""),
            fontsize=12,
            labelpad=10,
        )
        ax.set_title(title, fontsize=12, pad=15)

        legend = ax.legend(
            fontsize=10,
            ncol=3,
            loc="upper right" if not cumulative else "upper left",
            bbox_to_anchor=(1, 1.02) if not cumulative else (0, 1.02),
            frameon=True,
            facecolor="none",
            edgecolor="none",
        )
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_alpha(0)

    def _plot_shap_values(self, fig: plt.Figure, ax_pos: plt.GridSpec) -> None:
        """Plot SHAP values over time."""
        ax = fig.add_subplot(ax_pos)

        # Plot SHAP values for each feature
        for i, name in enumerate(self.martingales["reset"].keys()):
            ax.plot(
                self.shap_values[:, i],
                label=name.capitalize(),
                color=self.colors[i],
                linewidth=1.5,
                alpha=0.7,
            )

        # Add change point indicators
        for cp in self.change_points:
            ax.axvline(x=cp, color="red", linestyle="--", alpha=0.3)

        ax.set_title("SHAP Values Over Time", fontsize=12, pad=20)
        ax.set_xlabel("Time Steps", fontsize=10)
        ax.set_ylabel("Feature Importance", fontsize=10)
        ax.legend(fontsize=8, title="Centrality Measures")
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, fig: plt.Figure, ax_pos: plt.GridSpec) -> None:
        """Plot feature importance heatmap."""
        ax = fig.add_subplot(ax_pos)

        # Get feature names from reset martingales only
        feature_names = list(self.martingales["reset"].keys())

        # Use raw SHAP values without normalization
        shap_values = self.shap_values

        # Get the absolute maximum for symmetric colormap
        vmax = max(abs(np.min(shap_values)), abs(np.max(shap_values)))
        vmin = -vmax

        # Create heatmap with horizontal colorbar at bottom
        sns.heatmap(
            shap_values.T,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=50,
            yticklabels=[name.capitalize() for name in feature_names],
            cbar_kws={
                "label": "SHAP Value",
                "orientation": "horizontal",
                "pad": 0.2,
                "format": "%.2f",
            },
        )

        ax.set_title(
            "Feature Importance Over Time\n(Red = Positive Impact, Blue = Negative Impact)",
            fontsize=12,
            pad=15,
        )
        ax.set_xlabel("Time Steps", fontsize=10, labelpad=10)

        # Move y-axis labels to the right
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
