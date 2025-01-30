# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results."""

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import compute_shap_values


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    # Figure sizes (in inches)
    single_column_width: float = 3.3
    double_column_width: float = 7.0
    standard_height: float = 2.5
    grid_height: float = 3.3

    # Grid parameters
    grid_spacing: float = 0.4  # Space between subplots

    # Font sizes
    title_size: int = 8
    label_size: int = 8
    tick_size: int = 6
    legend_size: int = 6
    annotation_size: int = 6

    # Line parameters
    line_width: float = 0.8
    line_alpha: float = 0.8
    grid_alpha: float = 0.3
    grid_width: float = 0.5

    # Colors
    colors: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default colors if not provided."""
        if self.colors is None:
            self.colors = {
                "actual": "blue",
                "predicted": "orange",
                "average": "#2ecc71",
                "pred_avg": "#9b59b6",
                "threshold": "#FF7F7F",
                "change_point": "red",
            }


class MartingaleVisualizer:
    """Visualization class for martingale analysis results."""

    def __init__(
        self,
        martingales: Dict[str, Dict[str, Any]],
        change_points: List[int],
        threshold: float,
        epsilon: float,
        output_dir: str = "results",
        prefix: str = "",
    ):
        """Initialize the visualizer.

        Args:
            martingales: Dictionary of martingale results for each feature
            change_points: List of true change points
            threshold: Detection threshold
            epsilon: Sensitivity parameter
            output_dir: Directory to save visualizations
            prefix: Prefix for output filenames (e.g., "horizon_" for horizon martingales)
        """
        self.martingales = martingales
        self.change_points = change_points
        self.threshold = threshold
        self.epsilon = epsilon
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.vis_config = VisualizationConfig()

        # Compute SHAP values
        sequence_length = len(next(iter(martingales.values()))["martingales"])
        self.shap_values, self.feature_names = compute_shap_values(
            martingales=martingales,
            change_points=change_points,
            sequence_length=sequence_length,
            threshold=threshold,
        )

        # Set paper-style parameters
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(
            {
                "font.size": self.vis_config.label_size,
                "axes.titlesize": self.vis_config.title_size,
                "axes.labelsize": self.vis_config.label_size,
                "xtick.labelsize": self.vis_config.tick_size,
                "ytick.labelsize": self.vis_config.tick_size,
                "legend.fontsize": self.vis_config.legend_size,
                "lines.linewidth": self.vis_config.line_width,
                "grid.alpha": self.vis_config.grid_alpha,
                "grid.linewidth": self.vis_config.grid_width,
            }
        )

    def create_visualization(self) -> None:
        """Create all visualizations."""
        self._plot_feature_martingales()
        self._plot_combined_martingales()
        self._plot_overlaid_martingales()
        self._plot_shap_values()

    def _plot_feature_martingales(self) -> None:
        """Create grid of individual feature martingale plots."""
        fig = plt.figure(
            figsize=(
                self.vis_config.double_column_width,
                self.vis_config.grid_height * 2,
            )
        )
        gs = GridSpec(
            4,
            2,
            figure=fig,
            hspace=self.vis_config.grid_spacing,
            wspace=self.vis_config.grid_spacing,
        )

        main_features = [
            "degree",
            "density",
            "clustering",
            "betweenness",
            "eigenvector",
            "closeness",
            "singular_value",
            "laplacian",
        ]

        for idx, feature in enumerate(main_features):
            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs[row, col])

            if feature in self.martingales:
                results = self.martingales[feature]
                martingale_values = np.array(
                    [
                        x.item() if isinstance(x, np.ndarray) else x
                        for x in results["martingales"]
                    ]
                )

                ax.plot(
                    martingale_values,
                    color=self.vis_config.colors["actual"],
                    linewidth=self.vis_config.line_width,
                    alpha=self.vis_config.line_alpha,
                    label="Mart.",
                )

                # Dynamic y-axis range based on actual martingale values
                max_mart = np.max(martingale_values)
                if max_mart < 0.1:  # Very small values
                    y_max = max(max_mart * 5, 0.1)
                    n_ticks = 4
                elif max_mart < 1:  # Small values
                    y_max = max(max_mart * 2, 1)
                    n_ticks = 5
                elif max_mart < 10:  # Medium values
                    y_max = max(max_mart * 1.5, 5)
                    n_ticks = 6
                else:  # Large values
                    y_max = max_mart * 1.2
                    n_ticks = min(7, int(y_max / 5) + 1)

                # Calculate negative range as a small percentage of y_max
                y_min = -y_max * 0.05

                # Create ticks including the negative range
                y_ticks = np.linspace(y_min, y_max, n_ticks)

                ax.set_ylim(y_min, y_max)
                ax.set_yticks(y_ticks)

                # Format tick labels to avoid scientific notation and limit decimals
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda x, p: f"{x:.3f}" if abs(x) < 0.01 else f"{x:.2f}"
                    )
                )

            for cp in self.change_points:
                ax.axvline(
                    x=cp,
                    color=self.vis_config.colors["change_point"],
                    linestyle="--",
                    alpha=0.3,  # Reduced alpha for less visual clutter
                    linewidth=self.vis_config.grid_width,
                )

            ax.set_title(feature.title(), fontsize=self.vis_config.title_size, pad=3)
            ax.set_xlim(0, 200)
            ax.set_xticks(np.arange(0, 201, 50))

            if row == 3:
                ax.set_xlabel("Time", fontsize=self.vis_config.label_size)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel("Mart. Value", fontsize=self.vis_config.label_size)

            if idx == 0:  # Only show legend in first plot
                ax.legend(
                    fontsize=self.vis_config.legend_size,
                    ncol=1,
                    loc="upper right",
                    borderaxespad=0.1,
                    handlelength=1.0,
                    columnspacing=0.8,
                )
            ax.tick_params(
                axis="both", which="major", labelsize=self.vis_config.tick_size, pad=2
            )
            ax.grid(
                True,
                linestyle=":",
                alpha=self.vis_config.grid_alpha * 0.7,  # Reduced grid alpha
                linewidth=self.vis_config.grid_width * 0.8,  # Thinner grid lines
            )

        plt.tight_layout(pad=0.3)
        self._save_figure("feature_martingales.png")

    def _plot_combined_martingales(self) -> None:
        """Create plot for combined martingales."""
        fig = plt.figure(
            figsize=(self.vis_config.double_column_width, self.vis_config.grid_height)
        )
        ax = fig.add_subplot(111)
        combined_results = self.martingales["combined"]

        sum_martingale = combined_results["martingale_sum"]
        ax.plot(
            sum_martingale,
            color=self.vis_config.colors["actual"],
            label="Sum Mart.",
            linewidth=self.vis_config.line_width,
            alpha=self.vis_config.line_alpha,
            zorder=10,
        )

        avg_martingale = combined_results["martingale_avg"]
        ax.plot(
            avg_martingale,
            color=self.vis_config.colors["average"],
            label="Avg Mart.",
            linewidth=self.vis_config.line_width,
            linestyle="--",
            alpha=self.vis_config.line_alpha,
            zorder=8,
        )

        ax.axhline(
            y=self.threshold,
            color=self.vis_config.colors["threshold"],
            linestyle="--",
            alpha=0.7,
            linewidth=self.vis_config.line_width,
            label="Threshold",
            zorder=7,
        )

        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=self.vis_config.colors["change_point"],
                linestyle="--",
                alpha=0.5,
                linewidth=self.vis_config.grid_width,
                zorder=6,
            )

            detection_idx = next(
                (
                    i
                    for i in range(cp, len(sum_martingale))
                    if sum_martingale[i] > self.threshold
                ),
                None,
            )
            if detection_idx:
                delay = detection_idx - cp
                ax.annotate(
                    f"d:{delay}",
                    xy=(detection_idx, self.threshold),
                    xytext=(cp - 10, self.threshold * 1.1),
                    color=self.vis_config.colors["actual"],
                    fontsize=self.vis_config.annotation_size,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=self.vis_config.colors["actual"],
                        alpha=0.8,
                        linewidth=1.0,
                        connectionstyle="arc3,rad=-0.2",
                    ),
                )

        ax.grid(True, which="major", linestyle=":", alpha=self.vis_config.grid_alpha)
        ax.grid(
            True, which="minor", linestyle=":", alpha=self.vis_config.grid_alpha / 2
        )
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

        max_mart = max(np.max(sum_martingale), np.max(avg_martingale))
        y_max = max(max_mart + max_mart * 0.25, self.threshold * 1.25)
        ax.set_ylim(-5, y_max)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(25))

        ax.legend(
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            fontsize=self.vis_config.legend_size,
            frameon=True,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0,
        )

        ax.set_xlabel("Time", fontsize=self.vis_config.label_size, labelpad=4)
        ax.set_ylabel(
            "Martingale Value", fontsize=self.vis_config.label_size, labelpad=4
        )
        ax.tick_params(axis="both", which="major", labelsize=self.vis_config.tick_size)

        plt.tight_layout(pad=0.3)
        self._save_figure("combined_martingales.png")

    def _plot_overlaid_martingales(self) -> None:
        """Create plot overlaying individual feature martingales with combined martingale."""
        fig = plt.figure(
            figsize=(self.vis_config.double_column_width, self.vis_config.grid_height)
        )
        ax = fig.add_subplot(111)

        # Plot individual feature martingales and calculate true sum
        feature_names = [
            "degree",
            "density",
            "clustering",
            "betweenness",
            "eigenvector",
            "closeness",
            "singular_value",
            "laplacian",
        ]

        # Use a different color for each feature
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))

        # Initialize sum martingale array
        sum_martingale = None

        # First plot individual martingales and calculate true sum
        for i, feature in enumerate(feature_names):
            if feature in self.martingales:
                feature_mart = np.array(self.martingales[feature]["martingales"])

                # Initialize or add to sum_martingale
                if sum_martingale is None:
                    sum_martingale = feature_mart
                else:
                    sum_martingale += feature_mart

                # Plot individual feature
                ax.plot(
                    feature_mart,
                    color=colors[i],
                    label=f"{feature.replace('_', ' ').title()}",
                    linewidth=1.0,
                    alpha=0.3,  # Reduced alpha for better visibility of combined
                )

        # Plot the true sum martingale
        ax.plot(
            sum_martingale,
            color="blue",
            linestyle="--",
            label="Combined (Sum)",
            linewidth=1.5,
            alpha=0.8,
            zorder=10,
        )

        # Add threshold line - only relevant for combined martingale
        ax.axhline(
            y=self.threshold,
            color=self.vis_config.colors["threshold"],
            linestyle="--",
            alpha=0.7,
            linewidth=self.vis_config.line_width,
            label="Threshold (Combined)",
            zorder=7,
        )

        # Add true change points
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=self.vis_config.colors["change_point"],
                linestyle="--",
                alpha=0.5,
                linewidth=self.vis_config.grid_width,
                zorder=6,
            )

            detection_idx = next(
                (
                    i
                    for i in range(cp, len(sum_martingale))
                    if sum_martingale[i] > self.threshold
                ),
                None,
            )
            if detection_idx:
                delay = detection_idx - cp
                ax.annotate(
                    f"d:{delay}",
                    xy=(detection_idx, self.threshold),
                    xytext=(cp - 10, self.threshold * 1.1),
                    color=self.vis_config.colors["actual"],
                    fontsize=self.vis_config.annotation_size,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=self.vis_config.colors["actual"],
                        alpha=0.8,
                        linewidth=1.0,
                        connectionstyle="arc3,rad=-0.2",
                    ),
                )

        ax.grid(True, which="major", linestyle=":", alpha=self.vis_config.grid_alpha)
        ax.grid(
            True, which="minor", linestyle=":", alpha=self.vis_config.grid_alpha / 2
        )
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

        # Set y-axis limits based on the true sum martingale
        max_mart = np.max(sum_martingale)
        y_max = max(max_mart + max_mart * 0.25, self.threshold * 1.25)
        ax.set_ylim(-5, y_max)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(25))

        # Create two-column legend with slightly smaller font
        ax.legend(
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            fontsize=self.vis_config.legend_size * 0.8,
            frameon=True,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0,
        )

        ax.set_xlabel("Time", fontsize=self.vis_config.label_size, labelpad=4)
        ax.set_ylabel(
            "Martingale Value", fontsize=self.vis_config.label_size, labelpad=4
        )
        ax.tick_params(axis="both", which="major", labelsize=self.vis_config.tick_size)

        plt.tight_layout(pad=0.3)
        self._save_figure("overlaid_martingales.png")

    def _plot_shap_values(self) -> None:
        """Create plot for SHAP values."""
        fig = plt.figure(
            figsize=(
                self.vis_config.double_column_width,
                self.vis_config.grid_height
                * 0.8,  # Reduced height since we only have one panel
            )
        )
        ax = fig.add_subplot(111)

        # Add vertical "Actual" label
        ax.text(
            -0.1,
            0.5,
            "Actual",
            transform=ax.transAxes,
            fontsize=self.vis_config.label_size,
            va="center",
            ha="right",
            rotation=90,
        )

        # Create color map for available features
        n_features = len(self.feature_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_features))

        # Plot SHAP values for each feature
        for i, feature in enumerate(self.feature_names):
            ax.plot(
                self.shap_values[:, i],
                label=feature.title(),
                color=colors[i],
                linewidth=self.vis_config.line_width,
                alpha=self.vis_config.line_alpha,
            )

        # Add change point markers
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=self.vis_config.colors["change_point"],
                linestyle="--",
                alpha=0.3,
                linewidth=self.vis_config.grid_width,
            )

        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

        # Dynamic y-axis range with small margin
        y_vals = self.shap_values
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        y_margin = max((y_max - y_min) * 0.1, 0.1)  # At least 0.1 margin
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add grid with reduced alpha
        ax.grid(
            True,
            linestyle=":",
            alpha=self.vis_config.grid_alpha * 0.7,
            linewidth=self.vis_config.grid_width * 0.8,
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.vis_config.tick_size,
            pad=2,
        )

        # Legend configuration
        ax.legend(
            ncol=2,
            loc="upper right",
            fontsize=self.vis_config.legend_size,
            borderaxespad=0.1,
            handlelength=1.0,
            columnspacing=0.8,
        )

        ax.set_xlabel("Time", fontsize=self.vis_config.label_size)
        plt.tight_layout(pad=0.3)
        self._save_figure("shap_values.png")

    def _save_figure(self, filename: str) -> None:
        """Save figure with publication-quality settings."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_filename = f"{self.prefix}{filename}" if self.prefix else filename
        plt.savefig(
            self.output_dir / final_filename,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close()
