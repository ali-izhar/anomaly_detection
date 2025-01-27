# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results."""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.gridspec import GridSpec

from .utils import compute_shap_values


class MartingaleVisualizer:
    """Visualization class for martingale analysis results."""

    def __init__(
        self,
        martingales: Dict[str, Dict[str, Any]],
        change_points: List[int],
        threshold: float = 30,
        epsilon: float = 0.8,
        output_dir: str = "martingale_outputs",
    ):
        """Initialize visualizer with analysis results."""
        self.martingales = martingales
        self.change_points = change_points
        self.threshold = threshold
        self.epsilon = epsilon
        self.output_dir = Path(output_dir)

        # Compute SHAP values
        sequence_length = len(next(iter(martingales.values()))["martingales"])
        self.shap_values = compute_shap_values(
            martingales=martingales,
            change_points=change_points,
            sequence_length=sequence_length,
            threshold=threshold,
        )

        # Set paper-style parameters
        plt.style.use("seaborn-v0_8-paper")
        sns.set_style("whitegrid", {"grid.linestyle": ":"})
        sns.set_context("paper", font_scale=1.0)

        # Paper-specific sizes (in inches)
        self.SINGLE_COLUMN_WIDTH = 3.5
        self.DOUBLE_COLUMN_WIDTH = 7.0
        self.STANDARD_HEIGHT = 2.5
        self.GRID_HEIGHT = 2.0

        # Font sizes for paper
        self.TITLE_SIZE = 9
        self.LABEL_SIZE = 8
        self.TICK_SIZE = 7
        self.LEGEND_SIZE = 7
        self.ANNOTATION_SIZE = 6

        # Line properties
        self.LINE_WIDTH = 1.0
        self.LINE_ALPHA = 0.8
        self.GRID_ALPHA = 0.15  # Lighter grid
        self.GRID_WIDTH = 0.5

        # Colors for features and combined metrics
        self.feature_colors = {
            "degree": "#1f77b4",  # Blue
            "clustering": "#ff7f0e",  # Orange
            "betweenness": "#2ca02c",  # Green
            "closeness": "#d62728",  # Red
            "eigenvector": "#9467bd",  # Purple
            "density": "#8c564b",  # Brown
            "singular_value": "#e377c2",  # Pink
            "laplacian": "#7f7f7f",  # Gray
        }

        # Special colors for combined metrics
        self.combined_colors = {
            "average": "#2ca02c",  # Green
            "sum": "#1f77b4",  # Blue
        }

    def create_visualization(self) -> None:
        """Generate separate visualizations for features and combined analysis."""
        self._plot_feature_grid()
        self._plot_combined_martingales()
        self._plot_shap_values()

    def _plot_feature_grid(self) -> None:
        """Create grid of individual feature martingale plots."""
        # Select main features to plot (2x2 grid)
        main_features = ["degree", "clustering", "betweenness", "closeness"]

        fig = plt.figure(figsize=(self.DOUBLE_COLUMN_WIDTH, self.GRID_HEIGHT * 2))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

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

                # Plot martingale values
                ax.plot(
                    martingale_values,
                    color=self.feature_colors[feature],
                    linewidth=self.LINE_WIDTH,
                    alpha=self.LINE_ALPHA,
                    label="Act.",
                )

            # Add change points
            for cp in self.change_points:
                ax.axvline(
                    x=cp,
                    color="red",
                    linestyle="--",
                    alpha=0.3,
                    linewidth=self.GRID_WIDTH,
                )

            self._customize_plot(
                ax,
                feature.title(),
                "Time" if row == 1 else "",
                "Mart. Value" if col == 0 else "",
                legend_cols=1,
                feature_plot=True,
            )

        # Save with high quality
        self._save_figure(fig, "feature_martingales.png")

    def _plot_combined_martingales(self) -> None:
        """Create plot for combined martingales."""
        # Create figure with black border
        fig = plt.figure(
            figsize=(self.DOUBLE_COLUMN_WIDTH, self.GRID_HEIGHT)
        )  # More compact height
        fig.patch.set_facecolor("white")
        fig.patch.set_edgecolor("black")
        fig.patch.set_linewidth(1.0)

        ax = fig.add_subplot(111)

        combined_results = self.martingales["combined"]

        # Plot sum martingale with blue color
        sum_martingale = combined_results["martingale_sum"]
        ax.plot(
            sum_martingale,
            color="#0000FF",  # Solid blue
            label="Sum Mart.",
            linewidth=1.2,
            alpha=1.0,
            zorder=10,
        )

        # Plot average martingale with green dashed line
        avg_martingale = combined_results["martingale_avg"]
        ax.plot(
            avg_martingale,
            color="#00FF00",  # Green
            label="Avg Mart.",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            zorder=8,
        )

        # Add threshold line in gray dashed
        ax.axhline(
            y=self.threshold,
            color="#666666",  # Gray
            linestyle="--",
            alpha=0.7,
            linewidth=1.0,
            label="Threshold",
            zorder=7,
        )

        # Add change points as vertical red dashed lines
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color="#FF9999",  # Light red
                linestyle="--",
                alpha=0.5,
                linewidth=1.0,
                zorder=6,
            )

        # Customize grid and spines
        ax.grid(True, which="major", linestyle=":", alpha=0.2, color="gray")
        ax.grid(True, which="minor", linestyle=":", alpha=0.1, color="gray")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

        # Set x-axis ticks at intervals of 50
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

        # Calculate dynamic y-axis limits based on martingale values
        max_mart = max(np.max(sum_martingale), np.max(avg_martingale))
        min_mart = min(np.min(sum_martingale), np.min(avg_martingale))
        y_range = max_mart - min_mart

        # Add padding (25% of range) and ensure threshold is visible
        y_padding = y_range * 0.25
        y_min = min(min_mart - y_padding, -y_padding)  # Ensure some negative space
        y_max = max(
            max_mart + y_padding, self.threshold + y_padding
        )  # Ensure threshold is visible

        # Round to nearest 50 for clean tick values
        y_min = np.floor(y_min / 50) * 50
        y_max = np.ceil(y_max / 50) * 50

        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(25))

        # Add detection delay annotations
        # Find the first threshold crossing after each change point
        for cp in self.change_points:
            # Find the detection point (first threshold crossing after change point)
            detection_idx = None
            for i in range(cp, len(sum_martingale)):
                if sum_martingale[i] > self.threshold:
                    detection_idx = i
                    break

            if detection_idx is not None:
                delay = detection_idx - cp
                # Position the annotation above the threshold crossing point
                cross_value = sum_martingale[detection_idx]

                # Calculate arrow positions to extend before CP and above threshold
                arrow_start_x = cp - 10  # Start before the change point
                arrow_start_y = (
                    self.threshold + y_padding / 2
                )  # Position above threshold proportionally

                ax.annotate(
                    f"d:{delay}",
                    xy=(detection_idx, cross_value),  # Point to threshold crossing
                    xytext=(arrow_start_x, arrow_start_y),  # Extended position
                    color="#0000FF",
                    fontsize=8,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#0000FF",
                        alpha=0.8,
                        linewidth=1.0,
                        connectionstyle="arc3,rad=0.2",  # Add slight curve to arrow
                    ),
                )

        # Customize legend
        legend = ax.legend(
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0,
        )
        legend.get_frame().set_alpha(0.9)

        # Labels
        ax.set_xlabel("Time", fontsize=9, labelpad=4)
        ax.set_ylabel("Martingale Value", fontsize=9, labelpad=4)
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Adjust layout to ensure the border is visible
        plt.tight_layout(pad=0.3)

        # Save with high quality
        self._save_figure(fig, "combined_martingales.png")

    def _plot_shap_values(self) -> None:
        """Create plot for SHAP values."""
        fig = plt.figure(figsize=(self.DOUBLE_COLUMN_WIDTH, self.GRID_HEIGHT * 1.2))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.1)

        # Plot actual SHAP values
        ax1 = fig.add_subplot(gs[0])
        ax1.text(
            -0.1,
            0.5,
            "Actual",
            transform=ax1.transAxes,
            fontsize=self.LABEL_SIZE,
            va="center",
        )

        # Plot predicted SHAP values (placeholder for now)
        ax2 = fig.add_subplot(gs[1])
        ax2.text(
            -0.1,
            0.5,
            "Predicted",
            transform=ax2.transAxes,
            fontsize=self.LABEL_SIZE,
            va="center",
        )

        for ax in [ax1, ax2]:
            i = 0
            for name in ["degree", "clustering", "betweenness", "closeness"]:
                if name in self.martingales and name != "combined":
                    ax.plot(
                        self.shap_values[:, i],
                        label=name.title(),
                        color=self.feature_colors[name],
                        linewidth=self.LINE_WIDTH,
                        alpha=self.LINE_ALPHA,
                    )
                    i += 1

            # Add change points
            for cp in self.change_points:
                ax.axvline(
                    x=cp,
                    color="red",
                    linestyle="--",
                    alpha=0.3,
                    linewidth=self.GRID_WIDTH,
                )

            self._customize_plot(
                ax,
                "",
                "Time" if ax == ax2 else "",
                "",
                legend_cols=2 if ax == ax1 else 0,
                ylim=(-0.1, 1.1),
            )

        # Save with high quality
        self._save_figure(fig, "shap_values.png")

    def _customize_plot(
        self,
        ax: plt.Axes,
        title: str,
        xlabel: str,
        ylabel: str,
        legend_cols: int = 1,
        feature_plot: bool = False,
        ylim: tuple = None,
    ) -> None:
        """Apply consistent styling to plots."""
        # Grid styling
        ax.grid(True, linestyle=":", alpha=self.GRID_ALPHA, linewidth=self.GRID_WIDTH)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        # Labels with minimal padding
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.LABEL_SIZE, labelpad=2)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.LABEL_SIZE, labelpad=2)
        if title:
            ax.set_title(title, fontsize=self.TITLE_SIZE, pad=4)

        ax.tick_params(axis="both", which="major", labelsize=self.TICK_SIZE, pad=2)

        # Set y-limits if specified
        if ylim is not None:
            ax.set_ylim(ylim)

        # Compact legend if needed
        if legend_cols > 0:
            legend = ax.legend(
                fontsize=self.LEGEND_SIZE,
                ncol=legend_cols,
                loc="upper right",
                bbox_to_anchor=(1, 1.02),
                frameon=True,
                facecolor="white",
                edgecolor="none",
                borderaxespad=0.1,
                handlelength=1.0,
                columnspacing=0.8,
            )
            legend.get_frame().set_alpha(0.8)

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure with publication-quality settings."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.02,
        )
        plt.close(fig)
