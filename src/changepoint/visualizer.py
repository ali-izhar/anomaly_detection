# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results."""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

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
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.0)

        # Paper-specific sizes
        self.SINGLE_COLUMN_WIDTH = 3.5  # inches
        self.DOUBLE_COLUMN_WIDTH = 7.0  # inches
        self.STANDARD_HEIGHT = 2.5  # inches

        # Font sizes for paper
        self.TITLE_SIZE = 9
        self.LABEL_SIZE = 8
        self.TICK_SIZE = 7
        self.LEGEND_SIZE = 7
        self.ANNOTATION_SIZE = 6

        # Line properties
        self.LINE_WIDTH = 1.0
        self.LINE_ALPHA = 0.8
        self.GRID_ALPHA = 0.3
        self.GRID_WIDTH = 0.5

        # Use husl palette for features - it gives beautiful distinct colors
        self.colors = sns.husl_palette(n_colors=len(martingales), h=0.01, s=0.9, l=0.65)

        # Special colors for combined metrics
        self.combined_colors = {
            "average": "#FF1F5B",  # vibrant pink-red
            "sum": "#1D2B5E",  # deep navy
        }

    def create_visualization(self) -> None:
        """Generate comprehensive visualization of martingale analysis."""
        # Create figure with compact research paper dimensions
        fig = plt.figure(figsize=(self.DOUBLE_COLUMN_WIDTH, self.STANDARD_HEIGHT * 1.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

        # Combined plot for individual and combined martingales
        ax_martingales = fig.add_subplot(gs[0, 0])

        # Plot individual martingales
        for (name, results), color in zip(self.martingales.items(), self.colors):
            if name != "combined":  # Skip combined for now
                martingale_values = np.array(
                    [
                        x.item() if isinstance(x, np.ndarray) else x
                        for x in results["martingales"]
                    ]
                )
                ax_martingales.plot(
                    martingale_values,
                    color=color,
                    label=f"{name.replace('_', ' ').title()}",
                    linewidth=self.LINE_WIDTH,
                    alpha=self.LINE_ALPHA,
                )

        # Add combined martingales (sum and average)
        combined_results = self.martingales["combined"]
        ax_martingales.plot(
            combined_results["martingale_avg"],
            color=self.combined_colors["average"],
            label="Combined (Average)",
            linewidth=self.LINE_WIDTH * 1.5,
            alpha=self.LINE_ALPHA,
            zorder=10,
        )
        ax_martingales.plot(
            combined_results["martingale_sum"],
            color=self.combined_colors["sum"],
            label="Combined (Sum)",
            linewidth=self.LINE_WIDTH * 1.5,
            alpha=self.LINE_ALPHA,
            zorder=10,
        )

        # Add change point indicators and threshold line
        for cp in self.change_points:
            ax_martingales.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)
        ax_martingales.axhline(
            y=self.threshold,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=self.LINE_WIDTH,
        )

        self._customize_martingale_plot(ax_martingales, "Martingale Analysis")

        # SHAP values over time
        self._plot_shap_values(fig.add_subplot(gs[0, 1]))

        fig.suptitle(
            f"Martingale Analysis (ε={self.epsilon}, threshold={self.threshold})",
            fontsize=self.TITLE_SIZE,
            y=0.95,
        )

        # Save with paper-quality settings
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / "martingale_analysis.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.02,
        )
        plt.close()

    def _customize_martingale_plot(
        self,
        ax: plt.Axes,
        title: str,
    ) -> None:
        """Customize martingale plot appearance."""
        ax.grid(True, linestyle=":", alpha=self.GRID_ALPHA, linewidth=self.GRID_WIDTH)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        ax.set_xlabel("Time Steps", fontsize=self.LABEL_SIZE, labelpad=2)
        ax.set_ylabel("Martingale Values", fontsize=self.LABEL_SIZE, labelpad=2)
        ax.set_title(title, fontsize=self.TITLE_SIZE, pad=4)
        ax.tick_params(axis="both", which="major", labelsize=self.TICK_SIZE, pad=2)

        legend = ax.legend(
            fontsize=self.LEGEND_SIZE,
            ncol=2,
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

    def _plot_shap_values(self, ax: plt.Axes) -> None:
        """Plot SHAP values over time."""
        # Plot SHAP values for each feature
        i = 0  # Keep track of feature index separately
        for name in self.martingales.keys():
            if name != "combined":  # Skip combined feature
                ax.plot(
                    self.shap_values[:, i],
                    label=name.replace("_", " ").title(),
                    color=self.colors[i],
                    linewidth=self.LINE_WIDTH,
                    alpha=self.LINE_ALPHA,
                )
                i += 1

        # Add change point indicators
        for cp in self.change_points:
            ax.axvline(
                x=cp, color="red", linestyle="--", alpha=0.3, linewidth=self.GRID_WIDTH
            )

        ax.set_title("SHAP Values Over Time", fontsize=self.TITLE_SIZE, pad=4)
        ax.set_xlabel("Time Steps", fontsize=self.LABEL_SIZE, labelpad=2)
        ax.set_ylabel("Feature Importance", fontsize=self.LABEL_SIZE, labelpad=2)
        ax.tick_params(axis="both", which="major", labelsize=self.TICK_SIZE, pad=2)
        ax.grid(True, alpha=self.GRID_ALPHA, linestyle=":", linewidth=self.GRID_WIDTH)

        ax.legend(
            fontsize=self.LEGEND_SIZE,
            ncol=2,
            title="Features",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.1,
            handlelength=1.0,
            columnspacing=0.8,
        )
