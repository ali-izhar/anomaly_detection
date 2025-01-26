# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results."""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

from .utils import compute_shap_values, compute_combined_martingales


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
        """Initialize visualizer with analysis results.

        Args:
            martingales: Dictionary containing martingale values for each feature
            change_points: True change point indices
            threshold: Threshold value for change detection
            epsilon: Sensitivity parameter for martingale
            output_dir: Directory to save outputs
        """
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
            n_colors=len(martingales),
        )

    def create_visualization(self) -> None:
        """Generate comprehensive visualization of martingale analysis."""
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Individual martingales
        self._plot_martingale_sequence(
            fig.add_subplot(gs[0, 0]),
            self.martingales,
            "Individual Feature Martingales",
        )

        # Combined martingales
        self._plot_combined_martingales(
            fig.add_subplot(gs[0, 1]), "Combined Martingales"
        )

        # SHAP values over time
        self._plot_shap_values(fig.add_subplot(gs[1, 0]))

        # Feature importance heatmap
        self._plot_feature_importance(fig.add_subplot(gs[1, 1]))

        fig.suptitle(
            f"Martingale Analysis (ε={self.epsilon}, threshold={self.threshold})",
            fontsize=14,
            y=0.95,
        )

        # Save
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / "martingale_analysis.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

    def _plot_martingale_sequence(
        self,
        ax: plt.Axes,
        martingales: Dict[str, Dict[str, Any]],
        title: str,
    ) -> None:
        """Plot individual martingale sequences."""
        # Add change point indicators
        for cp in self.change_points:
            ax.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

        # Add threshold line
        ax.axhline(y=self.threshold, color="gray", linestyle="--", alpha=0.5)

        # Plot individual martingales
        for (name, results), color in zip(martingales.items(), self.colors):
            martingales = np.array(
                [
                    x.item() if isinstance(x, np.ndarray) else x
                    for x in results["martingales"]
                ]
            )
            ax.plot(
                martingales,
                color=color,
                label=name.capitalize(),
                linewidth=1.5,
                alpha=0.6,
            )

        self._customize_martingale_plot(ax, title)

    def _plot_combined_martingales(
        self,
        ax: plt.Axes,
        title: str,
    ) -> None:
        """Plot combined (sum and average) martingales."""
        # Add change point indicators
        for cp in self.change_points:
            ax.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

        # Add threshold line
        ax.axhline(y=self.threshold, color="gray", linestyle="--", alpha=0.5)

        # Plot combined martingales
        M_sum, M_avg = compute_combined_martingales(self.martingales)

        ax.plot(M_avg, color="#FF4B4B", label="Average", linewidth=2.0, alpha=0.9)
        ax.plot(M_sum, color="#2F2F2F", label="Sum", linewidth=2.0, alpha=0.8)

        self._customize_martingale_plot(ax, title)

    def _customize_martingale_plot(
        self,
        ax: plt.Axes,
        title: str,
    ) -> None:
        """Customize martingale plot appearance."""
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        ax.set_xlabel("Time Steps", fontsize=10, labelpad=5)
        ax.set_ylabel("Martingale Values", fontsize=10, labelpad=5)
        ax.set_title(title, fontsize=12, pad=10)

        legend = ax.legend(
            fontsize=8,
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(1, 1.02),
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )
        legend.get_frame().set_alpha(0.8)

    def _plot_shap_values(self, ax: plt.Axes) -> None:
        """Plot SHAP values over time."""
        # Plot SHAP values for each feature
        for i, name in enumerate(self.martingales.keys()):
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

        ax.set_title("SHAP Values Over Time", fontsize=12, pad=10)
        ax.set_xlabel("Time Steps", fontsize=10)
        ax.set_ylabel("Feature Importance", fontsize=10)
        ax.legend(fontsize=8, ncol=2, title="Features")
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, ax: plt.Axes) -> None:
        """Plot feature importance heatmap."""
        # Get feature names
        feature_names = list(self.martingales.keys())

        # Get the absolute maximum for symmetric colormap
        vmax = max(abs(np.min(self.shap_values)), abs(np.max(self.shap_values)))
        vmin = -vmax

        # Create heatmap
        sns.heatmap(
            self.shap_values.T,
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
            pad=10,
        )
        ax.set_xlabel("Time Steps", fontsize=10, labelpad=5)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
