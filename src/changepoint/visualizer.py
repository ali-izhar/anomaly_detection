# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results."""

from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .threshold import CustomThresholdModel
from ..configs.plotting import (
    FIGURE_DIMENSIONS as FD,
    TYPOGRAPHY as TYPO,
    LINE_STYLE as LS,
    COLORS,
    LEGEND_STYLE as LEGEND,
    GRID_STYLE as GRID,
    EXPORT_SETTINGS as EXPORT,
    get_matplotlib_rc_params,
)


class MartingaleVisualizer:
    """Visualization class for martingale analysis results."""

    def __init__(
        self,
        martingales: Dict[str, Any],
        change_points: List[int],
        threshold: float,
        epsilon: float,
        output_dir: str = "results",
        prefix: str = "",
        skip_shap: bool = False,
        method: str = "multiview",
    ):
        """Initialize the visualizer.

        Args:
            martingales: Dictionary containing martingale results from detector
            change_points: List of true change points
            threshold: Detection threshold
            epsilon: Sensitivity parameter
            output_dir: Directory to save visualizations
            prefix: Prefix for output files
            skip_shap: Whether to skip SHAP value plots
            method: Detection method ('single_view' or 'multiview')
        """
        self.change_points = change_points
        self.threshold = threshold
        self.epsilon = epsilon
        self.output_dir = output_dir
        self.prefix = prefix
        self.skip_shap = skip_shap
        self.method = method

        # Process martingales into a format suitable for plotting
        self.martingales = self._process_martingales(martingales)

        # Initialize SHAP values
        self.shap_values = None
        self.feature_names = None
        self.prediction_shap_values = None

        # Compute SHAP values if needed
        if not skip_shap and method == "multiview":
            try:
                # Get sequence length from combined martingales
                sequence_length = len(self.martingales["combined"]["martingales_sum"])

                # Get feature names
                self.feature_names = [
                    "mean_degree",
                    "density",
                    "mean_clustering",
                    "mean_betweenness",
                    "mean_eigenvector",
                    "mean_closeness",
                    "max_singular_value",
                    "min_nonzero_laplacian",
                ]

                # Prepare data for SHAP computation
                traditional_data = []
                prediction_data = []

                for feature in self.feature_names:
                    if feature in self.martingales:
                        trad_vals = self.martingales[feature]["martingales"]
                        pred_vals = self.martingales[feature].get(
                            "prediction_martingales", []
                        )

                        traditional_data.append(trad_vals)
                        if len(pred_vals) > 0:
                            prediction_data.append(pred_vals)

                # Convert to numpy arrays and transpose to get (time, features) shape
                if traditional_data:
                    self.shap_values = np.array(traditional_data).T
                if prediction_data:
                    self.prediction_shap_values = np.array(prediction_data).T

            except (ValueError, KeyError) as e:
                print(f"Warning: Could not compute SHAP values: {str(e)}")
                self.shap_values = None
                self.feature_names = None
                self.prediction_shap_values = None

        # Set paper-style parameters
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(get_matplotlib_rc_params())

    def _process_martingales(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process detector results into a format suitable for plotting."""
        processed = {"combined": {}}

        # Get the sequence length from the combined martingales
        if self.method == "multiview":
            seq_len = len(detection_results.get("traditional_sum_martingales", []))
        else:
            seq_len = len(detection_results.get("traditional_martingales", []))

        if self.method == "multiview":
            # Process multiview results
            processed["combined"].update(
                {
                    "martingales_sum": detection_results.get(
                        "traditional_sum_martingales", []
                    )[:seq_len],
                    "martingales_avg": detection_results.get(
                        "traditional_avg_martingales", []
                    )[:seq_len],
                    "prediction_martingale_sum": detection_results.get(
                        "horizon_sum_martingales", []
                    ),
                    "prediction_martingale_avg": detection_results.get(
                        "horizon_avg_martingales", []
                    ),
                }
            )

            # Process individual feature martingales
            individual_traditional = detection_results.get(
                "individual_traditional_martingales", []
            )
            individual_horizon = detection_results.get(
                "individual_horizon_martingales", []
            )

            features = [
                "mean_degree",
                "density",
                "mean_clustering",
                "mean_betweenness",
                "mean_eigenvector",
                "mean_closeness",
                "max_singular_value",
                "min_nonzero_laplacian",
            ]

            for i, feature in enumerate(features):
                if i < len(individual_traditional):
                    # Use actual sequence length
                    traditional_values = individual_traditional[i][:seq_len]
                    processed[feature] = {
                        "martingales": traditional_values,
                        "prediction_martingales": (
                            individual_horizon[i] if individual_horizon else None
                        ),
                    }

        else:  # single_view
            # Process single-view results
            processed["combined"].update(
                {
                    "martingales": detection_results.get("traditional_martingales", [])[
                        :seq_len
                    ],
                    "prediction_martingales": detection_results.get(
                        "horizon_martingales", []
                    ),
                }
            )

        return processed

    def create_visualization(self) -> None:
        """Create all visualizations."""
        # Always plot combined martingales
        self._plot_combined_martingales()

        # Only create feature plots for multiview results with individual martingales
        if self.method == "multiview" and len(self.martingales) > 1:
            self._plot_feature_martingales()
            self._plot_overlaid_martingales()

            # Plot SHAP values if available and not explicitly skipped
            if (
                not self.skip_shap
                and hasattr(self, "shap_values")
                and self.shap_values is not None
            ):
                self._plot_shap_values()

    def _plot_feature_martingales(self) -> None:
        """Create grid of individual feature martingale plots."""
        fig = plt.figure(
            figsize=(
                FD["DOUBLE_COLUMN_WIDTH"],
                FD["GRID_HEIGHT"] * 1.5,
            ),  # Reduced height
        )
        gs = GridSpec(
            4,
            2,
            figure=fig,
            hspace=FD["GRID_SPACING"],
            wspace=FD["GRID_SPACING"],
        )

        features = [
            "mean_degree",
            "density",
            "mean_clustering",
            "mean_betweenness",
            "mean_eigenvector",
            "mean_closeness",
            "max_singular_value",
            "min_nonzero_laplacian",
        ]

        max_mart_global = 0

        # First pass to find global maximum
        for feature in features:
            if feature in self.martingales:
                results = self.martingales[feature]
                martingales = results.get("martingales", [])
                pred_martingales = results.get("prediction_martingales", [])

                if isinstance(martingales, (list, np.ndarray)) and len(martingales) > 0:
                    max_mart_global = max(max_mart_global, np.max(martingales))
                if (
                    isinstance(pred_martingales, (list, np.ndarray))
                    and len(pred_martingales) > 0
                ):
                    max_mart_global = max(max_mart_global, np.max(pred_martingales))

        # Now plot each feature
        for idx, feature in enumerate(features):
            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs[row, col])

            if feature in self.martingales:
                results = self.martingales[feature]
                martingales = results.get("martingales", [])
                pred_martingales = results.get("prediction_martingales", [])

                # Plot traditional martingale
                if isinstance(martingales, (list, np.ndarray)) and len(martingales) > 0:
                    ax.plot(
                        martingales,
                        color=COLORS["actual"],
                        linewidth=LS["LINE_WIDTH"],
                        alpha=LS["LINE_ALPHA"],
                        label="Trad.",  # Shortened label
                        zorder=10,
                    )

                # Plot prediction martingale
                if (
                    isinstance(pred_martingales, (list, np.ndarray))
                    and len(pred_martingales) > 0
                ):
                    history_size = len(martingales) - len(pred_martingales)
                    time_points = np.arange(
                        history_size, history_size + len(pred_martingales)
                    )
                    ax.plot(
                        time_points,
                        pred_martingales,
                        color=COLORS["predicted"],
                        linewidth=LS["LINE_WIDTH"],
                        linestyle="--",
                        alpha=LS["LINE_ALPHA"],
                        label="Pred.",  # Shortened label
                        zorder=9,
                    )

                # Add change points with reduced alpha
                for cp in self.change_points:
                    ax.axvline(
                        x=cp,
                        color=COLORS["change_point"],
                        linestyle="--",
                        alpha=0.2,  # Reduced alpha
                        linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner lines
                        zorder=1,
                    )

                # Add threshold line with reduced alpha
                ax.axhline(
                    y=self.threshold,
                    color=COLORS["threshold"],
                    linestyle="--",
                    alpha=0.2,  # Reduced alpha
                    linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner lines
                    zorder=2,
                )

                # Set consistent y-axis limits based on global maximum
                y_max = max_mart_global * 1.05  # Reduced margin
                y_min = 0  # Start from 0
                ax.set_ylim(y_min, y_max)

                # Create evenly spaced ticks
                n_ticks = 4  # Reduced number of ticks
                y_ticks = np.linspace(0, y_max, n_ticks)
                ax.set_yticks(y_ticks)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"{x:.1f}")  # Simplified format
                )

            # Simplified title
            ax.set_title(
                feature.replace("mean_", "").replace("_", " ").title(),
                fontsize=TYPO["TITLE_SIZE"],
                pad=1,  # Reduced padding
            )

            # X-axis settings
            ax.set_xlim(0, 200)
            ax.set_xticks(np.arange(0, 201, 100))  # Reduced number of ticks

            if row == 3:  # Bottom row
                ax.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"], labelpad=1)
            else:
                ax.set_xticklabels([])

            if col == 0:  # Left column
                ax.set_ylabel("Value", fontsize=TYPO["LABEL_SIZE"], labelpad=1)

            if idx == 0:  # Legend only in first plot
                ax.legend(
                    fontsize=TYPO["LEGEND_SIZE"],
                    ncol=2,
                    loc="upper right",
                    borderaxespad=0.1,
                    handlelength=LEGEND["HANDLE_LENGTH"],
                    columnspacing=LEGEND["COLUMN_SPACING"],
                )

            # Tick parameters
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=TYPO["TICK_SIZE"],
                pad=1,
                length=2,
            )

            # Grid settings
            ax.grid(
                True,
                linestyle=":",
                alpha=GRID["MAJOR_ALPHA"] * 0.5,  # Reduced alpha
                linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner lines
            )

        plt.tight_layout(pad=0.1)  # Tighter layout
        self._save_figure("feature_martingales.png")  # Save as PNG

    def _plot_combined_martingales(self) -> None:
        """Create plot for combined martingales."""
        fig = plt.figure(figsize=(FD["SINGLE_COLUMN_WIDTH"], FD["STANDARD_HEIGHT"]))
        ax = fig.add_subplot(111)
        combined_results = self.martingales["combined"]

        if self.method == "multiview":
            # Plot traditional martingales
            sum_martingale = combined_results.get("martingales_sum")
            avg_martingale = combined_results.get("martingales_avg")

            if (
                sum_martingale is not None
                and isinstance(sum_martingale, (list, np.ndarray))
                and len(sum_martingale) > 0
            ):
                ax.plot(
                    sum_martingale,
                    color=COLORS["actual"],
                    label="Sum",  # Shortened label
                    linewidth=LS["LINE_WIDTH"],
                    alpha=LS["LINE_ALPHA"],
                    zorder=10,
                )

            if (
                avg_martingale is not None
                and isinstance(avg_martingale, (list, np.ndarray))
                and len(avg_martingale) > 0
            ):
                ax.plot(
                    avg_martingale,
                    color=COLORS["average"],
                    label="Avg",  # Shortened label
                    linewidth=LS["LINE_WIDTH"],
                    linestyle="--",
                    alpha=LS["LINE_ALPHA"],
                    zorder=8,
                )

            # Plot prediction martingales
            pred_sum_martingale = combined_results.get("prediction_martingale_sum")
            pred_avg_martingale = combined_results.get("prediction_martingale_avg")

            if (
                pred_sum_martingale is not None
                and isinstance(pred_sum_martingale, (list, np.ndarray))
                and len(pred_sum_martingale) > 0
            ):
                history_size = len(sum_martingale) - len(pred_sum_martingale)
                time_points = np.arange(
                    history_size, history_size + len(pred_sum_martingale)
                )
                ax.plot(
                    time_points,
                    pred_sum_martingale,
                    color=COLORS["predicted"],
                    label="P.Sum",  # Shortened label
                    linewidth=LS["LINE_WIDTH"],
                    alpha=LS["LINE_ALPHA"],
                    zorder=9,
                )

            if (
                pred_avg_martingale is not None
                and isinstance(pred_avg_martingale, (list, np.ndarray))
                and len(pred_avg_martingale) > 0
            ):
                history_size = len(avg_martingale) - len(pred_avg_martingale)
                time_points = np.arange(
                    history_size, history_size + len(pred_avg_martingale)
                )
                ax.plot(
                    time_points,
                    pred_avg_martingale,
                    color=COLORS["pred_avg"],
                    label="P.Avg",  # Shortened label
                    linewidth=LS["LINE_WIDTH"],
                    linestyle="--",
                    alpha=LS["LINE_ALPHA"],
                    zorder=7,
                )

        # Add threshold and change points with reduced visibility
        ax.axhline(
            y=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            alpha=0.2,  # Reduced alpha
            linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner line
            label="Thresh",  # Shortened label
            zorder=5,
        )

        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.2,  # Reduced alpha
                linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner line
                zorder=4,
            )

        # Grid settings
        ax.grid(
            True,
            which="major",
            linestyle=":",
            alpha=GRID["MAJOR_ALPHA"] * 0.5,  # Reduced alpha
            linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner lines
        )

        # Axis settings
        ax.set_xlim(0, 200)
        ax.set_xticks(np.arange(0, 201, 100))  # Reduced number of ticks

        # Dynamic y-axis range
        if self.method == "multiview":
            mart_seq = combined_results.get("martingales_sum", [])
            pred_mart_seq = combined_results.get("prediction_martingale_sum", [])
        else:
            mart_seq = combined_results.get("martingales", [])
            pred_mart_seq = combined_results.get("prediction_martingales", [])

        max_mart = 0
        if isinstance(mart_seq, (list, np.ndarray)) and len(mart_seq) > 0:
            max_mart = max(max_mart, np.max(mart_seq))
        if isinstance(pred_mart_seq, (list, np.ndarray)) and len(pred_mart_seq) > 0:
            max_mart = max(max_mart, np.max(pred_mart_seq))

        if max_mart > 0:
            y_max = max(max_mart * 1.05, self.threshold * 1.05)  # Reduced margin
            ax.set_ylim(0, y_max)  # Start from 0
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduced number of ticks

        # Legend
        ax.legend(
            ncol=3,  # Increased columns for compactness
            loc="upper right",
            fontsize=TYPO["LEGEND_SIZE"],
            frameon=True,
            handlelength=LEGEND["HANDLE_LENGTH"],
            handletextpad=0.2,  # Reduced padding
            columnspacing=LEGEND["COLUMN_SPACING"],
            borderaxespad=0.1,  # Reduced padding
        )

        # Labels
        ax.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"], labelpad=1)
        ax.set_ylabel("Value", fontsize=TYPO["LABEL_SIZE"], labelpad=1)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TYPO["TICK_SIZE"],
            pad=1,
            length=2,
        )

        plt.tight_layout(pad=0.1)  # Tighter layout
        self._save_figure("combined_martingales.png")  # Save as PNG

    def _plot_overlaid_martingales(self) -> None:
        """Create plot overlaying individual feature martingales with combined martingale."""
        fig = plt.figure(figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 2))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        features = [
            "mean_degree",
            "density",
            "mean_clustering",
            "mean_betweenness",
            "mean_eigenvector",
            "mean_closeness",
            "max_singular_value",
            "min_nonzero_laplacian",
        ]

        # Use a different color for each feature
        colors = plt.cm.tab10(np.linspace(0, 1, len(features)))

        # Plot traditional martingales (top subplot)
        ax1 = fig.add_subplot(gs[0])

        # Plot individual feature martingales
        for i, feature in enumerate(features):
            if feature in self.martingales:
                martingales = self.martingales[feature].get("martingales", [])
                if isinstance(martingales, (list, np.ndarray)) and len(martingales) > 0:
                    ax1.plot(
                        martingales,
                        color=colors[i],
                        label=feature.replace("_", " ").title(),
                        linewidth=1.0,
                        alpha=0.3,
                    )

        # Plot the combined sum martingale
        combined_sum = self.martingales["combined"].get("martingales_sum", [])
        if isinstance(combined_sum, (list, np.ndarray)) and len(combined_sum) > 0:
            ax1.plot(
                combined_sum,
                color="blue",
                linestyle="--",
                label="Combined (Sum)",
                linewidth=1.5,
                alpha=0.8,
                zorder=10,
            )

        # Add threshold and change points to top subplot
        self._add_threshold_and_changes(ax1)
        ax1.set_title("Traditional Martingales", fontsize=TYPO["TITLE_SIZE"])

        # Plot prediction martingales (bottom subplot)
        ax2 = fig.add_subplot(gs[1])

        # Plot individual feature prediction martingales
        for i, feature in enumerate(features):
            if feature in self.martingales:
                pred_martingales = self.martingales[feature].get(
                    "prediction_martingales", []
                )
                if (
                    isinstance(pred_martingales, (list, np.ndarray))
                    and len(pred_martingales) > 0
                ):
                    # Get traditional martingales to calculate history size
                    martingales = self.martingales[feature].get("martingales", [])
                    history_size = len(martingales) - len(pred_martingales)
                    time_points = np.arange(
                        history_size, history_size + len(pred_martingales)
                    )
                    ax2.plot(
                        time_points,
                        pred_martingales,
                        color=colors[i],
                        label=feature.replace("_", " ").title(),
                        linewidth=1.0,
                        alpha=0.3,
                    )

        # Plot the combined prediction sum martingale
        combined_pred_sum = self.martingales["combined"].get(
            "prediction_martingale_sum", []
        )
        if (
            isinstance(combined_pred_sum, (list, np.ndarray))
            and len(combined_pred_sum) > 0
        ):
            # Get traditional sum martingale to calculate history size
            combined_sum = self.martingales["combined"].get("martingales_sum", [])
            history_size = len(combined_sum) - len(combined_pred_sum)
            time_points = np.arange(history_size, history_size + len(combined_pred_sum))
            ax2.plot(
                time_points,
                combined_pred_sum,
                color="orange",
                linestyle="--",
                label="Combined (Sum)",
                linewidth=1.5,
                alpha=0.8,
                zorder=10,
            )

        # Add threshold and change points to bottom subplot
        self._add_threshold_and_changes(ax2)
        ax2.set_title("Prediction Martingales", fontsize=TYPO["TITLE_SIZE"])

        # Set common parameters for both subplots
        for ax in [ax1, ax2]:
            ax.grid(True, which="major", linestyle=":", alpha=GRID["MAJOR_ALPHA"])
            ax.grid(True, which="minor", linestyle=":", alpha=GRID["MAJOR_ALPHA"] / 2)
            ax.set_xticks(np.arange(0, 201, 50))
            ax.set_xlim(0, 200)

            # Create consistent legend positioning for both subplots
            ax.legend(
                ncol=2,
                loc="upper right",
                fontsize=TYPO["LEGEND_SIZE"] * 0.8,
                frameon=True,
                handlelength=LEGEND["HANDLE_LENGTH"],
                handletextpad=0.5,
                columnspacing=LEGEND["COLUMN_SPACING"],
                borderaxespad=0.1,  # Consistent padding
            )

            ax.tick_params(axis="both", which="major", labelsize=TYPO["TICK_SIZE"])

        # Set labels only for bottom subplot
        ax2.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"], labelpad=4)
        for ax in [ax1, ax2]:
            ax.set_ylabel("Martingale Value", fontsize=TYPO["LABEL_SIZE"], labelpad=4)

        plt.tight_layout(pad=0.3)
        self._save_figure("overlaid_martingales.png")

    def _add_threshold_and_changes(self, ax) -> None:
        """Add threshold line and change point markers to an axis."""
        # Add threshold line
        ax.axhline(
            y=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            alpha=0.3,
            linewidth=GRID["MAJOR_LINE_WIDTH"],
        )

        # Add change points
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.3,
                linewidth=GRID["MAJOR_LINE_WIDTH"],
            )

        # Set y-axis limits based on data
        lines = ax.get_lines()
        if lines:
            y_data = []
            for line in lines:
                data = line.get_ydata()
                if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    y_data.append(data)

            if y_data:
                y_data = np.concatenate(y_data)
                y_data = y_data[~np.isnan(y_data)]  # Remove NaN values
                if len(y_data) > 0:
                    y_min, y_max = np.min(y_data), np.max(y_data)
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)

    def _plot_shap_values(self) -> None:
        """Create plot for SHAP values with two subplots: traditional and prediction."""
        if self.shap_values is None or self.feature_names is None:
            return

        fig = plt.figure(figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 1.6))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        # Create color map for available features
        n_features = len(self.feature_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_features))

        # Plot traditional SHAP values (top subplot)
        ax1 = fig.add_subplot(gs[0])
        self._plot_shap_subplot(
            ax1, self.shap_values, self.feature_names, colors, "Traditional SHAP Values"
        )

        # Plot prediction SHAP values if available (bottom subplot)
        ax2 = fig.add_subplot(gs[1])
        if (
            hasattr(self, "prediction_shap_values")
            and self.prediction_shap_values is not None
        ):
            self._plot_shap_subplot(
                ax2,
                self.prediction_shap_values,
                self.feature_names,
                colors,
                "Prediction SHAP Values",
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No prediction SHAP values available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=TYPO["LABEL_SIZE"],
            )
            ax2.set_title("Prediction SHAP Values", fontsize=TYPO["TITLE_SIZE"])

        # Set common parameters
        for ax in [ax1, ax2]:
            ax.set_xticks(np.arange(0, 201, 50))
            ax.set_xlim(0, 200)
            ax.tick_params(axis="both", which="major", labelsize=TYPO["TICK_SIZE"])
            ax.grid(True, linestyle=":", alpha=GRID["MAJOR_ALPHA"] * 0.7)

            # Add threshold and change points
            self._add_threshold_and_changes(ax)

        # Set labels
        ax2.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"], labelpad=1)
        for ax in [ax1, ax2]:
            ax.set_ylabel("SHAP Value", fontsize=TYPO["LABEL_SIZE"], labelpad=1)

        plt.tight_layout(pad=0.1)  # Tighter layout
        self._save_figure("shap_values.png")

    def _plot_shap_subplot(self, ax, shap_values, feature_names, colors, title):
        """Helper method to plot SHAP values in a subplot."""
        if not isinstance(shap_values, (list, np.ndarray)) or len(shap_values) == 0:
            ax.text(
                0.5,
                0.5,
                "No SHAP values available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=TYPO["LABEL_SIZE"],
            )
            ax.set_title(title, fontsize=TYPO["TITLE_SIZE"])
            return

        # Plot SHAP values for each feature
        for i, feature in enumerate(feature_names):
            if i < shap_values.shape[1]:  # Only plot if we have values for this feature
                ax.plot(
                    shap_values[:, i],
                    label=feature.replace("mean_", "").replace("_", " ").title(),
                    color=colors[i],
                    linewidth=LS["LINE_WIDTH"],
                    alpha=LS["LINE_ALPHA"],
                )

        # Add change point markers
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.2,  # Reduced alpha
                linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner line
            )

        # Add threshold line
        ax.axhline(
            y=0,  # SHAP values are centered around 0
            color=COLORS["threshold"],
            linestyle="--",
            alpha=0.2,  # Reduced alpha
            linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.5,  # Thinner line
        )

        # Dynamic y-axis range with small margin
        if isinstance(shap_values, (list, np.ndarray)) and len(shap_values) > 0:
            y_min, y_max = np.nanmin(shap_values), np.nanmax(shap_values)
            if not (np.isnan(y_min) or np.isnan(y_max)):
                y_margin = max((y_max - y_min) * 0.1, 0.1)  # At least 0.1 margin
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Legend configuration
        if len(ax.get_lines()) > 0:  # Only add legend if we have plotted lines
            ax.legend(
                ncol=2,
                loc="upper right",
                fontsize=TYPO["LEGEND_SIZE"] * 0.8,
                frameon=True,
                handlelength=LEGEND["HANDLE_LENGTH"],
                handletextpad=0.2,
                columnspacing=LEGEND["COLUMN_SPACING"],
                borderaxespad=0.1,
            )

        ax.set_title(title, fontsize=TYPO["TITLE_SIZE"])

    def _save_figure(self, filename: str) -> None:
        """Save figure with publication-quality settings."""
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        final_filename = f"{self.prefix}{filename}" if self.prefix else filename
        plt.savefig(
            os.path.join(self.output_dir, final_filename),
            dpi=EXPORT["DPI"],
            bbox_inches=EXPORT["BBOX_INCHES"],
            pad_inches=EXPORT["PAD_INCHES"],
            format=EXPORT["FORMAT"],
            transparent=EXPORT["TRANSPARENT"],
        )
