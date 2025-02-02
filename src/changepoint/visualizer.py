# src/changepoint/visualizer.py

"""Visualizer for changepoint analysis results."""

from pathlib import Path
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
        martingales: Dict[str, Dict[str, Any]],
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
            martingales: Dictionary of martingale results for each feature or pipeline results
            change_points: List of true change points
            threshold: Detection threshold
            epsilon: Sensitivity parameter
            output_dir: Directory to save visualizations
            prefix: Prefix for output filenames
            skip_shap: Whether to skip SHAP value computation
            method: Detection method ('single_view' or 'multiview')
        """
        # Set basic parameters
        self.threshold = threshold
        self.epsilon = epsilon
        self.change_points = change_points
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.method = method

        # Process martingales
        self.martingales = self._process_martingales(martingales)

        # Compute SHAP values if needed
        if not skip_shap:
            try:
                sequence_length = len(
                    next(iter(self.martingales.values()))["martingales"]
                )
                model = CustomThresholdModel(threshold=threshold)
                self.shap_values, self.feature_names = (
                    model.compute_martingale_shap_values(
                        martingales=self.martingales,
                        change_points=change_points,
                        sequence_length=sequence_length,
                        threshold=threshold,
                    )
                )
            except (ValueError, KeyError):
                self.shap_values = None
                self.feature_names = None

        # Set paper-style parameters
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(get_matplotlib_rc_params())

    def _process_martingales(
        self, martingales: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Process martingales to ensure consistent format."""
        # Check if this is pipeline output
        if "features_raw" in martingales.get("combined", {}):
            processed = {}
            combined_result = martingales["combined"]

            # Process combined results based on whether it's single-view or multiview
            is_multiview = self.method == "multiview"

            if is_multiview:
                # Multiview case
                processed["combined"] = {
                    "martingales": combined_result.get(
                        "martingale_sum"
                    ),  # Use sum as main martingale
                    "martingales_sum": combined_result.get("martingale_sum"),
                    "martingales_avg": combined_result.get("martingale_avg"),
                    "pvalues": combined_result.get("pvalues"),  # Fixed key name
                    "strangeness": combined_result.get("strangeness"),
                }

                # Add prediction martingales if available
                if "prediction_martingale_sum" in combined_result:
                    processed["combined"].update(
                        {
                            "prediction_martingales": combined_result[
                                "prediction_martingale_sum"
                            ],  # Use sum as main prediction
                            "prediction_martingale_sum": combined_result[
                                "prediction_martingale_sum"
                            ],
                            "prediction_martingale_avg": combined_result[
                                "prediction_martingale_avg"
                            ],
                            "prediction_pvalues": combined_result.get(
                                "prediction_pvalues"
                            ),
                            "prediction_strangeness": combined_result.get(
                                "prediction_strangeness"
                            ),
                        }
                    )

                # Extract individual feature results
                features = [
                    "degree",
                    "density",
                    "clustering",
                    "betweenness",
                    "eigenvector",
                    "closeness",
                    "singular_value",
                    "laplacian",
                ]

                # Get individual martingales and prediction martingales
                individual_martingales = combined_result.get(
                    "individual_martingales", []
                )
                prediction_individual_martingales = combined_result.get(
                    "prediction_individual_martingales", []
                )
                num_horizons = (
                    len(prediction_individual_martingales) // len(features)
                    if prediction_individual_martingales
                    else 0
                )

                for i, feature in enumerate(features):
                    if i < len(individual_martingales):
                        feature_dict = {
                            "martingales": individual_martingales[i],
                            "pvalues": combined_result.get(
                                "pvalues", [None] * len(features)
                            )[i],
                            "strangeness": combined_result.get(
                                "strangeness", [None] * len(features)
                            )[i],
                        }

                        # Add prediction martingales for this feature
                        if prediction_individual_martingales and num_horizons > 0:
                            feature_start_idx = i * num_horizons
                            feature_end_idx = (i + 1) * num_horizons

                            if feature_start_idx < len(
                                prediction_individual_martingales
                            ):
                                # Get all horizon predictions for this feature
                                feature_predictions = prediction_individual_martingales[
                                    feature_start_idx:feature_end_idx
                                ]
                                if feature_predictions and any(
                                    len(m) > 0 for m in feature_predictions
                                ):
                                    # Store each horizon's predictions separately
                                    feature_dict["prediction_martingales_horizons"] = (
                                        feature_predictions
                                    )
                                    # Also store the sum for backward compatibility
                                    feature_dict["prediction_martingales"] = (
                                        np.sum(
                                            [
                                                np.array(m)
                                                for m in feature_predictions
                                                if len(m) > 0
                                            ],
                                            axis=0,
                                        )
                                        if feature_predictions
                                        else []
                                    )
                                    feature_dict["prediction_pvalues"] = (
                                        combined_result.get(
                                            "prediction_pvalues", [None] * len(features)
                                        )[i]
                                    )
                                    feature_dict["prediction_strangeness"] = (
                                        combined_result.get(
                                            "prediction_strangeness",
                                            [None] * len(features),
                                        )[i]
                                    )

                        processed[feature] = feature_dict

            else:  # single_view
                # Single-view case
                processed["combined"] = {
                    "martingales": combined_result.get("martingales", []),
                    "pvalues": combined_result.get("pvalues", []),  # Fixed key name
                    "strangeness": combined_result.get("strangeness", []),
                }

                # Add prediction martingales if available
                if "prediction_martingales" in combined_result:
                    processed["combined"].update(
                        {
                            "prediction_martingales": combined_result[
                                "prediction_martingales"
                            ],
                            "prediction_pvalues": combined_result.get(
                                "prediction_pvalues", []
                            ),
                            "prediction_strangeness": combined_result.get(
                                "prediction_strangeness", []
                            ),
                        }
                    )

            return processed

        return martingales

    def create_visualization(self) -> None:
        """Create all visualizations."""
        # Always plot combined martingales
        self._plot_combined_martingales()

        # Only create feature plots for multiview results
        is_multiview = self.martingales["combined"].get("martingales_sum") is not None
        if is_multiview and len(self.martingales) > 1:
            self._plot_feature_martingales()
            self._plot_overlaid_martingales()

        # Only create SHAP plot if we have SHAP values and it's multiview
        if (
            is_multiview
            and self.shap_values is not None
            and self.feature_names is not None
        ):
            self._plot_shap_values()

    def _plot_feature_martingales(self) -> None:
        """Create grid of individual feature martingale plots, including both traditional and prediction."""
        fig = plt.figure(
            figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 2),
        )
        gs = GridSpec(
            4,
            2,
            figure=fig,
            hspace=FD["GRID_SPACING"],
            wspace=FD["GRID_SPACING"],
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

        max_mart_global = 0  # Track global maximum for consistent y-axis

        # First pass to find global maximum
        for feature in main_features:
            if feature in self.martingales:
                results = self.martingales[feature]

                # Check traditional martingales
                if "martingales" in results and results["martingales"] is not None:
                    martingale_values = np.array(
                        [
                            x.item() if isinstance(x, np.ndarray) else x
                            for x in results["martingales"]
                        ]
                    )
                    if len(martingale_values) > 0:
                        max_mart_global = max(
                            max_mart_global, np.max(martingale_values)
                        )

                # Check prediction martingales (both summed and individual horizons)
                if (
                    "prediction_martingales_horizons" in results
                    and results["prediction_martingales_horizons"] is not None
                ):
                    for horizon_pred in results["prediction_martingales_horizons"]:
                        pred_values = np.array(
                            [
                                x.item() if isinstance(x, np.ndarray) else x
                                for x in horizon_pred
                            ]
                        )
                        if len(pred_values) > 0:
                            max_mart_global = max(max_mart_global, np.max(pred_values))

        # Now plot each feature
        for idx, feature in enumerate(main_features):
            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs[row, col])

            if feature in self.martingales:
                results = self.martingales[feature]

                # Plot traditional martingale
                if "martingales" in results and results["martingales"] is not None:
                    martingale_values = np.array(
                        [
                            x.item() if isinstance(x, np.ndarray) else x
                            for x in results["martingales"]
                        ]
                    )
                    if len(martingale_values) > 0:
                        ax.plot(
                            martingale_values,
                            color=COLORS["actual"],
                            linewidth=LS["LINE_WIDTH"],
                            alpha=LS["LINE_ALPHA"],
                            label="Trad. Mart.",
                            zorder=10,
                        )

                # Plot individual horizon predictions
                if (
                    "prediction_martingales_horizons" in results
                    and results["prediction_martingales_horizons"] is not None
                ):
                    # Create color gradient for different horizons
                    num_horizons = len(results["prediction_martingales_horizons"])
                    colors = plt.cm.Oranges(np.linspace(0.3, 0.8, num_horizons))

                    # Get sequence length from traditional martingales
                    seq_length = (
                        len(martingale_values) if "martingales" in results else 0
                    )
                    history_size = 10  # Default history size

                    # Find actual history size from the data
                    if "combined" in self.martingales:
                        combined_trad = self.martingales["combined"].get(
                            "martingales", []
                        )
                        combined_pred = self.martingales["combined"].get(
                            "prediction_martingales", []
                        )
                        if len(combined_trad) > 0 and len(combined_pred) > 0:
                            # Calculate history size based on the difference in lengths
                            history_size = len(combined_trad) - len(combined_pred)

                    for h, horizon_pred in enumerate(
                        results["prediction_martingales_horizons"]
                    ):
                        pred_values = np.array(
                            [
                                x.item() if isinstance(x, np.ndarray) else x
                                for x in horizon_pred
                            ]
                        )
                        if len(pred_values) > 0:
                            # Create time points starting from history_size
                            time_points = np.arange(
                                history_size, history_size + len(pred_values)
                            )
                            ax.plot(
                                time_points,
                                pred_values,
                                color=colors[h],
                                linewidth=LS["LINE_WIDTH"] * 0.8,
                                linestyle="--",
                                alpha=LS["LINE_ALPHA"] * 0.7,
                                label=(
                                    f"H{h+1} Mart." if idx == 0 else None
                                ),  # Only show legend in first plot
                                zorder=5 - h,  # Earlier horizons on top
                            )

                # Add change points
                for cp in self.change_points:
                    ax.axvline(
                        x=cp,
                        color=COLORS["change_point"],
                        linestyle="--",
                        alpha=0.3,
                        linewidth=GRID["MAJOR_LINE_WIDTH"],
                        zorder=1,
                    )

                # Add threshold line
                ax.axhline(
                    y=self.threshold,
                    color=COLORS["threshold"],
                    linestyle="--",
                    alpha=0.3,
                    linewidth=GRID["MAJOR_LINE_WIDTH"],
                    zorder=2,
                )

                # Set consistent y-axis limits based on global maximum
                y_max = max_mart_global * 1.1  # Add 10% margin
                y_min = -y_max * 0.05  # Small negative range
                ax.set_ylim(y_min, y_max)

                # Create evenly spaced ticks
                n_ticks = 6
                y_ticks = np.linspace(0, y_max, n_ticks)
                ax.set_yticks(y_ticks)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(
                        lambda x, p: f"{x:.2f}" if abs(x) < 0.01 else f"{x:.1f}"
                    )
                )

            ax.set_title(feature.title(), fontsize=TYPO["TITLE_SIZE"], pad=3)
            ax.set_xlim(0, 200)
            ax.set_xticks(np.arange(0, 201, 50))

            if row == 3:
                ax.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"])
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel("Mart. Value", fontsize=TYPO["LABEL_SIZE"])

            if idx == 0:  # Only show legend in first plot
                ax.legend(
                    fontsize=LEGEND["LEGEND_SIZE"],
                    ncol=2,
                    loc="upper right",
                    borderaxespad=0.1,
                    handlelength=1.0,
                    columnspacing=0.8,
                )
            ax.tick_params(
                axis="both", which="major", labelsize=TYPO["TICK_SIZE"], pad=2
            )
            ax.grid(
                True,
                linestyle=":",
                alpha=GRID["MAJOR_ALPHA"] * 0.7,
                linewidth=GRID["MAJOR_LINE_WIDTH"] * 0.8,
            )

        plt.tight_layout(pad=0.3)
        self._save_figure("feature_martingales.png")

    def _plot_combined_martingales(self) -> None:
        """Create plot for combined martingales, including both traditional and prediction."""
        fig = plt.figure(figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"]))
        ax = fig.add_subplot(111)
        combined_results = self.martingales["combined"]
        is_multiview = combined_results.get("martingales_sum") is not None

        if is_multiview:
            # Plot traditional martingales
            sum_martingale = combined_results.get("martingales_sum")
            avg_martingale = combined_results.get("martingales_avg")

            if sum_martingale is not None:
                ax.plot(
                    sum_martingale,
                    color=COLORS["actual"],
                    label="Sum Mart.",
                    linewidth=LS["LINE_WIDTH"],
                    alpha=LS["LINE_ALPHA"],
                    zorder=10,
                )

            if avg_martingale is not None:
                ax.plot(
                    avg_martingale,
                    color=COLORS["average"],
                    label="Avg Mart.",
                    linewidth=LS["LINE_WIDTH"],
                    linestyle="--",
                    alpha=LS["LINE_ALPHA"],
                    zorder=8,
                )

            # Plot prediction martingales if available
            if "prediction_martingale_sum" in combined_results:
                pred_sum_martingale = combined_results["prediction_martingale_sum"]
                pred_avg_martingale = combined_results["prediction_martingale_avg"]

                if pred_sum_martingale is not None:
                    ax.plot(
                        pred_sum_martingale,
                        color=COLORS["predicted"],
                        label="Pred. Sum Mart.",
                        linewidth=LS["LINE_WIDTH"],
                        alpha=LS["LINE_ALPHA"],
                        zorder=9,
                    )

                if pred_avg_martingale is not None:
                    ax.plot(
                        pred_avg_martingale,
                        color=COLORS["pred_avg"],
                        label="Pred. Avg Mart.",
                        linewidth=LS["LINE_WIDTH"],
                        linestyle="--",
                        alpha=LS["LINE_ALPHA"],
                        zorder=7,
                    )
        else:
            # Plot single martingale for single-view
            martingale = combined_results.get("martingales")
            if martingale is not None:
                ax.plot(
                    martingale,
                    color=COLORS["actual"],
                    label="Martingale",
                    linewidth=LS["LINE_WIDTH"],
                    alpha=LS["LINE_ALPHA"],
                    zorder=10,
                )

            # Plot prediction martingale if available
            if "prediction_martingales" in combined_results:
                pred_martingale = combined_results["prediction_martingales"]
                if pred_martingale is not None:
                    ax.plot(
                        pred_martingale,
                        color=COLORS["predicted"],
                        label="Pred. Mart.",
                        linewidth=LS["LINE_WIDTH"],
                        linestyle="--",
                        alpha=LS["LINE_ALPHA"],
                        zorder=9,
                    )

        # Add threshold line
        ax.axhline(
            y=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            alpha=0.7,
            linewidth=LS["LINE_WIDTH"],
            label="Threshold",
            zorder=5,
        )

        # Add change points and delay annotations
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.5,
                linewidth=GRID["MAJOR_LINE_WIDTH"],
                zorder=4,
            )

            # Get the relevant martingale sequences for delay calculation
            mart_seq = combined_results.get(
                "martingales_sum" if is_multiview else "martingales"
            )
            pred_mart_seq = combined_results.get(
                "prediction_martingale_sum"
                if is_multiview
                else "prediction_martingales"
            )

            # Traditional detection delay
            if mart_seq is not None:
                detection_idx = next(
                    (
                        i
                        for i in range(cp, len(mart_seq))
                        if mart_seq[i] > self.threshold
                    ),
                    None,
                )
                if detection_idx:
                    delay = detection_idx - cp
                    ax.annotate(
                        f"d:{delay}",
                        xy=(detection_idx, self.threshold),
                        xytext=(cp - 10, self.threshold * 1.1),
                        color=COLORS["actual"],
                        fontsize=LEGEND["ANNOTATION_SIZE"],
                        arrowprops=dict(
                            arrowstyle="->",
                            color=COLORS["actual"],
                            alpha=0.8,
                            linewidth=1.0,
                            connectionstyle="arc3,rad=-0.2",
                        ),
                    )

            # Prediction detection delay
            if pred_mart_seq is not None:
                pred_detection_idx = next(
                    (
                        i
                        for i in range(cp, len(pred_mart_seq))
                        if pred_mart_seq[i] > self.threshold
                    ),
                    None,
                )
                if pred_detection_idx:
                    pred_delay = pred_detection_idx - cp
                    ax.annotate(
                        f"pd:{pred_delay}",
                        xy=(pred_detection_idx, self.threshold),
                        xytext=(cp - 10, self.threshold * 1.3),
                        color=COLORS["predicted"],
                        fontsize=LEGEND["ANNOTATION_SIZE"],
                        arrowprops=dict(
                            arrowstyle="->",
                            color=COLORS["predicted"],
                            alpha=0.8,
                            linewidth=1.0,
                            connectionstyle="arc3,rad=0.2",
                        ),
                    )

        # Grid settings
        ax.grid(True, which="major", linestyle=":", alpha=GRID["MAJOR_ALPHA"])
        ax.grid(True, which="minor", linestyle=":", alpha=GRID["MAJOR_ALPHA"] / 2)

        # Axis settings
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

        # Dynamic y-axis range
        mart_seq = combined_results.get(
            "martingales_sum" if is_multiview else "martingales"
        )
        pred_mart_seq = combined_results.get(
            "prediction_martingale_sum" if is_multiview else "prediction_martingales"
        )

        if mart_seq is not None and len(mart_seq) > 0:
            max_mart = np.max(mart_seq)
            if pred_mart_seq is not None and len(pred_mart_seq) > 0:
                max_mart = max(max_mart, np.max(pred_mart_seq))

            y_max = max(max_mart + max_mart * 0.25, self.threshold * 1.25)
            ax.set_ylim(-5, y_max)
            ax.yaxis.set_major_locator(plt.MultipleLocator(50))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(25))

        # Legend
        n_cols = 4 if pred_mart_seq is not None else (3 if is_multiview else 2)
        ax.legend(
            ncol=n_cols,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            fontsize=LEGEND["LEGEND_SIZE"],
            frameon=True,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=1.0,
        )

        # Labels
        ax.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"], labelpad=4)
        ax.set_ylabel("Martingale Value", fontsize=TYPO["LABEL_SIZE"], labelpad=4)
        ax.tick_params(axis="both", which="major", labelsize=TYPO["TICK_SIZE"])

        plt.tight_layout(pad=0.3)
        self._save_figure("combined_martingales.png")

    def _plot_overlaid_martingales(self) -> None:
        """Create plot overlaying individual feature martingales with combined martingale.
        Creates two subplots: one for traditional martingales and one for prediction martingales.
        """
        fig = plt.figure(figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 2))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

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

        # Plot traditional martingales (top subplot)
        ax1 = fig.add_subplot(gs[0])
        sum_martingale = None

        for i, feature in enumerate(feature_names):
            if feature in self.martingales:
                feature_mart = np.array(self.martingales[feature]["martingales"])
                if len(feature_mart) > 0:
                    # Initialize or add to sum_martingale
                    if sum_martingale is None:
                        sum_martingale = feature_mart
                    else:
                        sum_martingale += feature_mart

                    # Plot individual feature
                    ax1.plot(
                        feature_mart,
                        color=colors[i],
                        label=f"{feature.replace('_', ' ').title()}",
                        linewidth=1.0,
                        alpha=0.3,
                    )

        # Plot the true sum martingale from combined results
        combined_sum = self.martingales["combined"].get("martingales_sum")
        if combined_sum is not None and len(combined_sum) > 0:
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
        self._add_threshold_and_changes(ax1, combined_sum)
        ax1.set_title("Traditional Martingales", fontsize=TYPO["TITLE_SIZE"])

        # Plot prediction martingales (bottom subplot)
        ax2 = fig.add_subplot(gs[1])
        pred_sum_martingale = None

        for i, feature in enumerate(feature_names):
            if (
                feature in self.martingales
                and "prediction_martingales" in self.martingales[feature]
            ):
                pred_feature_mart = np.array(
                    self.martingales[feature]["prediction_martingales"]
                )
                if len(pred_feature_mart) > 0:
                    # Initialize or add to prediction sum_martingale
                    if pred_sum_martingale is None:
                        pred_sum_martingale = pred_feature_mart
                    else:
                        pred_sum_martingale += pred_feature_mart

                    # Plot individual prediction feature
                    ax2.plot(
                        pred_feature_mart,
                        color=colors[i],
                        label=f"{feature.replace('_', ' ').title()}",
                        linewidth=1.0,
                        alpha=0.3,
                    )

        # Plot the prediction sum martingale from combined results
        combined_pred_sum = self.martingales["combined"].get(
            "prediction_martingale_sum"
        )
        if combined_pred_sum is not None and len(combined_pred_sum) > 0:
            ax2.plot(
                combined_pred_sum,
                color="orange",
                linestyle="--",
                label="Combined (Sum)",
                linewidth=1.5,
                alpha=0.8,
                zorder=10,
            )

        # Add threshold and change points to bottom subplot
        self._add_threshold_and_changes(ax2, combined_pred_sum)
        ax2.set_title("Prediction Martingales", fontsize=TYPO["TITLE_SIZE"])

        # Set common parameters for both subplots
        for ax in [ax1, ax2]:
            ax.grid(True, which="major", linestyle=":", alpha=GRID["MAJOR_ALPHA"])
            ax.grid(True, which="minor", linestyle=":", alpha=GRID["MAJOR_ALPHA"] / 2)
            ax.set_xticks(np.arange(0, 201, 50))
            ax.set_xlim(0, 200)

            # Create two-column legend with slightly smaller font
            ax.legend(
                ncol=2,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.02),
                fontsize=LEGEND["LEGEND_SIZE"] * 0.8,
                frameon=True,
                handlelength=1.5,
                handletextpad=0.5,
                columnspacing=1.0,
            )

            ax.tick_params(axis="both", which="major", labelsize=TYPO["TICK_SIZE"])

        # Set labels only for bottom subplot
        ax2.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"], labelpad=4)
        for ax in [ax1, ax2]:
            ax.set_ylabel("Martingale Value", fontsize=TYPO["LABEL_SIZE"], labelpad=4)

        plt.tight_layout(pad=0.3)
        self._save_figure("overlaid_martingales.png")

    def _add_threshold_and_changes(self, ax, martingale_values):
        """Helper method to add threshold line and change points to a subplot."""
        # Add threshold line
        ax.axhline(
            y=self.threshold,
            color=COLORS["threshold"],
            linestyle="--",
            alpha=0.7,
            linewidth=LS["LINE_WIDTH"],
            label="Threshold",
            zorder=7,
        )

        # Add change points and delay annotations
        for cp in self.change_points:
            ax.axvline(
                x=cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.5,
                linewidth=GRID["MAJOR_LINE_WIDTH"],
                zorder=6,
            )

            if martingale_values is not None and len(martingale_values) > 0:
                detection_idx = next(
                    (
                        i
                        for i in range(cp, len(martingale_values))
                        if martingale_values[i] > self.threshold
                    ),
                    None,
                )
                if detection_idx:
                    delay = detection_idx - cp
                    ax.annotate(
                        f"d:{delay}",
                        xy=(detection_idx, self.threshold),
                        xytext=(cp - 10, self.threshold * 1.1),
                        color=COLORS["actual"],
                        fontsize=LEGEND["ANNOTATION_SIZE"],
                        arrowprops=dict(
                            arrowstyle="->",
                            color=COLORS["actual"],
                            alpha=0.8,
                            linewidth=1.0,
                            connectionstyle="arc3,rad=-0.2",
                        ),
                    )

        # Set y-axis limits based on martingale values
        if martingale_values is not None and len(martingale_values) > 0:
            max_mart = np.max(martingale_values)
            if not np.isnan(max_mart):
                y_max = max(max_mart + max_mart * 0.25, self.threshold * 1.25)
                ax.set_ylim(-5, y_max)
                ax.yaxis.set_major_locator(plt.MultipleLocator(50))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
        else:
            # Set default y-axis limits if no martingale values
            ax.set_ylim(-5, self.threshold * 2)
            ax.yaxis.set_major_locator(plt.MultipleLocator(20))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(10))

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
        if hasattr(self, "prediction_shap_values"):
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

        # Set labels
        ax2.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"])
        for ax in [ax1, ax2]:
            ax.set_ylabel("SHAP Value", fontsize=TYPO["LABEL_SIZE"])

        plt.tight_layout(pad=0.3)
        self._save_figure("shap_values.png")

    def _plot_shap_subplot(self, ax, shap_values, feature_names, colors, title):
        """Helper method to plot SHAP values in a subplot."""
        if shap_values is None or len(shap_values) == 0:
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

        # Add vertical "Actual" label
        ax.text(
            -0.1,
            0.5,
            "Actual",
            transform=ax.transAxes,
            fontsize=TYPO["LABEL_SIZE"],
            va="center",
            ha="right",
            rotation=90,
        )

        # Plot SHAP values for each feature
        for i, feature in enumerate(feature_names):
            if i < shap_values.shape[1]:  # Only plot if we have values for this feature
                ax.plot(
                    shap_values[:, i],
                    label=feature.title(),
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
                alpha=0.3,
                linewidth=GRID["MAJOR_LINE_WIDTH"],
            )

        # Dynamic y-axis range with small margin
        if len(shap_values) > 0:
            y_min, y_max = np.nanmin(shap_values), np.nanmax(shap_values)
            if not (np.isnan(y_min) or np.isnan(y_max)):
                y_margin = max((y_max - y_min) * 0.1, 0.1)  # At least 0.1 margin
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Legend configuration
        if len(ax.get_lines()) > 0:  # Only add legend if we have plotted lines
            ax.legend(
                ncol=2,
                loc="upper right",
                fontsize=LEGEND["LEGEND_SIZE"],
                borderaxespad=0.1,
                handlelength=1.0,
                columnspacing=0.8,
            )

        ax.set_title(title, fontsize=TYPO["TITLE_SIZE"])

    def _save_figure(self, filename: str) -> None:
        """Save figure with publication-quality settings."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_filename = f"{self.prefix}{filename}" if self.prefix else filename
        plt.savefig(
            self.output_dir / final_filename,
            dpi=EXPORT["DPI"],
            bbox_inches=EXPORT["BBOX_INCHES"],
            pad_inches=EXPORT["PAD_INCHES"],
            format=EXPORT["FORMAT"],
            transparent=EXPORT["TRANSPARENT"],
        )
        plt.close()
