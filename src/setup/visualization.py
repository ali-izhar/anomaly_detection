# src/setup/visualization.py

"""Network visualization and plotting functionality."""

from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.setup.config import (
    ExperimentConfig,
    OutputConfig,
    VisualizationConfig,
)
from src.setup.metrics import MetricComputer

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualizes experiment results and metrics."""

    def __init__(
        self,
        vis_config: Optional[VisualizationConfig] = None,
        output_config: Optional[OutputConfig] = None,
        metric_computer: Optional["MetricComputer"] = None,
    ):
        """Initialize visualizer with configurations."""
        self.vis_config = vis_config or VisualizationConfig()
        self.output_config = output_config or OutputConfig()
        self.metric_computer = metric_computer
        plt.style.use("seaborn-v0_8-paper")

    def visualize_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        config: ExperimentConfig,
    ):
        """Generate visualizations for single experiment results."""
        # Step 1: Plot metric evolution
        plt.figure(
            figsize=(
                self.vis_config.single_column_width,
                self.vis_config.standard_height,
            )
        )
        self.plot_metric_evolution(
            results["graphs"],
            results["forecast_metrics"][0],
            config.min_history,
            model_type=config.model,
            change_points=results["ground_truth"]["change_points"],
        )
        self._save_figure(output_dir / "metric_evolution.png")

        # Step 2: Plot performance extremes
        plt.figure(
            figsize=(
                self.vis_config.double_column_width,
                self.vis_config.standard_height * 2,
            )
        )
        self.plot_performance_extremes(
            results["graphs"][
                config.min_history : config.min_history
                + len(results["forecast_metrics"][0])
            ],
            results["forecast_metrics"][0],
            min_history=config.min_history,
            model_type=config.model,
            change_points=results["ground_truth"]["change_points"],
        )
        self._save_figure(output_dir / "performance_extremes.png")

        # Step 3: Plot martingale comparison dashboard
        self._set_plot_style()
        self.create_martingale_comparison_dashboard(
            network_series=results["graphs"],
            actual_martingales=results["actual_metrics"][1],
            pred_martingales=results["forecast_metrics"][2],
            actual_shap=results["actual_metrics"][2],
            pred_shap=results["forecast_metrics"][3],
            output_path=output_dir / "martingale_comparison_dashboard.png",
            threshold=config.martingale_threshold,
            epsilon=config.martingale_epsilon,
            change_points=results["ground_truth"]["change_points"],
            prediction_window=config.prediction_window,
        )
        plt.rcdefaults()

        # Step 4: Plot SHAP values
        self._plot_shap_values(
            results["actual_metrics"][2],
            results["forecast_metrics"][3],
            output_dir,
        )

    def visualize_aggregated_results(
        self,
        aggregated: Dict[str, Any],
        output_dir: Path,
        config: ExperimentConfig,
    ):
        """Visualize aggregated results from multiple runs."""
        # Step 1: Plot feature evolution
        self._plot_feature_evolution(aggregated, config, output_dir)

        # Step 2: Calculate martingale statistics
        time_points_actual, time_points_pred = self._get_time_points(aggregated, config)
        actual_sum, actual_sum_std, pred_sum, pred_sum_std = (
            self._calculate_martingale_sums(
                aggregated, time_points_actual, time_points_pred
            )
        )
        actual_avg, actual_avg_std, pred_avg, pred_avg_std = (
            self._calculate_martingale_averages(
                actual_sum, actual_sum_std, pred_sum, pred_sum_std
            )
        )

        # Step 3: Calculate delays
        delays = self.metric_computer.calculate_cp_delays(
            actual_sum=actual_sum,
            pred_sum=pred_sum,
            time_points_actual=time_points_actual,
            time_points_pred=time_points_pred,
            change_points=aggregated["change_points"]["actual"]["positions"],
            threshold=config.martingale_threshold,
            min_segment=config.params.min_segment,
        )

        # Step 4: Plot martingale summary
        fig_main = plt.figure(
            figsize=(
                self.vis_config.single_column_width,
                self.vis_config.standard_height,
            )
        )
        ax_main = fig_main.add_subplot(111)
        self._plot_martingale_summary(
            ax_main,
            time_points_actual,
            time_points_pred,
            actual_sum,
            actual_sum_std,
            pred_sum,
            pred_sum_std,
            actual_avg,
            actual_avg_std,
            pred_avg,
            pred_avg_std,
            aggregated["change_points"]["actual"]["positions"],
            config.martingale_threshold,
            delays,
        )
        self._save_figure(output_dir / "aggregated_martingales_main.png")

        # Step 5: Plot individual feature martingales
        self._plot_feature_martingales(
            ["degree", "clustering", "betweenness", "closeness"],
            time_points_actual,
            time_points_pred,
            aggregated["martingale_values"],
            aggregated["change_points"]["actual"]["positions"],
            output_dir,
        )

        # Step 6: Plot SHAP values time series
        if "shap_values" in aggregated:
            self._plot_shap_time_series(
                aggregated["shap_values"]["actual"],
                aggregated["shap_values"]["predicted"],
                output_dir,
            )
        else:
            logger.warning("No SHAP values available in aggregated results")

    def _set_plot_style(self):
        """Set matplotlib plot style parameters."""
        plt.rcParams.update(
            {
                "figure.figsize": (
                    self.vis_config.double_column_width,
                    self.vis_config.grid_height,
                ),
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

    def _save_figure(self, path: Path):
        """Save figure with standard parameters."""
        plt.savefig(
            path,
            dpi=self.output_config.dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            format=self.output_config.figure_format,
            metadata=self.output_config.figure_metadata,
        )
        plt.close()

    def _get_time_points(
        self, aggregated: Dict[str, Any], config: ExperimentConfig
    ) -> tuple[range, range]:
        """Calculate time points for actual and predicted values."""
        actual_len = len(
            aggregated["martingale_values"]["actual"]["reset"]["degree"][
                "martingale_values"
            ]
        )
        pred_len = len(
            aggregated["martingale_values"]["predicted"]["reset"]["degree"][
                "martingale_values"
            ]
        )

        # Actual martingales: start from 0 (first 20 points are 1s)
        time_points_actual = range(
            0,  # Start from 0 to show initial ones
            actual_len + 20,  # Add 20 to account for initial padding
        )

        # Predicted martingales: shift right by min_history to show predictions between change points and detections
        time_points_pred = range(
            config.min_history
            * 2,  # Start from 2*min_history to shift predictions right
            config.min_history * 2 + pred_len + 10,
        )

        return time_points_actual, time_points_pred

    def _calculate_martingale_sums(
        self,
        aggregated: Dict[str, Any],
        time_points_actual: range,
        time_points_pred: range,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate sums of martingales across features."""
        features = ["degree", "clustering", "betweenness", "closeness"]

        # Initialize arrays with ones for padding
        actual_sum = np.ones(len(time_points_actual))
        actual_sum_std = np.zeros(len(time_points_actual))
        pred_sum = np.ones(len(time_points_pred))
        pred_sum_std = np.zeros(len(time_points_pred))

        # Calculate start and end indices for actual values
        actual_start_idx = 20  # First 20 points are ones
        actual_end_idx = len(time_points_actual)

        # Calculate start and end indices for predicted values
        pred_start_idx = 0
        pred_end_idx = len(time_points_pred) - 10  # Last 10 points are ones

        # Reset the summed regions to zero before adding feature values
        actual_sum[actual_start_idx:actual_end_idx] = 0
        pred_sum[pred_start_idx:pred_end_idx] = 0

        for feature in features:
            if feature in aggregated["martingale_values"]["actual"]["reset"]:
                actual_mart = np.array(
                    aggregated["martingale_values"]["actual"]["reset"][feature][
                        "martingale_values"
                    ]
                )
                actual_std = np.array(
                    aggregated["martingale_values"]["actual"]["reset"][feature]["std"]
                )
                actual_sum[actual_start_idx:actual_end_idx] += actual_mart
                actual_sum_std[actual_start_idx:actual_end_idx] += actual_std**2

                pred_mart = np.array(
                    aggregated["martingale_values"]["predicted"]["reset"][feature][
                        "martingale_values"
                    ]
                )
                pred_std = np.array(
                    aggregated["martingale_values"]["predicted"]["reset"][feature][
                        "std"
                    ]
                )
                pred_sum[pred_start_idx:pred_end_idx] += pred_mart
                pred_sum_std[pred_start_idx:pred_end_idx] += pred_std**2

        return actual_sum, actual_sum_std, pred_sum, pred_sum_std

    def _calculate_martingale_averages(
        self,
        actual_sum: np.ndarray,
        actual_sum_std: np.ndarray,
        pred_sum: np.ndarray,
        pred_sum_std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate averages of martingales."""
        n_features = 4  # degree, clustering, betweenness, closeness
        actual_avg = actual_sum / n_features
        actual_avg_std = np.sqrt(actual_sum_std) / n_features
        pred_avg = pred_sum / n_features
        pred_avg_std = np.sqrt(pred_sum_std) / n_features

        return actual_avg, actual_avg_std, pred_avg, pred_avg_std

    def _plot_feature_evolution(
        self,
        aggregated: Dict[str, Any],
        config: ExperimentConfig,
        output_dir: Path,
    ):
        """Plot feature evolution in a grid."""
        plt.figure(
            figsize=(self.vis_config.single_column_width, self.vis_config.grid_height)
        )
        gs = plt.GridSpec(
            2,
            2,
            hspace=self.vis_config.grid_spacing,
            wspace=self.vis_config.grid_spacing,
        )

        for i, feature in enumerate(aggregated["features"]["actual"].keys()):
            ax = plt.subplot(gs[i // 2, i % 2])
            self._plot_single_feature(
                ax,
                feature,
                aggregated,
                config,
                aggregated["features"]["actual"][feature],
                aggregated["features"]["predicted"][feature],
            )

        plt.gcf().set_constrained_layout(True)
        self._save_figure(output_dir / "aggregated_features.png")

    def _plot_single_feature(
        self,
        ax: plt.Axes,
        feature: str,
        aggregated: Dict[str, Any],
        config: ExperimentConfig,
        actual_values: List[float],
        predicted_values: List[float],
    ):
        """Plot a single feature's evolution."""
        time_points = range(
            config.min_history,
            config.min_history + len(actual_values),
        )

        # Calculate metrics
        actual = np.array(actual_values)
        predicted = np.array(predicted_values)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # Plot actual and predicted
        ax.plot(
            time_points,
            actual,
            label="Actual",
            linewidth=self.vis_config.line_width,
            color=self.vis_config.colors["actual"],
            alpha=self.vis_config.line_alpha,
        )
        ax.plot(
            time_points,
            predicted,
            label="Predicted",
            linewidth=self.vis_config.line_width,
            color=self.vis_config.colors["predicted"],
            alpha=self.vis_config.line_alpha,
        )

        # Add change point markers and metrics
        ax.plot(
            [],
            [],
            "--",
            label="Change Point",
            linewidth=self.vis_config.line_width,
            color=self.vis_config.colors["change_point"],
            alpha=0.5,
        )
        ax.plot([], [], " ", label=f"MAE = {mae:.3f}")
        ax.plot([], [], " ", label=f"RMSE = {rmse:.3f}")

        # Configure plot
        ax.set_title(
            f"Avg. {feature.capitalize()}", fontsize=self.vis_config.title_size, pad=3
        )
        if list(aggregated["features"]["actual"].keys()).index(feature) >= 2:
            ax.set_xlabel("Time", fontsize=self.vis_config.label_size)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        # Add change points
        for pos, freq in aggregated["change_points"]["actual"]["positions"].items():
            ax.axvline(
                x=pos,
                color=self.vis_config.colors["change_point"],
                alpha=freq * 0.5,
                linestyle="--",
                linewidth=self.vis_config.grid_width,
            )

        # Configure axes and legend
        ax.set_xlim(0, 200)
        ax.set_xticks(np.arange(0, 201, 50))
        self._configure_legend(ax, actual, predicted, time_points)
        ax.tick_params(
            axis="both", which="major", labelsize=self.vis_config.tick_size, pad=2
        )
        ax.grid(
            True,
            linestyle=":",
            alpha=self.vis_config.grid_alpha,
            linewidth=self.vis_config.grid_width,
        )

    def _configure_legend(
        self,
        ax: plt.Axes,
        actual: np.ndarray,
        predicted: np.ndarray,
        time_points: range,
    ):
        """Configure legend placement based on data distribution."""
        ymin, ymax = ax.get_ylim()
        data = np.array([actual, predicted])

        regions = {
            "upper right": np.sum(
                (
                    data[:, int(0.75 * len(time_points)) :]
                    > (ymin + 0.75 * (ymax - ymin))
                ).any(axis=0)
            ),
            "upper left": np.sum(
                (
                    data[:, : int(0.25 * len(time_points))]
                    > (ymin + 0.75 * (ymax - ymin))
                ).any(axis=0)
            ),
            "lower right": np.sum(
                (
                    data[:, int(0.75 * len(time_points)) :]
                    < (ymin + 0.25 * (ymax - ymin))
                ).any(axis=0)
            ),
            "lower left": np.sum(
                (
                    data[:, : int(0.25 * len(time_points))]
                    < (ymin + 0.25 * (ymax - ymin))
                ).any(axis=0)
            ),
        }

        best_loc = min(regions.items(), key=lambda x: x[1])[0]
        ax.legend(
            fontsize=self.vis_config.legend_size,
            ncol=1,
            loc=best_loc,
            borderaxespad=0.1,
            handlelength=1.0,
            columnspacing=0.8,
        )

    def _plot_martingale_summary(
        self,
        ax: plt.Axes,
        time_points_actual: range,
        time_points_pred: range,
        actual_sum: np.ndarray,
        actual_sum_std: np.ndarray,
        pred_sum: np.ndarray,
        pred_sum_std: np.ndarray,
        actual_avg: np.ndarray,
        actual_avg_std: np.ndarray,
        pred_avg: np.ndarray,
        pred_avg_std: np.ndarray,
        change_points: Dict[int, float],
        threshold: float,
        delays: Dict[str, Dict[str, Dict[str, float]]],
    ):
        """Plot summary of martingale values."""
        # Plot sums
        ax.plot(
            time_points_actual,
            actual_sum,
            label="Sum Mart.",
            color=self.vis_config.colors["actual"],
            linestyle="-",
            linewidth=self.vis_config.line_width,
            alpha=self.vis_config.line_alpha,
        )
        ax.fill_between(
            time_points_actual,
            actual_sum - np.sqrt(actual_sum_std),
            actual_sum + np.sqrt(actual_sum_std),
            color=self.vis_config.colors["actual"],
            alpha=0.1,
        )

        ax.plot(
            time_points_pred,
            pred_sum,
            label="Pred. Sum",
            color=self.vis_config.colors["predicted"],
            linestyle="-",
            linewidth=self.vis_config.line_width,
            alpha=self.vis_config.line_alpha,
        )
        ax.fill_between(
            time_points_pred,
            pred_sum - np.sqrt(pred_sum_std),
            pred_sum + np.sqrt(pred_sum_std),
            color=self.vis_config.colors["predicted"],
            alpha=0.1,
        )

        # Plot averages
        ax.plot(
            time_points_actual,
            actual_avg,
            label="Avg Mart.",
            color=self.vis_config.colors["average"],
            linestyle="--",
            linewidth=self.vis_config.line_width,
            alpha=self.vis_config.line_alpha,
        )
        ax.fill_between(
            time_points_actual,
            actual_avg - actual_avg_std,
            actual_avg + actual_avg_std,
            color=self.vis_config.colors["average"],
            alpha=0.1,
        )

        ax.plot(
            time_points_pred,
            pred_avg,
            label="Pred. Avg",
            color=self.vis_config.colors["pred_avg"],
            linestyle="--",
            linewidth=self.vis_config.line_width,
            alpha=self.vis_config.line_alpha,
        )
        ax.fill_between(
            time_points_pred,
            pred_avg - pred_avg_std,
            pred_avg + pred_avg_std,
            color=self.vis_config.colors["pred_avg"],
            alpha=0.1,
        )

        # Add change points and delays
        for pos, freq in change_points.items():
            ax.axvline(
                x=int(pos),
                color=self.vis_config.colors["change_point"],
                alpha=freq * 0.5,
                linestyle="--",
                linewidth=self.vis_config.grid_width,
            )

            # Add delay annotations if available
            if str(pos) in delays.get("detection", {}) and str(pos) in delays.get(
                "prediction", {}
            ):
                det_delay = delays["detection"][str(pos)]
                pred_delay = delays["prediction"][str(pos)]

                base_y = threshold * 1.1
                spacing = threshold * 0.15

                # Detection delay
                ax.annotate(
                    f'd:{det_delay["mean"]:.1f}',
                    xy=(int(pos) + det_delay["mean"], base_y),
                    xytext=(int(pos) - 20, base_y + spacing),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=self.vis_config.colors["actual"],
                        connectionstyle="arc3,rad=-0.2",
                        linewidth=0.5,
                    ),
                    color=self.vis_config.colors["actual"],
                    fontsize=self.vis_config.annotation_size,
                    ha="right",
                    va="bottom",
                )

                # Prediction delay
                ax.annotate(
                    f'p:{pred_delay["mean"]:.1f}',
                    xy=(int(pos) + pred_delay["mean"], base_y),
                    xytext=(int(pos) - 20, base_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=self.vis_config.colors["predicted"],
                        connectionstyle="arc3,rad=-0.2",
                        linewidth=0.5,
                    ),
                    color=self.vis_config.colors["predicted"],
                    fontsize=self.vis_config.annotation_size,
                    ha="right",
                    va="top",
                )

        # Add threshold line
        ax.axhline(
            y=threshold,
            color=self.vis_config.colors["threshold"],
            linestyle="--",
            alpha=0.5,
            linewidth=self.vis_config.line_width,
            label="Threshold",
        )

        # Configure plot
        ax.set_xlabel("Time", fontsize=self.vis_config.label_size)
        ax.set_ylabel("Martingale Value", fontsize=self.vis_config.label_size)
        ax.legend(
            fontsize=self.vis_config.legend_size,
            ncol=2,
            loc="upper right",
        )
        ax.tick_params(axis="both", which="major", labelsize=self.vis_config.tick_size)
        ax.grid(True, linestyle=":", alpha=self.vis_config.grid_alpha)

    def _plot_feature_martingales(
        self,
        features: List[str],
        time_points_actual: range,
        time_points_pred: range,
        martingales: Dict[str, Dict[str, Dict[str, Any]]],
        change_points: Dict[int, float],
        output_dir: Path,
    ):
        """Plot individual feature martingales."""
        plt.figure(
            figsize=(self.vis_config.single_column_width, self.vis_config.grid_height)
        )
        gs = plt.GridSpec(
            2,
            2,
            hspace=self.vis_config.grid_spacing,
            wspace=self.vis_config.grid_spacing,
        )

        for i, feature in enumerate(features):
            if feature in martingales["actual"]["reset"]:
                ax = plt.subplot(gs[i // 2, i % 2])

                # Create padded arrays for actual martingales
                actual_mart = np.ones(len(time_points_actual))
                actual_std = np.zeros(len(time_points_actual))
                mart_values = martingales["actual"]["reset"][feature][
                    "martingale_values"
                ]
                std_values = martingales["actual"]["reset"][feature]["std"]
                actual_mart[20 : 20 + len(mart_values)] = mart_values
                actual_std[20 : 20 + len(std_values)] = std_values

                # Plot actual martingales
                ax.plot(
                    time_points_actual,
                    actual_mart,
                    label="Act.",
                    color=self.vis_config.colors["actual"],
                    linewidth=self.vis_config.line_width,
                    alpha=self.vis_config.line_alpha,
                )
                ax.fill_between(
                    time_points_actual,
                    actual_mart - actual_std,
                    actual_mart + actual_std,
                    color=self.vis_config.colors["actual"],
                    alpha=0.2,
                )

                # Create padded arrays for predicted martingales
                pred_mart = np.ones(len(time_points_pred))
                pred_std = np.zeros(len(time_points_pred))
                mart_values = martingales["predicted"]["reset"][feature][
                    "martingale_values"
                ]
                std_values = martingales["predicted"]["reset"][feature]["std"]
                pred_mart[: len(mart_values)] = mart_values
                pred_std[: len(std_values)] = std_values

                # Plot predicted martingales
                ax.plot(
                    time_points_pred,
                    pred_mart,
                    label="Pred.",
                    color=self.vis_config.colors["predicted"],
                    linewidth=self.vis_config.line_width,
                    alpha=self.vis_config.line_alpha,
                )
                ax.fill_between(
                    time_points_pred,
                    pred_mart - pred_std,
                    pred_mart + pred_std,
                    color=self.vis_config.colors["predicted"],
                    alpha=0.2,
                )

                # Add change points
                for pos, freq in change_points.items():
                    ax.axvline(
                        x=int(pos),
                        color=self.vis_config.colors["change_point"],
                        alpha=freq * 0.5,
                        linestyle="--",
                        linewidth=self.vis_config.grid_width,
                    )

                ax.set_title(
                    feature.capitalize(), fontsize=self.vis_config.title_size, pad=3
                )
                if i >= 2:  # Bottom row
                    ax.set_xlabel("Time", fontsize=self.vis_config.label_size)
                else:  # Top row
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                ax.set_ylabel("Mart. Value", fontsize=self.vis_config.label_size)
                ax.legend(
                    fontsize=self.vis_config.legend_size,
                    ncol=2,
                    loc="upper right",
                    borderaxespad=0.1,
                    handlelength=1.0,
                    columnspacing=0.8,
                )
                ax.tick_params(
                    axis="both",
                    which="major",
                    labelsize=self.vis_config.tick_size,
                    pad=2,
                )
                ax.grid(
                    True,
                    linestyle=":",
                    alpha=self.vis_config.grid_alpha,
                    linewidth=self.vis_config.grid_width,
                )
                ax.set_xlim(0, 200)
                ax.set_xticks(np.arange(0, 201, 50))

        plt.gcf().set_constrained_layout(True)
        self._save_figure(output_dir / "aggregated_martingales_features.png")

    def _plot_shap_time_series(
        self,
        actual_shap: Dict[str, Dict[str, Any]],
        predicted_shap: Dict[str, Dict[str, Any]],
        output_dir: Path,
    ):
        """Plot SHAP values over time."""
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create figure with compact research paper dimensions
        fig = plt.figure(
            figsize=(self.vis_config.single_column_width, self.vis_config.grid_height)
        )
        gs = plt.GridSpec(
            2,
            1,
            hspace=self.vis_config.grid_spacing,
        )

        # Calculate time points similar to martingales
        actual_len = len(next(iter(actual_shap.values()))["values"])
        pred_len = len(next(iter(predicted_shap.values()))["values"])

        # Actual SHAP: start from 0 (first 20 points are 1s)
        time_points_actual = range(0, actual_len + 20)

        # Predicted SHAP: shift right by min_history
        time_points_pred = range(10, 10 + pred_len)

        # Plot actual SHAP values
        ax_actual_shap = fig.add_subplot(gs[0])
        self._plot_feature_shap_values(
            ax_actual_shap,
            actual_shap,
            time_points_actual,
            "Actual",
            change_points=[50, 150],
            show_legend=True,
        )

        # Plot predicted SHAP values
        ax_pred_shap = fig.add_subplot(gs[1])
        self._plot_feature_shap_values(
            ax_pred_shap,
            predicted_shap,
            time_points_pred,
            "Predicted",
            change_points=[50, 150],
            show_legend=False,
        )

        plt.gcf().set_constrained_layout(True)
        self._save_figure(output_dir / "shap_time_series.png")

    def _plot_feature_shap_values(
        self, ax, shap_values, time_points, title, change_points=None, show_legend=False
    ):
        """Plot SHAP values for each feature over time."""
        features = ["degree", "clustering", "betweenness", "closeness"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(features)))

        # Plot SHAP values for each feature
        for idx, feature in enumerate(features):
            if feature in shap_values:
                values = np.ones(len(time_points))  # Initialize with ones
                feature_values = shap_values[feature]["values"]

                if len(time_points) > len(feature_values):
                    # For actual values: place after initial padding
                    values[20 : 20 + len(feature_values)] = feature_values
                else:
                    # For predicted values: place at start
                    values[: len(feature_values)] = feature_values

                ax.plot(
                    time_points,
                    values,
                    label=feature.capitalize(),
                    color=colors[idx],
                    linewidth=self.vis_config.line_width,
                    alpha=self.vis_config.line_alpha,
                )

        # Add change point indicators if provided
        if change_points:
            for cp in change_points:
                ax.axvline(
                    x=cp,
                    color=self.vis_config.colors["change_point"],
                    linestyle="--",
                    alpha=0.3,
                    linewidth=self.vis_config.grid_width,
                )

        # Add title in top left corner
        ax.text(
            0.02,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=self.vis_config.legend_size,
            ha="left",
            va="top",
        )

        # Only show x-axis label on bottom plot (Predicted)
        if title == "Predicted":
            ax.set_xlabel("Time", fontsize=self.vis_config.label_size)
        else:
            ax.set_xlabel("")

        if show_legend:
            ax.legend(
                fontsize=self.vis_config.legend_size,
                ncol=2,
                loc="upper right",
                borderaxespad=0.1,
                handlelength=1.0,
                columnspacing=0.8,
            )

        ax.grid(
            True,
            alpha=self.vis_config.grid_alpha,
            linestyle=":",
            linewidth=self.vis_config.grid_width,
        )
        ax.tick_params(
            axis="both", which="major", labelsize=self.vis_config.tick_size, pad=2
        )
        ax.set_xlim(0, 200)
        ax.set_xticks(np.arange(0, 201, 50))
