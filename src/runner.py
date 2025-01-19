# src/runner.py

"""Utility functions for network prediction and analysis."""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import networkx as nx
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from changepoint.detector import ChangePointDetector
from changepoint.threshold import CustomThresholdModel
from predictor.visualizer import Visualizer
from analysis import analyze_martingale_distributions

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for network experiments."""

    model: str
    params: Dict[str, Any]
    min_history: int
    prediction_window: int
    martingale_threshold: float = 60.0
    martingale_epsilon: float = 0.85
    shap_threshold: float = 30.0
    shap_window_size: int = 5
    n_runs: int = 1
    save_individual: bool = False
    visualize_individual: bool = False


class ExperimentRunner:
    """Class to run network prediction and analysis experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[Path] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the experiment runner.

        Args:
            config: ExperimentConfig object containing experiment parameters
            output_dir: Optional output directory for results
            seed: Optional random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.output_dir = output_dir or self._create_output_dir()

        # Initialize components
        self.generator = GraphGenerator()
        self.feature_extractor = NetworkFeatureExtractor()
        self.detector = ChangePointDetector()

        # Feature weights for martingale analysis
        self.feature_weights = {
            "betweenness": 1.0,  # Most reliable
            "clustering": 0.85,  # Most consistent
            "closeness": 0.7,  # Moderate reliability
            "degree": 0.5,  # Least reliable
        }

    def _create_output_dir(self) -> Path:
        """Create and return output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.n_runs > 1:
            base_dir = Path(f"results/multi_{self.config.model}_{timestamp}")
        else:
            base_dir = Path(f"results/{self.config.model}_{timestamp}")

        base_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_dir = base_dir  # Use the same directory for all results
        return base_dir

    def run(self) -> Dict[str, Any]:
        """Run the experiment(s).

        This is the main entry point that handles both single and multiple runs.

        Returns:
            Dict containing results (single run) or aggregated results (multiple runs)
        """
        if self.config.n_runs > 1:
            return self._run_multiple()
        else:
            return self._run_single()

    def _run_single(
        self, run_number: Optional[int] = None, reuse_dir: bool = False
    ) -> Dict[str, Any]:
        """Run a single experiment.

        Args:
            run_number: Optional run number for multiple experiments
            reuse_dir: Whether to reuse the existing output directory

        Returns:
            Dict containing experiment results
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate data
        graphs, ground_truth = self._generate_data()

        # Initialize predictor
        predictor = self._initialize_predictor()

        # Compute metrics
        actual_metrics = self._compute_actual_metrics(graphs)
        forecast_metrics = self._compute_forecast_metrics(graphs, predictor)

        results = {
            "graphs": graphs,
            "ground_truth": ground_truth,
            "actual_metrics": actual_metrics,
            "forecast_metrics": forecast_metrics,
            "seed": self.seed,
            "run_number": run_number,
        }

        # Only save and visualize individual results if this is a single run
        # or if explicitly requested for multiple runs
        if self.config.n_runs == 1 or self.config.save_individual:
            self._save_results(results)
        if self.config.n_runs == 1 or self.config.visualize_individual:
            self.visualize_results(results)

        return results

    def _run_multiple(self) -> Dict[str, Any]:
        """Run multiple experiments and aggregate results."""
        all_results = []
        base_seed = self.seed if self.seed is not None else np.random.randint(0, 10000)

        # Store aggregation data
        aggregated_features = {
            "actual": {
                "degree": [],
                "clustering": [],
                "betweenness": [],
                "closeness": [],
            },
            "predicted": {
                "degree": [],
                "clustering": [],
                "betweenness": [],
                "closeness": [],
            },
        }
        aggregated_martingales = {
            "actual": {"reset": {}, "cumulative": {}},
            "predicted": {"reset": {}, "cumulative": {}},
        }
        all_change_points = {
            "actual": [],  # Ground truth CPs
            "detected": [],  # CPs detected from actual features
            "predicted": [],  # CPs predicted from predicted features
        }

        for i in range(self.config.n_runs):
            logger.info(f"\nRunning experiment {i+1}/{self.config.n_runs}")
            run_seed = base_seed + i if base_seed is not None else None

            # Run single experiment with the current seed
            results = self._run_single(run_number=i + 1, reuse_dir=True)
            all_results.append(results)

            # Collect actual features
            for feature in aggregated_features["actual"].keys():
                actual_feature_values = [
                    state["metrics"][
                        f"avg_{feature}" if feature != "clustering" else feature
                    ]
                    for state in results["graphs"][self.config.min_history :]
                ]
                predicted_feature_values = [
                    state["metrics"][
                        f"avg_{feature}" if feature != "clustering" else feature
                    ]
                    for state in results["forecast_metrics"][0]
                ]

                logger.debug(
                    f"Feature {feature} values shape - Actual: {len(actual_feature_values)}, Predicted: {len(predicted_feature_values)}"
                )

                aggregated_features["actual"][feature].append(actual_feature_values)
                aggregated_features["predicted"][feature].append(
                    predicted_feature_values
                )

            # Collect martingales with debug logging
            for feature in results["actual_metrics"][1]["reset"].keys():
                # Initialize lists if not present
                if feature not in aggregated_martingales["actual"]["reset"]:
                    aggregated_martingales["actual"]["reset"][feature] = []
                    aggregated_martingales["actual"]["cumulative"][feature] = []
                    aggregated_martingales["predicted"]["reset"][feature] = []
                    aggregated_martingales["predicted"]["cumulative"][feature] = []

                # Get actual martingales with shape logging
                actual_reset = results["actual_metrics"][1]["reset"][feature][
                    "martingales"
                ]
                actual_cumul = results["actual_metrics"][1]["cumulative"][feature][
                    "martingales"
                ]
                logger.debug(f"Run {i+1}, Feature {feature} martingale shapes:")
                logger.debug(f"  Actual reset: {np.array(actual_reset).shape}")
                logger.debug(f"  Actual cumulative: {np.array(actual_cumul).shape}")

                # Get predicted martingales
                pred_reset = results["forecast_metrics"][2]["reset"][feature][
                    "martingales"
                ]
                pred_cumul = results["forecast_metrics"][2]["cumulative"][feature][
                    "martingales"
                ]
                logger.debug(f"  Predicted reset: {np.array(pred_reset).shape}")
                logger.debug(f"  Predicted cumulative: {np.array(pred_cumul).shape}")

                # Append to aggregation lists
                aggregated_martingales["actual"]["reset"][feature].append(actual_reset)
                aggregated_martingales["actual"]["cumulative"][feature].append(
                    actual_cumul
                )
                aggregated_martingales["predicted"]["reset"][feature].append(pred_reset)
                aggregated_martingales["predicted"]["cumulative"][feature].append(
                    pred_cumul
                )

            # Collect change points
            actual_cps = results["ground_truth"]["change_points"]
            all_change_points["actual"].append(actual_cps)

            detected_cps = self._get_detected_change_points(
                results["actual_metrics"][1]["reset"]
            )
            all_change_points["detected"].append(detected_cps)

            predicted_cps = self._get_detected_change_points(
                results["forecast_metrics"][2]["reset"]
            )
            all_change_points["predicted"].append(predicted_cps)

        # Create final aggregated results
        aggregated = self._create_aggregated_results(
            all_results,
            aggregated_features,
            aggregated_martingales,
            all_change_points,
            {},  # Empty delays dict
            {},  # Empty delays_per_cp dict
        )

        # Save aggregated results
        self._save_aggregated_results(aggregated)

        # Create and save aggregated visualizations
        self._visualize_aggregated_results(aggregated)

        # Compute KL divergence between actual and predicted martingales
        # Calculate proper time indices for actual and predicted values
        time_points_actual = range(
            self.config.min_history,
            self.config.min_history
            + len(
                aggregated["martingales"]["actual"]["reset"]["degree"]["martingales"]
            ),
        )
        time_points_pred = range(
            self.config.min_history - self.config.prediction_window,
            self.config.min_history
            - self.config.prediction_window
            + len(
                aggregated["martingales"]["predicted"]["reset"]["degree"]["martingales"]
            ),
        )

        # Calculate sum of martingales across features
        features = ["degree", "clustering", "betweenness", "closeness"]
        actual_sum = np.zeros(len(time_points_actual))
        pred_sum = np.zeros(len(time_points_pred))

        # Also track averages
        actual_avg = np.zeros(len(time_points_actual))
        pred_avg = np.zeros(len(time_points_pred))
        n_features = 0

        for feature in features:
            if feature in aggregated["martingales"]["actual"]["reset"]:
                n_features += 1
                # Add to sums
                actual_sum += np.array(
                    aggregated["martingales"]["actual"]["reset"][feature]["martingales"]
                )
                pred_sum += np.array(
                    aggregated["martingales"]["predicted"]["reset"][feature][
                        "martingales"
                    ]
                )

                # Add to running averages
                actual_avg = actual_sum / n_features
                pred_avg = pred_sum / n_features

        # Compute distribution analysis
        distribution_analysis = analyze_martingale_distributions(
            actual_sum=actual_sum,
            pred_sum=pred_sum,
            time_points_actual=time_points_actual,
            time_points_pred=time_points_pred,
            prediction_window=self.config.prediction_window,
        )

        # Create distribution visualization
        from analysis import visualize_distribution_analysis

        visualize_distribution_analysis(
            actual_sum=actual_sum,
            pred_sum=pred_sum,
            actual_avg=actual_avg,
            pred_avg=pred_avg,
            time_points_actual=time_points_actual,
            time_points_pred=time_points_pred,
            prediction_window=self.config.prediction_window,
            output_path=self.aggregated_dir / "martingale_distributions.png",
        )

        # Log the analysis results
        logger.info("\nMartingale Distribution Analysis:")
        logger.info("-" * 50)
        logger.info(
            f"KL Divergence (Histogram): {distribution_analysis['kl_div_hist']:.3f}"
        )
        logger.info(f"KL Divergence (KDE): {distribution_analysis['kl_div_kde']:.3f}")
        logger.info(f"Jensen-Shannon Divergence: {distribution_analysis['js_div']:.3f}")
        logger.info(f"Correlation: {distribution_analysis['correlation']:.3f}")

        # Add distribution analysis to aggregated results
        aggregated["distribution_analysis"] = distribution_analysis

        return aggregated

    def _get_detected_change_points(self, martingales: Dict[str, Dict]) -> List[int]:
        """Get unique change points detected across all features."""
        all_cps = []
        for feature_data in martingales.values():
            if "change_detected_instant" in feature_data:
                all_cps.extend(feature_data["change_detected_instant"])
        return sorted(list(set(all_cps)))

    def _calculate_cp_delays(
        self,
        actual_sum: np.ndarray,
        pred_sum: np.ndarray,
        time_points_actual: range,
        time_points_pred: range,
        change_points: Dict[int, float],
        threshold: float,
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Calculate delays based on threshold crossings for each change point.

        Args:
            actual_sum: Array of actual sum martingale values
            pred_sum: Array of predicted sum martingale values
            time_points_actual: Range of timesteps for actual values
            time_points_pred: Range of timesteps for predicted values
            change_points: Dict mapping change points to their frequencies
            threshold: Martingale threshold value

        Returns:
            Dict containing detection and prediction delays for each CP
        """
        delays = {
            "detection": {},  # CP -> {"mean": float, "std": float}
            "prediction": {},
        }

        # Convert ranges to lists for easier indexing
        actual_times = list(time_points_actual)
        pred_times = list(time_points_pred)

        for cp in change_points.keys():
            # Find first threshold crossing after CP
            detection_times = []
            prediction_times = []

            # Look for threshold crossings within min_segment//2 steps after CP
            window = self.config.params.min_segment // 2

            # For actual detections
            try:
                cp_idx = actual_times.index(cp)
                window_end = min(cp_idx + window, len(actual_sum))
                for i in range(cp_idx, window_end):
                    if actual_sum[i] > threshold:
                        detection_times.append(actual_times[i] - cp)
                        break
            except ValueError:
                pass  # CP not in actual time range

            # For predictions
            try:
                cp_idx = pred_times.index(cp)
                window_end = min(cp_idx + window, len(pred_sum))
                for i in range(cp_idx, window_end):
                    if pred_sum[i] > threshold:
                        prediction_times.append(pred_times[i] - cp)
                        break
            except ValueError:
                pass  # CP not in prediction time range

            # Calculate statistics if we found any crossings
            if detection_times:
                delays["detection"][cp] = {
                    "mean": float(np.mean(detection_times)),
                    "std": (
                        float(np.std(detection_times))
                        if len(detection_times) > 1
                        else 0.0
                    ),
                }

            if prediction_times:
                delays["prediction"][cp] = {
                    "mean": float(np.mean(prediction_times)),
                    "std": (
                        float(np.std(prediction_times))
                        if len(prediction_times) > 1
                        else 0.0
                    ),
                }

        return delays

    def _create_aggregated_results(
        self,
        all_results: List[Dict],
        aggregated_features: Dict,
        aggregated_martingales: Dict,
        all_change_points: Dict,
        earliest_delays: Dict,
        delays_per_cp: Dict[str, Dict[int, List[float]]],
    ) -> Dict[str, Any]:
        """Create final aggregated results with averages and statistics."""
        # Average features
        avg_features = {
            "actual": {
                feature: np.mean(values, axis=0).tolist()
                for feature, values in aggregated_features["actual"].items()
            },
            "predicted": {
                feature: np.mean(values, axis=0).tolist()
                for feature, values in aggregated_features["predicted"].items()
            },
        }

        # Average martingales
        avg_martingales = {
            "actual": {
                reset_type: {
                    feature: {
                        "martingales": self._safe_aggregate_values(values, "mean"),
                        "std": self._safe_aggregate_values(values, "std"),
                    }
                    for feature, values in feature_data.items()
                    if values  # Only process features with data
                }
                for reset_type, feature_data in aggregated_martingales["actual"].items()
            },
            "predicted": {
                reset_type: {
                    feature: {
                        "martingales": self._safe_aggregate_values(values, "mean"),
                        "std": self._safe_aggregate_values(values, "std"),
                    }
                    for feature, values in feature_data.items()
                    if values  # Only process features with data
                }
                for reset_type, feature_data in aggregated_martingales[
                    "predicted"
                ].items()
            },
        }

        # Change point statistics
        cp_stats = {
            "actual": {
                "mean_count": float(
                    np.mean([len(cps) for cps in all_change_points["actual"]])
                ),
                "std_count": float(
                    np.std([len(cps) for cps in all_change_points["actual"]])
                ),
                "positions": self._aggregate_cp_positions(all_change_points["actual"]),
            },
            "detected": {
                "mean_count": float(
                    np.mean([len(cps) for cps in all_change_points["detected"]])
                ),
                "std_count": float(
                    np.std([len(cps) for cps in all_change_points["detected"]])
                ),
                "positions": self._aggregate_cp_positions(
                    all_change_points["detected"]
                ),
            },
            "predicted": {
                "mean_count": float(
                    np.mean([len(cps) for cps in all_change_points["predicted"]])
                ),
                "std_count": float(
                    np.std([len(cps) for cps in all_change_points["predicted"]])
                ),
                "positions": self._aggregate_cp_positions(
                    all_change_points["predicted"]
                ),
            },
        }

        return {
            "n_runs": self.config.n_runs,
            "features": avg_features,
            "martingales": avg_martingales,
            "change_points": cp_stats,
            "config": self.config,
        }

    def _safe_aggregate_values(
        self, values: List[Any], operation: str
    ) -> Union[float, List[float]]:
        """Safely aggregate values handling both single values and arrays."""
        if not values:
            return None

        # Debug log input values
        logger.debug(f"\nAggregating values for operation {operation}")
        logger.debug(f"Input values type: {type(values)}")
        logger.debug(f"First value type: {type(values[0]) if values else 'None'}")

        # Convert values to numpy array
        try:
            # Convert each value to float first if they're arrays
            if isinstance(values[0], (list, np.ndarray)):
                # Make sure all arrays have the same length
                first_len = len(values[0])
                if not all(len(x) == first_len for x in values):
                    logger.error("Arrays have different lengths")
                    return None
                # Convert to 2D array
                arr = np.array(values, dtype=float)
            else:
                arr = np.array(values, dtype=float)

            logger.debug(f"Converted array shape: {arr.shape}")
            logger.debug(f"Array data type: {arr.dtype}")
        except Exception as e:
            logger.error(f"Could not convert values to array: {e}")
            logger.debug(f"Problematic values: {values}")
            return None

        # Handle single values vs arrays
        try:
            if arr.ndim == 1:
                logger.debug("Processing as 1D array (single values)")
                if operation == "mean":
                    result = float(np.mean(arr))
                else:  # std
                    result = float(np.std(arr))
                logger.debug(f"Result: {result}")
                return result
            else:
                logger.debug(f"Processing as {arr.ndim}D array")
                if operation == "mean":
                    result = np.mean(arr, axis=0)
                else:  # std
                    # Compute std along axis 0 (across runs)
                    result = np.std(
                        arr, axis=0, ddof=1
                    )  # Use ddof=1 for sample standard deviation
                logger.debug(f"Result shape: {result.shape}")
                return result.tolist()
        except Exception as e:
            logger.error(f"Error in computation: {e}")
            return None

    def _aggregate_cp_positions(self, cp_lists: List[List[int]]) -> Dict[int, float]:
        """Aggregate change point positions across runs into a frequency map."""
        all_positions = []
        for cps in cp_lists:
            all_positions.extend(cps)

        unique_positions = sorted(set(all_positions))
        return {
            pos: all_positions.count(pos) / len(cp_lists) for pos in unique_positions
        }

    def _visualize_aggregated_results(self, aggregated: Dict[str, Any]):
        """Create visualizations for aggregated results."""
        # Create aggregated feature evolution plot
        plt.figure(figsize=(15, 10))
        for feature in ["degree", "clustering", "betweenness", "closeness"]:
            if feature in aggregated["features"]["actual"]:
                plt.subplot(
                    2,
                    2,
                    list(aggregated["features"]["actual"].keys()).index(feature) + 1,
                )
                # Create proper time indices starting from min_history
                time_points = range(
                    self.config.min_history,
                    self.config.min_history
                    + len(aggregated["features"]["actual"][feature]),
                )

                # Calculate error metrics for this feature
                actual = np.array(aggregated["features"]["actual"][feature])
                predicted = np.array(aggregated["features"]["predicted"][feature])
                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))

                plt.plot(
                    time_points,
                    aggregated["features"]["actual"][feature],
                    label="Actual",
                )
                plt.plot(
                    time_points,
                    aggregated["features"]["predicted"][feature],
                    label="Predicted",
                )

                # Add change point marker explanation
                plt.plot([], [], "r--", alpha=0.5, label="Change Point")

                # Add metrics to legend
                plt.plot([], [], " ", label=f"MAE = {mae:.3f}")
                plt.plot([], [], " ", label=f"RMSE = {rmse:.3f}")

                plt.title(f"Average {feature.capitalize()} Evolution")
                plt.xlabel("Time")
                plt.ylabel(feature.capitalize())
                plt.legend()

                # Add change point frequencies as vertical lines
                for pos, freq in aggregated["change_points"]["actual"][
                    "positions"
                ].items():
                    plt.axvline(x=int(pos), color="r", alpha=freq * 0.5, linestyle="--")

        plt.tight_layout()
        plt.savefig(
            self.aggregated_dir / "aggregated_features.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Calculate proper time indices for actual and predicted values
        time_points_actual = range(
            self.config.min_history,
            self.config.min_history
            + len(
                aggregated["martingales"]["actual"]["reset"]["degree"]["martingales"]
            ),
        )
        time_points_pred = range(
            self.config.min_history - self.config.prediction_window,
            self.config.min_history
            - self.config.prediction_window
            + len(
                aggregated["martingales"]["predicted"]["reset"]["degree"]["martingales"]
            ),
        )

        # Calculate sum and average of martingales across features
        features = ["degree", "clustering", "betweenness", "closeness"]
        actual_sum = np.zeros(len(time_points_actual))
        actual_sum_std = np.zeros(len(time_points_actual))
        pred_sum = np.zeros(len(time_points_pred))
        pred_sum_std = np.zeros(len(time_points_pred))

        # Calculate sums and averages
        for feature in features:
            if feature in aggregated["martingales"]["actual"]["reset"]:
                # Add to sum for actual
                actual_mart = np.array(
                    aggregated["martingales"]["actual"]["reset"][feature]["martingales"]
                )
                actual_std = np.array(
                    aggregated["martingales"]["actual"]["reset"][feature]["std"]
                )
                actual_sum += actual_mart
                actual_sum_std += actual_std**2  # Add variances

                # Add to sum for predicted
                pred_mart = np.array(
                    aggregated["martingales"]["predicted"]["reset"][feature][
                        "martingales"
                    ]
                )
                pred_std = np.array(
                    aggregated["martingales"]["predicted"]["reset"][feature]["std"]
                )
                pred_sum += pred_mart
                pred_sum_std += pred_std**2  # Add variances

        # Calculate averages
        actual_avg = actual_sum / len(features)
        actual_avg_std = np.sqrt(actual_sum_std) / len(features)
        pred_avg = pred_sum / len(features)
        pred_avg_std = np.sqrt(pred_sum_std) / len(features)

        # Calculate delays
        delays = self._calculate_cp_delays(
            actual_sum=actual_sum,
            pred_sum=pred_sum,
            time_points_actual=time_points_actual,
            time_points_pred=time_points_pred,
            change_points=aggregated["change_points"]["actual"]["positions"],
            threshold=self.config.martingale_threshold,
        )

        # 1. Main plot (Sum and Average)
        fig_main = plt.figure(figsize=(15, 8))
        ax_main = fig_main.add_subplot(111)

        # Plot actual and predicted sums
        ax_main.plot(
            time_points_actual,
            actual_sum,
            label="Sum Martingale",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        ax_main.fill_between(
            time_points_actual,
            actual_sum - np.sqrt(actual_sum_std),
            actual_sum + np.sqrt(actual_sum_std),
            color="blue",
            alpha=0.1,
        )
        ax_main.plot(
            time_points_pred,
            pred_sum,
            label="Predicted Sum",
            color="orange",
            linestyle="-",
            linewidth=2,
        )
        ax_main.fill_between(
            time_points_pred,
            pred_sum - np.sqrt(pred_sum_std),
            pred_sum + np.sqrt(pred_sum_std),
            color="orange",
            alpha=0.1,
        )

        # Plot averages with different colors
        ax_main.plot(
            time_points_actual,
            actual_avg,
            label="Avg Martingale",
            color="#2ecc71",
            linestyle="--",
            linewidth=2,
        )
        ax_main.fill_between(
            time_points_actual,
            actual_avg - actual_avg_std,
            actual_avg + actual_avg_std,
            color="#2ecc71",
            alpha=0.1,
        )
        ax_main.plot(
            time_points_pred,
            pred_avg,
            label="Predicted Avg",
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
        )
        ax_main.fill_between(
            time_points_pred,
            pred_avg - pred_avg_std,
            pred_avg + pred_avg_std,
            color="#e74c3c",
            alpha=0.1,
        )

        # Add threshold line and change points
        ax_main.axhline(
            y=self.config.martingale_threshold,
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Threshold",
        )
        for pos, freq in aggregated["change_points"]["actual"]["positions"].items():
            ax_main.axvline(x=int(pos), color="g", alpha=freq * 0.5, linestyle="--")

        # Add arrows and annotations for delays
        max_mart_value = max(np.max(actual_sum), np.max(pred_sum))
        threshold = self.config.martingale_threshold

        for cp in delays["detection"].keys():
            # Only add arrows if we have both detection and prediction delays for this CP
            if cp in delays["prediction"]:
                det_delay = delays["detection"][cp]
                pred_delay = delays["prediction"][cp]

                # For predictions
                arrow_start = cp - 20  # Start 20 steps before CP
                y_start = threshold * 1.3  # Position above threshold
                y_end = threshold * 1.5

                # Add prediction arrow
                ax_main.annotate(
                    f'prediction\ndelay: {pred_delay["mean"]:.1f}±{pred_delay["std"]:.1f}',
                    xy=(cp + pred_delay["mean"], y_start),  # Arrow end
                    xytext=(arrow_start, y_end),  # Text position
                    arrowprops=dict(
                        arrowstyle="->",
                        color="orange",
                        connectionstyle="arc3,rad=-0.3",
                        linewidth=2,
                        mutation_scale=15,
                    ),
                    color="orange",
                    fontsize=10,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="orange",
                        alpha=0.8,
                        pad=0.5,
                        boxstyle="round,pad=0.5",
                    ),
                )

                # For detections (positioned higher)
                y_start = threshold * 1.8
                y_end = threshold * 2.0

                # Add detection arrow
                ax_main.annotate(
                    f'detection\ndelay: {det_delay["mean"]:.1f}±{det_delay["std"]:.1f}',
                    xy=(cp + det_delay["mean"], y_start),  # Arrow end
                    xytext=(arrow_start, y_end),  # Text position
                    arrowprops=dict(
                        arrowstyle="->",
                        color="blue",
                        connectionstyle="arc3,rad=-0.3",
                        linewidth=2,
                        mutation_scale=15,
                    ),
                    color="blue",
                    fontsize=10,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="blue",
                        alpha=0.8,
                        pad=0.5,
                        boxstyle="round,pad=0.5",
                    ),
                )

        ax_main.set_xlabel("Time", fontsize=12)
        ax_main.set_ylabel("Martingale Value", fontsize=12)
        ax_main.legend(loc="upper right", fontsize=10)
        ax_main.tick_params(axis="both", which="major", labelsize=10)

        plt.tight_layout()
        plt.savefig(
            self.aggregated_dir / "aggregated_martingales_main.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Individual feature plots (2x2 grid)
        fig_features = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig_features)

        for i, feature in enumerate(features):
            if feature in aggregated["martingales"]["actual"]["reset"]:
                ax = fig_features.add_subplot(gs[i // 2, i % 2])

                # Plot actual martingales
                actual_mart = aggregated["martingales"]["actual"]["reset"][feature][
                    "martingales"
                ]
                actual_std = aggregated["martingales"]["actual"]["reset"][feature][
                    "std"
                ]
                ax.plot(
                    time_points_actual,
                    actual_mart,
                    label="Actual",
                    color="blue",
                    linewidth=1,
                )
                ax.fill_between(
                    time_points_actual,
                    [m - s for m, s in zip(actual_mart, actual_std)],
                    [m + s for m, s in zip(actual_mart, actual_std)],
                    color="blue",
                    alpha=0.2,
                )

                # Plot predicted martingales (shifted left)
                pred_mart = aggregated["martingales"]["predicted"]["reset"][feature][
                    "martingales"
                ]
                pred_std = aggregated["martingales"]["predicted"]["reset"][feature][
                    "std"
                ]
                ax.plot(
                    time_points_pred,
                    pred_mart,
                    label="Predicted",
                    color="orange",
                    linewidth=1,
                )
                ax.fill_between(
                    time_points_pred,
                    [m - s for m, s in zip(pred_mart, pred_std)],
                    [m + s for m, s in zip(pred_mart, pred_std)],
                    color="orange",
                    alpha=0.2,
                )

                # Add change point frequencies as vertical lines
                for pos, freq in aggregated["change_points"]["actual"][
                    "positions"
                ].items():
                    ax.axvline(x=int(pos), color="g", alpha=freq * 0.5, linestyle="--")

                ax.set_title(f"{feature.capitalize()} Martingales", fontsize=10)
                ax.set_xlabel("Time", fontsize=8)
                ax.set_ylabel("Martingale Value", fontsize=8)
                # Add change point to legend
                ax.plot([], [], "g--", alpha=0.5, label="Change Point")
                ax.legend(fontsize=8)
                ax.tick_params(axis="both", which="major", labelsize=8)

        plt.tight_layout()
        plt.savefig(
            self.aggregated_dir / "aggregated_martingales_features.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Print summary statistics
        logger.info("\nMartingale Distribution Analysis:")
        logger.info("-" * 50)
        if "distribution_analysis" in aggregated:
            dist_analysis = aggregated["distribution_analysis"]
            logger.info(
                f"KL Divergence (Histogram): {dist_analysis['kl_div_hist']:.3f}"
            )
            logger.info(f"KL Divergence (KDE): {dist_analysis['kl_div_kde']:.3f}")
            logger.info(f"Jensen-Shannon Divergence: {dist_analysis['js_div']:.3f}")
            logger.info(f"Correlation: {dist_analysis['correlation']:.3f}")

    def _generate_data(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate synthetic network data."""
        logger.info(f"Generating {self.config.model} network time series...")
        result = self.generator.generate_sequence(
            model=self.config.model, params=self.config.params, seed=self.seed
        )

        # Map parameters to timesteps
        param_map = {}
        current_params = result["parameters"][0]
        for i in range(self.config.params.seq_len):
            if i in result["change_points"]:
                current_params = result["parameters"][
                    result["change_points"].index(i) + 1
                ]
            param_map[i] = current_params

        # Convert to network states
        network_states = []
        for i, adj in enumerate(result["graphs"]):
            G = nx.from_numpy_array(adj)
            network_states.append(
                {
                    "time": i,
                    "adjacency": adj,
                    "graph": G,
                    "metrics": self.feature_extractor.get_all_metrics(G).__dict__,
                    "params": param_map[i],
                    "is_change_point": i in result["change_points"],
                }
            )

        ground_truth = {
            "change_points": result["change_points"],
            "parameters": result["parameters"],
            "metadata": result["metadata"],
            "model": result["model"],
            "num_changes": result["num_changes"],
            "n": result["n"],
            "sequence_length": result["sequence_length"],
        }

        return network_states, ground_truth

    def _initialize_predictor(self):
        """Initialize the appropriate predictor based on configuration.

        Returns:
            Initialized predictor object
        """
        from predictor.weighted import WeightedPredictor
        from predictor.hybrid import (
            BAPredictor,
            SBMPredictor,
            WSPredictor,
            ERPredictor,
        )

        PREDICTOR_MAP = {
            "weighted": WeightedPredictor,
            "hybrid": {
                "ba": BAPredictor,
                "ws": WSPredictor,
                "er": ERPredictor,
                "sbm": SBMPredictor,
            },
        }

        if not hasattr(self.config, "predictor_type"):
            # Default to weighted predictor if not specified
            logger.info("No predictor type specified, using weighted predictor")
            return WeightedPredictor()

        if self.config.predictor_type == "weighted":
            return WeightedPredictor()
        elif self.config.predictor_type == "hybrid":
            # Get the appropriate hybrid predictor for the model type
            predictor_class = PREDICTOR_MAP["hybrid"].get(self.config.model)
            if predictor_class is None:
                raise ValueError(
                    f"No hybrid predictor available for model type {self.config.model}"
                )
            return predictor_class(config=self.config)
        else:
            raise ValueError(f"Unknown predictor type: {self.config.predictor_type}")

    def _compute_actual_metrics(self, graphs: List[Dict[str, Any]]) -> tuple:
        """Compute actual network metrics."""
        features = self.compute_network_features(graphs[self.config.min_history :])
        martingales = self.compute_martingales(features)
        shap_values = self.compute_shap_values(martingales, features)
        return features, martingales, shap_values

    def _compute_forecast_metrics(
        self, graphs: List[Dict[str, Any]], predictor
    ) -> tuple:
        """Compute forecasting metrics."""
        predictions = self.generate_predictions(graphs, predictor)
        features = self.compute_network_features(predictions)
        martingales = self.compute_martingales(features)
        shap_values = self.compute_shap_values(martingales, features)
        return predictions, features, martingales, shap_values

    def _save_results(self, results: Dict[str, Any]):
        """Save individual experiment results."""
        if not self.output_dir:
            return

        try:
            serializable_data = {
                "model": self.config.model,
                "parameters": self._convert_to_serializable(self.config),
                "ground_truth": self._convert_to_serializable(results["ground_truth"]),
                "seed": self.seed,
            }

            # For multiple runs, save individual results in numbered subdirectories
            if self.config.n_runs > 1 and self.config.save_individual:
                run_dir = (
                    self.output_dir / f"run_{results.get('run_number', 'unknown')}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                save_path = run_dir / "results.json"
            else:
                save_path = self.output_dir / "results.json"

            with open(save_path, "w") as f:
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            # Save in a simpler format
            simple_data = {
                "model": self.config.model,
                "seed": self.seed,
                "error": str(e),
            }
            error_path = (
                self.output_dir
                / f"run_{results.get('run_number', 'unknown')}"
                / "results_error.json"
                if self.config.n_runs > 1 and self.config.save_individual
                else self.output_dir / "results_error.json"
            )
            with open(error_path, "w") as f:
                json.dump(simple_data, f, indent=4)

    def _save_aggregated_results(self, aggregated: Dict[str, Any]):
        """Save aggregated results from multiple runs."""
        try:
            with open(self.aggregated_dir / "aggregated_results.json", "w") as f:
                json.dump(self._convert_to_serializable(aggregated), f, indent=4)
        except Exception as e:
            logger.error(f"Error saving aggregated results: {e}")
            # Save basic information if there's an error
            with open(self.aggregated_dir / "aggregated_results_error.json", "w") as f:
                json.dump(
                    {"error": str(e), "n_runs": aggregated.get("n_runs", None)},
                    f,
                    indent=4,
                )

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types and custom objects to Python native types for JSON serialization."""
        if obj is None:
            return None
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        elif hasattr(obj, "__dict__"):  # Handle custom classes and dataclasses
            return {
                key: ExperimentRunner._convert_to_serializable(value)
                for key, value in obj.__dict__.items()
                if not key.startswith("_")  # Skip private attributes
            }
        elif isinstance(obj, dict):
            return {
                str(key): ExperimentRunner._convert_to_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [ExperimentRunner._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [ExperimentRunner._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, "tolist"):  # Handle other array-like objects
            return obj.tolist()
        elif hasattr(obj, "__str__"):  # Fallback for other objects
            return str(obj)
        return obj

    def generate_predictions(
        self,
        network_series: List[Dict[str, Any]],
        predictor: Any,
    ) -> List[Dict[str, Any]]:
        """Generate predictions using the given predictor.

        Args:
            network_series: List of network states
            predictor: Network predictor object

        Returns:
            List of prediction results
        """
        predictions = []
        seq_len = len(network_series)  # Get sequence length from input data

        for t in range(self.config.min_history, seq_len):
            history = network_series[:t]
            predicted_adjs = predictor.predict(
                history, horizon=self.config.prediction_window
            )

            pred_graph = nx.from_numpy_array(predicted_adjs[0])
            predictions.append(
                {
                    "time": t,
                    "adjacency": predicted_adjs[0],
                    "graph": pred_graph,
                    "metrics": self.feature_extractor.get_all_metrics(
                        pred_graph
                    ).__dict__,
                    "history_size": len(history),
                }
            )

        return predictions

    def compute_network_features(
        self, graphs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute network features for analysis.

        Args:
            graphs: List of network states

        Returns:
            Dict of computed features
        """
        features = {
            "degree": [],
            "clustering": [],
            "betweenness": [],
            "closeness": [],
        }

        for state in graphs:
            G = state["graph"]
            metrics = self.feature_extractor.get_all_metrics(G)
            features["degree"].append(metrics.avg_degree)
            features["clustering"].append(metrics.clustering)
            features["betweenness"].append(metrics.avg_betweenness)
            features["closeness"].append(metrics.avg_closeness)

        return {k: np.array(v) for k, v in features.items()}

    def analyze_results(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze prediction accuracy across different phases.

        Args:
            results: Dict containing experiment results

        Returns:
            Dict containing analysis results
        """
        if not results["forecast_metrics"]:
            logger.warning("No forecast metrics to analyze")
            return {"phases": {}, "error": "No data to analyze"}

        # Define phases
        total_predictions = len(results["forecast_metrics"])
        if total_predictions < 3:
            phases = [(0, total_predictions, "All predictions")]
        else:
            third = total_predictions // 3
            phases = [
                (0, third, "First third"),
                (third, 2 * third, "Middle third"),
                (2 * third, None, "Last third"),
            ]

        results = {"phases": {}}

        # Analyze each phase
        for start, end, phase_name in phases:
            logger.info(f"\nAnalyzing {phase_name}:")
            phase_predictions = results["forecast_metrics"][start:end]
            phase_actuals = results["graphs"][
                self.config.min_history
                + start : self.config.min_history
                + (end if end else len(results["forecast_metrics"]))
            ]

            # Calculate errors
            all_errors = []
            for pred, actual in zip(phase_predictions, phase_actuals):
                pred_metrics = self.feature_extractor.get_all_metrics(
                    pred["graph"]
                ).__dict__
                actual_metrics = self.feature_extractor.get_all_metrics(
                    actual["graph"]
                ).__dict__
                all_errors.append(calculate_error_metrics(actual_metrics, pred_metrics))

            if not all_errors:
                logger.warning(f"No errors calculated for {phase_name}")
                continue

            # Average errors
            avg_errors = {
                metric: np.mean([e[metric] for e in all_errors])
                for metric in all_errors[0].keys()
            }

            results["phases"][phase_name] = {
                "start": start,
                "end": end,
                "errors": avg_errors,
            }

            for metric, error in avg_errors.items():
                logger.info(f"Average MAE for {metric}: {error:.3f}")

        return results

    def compute_martingales(
        self,
        features: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute martingales for network features.

        Args:
            features: Dict of network features

        Returns:
            Dict containing martingale analysis results
        """
        martingales = {"reset": {}, "cumulative": {}}

        for feature_name, feature_data in features.items():
            weight = self.feature_weights.get(feature_name, 1.0)
            adjusted_threshold = self.config.martingale_threshold / weight

            # Reset martingales
            reset_results = self.detector.detect_changes(
                data=feature_data.reshape(-1, 1),
                threshold=adjusted_threshold,
                epsilon=self.config.martingale_epsilon,
                reset=True,
            )

            weighted_martingales = reset_results["martingale_values"] * weight
            martingales["reset"][feature_name] = {
                "martingales": weighted_martingales,
                "change_detected_instant": reset_results["change_points"],
                "weight": weight,
                "adjusted_threshold": adjusted_threshold,
            }

            # Cumulative martingales
            cumul_results = self.detector.detect_changes(
                data=feature_data.reshape(-1, 1),
                threshold=adjusted_threshold,
                epsilon=self.config.martingale_epsilon,
                reset=False,
            )

            weighted_cumul_martingales = cumul_results["martingale_values"] * weight
            martingales["cumulative"][feature_name] = {
                "martingales": weighted_cumul_martingales,
                "change_detected_instant": cumul_results["change_points"],
                "weight": weight,
                "adjusted_threshold": adjusted_threshold,
            }

        return martingales

    def compute_shap_values(
        self,
        martingales: Dict[str, Dict[str, Dict[str, Any]]],
        features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute SHAP values for martingale analysis.

        Args:
            martingales: Dict of martingale results
            features: Dict of network features

        Returns:
            numpy array of SHAP values
        """
        model = CustomThresholdModel(threshold=self.config.shap_threshold)

        feature_matrix = np.column_stack(
            [
                martingales["reset"][feature]["martingales"]
                for feature in features.keys()
            ]
        )

        change_points = sorted(
            list(
                set(
                    cp
                    for m in martingales["reset"].values()
                    for cp in m["change_detected_instant"]
                )
            )
        )

        return model.compute_shap_values(
            X=feature_matrix,
            change_points=change_points,
            sequence_length=len(feature_matrix),
            window_size=self.config.shap_window_size,
        )

    def visualize_results(self, results: Dict[str, Any]):
        """Generate visualizations for the results."""
        visualizer = Visualizer()

        # 1. Metric evolution
        plt.figure(figsize=(12, 8))
        visualizer.plot_metric_evolution(
            results["graphs"],
            results["forecast_metrics"][0],  # predictions
            self.config.min_history,
            model_type=self.config.model,
            change_points=results["ground_truth"]["change_points"],
        )
        plt.savefig(
            self.output_dir / "metric_evolution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Performance extremes
        plt.figure(figsize=(20, 15))
        visualizer.plot_performance_extremes(
            results["graphs"][
                self.config.min_history : self.config.min_history
                + len(results["forecast_metrics"][0])
            ],
            results["forecast_metrics"][0],
            min_history=self.config.min_history,
            model_type=self.config.model,
            change_points=results["ground_truth"]["change_points"],
        )
        plt.savefig(
            self.output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Martingale comparison dashboard
        visualizer.create_martingale_comparison_dashboard(
            network_series=results["graphs"],
            actual_martingales=results["actual_metrics"][1],
            pred_martingales=results["forecast_metrics"][2],
            actual_shap=results["actual_metrics"][2],
            pred_shap=results["forecast_metrics"][3],
            output_path=self.output_dir / "martingale_comparison_dashboard.png",
            threshold=self.config.martingale_threshold,
            epsilon=self.config.martingale_epsilon,
            change_points=results["ground_truth"]["change_points"],
            prediction_window=self.config.prediction_window,
        )
