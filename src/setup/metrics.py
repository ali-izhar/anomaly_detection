# src/setup/metrics.py

"""Network metric computation and analysis."""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from changepoint.detector import ChangePointDetector
from changepoint.threshold import CustomThresholdModel

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Container for network metrics."""

    features: Dict[str, np.ndarray]
    martingales: Dict[str, Dict[str, Dict[str, Any]]]
    shap_values: Optional[np.ndarray] = None


class MetricComputer:
    """Computes and analyzes network metrics."""

    def __init__(
        self,
        feature_extractor: NetworkFeatureExtractor,
        detector: ChangePointDetector,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize metric computer."""
        self.feature_extractor = feature_extractor
        self.detector = detector
        self.feature_weights = feature_weights or {
            "degree": 1.0,
            "clustering": 1.0,
            "betweenness": 1.0,
            "closeness": 1.0,
        }

    def compute_network_features(
        self, graphs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Extract network features from graph states."""
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

    def compute_martingales(
        self,
        features: Dict[str, np.ndarray],
        threshold: float,
        epsilon: float,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute martingales for change point detection."""
        martingales = {"reset": {}, "cumulative": {}}

        # Step 1: Process each feature
        for feature_name, feature_values in features.items():
            # Step 2: Compute reset martingales
            reset_martingales = self.detector.detect_changes(
                data=feature_values.reshape(-1, 1),
                threshold=threshold,
                epsilon=epsilon,
                reset=True,
            )
            martingales["reset"][feature_name] = reset_martingales

            # Step 3: Compute cumulative martingales
            cumul_martingales = self.detector.detect_changes(
                data=feature_values.reshape(-1, 1),
                threshold=threshold,
                epsilon=epsilon,
                reset=False,
            )
            martingales["cumulative"][feature_name] = cumul_martingales

        return martingales

    def compute_shap_values(
        self,
        martingales: Dict[str, Dict[str, Dict[str, Any]]],
        features: Dict[str, np.ndarray],
        threshold: float,
        window_size: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute SHAP values for feature importance."""
        model = CustomThresholdModel(threshold=threshold)

        # Step 1: Create feature matrix
        feature_names = list(features.keys())
        feature_matrix = np.column_stack(
            [
                martingales["reset"][feature]["martingale_values"]
                for feature in feature_names
            ]
        )

        # Step 2: Get unique change points
        change_points = sorted(
            list(
                set(
                    cp
                    for m in martingales["reset"].values()
                    for cp in m["change_points"]
                )
            )
        )

        # Step 3: Compute SHAP values
        shap_values = model.compute_shap_values(
            X=feature_matrix,
            change_points=change_points,
            sequence_length=len(feature_matrix),
            window_size=window_size,
        )

        # Step 4: Structure SHAP values by feature
        result = {}
        for i, feature in enumerate(feature_names):
            result[feature] = {
                "values": shap_values[:, i].tolist(),
                "mean": float(np.mean(shap_values[:, i])),
                "std": float(np.std(shap_values[:, i])),
            }

        return result

    def compute_metrics(
        self,
        graphs: List[Dict[str, Any]],
        min_history: int,
        martingale_threshold: float,
        martingale_epsilon: float,
        shap_threshold: float,
        shap_window_size: int,
    ) -> NetworkMetrics:
        """Compute all metrics for a graph sequence."""
        # Step 1: Compute network features
        features = self.compute_network_features(graphs[min_history:])

        # Step 2: Compute martingales
        martingales = self.compute_martingales(
            features=features,
            threshold=martingale_threshold,
            epsilon=martingale_epsilon,
        )

        # Step 3: Compute SHAP values
        shap_values = self.compute_shap_values(
            martingales=martingales,
            features=features,
            threshold=shap_threshold,
            window_size=shap_window_size,
        )

        return NetworkMetrics(
            features=features,
            martingales=martingales,
            shap_values=shap_values,
        )

    def analyze_results(
        self,
        results: Dict[str, Any],
        min_history: int,
    ) -> Dict[str, Any]:
        """Analyze prediction accuracy across phases."""
        if not results["forecast_metrics"]:
            logger.warning("No forecast metrics to analyze")
            return {"phases": {}, "error": "No data to analyze"}

        # Step 1: Define analysis phases
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

        # Step 2: Analyze each phase
        analysis_results = {"phases": {}}
        for start, end, phase_name in phases:
            phase_predictions = results["forecast_metrics"][start:end]
            phase_actuals = results["graphs"][
                min_history
                + start : min_history
                + (end if end else len(results["forecast_metrics"]))
            ]

            # Calculate errors for each prediction
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

            # Average errors across predictions
            avg_errors = {
                metric: np.mean([e[metric] for e in all_errors])
                for metric in all_errors[0].keys()
            }

            analysis_results["phases"][phase_name] = {
                "start": start,
                "end": end,
                "errors": avg_errors,
            }

            for metric, error in avg_errors.items():
                logger.info(f"Average MAE for {metric}: {error:.3f}")

        return analysis_results

    def calculate_cp_delays(
        self,
        actual_sum: np.ndarray,
        pred_sum: np.ndarray,
        time_points_actual: range,
        time_points_pred: range,
        change_points: Dict[int, float],
        threshold: float,
        min_segment: int,
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Calculate detection and prediction delays for change points."""
        delays = {"detection": {}, "prediction": {}}
        actual_times = list(time_points_actual)
        pred_times = list(time_points_pred)

        # Step 1: Process each change point
        for cp in change_points:
            detection_times = []
            prediction_times = []
            window = min_segment // 2

            # Step 2: Find actual detections
            try:
                cp_idx = actual_times.index(cp)
                window_end = min(cp_idx + window, len(actual_sum))
                for i in range(cp_idx, window_end):
                    if actual_sum[i] > threshold:
                        detection_times.append(actual_times[i] - cp)
                        break
            except ValueError:
                pass

            # Step 3: Find predictions
            try:
                cp_idx = pred_times.index(cp)
                window_end = min(cp_idx + window, len(pred_sum))
                for i in range(cp_idx, window_end):
                    if pred_sum[i] > threshold:
                        prediction_times.append(pred_times[i] - cp)
                        break
            except ValueError:
                pass

            # Step 4: Calculate delay statistics
            if detection_times:
                delays["detection"][str(cp)] = {
                    "mean": round(float(np.mean(detection_times)), 2),
                    "std": (
                        round(float(np.std(detection_times)), 2)
                        if len(detection_times) > 1
                        else 0.0
                    ),
                }

            if prediction_times:
                delays["prediction"][str(cp)] = {
                    "mean": round(float(np.mean(prediction_times)), 2),
                    "std": (
                        round(float(np.std(prediction_times)), 2)
                        if len(prediction_times) > 1
                        else 0.0
                    ),
                }

        return delays
