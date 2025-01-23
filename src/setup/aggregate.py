# src/setup/aggregate.py

"""Module for aggregating experiment results."""

from typing import Dict, List, Any, Union
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregates and analyzes experiment results."""

    def __init__(self, n_runs: int, min_history: int):
        """Initialize aggregator.

        Args:
            n_runs: Number of experiment runs
            min_history: Minimum history length for predictions
        """
        self.n_runs = n_runs
        self.min_history = min_history

    def aggregate_results(
        self,
        all_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate results from multiple runs."""
        # Step 1: Initialize aggregation structures
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
            "detected": [],  # CPs from actual features
            "predicted": [],  # CPs from predicted features
        }

        earliest_delays = {"detection": {}, "prediction": {}}
        delays_per_cp = {"detection": {}, "prediction": {}}

        # Step 2: Collect data from all runs
        for result in all_results:
            # Collect features
            for feature in aggregated_features["actual"].keys():
                actual_feature_values = [
                    state["metrics"][
                        f"avg_{feature}" if feature != "clustering" else feature
                    ]
                    for state in result["graphs"][self.min_history :]
                ]
                predicted_feature_values = [
                    state["metrics"][
                        f"avg_{feature}" if feature != "clustering" else feature
                    ]
                    for state in result["forecast_metrics"][0]
                ]

                aggregated_features["actual"][feature].append(actual_feature_values)
                aggregated_features["predicted"][feature].append(
                    predicted_feature_values
                )

            # Collect martingales
            for feature in result["actual_metrics"][1]["reset"].keys():
                if feature not in aggregated_martingales["actual"]["reset"]:
                    aggregated_martingales["actual"]["reset"][feature] = []
                    aggregated_martingales["actual"]["cumulative"][feature] = []
                    aggregated_martingales["predicted"]["reset"][feature] = []
                    aggregated_martingales["predicted"]["cumulative"][feature] = []

                # Get actual martingales
                actual_reset = result["actual_metrics"][1]["reset"][feature][
                    "martingale_values"
                ]
                actual_cumul = result["actual_metrics"][1]["cumulative"][feature][
                    "martingale_values"
                ]

                # Get predicted martingales
                pred_reset = result["forecast_metrics"][2]["reset"][feature][
                    "martingale_values"
                ]
                pred_cumul = result["forecast_metrics"][2]["cumulative"][feature][
                    "martingale_values"
                ]

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
            actual_cps = result["ground_truth"]["change_points"]
            all_change_points["actual"].append(actual_cps)

            detected_cps = self._get_detected_change_points(
                result["actual_metrics"][1]["reset"]
            )
            all_change_points["detected"].append(detected_cps)

            predicted_cps = self._get_detected_change_points(
                result["forecast_metrics"][2]["reset"]
            )
            all_change_points["predicted"].append(predicted_cps)

            # Collect delays
            for delay_type in ["detection", "prediction"]:
                for cp in result["delays"][delay_type].keys():
                    delay = result["delays"][delay_type][cp]["mean"]

                    # Update earliest delay
                    if (
                        cp not in earliest_delays[delay_type]
                        or delay < earliest_delays[delay_type][cp]
                    ):
                        earliest_delays[delay_type][cp] = delay

                    # Collect all delays for statistics
                    if cp not in delays_per_cp[delay_type]:
                        delays_per_cp[delay_type][cp] = []
                    delays_per_cp[delay_type][cp].append(delay)

        # Step 3: Create final aggregated results
        return self._create_aggregated_results(
            all_results=all_results,
            aggregated_features=aggregated_features,
            aggregated_martingales=aggregated_martingales,
            all_change_points=all_change_points,
            earliest_delays=earliest_delays,
            delays_per_cp=delays_per_cp,
        )

    def _get_detected_change_points(self, martingales: Dict[str, Dict]) -> List[int]:
        """Extract unique change points detected across features."""
        all_cps = []
        for feature_data in martingales.values():
            if "change_points" in feature_data:
                all_cps.extend(feature_data["change_points"])
        return sorted(list(set(all_cps)))

    def _create_aggregated_results(
        self,
        all_results: List[Dict],
        aggregated_features: Dict,
        aggregated_martingales: Dict,
        all_change_points: Dict,
        earliest_delays: Dict[str, Dict[int, float]],
        delays_per_cp: Dict[str, Dict[int, List[float]]],
    ) -> Dict[str, Any]:
        """Create final aggregated results with statistics."""
        # Step 1: Average features
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

        # Step 2: Average martingales
        avg_martingales = {
            "actual": {
                reset_type: {
                    feature: {
                        "martingale_values": self._safe_aggregate_values(
                            values, "mean"
                        ),
                        "std": self._safe_aggregate_values(values, "std"),
                    }
                    for feature, values in feature_data.items()
                    if values
                }
                for reset_type, feature_data in aggregated_martingales["actual"].items()
            },
            "predicted": {
                reset_type: {
                    feature: {
                        "martingale_values": self._safe_aggregate_values(
                            values, "mean"
                        ),
                        "std": self._safe_aggregate_values(values, "std"),
                    }
                    for feature, values in feature_data.items()
                    if values
                }
                for reset_type, feature_data in aggregated_martingales[
                    "predicted"
                ].items()
            },
        }

        # Step 3: Change point statistics
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

        # Step 4: Delay statistics
        delay_stats = {
            "earliest": earliest_delays,
            "mean": {
                delay_type: {
                    str(cp): float(np.mean(delays)) for cp, delays in cp_delays.items()
                }
                for delay_type, cp_delays in delays_per_cp.items()
            },
            "std": {
                delay_type: {
                    str(cp): float(np.std(delays)) if len(delays) > 1 else 0.0
                    for cp, delays in cp_delays.items()
                }
                for delay_type, cp_delays in delays_per_cp.items()
            },
        }

        # Step 5: Aggregate SHAP values
        shap_values = {"actual": {}, "predicted": {}}
        features = ["degree", "clustering", "betweenness", "closeness"]

        for feature in features:
            # Collect actual SHAP values
            actual_shaps = []
            actual_means = []
            actual_stds = []
            for result in all_results:
                if (
                    result["actual_metrics"][2]
                    and feature in result["actual_metrics"][2]
                ):
                    actual_shaps.extend(result["actual_metrics"][2][feature]["values"])
                    actual_means.append(result["actual_metrics"][2][feature]["mean"])
                    actual_stds.append(result["actual_metrics"][2][feature]["std"])

            if actual_shaps:
                shap_values["actual"][feature] = {
                    "values": actual_shaps,
                    "mean": float(np.mean(actual_means)),
                    "std": float(np.mean(actual_stds)),  # Average of per-run stds
                }

            # Collect predicted SHAP values
            pred_shaps = []
            pred_means = []
            pred_stds = []
            for result in all_results:
                if (
                    result["forecast_metrics"][3]
                    and feature in result["forecast_metrics"][3]
                ):
                    pred_shaps.extend(result["forecast_metrics"][3][feature]["values"])
                    pred_means.append(result["forecast_metrics"][3][feature]["mean"])
                    pred_stds.append(result["forecast_metrics"][3][feature]["std"])

            if pred_shaps:
                shap_values["predicted"][feature] = {
                    "values": pred_shaps,
                    "mean": float(np.mean(pred_means)),
                    "std": float(np.mean(pred_stds)),  # Average of per-run stds
                }

        return {
            "n_runs": self.n_runs,
            "features": avg_features,
            "martingale_values": avg_martingales,
            "change_points": cp_stats,
            "delays": delay_stats,
            "shap_values": shap_values,
        }

    def _safe_aggregate_values(
        self, values: List[Any], operation: str
    ) -> Union[float, List[float]]:
        """Safely aggregate values handling both single values and arrays."""
        if not values:
            return None

        try:
            # Convert to numpy array
            if isinstance(values[0], (list, np.ndarray)):
                first_len = len(values[0])
                if not all(len(x) == first_len for x in values):
                    logger.error("Arrays have different lengths")
                    return None
                arr = np.array(values, dtype=float)
            else:
                arr = np.array(values, dtype=float)

            # Handle single values vs arrays
            if arr.ndim == 1:
                if operation == "mean":
                    return float(np.mean(arr))
                else:  # std
                    return float(np.std(arr))
            else:
                if operation == "mean":
                    result = np.mean(arr, axis=0)
                else:  # std
                    result = np.std(arr, axis=0, ddof=1)
                return result.tolist()

        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return None

    def _aggregate_cp_positions(self, cp_lists: List[List[int]]) -> Dict[int, float]:
        """Aggregate change point positions into a frequency map."""
        all_positions = []
        for cps in cp_lists:
            all_positions.extend(cps)

        unique_positions = sorted(set(all_positions))
        return {
            pos: all_positions.count(pos) / len(cp_lists) for pos in unique_positions
        }

    def save_aggregated_results(self, aggregated: Dict[str, Any], output_dir: Path):
        """Save aggregated results to file."""
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save main results
            with open(output_dir / "aggregated_results.json", "w") as f:
                json.dump(self._convert_to_serializable(aggregated), f, indent=4)
        except Exception as e:
            logger.error(f"Error saving aggregated results: {e}")
            try:
                # Ensure output directory exists for error file
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save error information
                with open(output_dir / "aggregated_results_error.json", "w") as f:
                    json.dump(
                        {"error": str(e), "n_runs": aggregated.get("n_runs", None)},
                        f,
                        indent=4,
                    )
            except Exception as e2:
                logger.error(f"Failed to save error information: {e2}")

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert objects to JSON serializable format."""
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
        elif hasattr(obj, "__dict__"):
            return {
                key: ResultAggregator._convert_to_serializable(value)
                for key, value in obj.__dict__.items()
                if not key.startswith("_")
            }
        elif isinstance(obj, dict):
            return {
                str(key): ResultAggregator._convert_to_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple, set)):
            return [ResultAggregator._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "__str__"):
            return str(obj)
        return obj
