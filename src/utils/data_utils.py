# src/utils/data_utils.py

"""Utilities for data processing, transformation, and visualization preparation."""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)


def normalize_features(
    features_numeric: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features by subtracting mean and dividing by standard deviation.

    Args:
        features_numeric: Array of numeric features with shape (n_samples, n_features)

    Returns:
        Tuple containing:
        - normalized_features: Normalized feature array
        - feature_means: Mean values used for normalization
        - feature_stds: Standard deviation values used for normalization
    """
    feature_means = np.mean(features_numeric, axis=0)
    feature_stds = np.std(features_numeric, axis=0)

    # Replace zero standard deviations with 1 to avoid division by zero
    feature_stds[feature_stds == 0] = 1.0

    normalized_features = (features_numeric - feature_means) / feature_stds

    return normalized_features, feature_means, feature_stds


def normalize_predictions(
    predicted_features: np.ndarray, feature_means: np.ndarray, feature_stds: np.ndarray
) -> Union[List, np.ndarray]:
    """Normalize predicted features using the same statistics as the main features.

    Args:
        predicted_features: Array of predicted features
        feature_means: Mean values from the main features
        feature_stds: Standard deviation values from the main features

    Returns:
        Normalized predictions in the same format as the input (list or np.ndarray)
    """
    predicted_normalized = []

    # For each timestep's predictions
    for t in range(len(predicted_features)):
        normalized_horizons = []
        # For each horizon
        for h in range(len(predicted_features[t])):
            # Normalize using the same stats as the main features
            normalized_horizon = (
                predicted_features[t][h] - feature_means
            ) / feature_stds
            normalized_horizons.append(normalized_horizon)
        predicted_normalized.append(normalized_horizons)

    # Convert back to numpy array if the input was a numpy array
    if isinstance(predicted_features, np.ndarray):
        predicted_normalized = np.array(predicted_normalized)

    return predicted_normalized


def prepare_result_data(
    sequence_result: Dict[str, Any],
    features_numeric: np.ndarray,
    features_raw: List,
    predicted_graphs: Optional[List] = None,
    trial_results: Optional[Dict[str, Any]] = None,
    predictor=None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Prepare and compile results data.

    Args:
        sequence_result: Dictionary containing graph sequence generation results
        features_numeric: Numeric feature values
        features_raw: Raw feature values
        predicted_graphs: Predicted graphs (optional)
        trial_results: Detection trial results (optional)
        predictor: Predictor instance (optional)
        config: Configuration dictionary (optional)

    Returns:
        Dictionary containing compiled results
    """
    if config is None:
        config = {}

    # Start with basic information
    results = {
        "true_change_points": sequence_result.get("change_points", []),
        "model_name": sequence_result.get("model_name", ""),
        "params": config,
    }

    # Add data if configured to save
    output_config = config.get("output", {})
    if output_config.get("save_features", False):
        results.update(
            {
                "features_raw": features_raw,
                "features_numeric": features_numeric,
            }
        )

    if predicted_graphs is not None and output_config.get("save_predictions", False):
        results.update(
            {
                "predicted_graphs": predicted_graphs,
            }
        )
        if predictor is not None and hasattr(predictor, "get_state"):
            results.update(
                {
                    "predictor_states": predictor.get_state(),
                }
            )

    # Add detection results if available
    if trial_results and output_config.get("save_martingales", False):
        if "aggregated" in trial_results:
            results.update(trial_results["aggregated"])
        # Add the individual trials data
        if "individual_trials" in trial_results:
            results["individual_trials"] = trial_results["individual_trials"]

    return results


def prepare_martingale_visualization_data(
    detection_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare detection results for visualization by ensuring all required fields exist.

    Args:
        detection_result: Dictionary containing detection results

    Returns:
        Dictionary with all necessary fields for visualization
    """
    # Initialize a copy of the detection result that we can modify
    complete_result = detection_result.copy()

    # Ensure basic martingale data exists
    if (
        "traditional_martingales" not in complete_result
        and "traditional_sum_martingales" in complete_result
    ):
        complete_result["traditional_martingales"] = complete_result[
            "traditional_sum_martingales"
        ].copy()

    # Add missing horizon martingale fields if needed
    if "traditional_sum_martingales" in complete_result:
        # Initialize standard horizon fields if missing
        horizon_pairs = [
            ("horizon_martingales", "traditional_sum_martingales"),
            ("horizon_sum_martingales", "traditional_sum_martingales"),
            ("horizon_avg_martingales", "traditional_avg_martingales"),
        ]

        for key, base_key in horizon_pairs:
            if key not in complete_result and base_key in complete_result:
                complete_result[key] = np.zeros_like(complete_result[base_key])

        # Add empty change points if missing
        if "horizon_change_points" not in complete_result:
            complete_result["horizon_change_points"] = []

        if "early_warnings" not in complete_result:
            complete_result["early_warnings"] = []

        # Initialize individual horizon martingales if needed
        if (
            "individual_horizon_martingales" not in complete_result
            and "individual_traditional_martingales" in complete_result
        ):
            n_features = len(complete_result["individual_traditional_martingales"])
            complete_result["individual_horizon_martingales"] = [
                np.zeros_like(feat_martingale)
                for feat_martingale in complete_result[
                    "individual_traditional_martingales"
                ]
            ]

    # Ensure all change points are lists (not NumPy arrays)
    change_point_keys = [
        "traditional_change_points",
        "horizon_change_points",
        "early_warnings",
    ]

    for key in change_point_keys:
        if key in complete_result:
            if isinstance(complete_result[key], np.ndarray):
                complete_result[key] = complete_result[key].tolist()
            if not isinstance(complete_result[key], list):
                complete_result[key] = []

    # Replace any scalar martingale values with arrays
    for key in complete_result:
        if key.endswith("_martingales") and np.isscalar(complete_result[key]):
            complete_result[key] = np.array([complete_result[key]])

    return complete_result
