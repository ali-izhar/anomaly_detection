# src/utils/data_utils.py

"""Utilities for data processing, transformation, and visualization preparation."""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

# Common keys used in the detection results
# These keys are used by multiple functions and extracted here for easier maintenance
RESULT_KEYS = {
    # Keys for traditional martingales
    "TRADITIONAL_MARTINGALES": "traditional_martingales",
    "TRADITIONAL_SUM_MARTINGALES": "traditional_sum_martingales",
    "TRADITIONAL_AVG_MARTINGALES": "traditional_avg_martingales",
    "TRADITIONAL_CHANGE_POINTS": "traditional_change_points",
    "INDIVIDUAL_TRADITIONAL_MARTINGALES": "individual_traditional_martingales",
    # Keys for horizon martingales
    "HORIZON_MARTINGALES": "horizon_martingales",
    "HORIZON_SUM_MARTINGALES": "horizon_sum_martingales",
    "HORIZON_AVG_MARTINGALES": "horizon_avg_martingales",
    "HORIZON_CHANGE_POINTS": "horizon_change_points",
    "INDIVIDUAL_HORIZON_MARTINGALES": "individual_horizon_martingales",
    "EARLY_WARNINGS": "early_warnings",
    # Keys for sequence results
    "CHANGE_POINTS": "change_points",
    "MODEL_NAME": "model_name",
    # Keys for output configuration
    "OUTPUT": "output",
    "SAVE_FEATURES": "save_features",
    "SAVE_PREDICTIONS": "save_predictions",
    "SAVE_MARTINGALES": "save_martingales",
}


# Utility functions for data normalization
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
        "true_change_points": sequence_result.get(RESULT_KEYS["CHANGE_POINTS"], []),
        "model_name": sequence_result.get(RESULT_KEYS["MODEL_NAME"], ""),
        "params": config,
    }

    # Add data if configured to save
    output_config = config.get(RESULT_KEYS["OUTPUT"], {})
    if output_config.get(RESULT_KEYS["SAVE_FEATURES"], False):
        results.update(
            {
                "features_raw": features_raw,
                "features_numeric": features_numeric,
            }
        )

    if predicted_graphs is not None and output_config.get(
        RESULT_KEYS["SAVE_PREDICTIONS"], False
    ):
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
    if trial_results and output_config.get(RESULT_KEYS["SAVE_MARTINGALES"], False):
        if "aggregated" in trial_results:
            results.update(trial_results["aggregated"])

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
    trad_mart = RESULT_KEYS["TRADITIONAL_MARTINGALES"]
    trad_sum_mart = RESULT_KEYS["TRADITIONAL_SUM_MARTINGALES"]

    if trad_mart not in complete_result and trad_sum_mart in complete_result:
        complete_result[trad_mart] = complete_result[trad_sum_mart].copy()

    # Add missing horizon martingale fields if needed
    if trad_sum_mart in complete_result:
        # Initialize standard horizon fields if missing
        horizon_pairs = [
            (RESULT_KEYS["HORIZON_MARTINGALES"], trad_sum_mart),
            (RESULT_KEYS["HORIZON_SUM_MARTINGALES"], trad_sum_mart),
            (
                RESULT_KEYS["HORIZON_AVG_MARTINGALES"],
                RESULT_KEYS["TRADITIONAL_AVG_MARTINGALES"],
            ),
        ]

        for key, base_key in horizon_pairs:
            if key not in complete_result and base_key in complete_result:
                complete_result[key] = np.zeros_like(complete_result[base_key])

        # Add empty change points if missing
        if RESULT_KEYS["HORIZON_CHANGE_POINTS"] not in complete_result:
            complete_result[RESULT_KEYS["HORIZON_CHANGE_POINTS"]] = []

        if RESULT_KEYS["EARLY_WARNINGS"] not in complete_result:
            complete_result[RESULT_KEYS["EARLY_WARNINGS"]] = []

        # Initialize individual horizon martingales if needed
        indiv_horizon = RESULT_KEYS["INDIVIDUAL_HORIZON_MARTINGALES"]
        indiv_trad = RESULT_KEYS["INDIVIDUAL_TRADITIONAL_MARTINGALES"]

        if indiv_horizon not in complete_result and indiv_trad in complete_result:
            n_features = len(complete_result[indiv_trad])
            complete_result[indiv_horizon] = [
                np.zeros_like(feat_martingale)
                for feat_martingale in complete_result[indiv_trad]
            ]

    # Ensure all change points are lists (not NumPy arrays)
    change_point_keys = [
        RESULT_KEYS["TRADITIONAL_CHANGE_POINTS"],
        RESULT_KEYS["HORIZON_CHANGE_POINTS"],
        RESULT_KEYS["EARLY_WARNINGS"],
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
