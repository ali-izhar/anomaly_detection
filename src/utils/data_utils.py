"""Utilities for data processing and transformation."""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union


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
    if config.get("output", {}).get("save_features", False):
        results.update(
            {
                "features_raw": features_raw,
                "features_numeric": features_numeric,
            }
        )

    if predicted_graphs is not None and config.get("output", {}).get(
        "save_predictions", False
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
    if trial_results and config.get("output", {}).get("save_martingales", False):
        if "aggregated" in trial_results:
            results.update(trial_results["aggregated"])

    return results
