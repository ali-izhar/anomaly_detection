"""Data transformation utilities."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to zero mean and unit variance."""
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds == 0] = 1.0
    normalized = (features - means) / stds
    return normalized, means, stds


def normalize_predictions(predictions: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Normalize predictions using existing statistics."""
    normalized = []
    for t in range(len(predictions)):
        horizons = [(predictions[t][h] - means) / stds for h in range(len(predictions[t]))]
        normalized.append(horizons)
    return np.array(normalized) if isinstance(predictions, np.ndarray) else normalized


def prepare_result_data(
    sequence_result: Dict[str, Any],
    features_numeric: np.ndarray,
    features_raw: Optional[List] = None,
    predicted_graphs: Optional[List] = None,
    trial_results: Optional[Dict[str, Any]] = None,
    predictor=None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile detection results into a single dictionary."""
    config = config or {}
    output = config.get("output", {})

    results = {
        "true_change_points": sequence_result.get("change_points", []),
        "model_name": sequence_result.get("model_name", ""),
        "params": config,
    }

    if output.get("save_features", False):
        results["features_raw"] = features_raw
        results["features_numeric"] = features_numeric

    if predicted_graphs is not None and output.get("save_predictions", False):
        results["predicted_graphs"] = predicted_graphs
        if predictor is not None and hasattr(predictor, "get_state"):
            results["predictor_states"] = predictor.get_state()

    if trial_results and output.get("save_martingales", False):
        if "aggregated" in trial_results:
            results.update(trial_results["aggregated"])
        if "individual_trials" in trial_results:
            results["individual_trials"] = trial_results["individual_trials"]

    return results
