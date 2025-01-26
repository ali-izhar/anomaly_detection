"""Utility functions for changepoint analysis."""

from typing import Dict, List, Any, Tuple
import numpy as np
from .threshold import CustomThresholdModel


def compute_shap_values(
    martingales: Dict[str, Dict[str, Any]],
    change_points: List[int],
    sequence_length: int,
    threshold: float,
    window_size: int = 5,
) -> np.ndarray:
    """Compute SHAP values using CustomThresholdModel on martingale values.

    Args:
        martingales: Dictionary containing martingale values for each feature
        change_points: True change point indices
        sequence_length: Length of the sequence
        threshold: Threshold value for change detection
        window_size: Window size for SHAP computation (default: 5)

    Returns:
        numpy.ndarray: SHAP values matrix of shape [n_timesteps x n_features]
    """
    # Convert martingale values to feature matrix
    feature_matrix = []

    for name, results in martingales.items():
        # Convert array of arrays to flat array
        martingales_array = np.array(
            [
                x.item() if isinstance(x, np.ndarray) else x
                for x in results["martingales"]
            ]
        )
        feature_matrix.append(martingales_array)

    X = np.vstack(feature_matrix).T  # [n_timesteps x n_features]

    model = CustomThresholdModel(threshold=threshold)
    return model.compute_shap_values(
        X=X,
        change_points=change_points,
        sequence_length=sequence_length,
        window_size=window_size,
    )


def compute_combined_martingales(
    martingales: Dict[str, Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sum and average of martingale values across features.

    Args:
        martingales: Dictionary containing martingale values for each feature

    Returns:
        Tuple[np.ndarray, np.ndarray]: (sum_martingale, avg_martingale)
            - sum_martingale: Sum of martingales across features
            - avg_martingale: Average of martingales across features
    """
    # Convert martingale sequences to arrays
    martingale_arrays = []
    for m in martingales.values():
        values = np.array(
            [x.item() if isinstance(x, np.ndarray) else x for x in m["martingales"]]
        )
        martingale_arrays.append(values)

    # Compute sum and average
    M_sum = np.sum(martingale_arrays, axis=0)
    M_avg = M_sum / len(martingales)

    return M_sum, M_avg
