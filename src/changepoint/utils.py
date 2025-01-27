"""Utility functions for changepoint analysis."""

from typing import Dict, List, Any
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

    # Process all features except 'combined'
    for name, results in martingales.items():
        if name != "combined":  # Skip the combined feature
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
