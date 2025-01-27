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
) -> tuple[np.ndarray, List[str]]:
    """Compute SHAP values using CustomThresholdModel on martingale values.

    Args:
        martingales: Dictionary containing martingale values for each feature
        change_points: True change point indices
        sequence_length: Length of the sequence
        threshold: Threshold value for change detection
        window_size: Window size for SHAP computation (default: 5)

    Returns:
        tuple: (shap_values, feature_names)
            - shap_values: numpy.ndarray of shape [n_timesteps x n_features]
            - feature_names: list of feature names in order matching shap_values columns
    """
    # Define fixed feature order to match visualization
    feature_order = [
        "degree",
        "density",
        "clustering",
        "betweenness",
        "eigenvector",
        "closeness",
        "singular_value",
        "laplacian",
    ]

    # Convert martingale values to feature matrix with consistent ordering
    feature_matrix = []
    feature_names = []  # Keep track of which features were actually found

    for feature in feature_order:
        if feature in martingales and feature != "combined":
            # Convert array of arrays to flat array
            martingales_array = np.array(
                [
                    x.item() if isinstance(x, np.ndarray) else x
                    for x in martingales[feature]["martingales"]
                ]
            )
            feature_matrix.append(martingales_array)
            feature_names.append(feature)

    if not feature_matrix:
        raise ValueError("No valid features found in martingales dictionary")

    X = np.vstack(feature_matrix).T  # [n_timesteps x n_features]

    model = CustomThresholdModel(threshold=threshold)
    shap_values = model.compute_shap_values(
        X=X,
        change_points=change_points,
        sequence_length=sequence_length,
        window_size=window_size,
    )

    return shap_values, feature_names
