"""Utilities for preparing and handling visualization data."""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


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
        for key, base_key in [
            ("horizon_martingales", "traditional_sum_martingales"),
            ("horizon_sum_martingales", "traditional_sum_martingales"),
            ("horizon_avg_martingales", "traditional_avg_martingales"),
        ]:
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
    for key in ["traditional_change_points", "horizon_change_points", "early_warnings"]:
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
