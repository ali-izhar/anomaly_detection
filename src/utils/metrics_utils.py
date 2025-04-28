# src/utils/metrics_utils.py

"""Utilities for calculating change point detection performance metrics."""

import numpy as np
from typing import List, Dict


def calculate_metrics(
    detected_change_points: List[int],
    true_change_points: List[int],
    total_steps: int,
    max_delay: int = 15,
) -> Dict[str, float]:
    """Calculate performance metrics for change point detection.

    Args:
        detected_change_points: List of indices where changes were detected
        true_change_points: List of indices where true changes occurred
        total_steps: Total number of time steps in the sequence
        max_delay: Maximum allowable delay to consider a detection as true positive

    Returns:
        Dictionary containing metrics:
        - tpr: True positive rate (recall)
        - fpr: False positive rate
        - avg_delay: Average detection delay for true positives
        - auc: Area under ROC curve (if applicable)
    """
    if not true_change_points:
        return {"tpr": 0.0, "fpr": 0.0, "avg_delay": 0.0, "auc": 0.0}

    # Initialize counters
    true_positives = 0
    false_positives = 0
    delays = []

    # Mark detected change points
    detected = set(detected_change_points)

    # Check each true change point
    for true_cp in true_change_points:
        # Find detections within acceptable delay window
        valid_detections = [
            d for d in detected_change_points if true_cp <= d <= true_cp + max_delay
        ]

        if valid_detections:
            # Count as true positive
            true_positives += 1
            # Record the delay using earliest detection
            earliest_detection = min(valid_detections)
            delay = earliest_detection - true_cp
            delays.append(delay)

            # Remove these detections from consideration to avoid double counting
            for d in valid_detections:
                if d in detected:
                    detected.remove(d)

    # Remaining detections are false positives
    false_positives = len(detected)

    # Non-detection points (excluding true change points)
    non_change_points = total_steps - len(true_change_points)

    # Calculate metrics
    tpr = true_positives / len(true_change_points) if true_change_points else 0.0
    fpr = false_positives / non_change_points if non_change_points > 0 else 0.0
    avg_delay = np.mean(delays) if delays else 0.0

    # AUC calculation - basic version
    # For proper AUC, would need full ROC curve points based on different thresholds
    # This is a simplified approximation
    auc = 0.5 * (1 + tpr - fpr) if tpr > 0 or fpr > 0 else 0.5

    return {"tpr": tpr, "fpr": fpr, "avg_delay": avg_delay, "auc": auc}


def calculate_metrics_for_horizon(
    traditional_change_points: List[int],
    horizon_change_points: List[int],
    true_change_points: List[int],
    total_steps: int,
    max_delay: int = 15,
) -> Dict[str, float]:
    """Calculate comparative metrics for traditional vs horizon detection.

    Args:
        traditional_change_points: Change points detected by traditional method
        horizon_change_points: Change points detected by horizon method
        true_change_points: True change points
        total_steps: Total time steps in sequence
        max_delay: Maximum delay to consider a detection valid

    Returns:
        Dictionary with metrics for both methods and improvement metrics
    """
    # Calculate metrics for traditional method
    trad_metrics = calculate_metrics(
        traditional_change_points, true_change_points, total_steps, max_delay
    )

    # Calculate metrics for horizon method
    horizon_metrics = calculate_metrics(
        horizon_change_points, true_change_points, total_steps, max_delay
    )

    # Calculate improvement
    tpr_improvement = horizon_metrics["tpr"] - trad_metrics["tpr"]
    delay_reduction = (
        (trad_metrics["avg_delay"] - horizon_metrics["avg_delay"])
        / trad_metrics["avg_delay"]
        if trad_metrics["avg_delay"] > 0
        else 0.0
    )

    # Combine results
    combined_metrics = {
        "traditional_tpr": trad_metrics["tpr"],
        "traditional_fpr": trad_metrics["fpr"],
        "traditional_avg_delay": trad_metrics["avg_delay"],
        "traditional_auc": trad_metrics["auc"],
        "horizon_tpr": horizon_metrics["tpr"],
        "horizon_fpr": horizon_metrics["fpr"],
        "horizon_avg_delay": horizon_metrics["avg_delay"],
        "horizon_auc": horizon_metrics["auc"],
        "tpr_improvement": tpr_improvement,
        "delay_reduction": delay_reduction,
    }

    return combined_metrics
