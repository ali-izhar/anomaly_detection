# src/utils/analysis_utils.py

"""Analyze change point detection results and generate tabular reports."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from tabulate import tabulate

logger = logging.getLogger(__name__)


def analyze_detection_results(
    results: Dict[str, Any], report_format: str = "rounded_grid"
) -> str:
    """Analyze change point detection results and generate a tabular report.

    Args:
        results: Dictionary containing detection results
        report_format: Tabulate table format (default: 'rounded_grid')

    Returns:
        Formatted string with tabular analysis
    """
    # Extract change points
    true_change_points = results.get("true_change_points", [])

    # Get detection points - removed the 1-index offset adjustment
    traditional_detected = results.get("traditional_change_points", [])
    horizon_detected = results.get("horizon_change_points", [])

    # If any of the arrays are numpy arrays, convert to list
    if isinstance(true_change_points, np.ndarray):
        true_change_points = true_change_points.tolist()
    if isinstance(traditional_detected, np.ndarray):
        traditional_detected = traditional_detected.tolist()
    if isinstance(horizon_detected, np.ndarray):
        horizon_detected = horizon_detected.tolist()

    # Calculate detection metrics for each true change point
    analysis_data = []
    for idx, cp in enumerate(true_change_points):
        # Find the closest traditional detection after the change point
        trad_delay, trad_detection = find_detection_delay(cp, traditional_detected)

        # Find the closest horizon detection after the change point
        horizon_delay, horizon_detection = find_detection_delay(cp, horizon_detected)

        # Calculate the delay reduction from using horizon detection
        delay_reduction = compute_delay_reduction(trad_delay, horizon_delay)

        analysis_data.append(
            [
                cp,
                trad_detection if trad_detection is not None else "Not detected",
                trad_delay if trad_delay is not None else "-",
                horizon_detection if horizon_detection is not None else "Not detected",
                horizon_delay if horizon_delay is not None else "-",
                delay_reduction,
            ]
        )

    # Generate summary statistics
    avg_trad_delay = compute_average_delay(true_change_points, traditional_detected)
    avg_horizon_delay = compute_average_delay(true_change_points, horizon_detected)
    avg_reduction = compute_average_reduction(avg_trad_delay, avg_horizon_delay)
    detection_rate_trad = compute_detection_rate(
        true_change_points, traditional_detected
    )
    detection_rate_horizon = compute_detection_rate(
        true_change_points, horizon_detected
    )

    # Create the table
    headers = [
        "True CP",
        "Traditional Detection",
        "Delay (steps)",
        "Horizon Detection",
        "Delay (steps)",
        "Delay Reduction",
    ]

    table = tabulate(analysis_data, headers=headers, tablefmt=report_format)

    # Create summary table
    summary_data = [
        [
            "Detection Rate",
            f"{detection_rate_trad:.1%}",
            f"{detection_rate_horizon:.1%}",
        ],
        [
            "Average Delay",
            f"{avg_trad_delay:.1f}" if avg_trad_delay is not None else "N/A",
            f"{avg_horizon_delay:.1f}" if avg_horizon_delay is not None else "N/A",
        ],
        [
            "Avg Delay Reduction",
            "",
            f"{avg_reduction:.1%}" if avg_reduction is not None else "N/A",
        ],
    ]

    summary_table = tabulate(
        summary_data,
        headers=["Metric", "Traditional", "Horizon"],
        tablefmt=report_format,
    )

    # Combine tables with headers
    report = (
        "Change Point Detection Analysis\n"
        "==============================\n\n"
        "Detection Details:\n"
        f"{table}\n\n"
        "Summary Statistics:\n"
        f"{summary_table}\n"
    )

    return report


def find_detection_delay(
    change_point: int, detections: List[int], max_delay: int = 50
) -> Tuple[Optional[int], Optional[int]]:
    """Find the delay between a change point and its detection.

    Args:
        change_point: The true change point index
        detections: List of detection indices
        max_delay: Maximum allowable delay to consider a detection valid

    Returns:
        Tuple of (delay, detection_point) or (None, None) if not detected
    """
    # Find detections that occur after the change point and within max_delay
    valid_detections = [
        d for d in detections if d >= change_point and d - change_point <= max_delay
    ]

    if valid_detections:
        # Find the earliest detection
        earliest = min(valid_detections)
        delay = earliest - change_point
        return delay, earliest

    return None, None


def compute_delay_reduction(
    traditional_delay: Optional[int], horizon_delay: Optional[int]
) -> str:
    """Compute the percentage reduction in delay from traditional to horizon detection.

    Args:
        traditional_delay: Delay in traditional detection (steps)
        horizon_delay: Delay in horizon detection (steps)

    Returns:
        String representing the delay reduction or appropriate message
    """
    if traditional_delay is None:
        return "No traditional detection"

    if horizon_delay is None:
        return "No horizon detection"

    if traditional_delay == 0:
        return "0% (no delay in traditional)"

    reduction = (traditional_delay - horizon_delay) / traditional_delay

    if reduction > 0:
        return f"{reduction:.1%}"
    elif reduction == 0:
        return "0%"
    else:
        return f"{reduction:.1%} (increased)"


def compute_average_delay(
    change_points: List[int], detections: List[int], max_delay: int = 50
) -> Optional[float]:
    """Compute the average delay across all detected change points.

    Args:
        change_points: List of true change points
        detections: List of detection points
        max_delay: Maximum delay to consider a detection valid

    Returns:
        Average delay or None if no valid detections
    """
    delays = []
    for cp in change_points:
        delay, _ = find_detection_delay(cp, detections, max_delay)
        if delay is not None:
            delays.append(delay)

    if delays:
        return sum(delays) / len(delays)

    return None


def compute_detection_rate(
    change_points: List[int], detections: List[int], max_delay: int = 50
) -> float:
    """Compute the percentage of change points that were detected.

    Args:
        change_points: List of true change points
        detections: List of detection points
        max_delay: Maximum delay to consider a detection valid

    Returns:
        Detection rate as a fraction (0.0 to 1.0)
    """
    if not change_points:
        return 0.0

    detected_count = 0
    for cp in change_points:
        delay, _ = find_detection_delay(cp, detections, max_delay)
        if delay is not None:
            detected_count += 1

    return detected_count / len(change_points)


def compute_average_reduction(
    avg_traditional_delay: Optional[float], avg_horizon_delay: Optional[float]
) -> Optional[float]:
    """Compute the average reduction in delay from traditional to horizon detection.

    Args:
        avg_traditional_delay: Average delay in traditional detection
        avg_horizon_delay: Average delay in horizon detection

    Returns:
        Average reduction as a fraction (0.0 to 1.0) or None if not applicable
    """
    if avg_traditional_delay is None or avg_horizon_delay is None:
        return None

    if avg_traditional_delay == 0:
        return 0.0

    return (avg_traditional_delay - avg_horizon_delay) / avg_traditional_delay


def print_analysis_report(
    results: Dict[str, Any], report_format: str = "rounded_grid"
) -> None:
    """Generate and print a tabular analysis report of detection results.

    Args:
        results: Dictionary containing detection results
        report_format: Tabulate table format (default: 'rounded_grid')
    """
    report = analyze_detection_results(results, report_format)
    print(report)
