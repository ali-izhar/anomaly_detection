# src/utils/analysis_utils.py

"""Analyze change point detection results and generate tabular reports."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from tabulate import tabulate
from collections import Counter

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

    # Check if we have multiple trials
    individual_trials = results.get("individual_trials", [])
    if individual_trials and len(individual_trials) > 1:
        return analyze_multiple_trials(results, report_format)

    # Single trial analysis
    # Get detection points
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
            f"{detection_rate_trad:.2%}",
            f"{detection_rate_horizon:.2%}",
        ],
        [
            "Average Delay",
            f"{avg_trad_delay:.2f}" if avg_trad_delay is not None else "N/A",
            f"{avg_horizon_delay:.2f}" if avg_horizon_delay is not None else "N/A",
        ],
        [
            "Avg Delay Reduction",
            "",
            f"{avg_reduction:.2%}" if avg_reduction is not None else "N/A",
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


def analyze_multiple_trials(
    results: Dict[str, Any], report_format: str = "rounded_grid"
) -> str:
    """Analyze results from multiple detection trials and generate a consolidated report.

    Args:
        results: Dictionary containing detection results with individual_trials
        report_format: Tabulate table format (default: 'rounded_grid')

    Returns:
        Formatted string with tabular analysis of multiple trials
    """
    # Extract change points and trials
    true_change_points = results.get("true_change_points", [])
    individual_trials = results.get("individual_trials", [])

    # Convert to list if numpy array
    if isinstance(true_change_points, np.ndarray):
        true_change_points = true_change_points.tolist()

    num_trials = len(individual_trials)

    # Log the received data for debugging
    logger.debug(f"Analyzing multiple trials data with {num_trials} trials")
    logger.debug(f"True change points: {true_change_points}")

    # Track per-trial detection statistics
    trial_statistics = []
    all_trad_detections = []
    all_horizon_detections = []
    all_trad_delays = []
    all_horizon_delays = []

    # For collecting data per trial and change point
    trial_data = []

    for trial_idx, trial in enumerate(individual_trials):
        trad_points = trial.get("traditional_change_points", [])
        horizon_points = trial.get("horizon_change_points", [])

        logger.debug(
            f"Trial {trial_idx+1}: Traditional points: {trad_points}, Horizon points: {horizon_points}"
        )

        # For each change point, find the corresponding detection
        cp_trad_delays = []
        cp_horizon_delays = []
        cp_trad_detections = []
        cp_horizon_detections = []

        for cp in true_change_points:
            trad_delay, trad_detection = find_detection_delay(cp, trad_points)
            horizon_delay, horizon_detection = find_detection_delay(cp, horizon_points)

            # Store delays and detections for this trial
            if trad_detection is not None:
                cp_trad_delays.append(trad_delay)
                cp_trad_detections.append(trad_detection)
                all_trad_detections.append(trad_detection)
                all_trad_delays.append(trad_delay)

            if horizon_detection is not None:
                cp_horizon_delays.append(horizon_delay)
                cp_horizon_detections.append(horizon_detection)
                all_horizon_detections.append(horizon_detection)
                all_horizon_delays.append(horizon_delay)

            # Add row for this CP and trial
            trial_data.append(
                [
                    f"Trial {trial_idx+1}",
                    cp,
                    trad_detection if trad_detection is not None else "Not detected",
                    trad_delay if trad_delay is not None else "-",
                    (
                        horizon_detection
                        if horizon_detection is not None
                        else "Not detected"
                    ),
                    horizon_delay if horizon_delay is not None else "-",
                    compute_delay_reduction(trad_delay, horizon_delay),
                ]
            )

        # Calculate trial-level statistics
        avg_trad_delay_trial = np.mean(cp_trad_delays) if cp_trad_delays else None
        avg_horizon_delay_trial = (
            np.mean(cp_horizon_delays) if cp_horizon_delays else None
        )
        trad_rate = (
            len(cp_trad_detections) / len(true_change_points)
            if true_change_points
            else 0.0
        )
        horizon_rate = (
            len(cp_horizon_detections) / len(true_change_points)
            if true_change_points
            else 0.0
        )

        trial_statistics.append(
            {
                "trial": trial_idx + 1,
                "trad_rate": trad_rate,
                "horizon_rate": horizon_rate,
                "trad_delay": avg_trad_delay_trial,
                "horizon_delay": avg_horizon_delay_trial,
            }
        )

    # Calculate global average detection times and delays
    avg_trad_detection = np.mean(all_trad_detections) if all_trad_detections else None
    avg_horizon_detection = (
        np.mean(all_horizon_detections) if all_horizon_detections else None
    )
    avg_trad_delay = np.mean(all_trad_delays) if all_trad_delays else None
    avg_horizon_delay = np.mean(all_horizon_delays) if all_horizon_delays else None

    # Log calculated values for debugging
    logger.debug(f"All traditional detections: {all_trad_detections}")
    logger.debug(f"All horizon detections: {all_horizon_detections}")
    logger.debug(f"Average traditional detection: {avg_trad_detection}")
    logger.debug(f"Average horizon detection: {avg_horizon_detection}")
    logger.debug(f"Average traditional delay: {avg_trad_delay}")
    logger.debug(f"Average horizon delay: {avg_horizon_delay}")

    # Calculate overall detection rates
    avg_trad_detection_rate = np.mean([stat["trad_rate"] for stat in trial_statistics])
    avg_horizon_detection_rate = np.mean(
        [stat["horizon_rate"] for stat in trial_statistics]
    )

    # Calculate delay reduction
    avg_reduction = compute_average_reduction(avg_trad_delay, avg_horizon_delay)

    # Create comprehensive summary table with all trials and aggregate data
    headers = [
        "Trial",
        "True CP",
        "Traditional Detection",
        "Delay (steps)",
        "Horizon Detection",
        "Delay (steps)",
        "Delay Reduction",
    ]

    # Add aggregate row
    trial_data.append(
        [
            "Aggregate",
            "/".join(str(cp) for cp in true_change_points),
            f"{avg_trad_detection:.2f}" if avg_trad_detection is not None else "N/A",
            f"{avg_trad_delay:.2f}" if avg_trad_delay is not None else "N/A",
            (
                f"{avg_horizon_detection:.2f}"
                if avg_horizon_detection is not None
                else "N/A"
            ),
            f"{avg_horizon_delay:.2f}" if avg_horizon_delay is not None else "N/A",
            f"{avg_reduction:.2%}" if avg_reduction is not None else "N/A",
        ]
    )

    # Create the table
    table = tabulate(trial_data, headers=headers, tablefmt=report_format)

    # Generate summary statistics table
    summary_data = [
        [
            "Detection Rate",
            f"{avg_trad_detection_rate:.2%}",
            f"{avg_horizon_detection_rate:.2%}",
        ],
        [
            "Average Delay",
            f"{avg_trad_delay:.2f}" if avg_trad_delay is not None else "N/A",
            f"{avg_horizon_delay:.2f}" if avg_horizon_delay is not None else "N/A",
        ],
        [
            "Avg Delay Reduction",
            "",
            f"{avg_reduction:.2%}" if avg_reduction is not None else "N/A",
        ],
    ]

    summary_table = tabulate(
        summary_data,
        headers=["Metric", "Traditional", "Horizon"],
        tablefmt=report_format,
    )

    # Create the report with the new table layout
    report = (
        f"Change Point Detection Analysis ({num_trials} Trials)\n"
        "==============================\n\n"
        "Detection Details (All Trials + Aggregate):\n"
        f"{table}\n\n"
        "Summary Statistics:\n"
        f"{summary_table}\n"
    )

    return report


def compute_consensus_points(
    detection_points: List[int], threshold: float = 0.3, tolerance: int = 3
) -> List[int]:
    """Compute consensus detection points from multiple trials.

    Args:
        detection_points: All detection points across trials
        threshold: Minimum fraction of trials required for consensus (0.0-1.0)
        tolerance: Points within this distance are considered the same detection

    Returns:
        List of consensus detection points
    """
    if not detection_points:
        return []

    # Group nearby points
    sorted_points = sorted(detection_points)
    clusters = []
    current_cluster = [sorted_points[0]]

    for point in sorted_points[1:]:
        if point - current_cluster[-1] <= tolerance:
            current_cluster.append(point)
        else:
            clusters.append(current_cluster)
            current_cluster = [point]

    # Add the last cluster
    if current_cluster:
        clusters.append(current_cluster)

    # Count frequencies of clusters
    cluster_counts = [len(cluster) for cluster in clusters]

    # Calculate median point for each significant cluster
    consensus_points = []
    for i, cluster in enumerate(clusters):
        if cluster_counts[i] / len(detection_points) >= threshold:
            # Use median as the representative point
            consensus_points.append(int(np.median(cluster)))

    return sorted(consensus_points)


def find_detection_delay(
    change_point: int,
    detections: List[int],
    max_delay: int = 50,
    is_traditional: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """Find the delay between a change point and its detection.

    Args:
        change_point: The true change point index
        detections: List of detection indices
        max_delay: Maximum allowable delay to consider a detection valid
        is_traditional: Deprecated parameter, kept for backward compatibility

    Returns:
        Tuple of (delay, detection_point) or (None, None) if not detected
    """
    if not detections:
        return None, None

    # Find detections that occur at or after the change point and within max_delay
    # For detections at the change point itself or 1 step before,
    # still count them but with a delay of 0
    valid_detections = []
    for d in detections:
        # Detection happens before or at the change point
        if d <= change_point and change_point - d <= 1:
            valid_detections.append(d)
        # Detection happens after the change point but within max_delay
        elif d > change_point and d - change_point <= max_delay:
            valid_detections.append(d)

    if valid_detections:
        # Find the earliest detection
        earliest = min(valid_detections)
        # Calculate delay (ensure it's not negative)
        delay = max(0, earliest - change_point)
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
        return f"{reduction:.2%}"
    elif reduction == 0:
        return "0%"
    else:
        return f"{reduction:.2%} (increased)"


def compute_average_delay(
    change_points: List[int],
    detections: List[int],
    max_delay: int = 50,
    is_traditional: bool = False,
) -> Optional[float]:
    """Compute the average delay across all detected change points.

    Args:
        change_points: List of true change points
        detections: List of detection points
        max_delay: Maximum delay to consider a detection valid
        is_traditional: Deprecated parameter, kept for backward compatibility

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
    change_points: List[int],
    detections: List[int],
    max_delay: int = 50,
    is_traditional: bool = False,
) -> float:
    """Compute the percentage of change points that were detected.

    Args:
        change_points: List of true change points
        detections: List of detection points
        max_delay: Maximum delay to consider a detection valid
        is_traditional: Deprecated parameter, kept for backward compatibility

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
