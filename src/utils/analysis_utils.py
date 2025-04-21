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

    # Collect all detection points across trials
    all_traditional_points = []
    all_horizon_points = []

    for trial in individual_trials:
        all_traditional_points.extend(trial.get("traditional_change_points", []))
        all_horizon_points.extend(trial.get("horizon_change_points", []))

    # Compute consensus detection points (points detected in multiple trials)
    consensus_traditional = compute_consensus_points(
        all_traditional_points, threshold=0.3, tolerance=3
    )
    consensus_horizon = compute_consensus_points(
        all_horizon_points, threshold=0.3, tolerance=3
    )

    # Track per-trial detection statistics
    trial_statistics = []
    for trial_idx, trial in enumerate(individual_trials):
        trad_points = trial.get("traditional_change_points", [])
        horizon_points = trial.get("horizon_change_points", [])

        # Calculate detection metrics for this trial
        trad_rate = compute_detection_rate(true_change_points, trad_points)
        horizon_rate = compute_detection_rate(true_change_points, horizon_points)
        trad_delay = compute_average_delay(true_change_points, trad_points)
        horizon_delay = compute_average_delay(true_change_points, horizon_points)

        trial_statistics.append(
            {
                "trial": trial_idx + 1,
                "trad_rate": trad_rate,
                "horizon_rate": horizon_rate,
                "trad_delay": trad_delay,
                "horizon_delay": horizon_delay,
                "trad_points": trad_points,
                "horizon_points": horizon_points,
            }
        )

    # Calculate detection metrics for each true change point using consensus detections
    analysis_data = []
    for idx, cp in enumerate(true_change_points):
        # Find the closest traditional detection after the change point
        trad_delay, trad_detection = find_detection_delay(cp, consensus_traditional)

        # Find the closest horizon detection after the change point
        horizon_delay, horizon_detection = find_detection_delay(cp, consensus_horizon)

        # Calculate the delay reduction from using horizon detection
        delay_reduction = compute_delay_reduction(trad_delay, horizon_delay)

        # Count how many trials detected this change point
        trad_detection_count = sum(
            1
            for stat in trial_statistics
            if any(abs(tp - cp) <= 5 for tp in stat["trad_points"])
        )
        horizon_detection_count = sum(
            1
            for stat in trial_statistics
            if any(abs(hp - cp) <= 5 for hp in stat["horizon_points"])
        )

        analysis_data.append(
            [
                cp,
                f"{trad_detection if trad_detection is not None else 'Not detected'} ({trad_detection_count}/{num_trials})",
                trad_delay if trad_delay is not None else "-",
                f"{horizon_detection if horizon_detection is not None else 'Not detected'} ({horizon_detection_count}/{num_trials})",
                horizon_delay if horizon_delay is not None else "-",
                delay_reduction,
            ]
        )

    # Generate overall summary statistics based on all trials
    avg_trad_detection_rate = np.mean([stat["trad_rate"] for stat in trial_statistics])
    avg_horizon_detection_rate = np.mean(
        [stat["horizon_rate"] for stat in trial_statistics]
    )

    # Average delays (only include trials with actual detections)
    valid_trad_delays = [
        stat["trad_delay"]
        for stat in trial_statistics
        if stat["trad_delay"] is not None
    ]
    valid_horizon_delays = [
        stat["horizon_delay"]
        for stat in trial_statistics
        if stat["horizon_delay"] is not None
    ]

    avg_trad_delay = np.mean(valid_trad_delays) if valid_trad_delays else None
    avg_horizon_delay = np.mean(valid_horizon_delays) if valid_horizon_delays else None
    avg_reduction = compute_average_reduction(avg_trad_delay, avg_horizon_delay)

    # Create the detection details table
    headers = [
        "True CP",
        "Traditional Detection (trials)",
        "Delay (steps)",
        "Horizon Detection (trials)",
        "Delay (steps)",
        "Delay Reduction",
    ]

    table = tabulate(analysis_data, headers=headers, tablefmt=report_format)

    # Create summary table
    summary_data = [
        [
            "Detection Rate",
            f"{avg_trad_detection_rate:.1%}",
            f"{avg_horizon_detection_rate:.1%}",
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

    # Create per-trial table for detailed comparison
    trial_data = []
    for stat in trial_statistics:
        trial_data.append(
            [
                stat["trial"],
                f"{stat['trad_rate']:.1%}",
                f"{stat['horizon_rate']:.1%}",
                (
                    f"{stat['trad_delay']:.1f}"
                    if stat["trad_delay"] is not None
                    else "N/A"
                ),
                (
                    f"{stat['horizon_delay']:.1f}"
                    if stat["horizon_delay"] is not None
                    else "N/A"
                ),
            ]
        )

    trial_table = tabulate(
        trial_data,
        headers=["Trial", "Trad. Rate", "Horiz. Rate", "Trad. Delay", "Horiz. Delay"],
        tablefmt=report_format,
    )

    # Combine tables with headers
    report = (
        f"Change Point Detection Analysis ({num_trials} Trials)\n"
        "==============================\n\n"
        "Consensus Detection Details:\n"
        f"{table}\n\n"
        "Summary Statistics (Avg. across trials):\n"
        f"{summary_table}\n\n"
        "Per-Trial Statistics:\n"
        f"{trial_table}\n"
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
