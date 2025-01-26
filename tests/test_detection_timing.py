# tests/test_detection_timing.py

"""Test script to analyze change point detection timing across multiple runs."""

import sys
import os
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, Any
import itertools
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.runner import (
    generate_network_series,
    compute_network_features,
    compute_martingales,
)
from src.graph.graph_config import GRAPH_CONFIGS
from src.changepoint.detector import ChangePointDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
N_RUNS = 30  # Increased number of runs for better statistics
N_NODES = 50  # Number of nodes in the network
SEQ_LEN = 200  # Increased sequence length
MIN_SEGMENT = 50  # Minimum segment length

# Parameter ranges to test
THRESHOLDS = [50.0, 70.0, 100.0, 150.0, 200.0]
EPSILONS = [0.6, 0.7, 0.8, 0.9]


def analyze_single_run(seed: int, threshold: float, epsilon: float) -> Dict[str, Any]:
    """Run a single experiment and analyze detection timing."""
    # Get SBM configuration
    config = GRAPH_CONFIGS["sbm"](
        n=N_NODES,
        seq_len=SEQ_LEN,
        min_segment=MIN_SEGMENT,
        min_changes=2,  # Force exactly 2 change points
        max_changes=2,
    )

    # Generate network series
    network_data = generate_network_series(config, seed=seed)
    graphs = network_data["graphs"]
    actual_cps = network_data["change_points"]

    # Compute features and martingales
    features = compute_network_features(graphs)
    detector = ChangePointDetector()
    martingale_results = compute_martingales(
        features, detector, threshold=threshold, epsilon=epsilon
    )

    # Extract detected change points from each feature
    detected_cps = defaultdict(list)
    for feature_name, result in martingale_results["reset"].items():
        if "change_detected_instant" in result:
            detected_cps[feature_name].extend(result["change_detected_instant"])

    # Also store the martingale values for analysis
    martingale_values = {
        feature: result["martingales"]
        for feature, result in martingale_results["reset"].items()
    }

    return {
        "seed": seed,
        "actual_cps": actual_cps,
        "detected_cps": dict(detected_cps),
        "martingale_values": martingale_values,
        "threshold": threshold,
        "epsilon": epsilon,
    }


def analyze_parameter_combinations():
    """Run experiments with different parameter combinations and analyze detection timing."""
    all_results = []
    parameter_stats = []

    # Run experiments for each parameter combination
    for threshold, epsilon in itertools.product(THRESHOLDS, EPSILONS):
        logger.info(f"\nTesting threshold={threshold}, epsilon={epsilon}")
        detection_offsets = defaultdict(list)
        early_detections = defaultdict(int)
        late_detections = defaultdict(int)
        false_positives = defaultdict(int)
        missed_detections = defaultdict(int)

        for run in range(N_RUNS):
            seed = run + 1
            result = analyze_single_run(seed, threshold, epsilon)
            all_results.append(result)
            actual_cps = result["actual_cps"]

            # Analyze detections for each feature
            for feature, detections in result["detected_cps"].items():
                matched_cps = set()

                # For each actual change point, find the closest detection
                for cp in actual_cps:
                    if detections:
                        closest_detection = min(detections, key=lambda x: abs(x - cp))
                        offset = closest_detection - cp

                        # Only count if within ±20 steps of the change point
                        if abs(offset) <= 20:
                            detection_offsets[feature].append(offset)
                            matched_cps.add(closest_detection)

                            if offset < 0:
                                early_detections[feature] += 1
                            else:
                                late_detections[feature] += 1

                # Count unmatched detections as false positives
                false_positives[feature] += len(set(detections) - matched_cps)

                # Count unmatched actual CPs as missed detections
                detected_cps = set(
                    d for d in detections if any(abs(d - cp) <= 20 for cp in actual_cps)
                )
                missed_detections[feature] += len(actual_cps) - len(detected_cps)

        # Compute statistics for this parameter combination
        for feature in detection_offsets.keys():
            offsets = detection_offsets[feature]
            if offsets:
                stats = {
                    "threshold": threshold,
                    "epsilon": epsilon,
                    "feature": feature,
                    "mean_offset": np.mean(offsets),
                    "std_offset": np.std(offsets),
                    "n_detections": len(offsets),
                    "early_detections": early_detections[feature],
                    "late_detections": late_detections[feature],
                    "false_positives": false_positives[feature],
                    "missed_detections": missed_detections[feature],
                    "late_to_early_ratio": late_detections[feature]
                    / (early_detections[feature] + 1e-6),
                }
                parameter_stats.append(stats)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(parameter_stats)

    # Create heatmaps for different metrics
    metrics = [
        "late_to_early_ratio",
        "mean_offset",
        "false_positives",
        "missed_detections",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        pivot_data = df.pivot_table(
            values=metric, index="threshold", columns="epsilon", aggfunc="mean"
        )

        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=axes[idx])
        axes[idx].set_title(f"{metric} by Parameters")

    plt.tight_layout()
    plt.savefig("parameter_analysis.png", dpi=300, bbox_inches="tight")

    # Find best parameter combinations for late detection
    best_params = df.sort_values(
        by=["late_to_early_ratio", "false_positives", "missed_detections"],
        ascending=[False, True, True],
    ).head(5)

    logger.info("\nBest parameter combinations for late detection:")
    logger.info(
        best_params[
            [
                "threshold",
                "epsilon",
                "late_to_early_ratio",
                "mean_offset",
                "false_positives",
                "missed_detections",
            ]
        ]
    )

    # Save detailed results
    df.to_csv("parameter_analysis.csv", index=False)
    logger.info("\nDetailed results saved to 'parameter_analysis.csv'")


if __name__ == "__main__":
    analyze_parameter_combinations()
