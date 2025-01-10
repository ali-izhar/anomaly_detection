"""Test script to analyze change point detection timing across multiple runs."""

import sys
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from predictor.utils import (
    generate_network_series,
    compute_network_features,
    compute_martingales,
)
from config.graph_configs import GRAPH_CONFIGS
from changepoint.detector import ChangePointDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
N_RUNS = 20  # Number of runs for the experiment
N_NODES = 50  # Number of nodes in the network
SEQ_LEN = 200  # Length of the sequence
MIN_SEGMENT = 50  # Minimum segment length
THRESHOLD = 70.0  # Detection threshold
EPSILON = 0.6  # Martingale sensitivity

def analyze_single_run(seed: int) -> Dict[str, Any]:
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
        features,
        detector,
        threshold=THRESHOLD,
        epsilon=EPSILON
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
        "martingale_values": martingale_values
    }

def analyze_detection_timing():
    """Run multiple experiments and analyze detection timing patterns."""
    results = []
    detection_offsets = defaultdict(list)
    
    # Run experiments
    for run in range(N_RUNS):
        seed = run + 1  # Use run number as seed for reproducibility
        logger.info(f"Running experiment {run+1}/{N_RUNS} with seed {seed}")
        
        result = analyze_single_run(seed)
        results.append(result)
        
        # Calculate detection offsets for each actual change point
        actual_cps = result["actual_cps"]
        for feature, detections in result["detected_cps"].items():
            for cp in actual_cps:
                # Find the closest detection to this change point
                if detections:
                    closest_detection = min(
                        detections,
                        key=lambda x: abs(x - cp)
                    )
                    # Only count if within ±10 steps of the change point
                    if abs(closest_detection - cp) <= 10:
                        offset = closest_detection - cp
                        detection_offsets[feature].append(offset)
    
    # Analyze results
    logger.info("\nDetection Timing Analysis:")
    for feature, offsets in detection_offsets.items():
        if offsets:
            mean_offset = np.mean(offsets)
            std_offset = np.std(offsets)
            logger.info(f"\n{feature}:")
            logger.info(f"  Mean offset: {mean_offset:.2f} steps")
            logger.info(f"  Std offset: {std_offset:.2f} steps")
            logger.info(f"  Number of detections: {len(offsets)}")
            
            # Count early vs late detections
            early = sum(1 for x in offsets if x < 0)
            late = sum(1 for x in offsets if x > 0)
            on_time = sum(1 for x in offsets if x == 0)
            logger.info(f"  Early detections: {early}")
            logger.info(f"  Late detections: {late}")
            logger.info(f"  On-time detections: {on_time}")
    
    # Plot distribution of detection offsets
    plt.figure(figsize=(12, 6))
    features = list(detection_offsets.keys())
    positions = range(len(features))
    
    plt.boxplot(
        [detection_offsets[f] for f in features],
        labels=features,
        showfliers=True
    )
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title("Distribution of Detection Offsets by Feature")
    plt.ylabel("Detection Offset (steps)")
    plt.xlabel("Feature")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig("detection_timing_analysis.png", dpi=300, bbox_inches="tight")
    logger.info("\nPlot saved as 'detection_timing_analysis.png'")

if __name__ == "__main__":
    analyze_detection_timing() 