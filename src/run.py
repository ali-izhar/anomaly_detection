# src/run.py

"""
Main script to run the forecast-based martingale detection algorithm on network sequences.
Implements the experimental setup from Section 6 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

Usage:
    python src/run.py <model_alias>
"""

import sys
import argparse
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import ObservedStream
from src.configs.loader import get_config
from src.graph.features import NetworkFeatureExtractor
from src.changepoint.visualizer import MartingaleVisualizer
from src.graph.generator import GraphGenerator

# from src.graph.visualizer import NetworkVisualizer


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Default parameters
DEFAULT_PARAMS = {
    "horizon": 5,
    "threshold": 60.0,
    "epsilon": 0.7,
    "window_size": 10,
    "predictor_type": "adaptive",
}

# Feature names used for detection
FEATURE_NAMES = [
    "degree",
    "density",
    "clustering",
    "betweenness",
    "eigenvector",
    "closeness",
    "singular_value",
    "laplacian",
]


def extract_numeric_features(feature_dict: dict) -> Dict[str, float]:
    """Extract numeric features from feature dictionary in a consistent order.

    Args:
        feature_dict: Dictionary containing raw feature values

    Returns:
        Dictionary mapping feature names to scalar values
    """
    # Extract basic metrics
    degrees = feature_dict.get("degrees", [])
    avg_degree = np.mean(degrees) if degrees else 0.0
    density = feature_dict.get("density", 0.0)
    clustering = feature_dict.get("clustering", [])
    avg_clustering = np.mean(clustering) if clustering else 0.0

    # Extract centrality metrics
    betweenness = feature_dict.get("betweenness", [])
    avg_betweenness = np.mean(betweenness) if betweenness else 0.0
    eigenvector = feature_dict.get("eigenvector", [])
    avg_eigenvector = np.mean(eigenvector) if eigenvector else 0.0
    closeness = feature_dict.get("closeness", [])
    avg_closeness = np.mean(closeness) if closeness else 0.0

    # Extract spectral metrics
    singular_values = feature_dict.get("singular_values", [])
    largest_sv = max(singular_values) if singular_values else 0.0
    laplacian_eigenvalues = feature_dict.get("laplacian_eigenvalues", [])
    smallest_nonzero_le = (
        min(x for x in laplacian_eigenvalues if x > 1e-10)
        if laplacian_eigenvalues
        else 0.0
    )

    return {
        "degree": avg_degree,
        "density": density,
        "clustering": avg_clustering,
        "betweenness": avg_betweenness,
        "eigenvector": avg_eigenvector,
        "closeness": avg_closeness,
        "singular_value": largest_sv,
        "laplacian": smallest_nonzero_le,
    }


def test_observed_stream(
    graphs: List[nx.Graph], true_change_points: List[int], params: dict
) -> Dict[str, Any]:
    """Test the ObservedStream class on a sequence of graphs.

    Args:
        graphs: List of networkx graphs
        true_change_points: List of true change point indices
        params: Dictionary of parameters

    Returns:
        Dictionary containing detection results
    """
    # Initialize feature extractor and observed stream
    feature_extractor = NetworkFeatureExtractor()
    observed_stream = ObservedStream(
        window_size=params["window_size"],
        threshold=params["threshold"],
        epsilon=params["epsilon"],
    )

    # Process each graph
    results = []
    detected_changes = set()
    feature_values = []

    logger.info("Processing graph sequence through ObservedStream...")
    pbar = tqdm(total=len(graphs), desc="Processing graphs", unit="graph")

    # Extract features for all graphs first
    features_raw = []
    for graph in graphs:
        # Extract all features at once
        features = feature_extractor.get_features(graph)
        features_raw.append(features)

    # Convert features to format expected by ObservedStream
    for t, raw_features in enumerate(features_raw):
        # Convert raw features to scalar values
        feature_dict = extract_numeric_features(raw_features)
        feature_values.append(feature_dict)

        # Update martingales
        result = observed_stream.update(feature_dict)
        results.append(result)

        # Track detected changes
        if result["changes"]:
            for feature, change_time in result["changes"].items():
                detected_changes.add(change_time)
                logger.info(f"Change detected at t={change_time} in feature {feature}")

        pbar.update(1)

    pbar.close()

    # Compile results
    detection_results = {
        "feature_values": feature_values,
        "martingale_results": results,
        "detected_changes": sorted(list(detected_changes)),
        "true_changes": true_change_points,
        "feature_martingales": {
            feature: [r["martingales"].get(feature, 1.0) for r in results]
            for feature in FEATURE_NAMES
        },
    }

    # Analyze detection performance
    if true_change_points:
        delays = []
        for true_cp in true_change_points:
            if detected_changes:
                closest_detection = min(
                    detected_changes, key=lambda x: abs(x - true_cp)
                )
                delay = closest_detection - true_cp
                delays.append(delay)
                logger.info(
                    f"Change point {true_cp}: detected at {closest_detection} (delay={delay})"
                )

        if delays:
            avg_delay = np.mean(delays)
            logger.info(f"Average detection delay: {avg_delay:.2f} time steps")

    return detection_results


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def run_detection(model_alias: str, output_dir: str = "results"):
    """Run the forecast-based martingale detection algorithm on a network sequence."""
    # 1. Setup
    model_name = get_full_model_name(model_alias)
    logger.info(f"Running detection on {model_name} network sequence...")

    # 2. Generate Network Sequence
    generator = GraphGenerator(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__

    # test: minimize params
    params.update(
        {
            "n": 30,
            "seq_len": 50,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 20,
        }
    )

    result = generator.generate_sequence(params)
    graphs = [nx.from_numpy_array(g) for g in result["graphs"]]
    true_change_points = result["change_points"]

    # 3. Run ObservedStream Detection
    logger.info("Running ObservedStream detection...")
    detection_results = test_observed_stream(
        graphs=graphs, true_change_points=true_change_points, params=DEFAULT_PARAMS
    )

    # 4. Visualize Results
    logger.info("Creating visualizations...")

    # Create martingales dictionary for visualization
    feature_martingales = {}
    for feature, martingale_values in detection_results["feature_martingales"].items():
        feature_martingales[feature] = {
            "martingales": martingale_values,
            "p_values": [1.0] * len(martingale_values),  # Placeholder
            "strangeness": [0.0] * len(martingale_values),  # Placeholder
        }

    # Create visualizations using MartingaleVisualizer
    martingale_viz = MartingaleVisualizer(
        martingales=feature_martingales,
        change_points=true_change_points,
        threshold=DEFAULT_PARAMS["threshold"],
        epsilon=DEFAULT_PARAMS["epsilon"],
        output_dir=output_dir,
        prefix="observed_",
    )
    martingale_viz.create_visualization()

    logger.info(f"Results saved to {output_dir}/")


def main():
    """Run detection based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run change point detection on network evolution."
    )
    parser.add_argument(
        "model",
        choices=["ba", "ws", "er", "sbm"],
        help="Model to analyze: ba (Barabási-Albert), ws (Watts-Strogatz), "
        "er (Erdős-Rényi), sbm (Stochastic Block Model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )

    args = parser.parse_args()
    run_detection(args.model, args.output_dir)


if __name__ == "__main__":
    main()
