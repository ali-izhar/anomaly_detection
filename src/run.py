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
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import os

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import ObservedStream
from src.configs.loader import get_config
from src.changepoint.visualizer import MartingaleVisualizer
from src.graph.generator import GraphGenerator

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


def test_observed_stream(
    graphs: List[np.ndarray], true_change_points: List[int], params: dict
) -> Dict[str, Any]:
    """Test the ObservedStream class on a sequence of graphs.

    Args:
        graphs: List of adjacency matrices
        true_change_points: List of true change point indices
        params: Dictionary of parameters

    Returns:
        Dictionary containing detection results
    """
    # Initialize observed stream with parameters
    observed_stream = ObservedStream(
        window_size=params["window_size"],
        threshold=params["threshold"],
        epsilon=params["epsilon"],
        martingale_method="multiview",  # Use multiview for better detection
        feature_set="all",  # Use all available features
        batch_size=1000,  # Add batch size for multiview
        max_martingale=None,  # Optional early stopping
        reset=True,  # Reset after detection
        random_state=42,
    )

    # Process each graph
    detected_changes = []
    logger.info("Processing graph sequence through ObservedStream...")
    pbar = tqdm(total=len(graphs), desc="Processing graphs", unit="graph")

    for t, adj_matrix in enumerate(graphs):
        # Update stream with new adjacency matrix
        result = observed_stream.update(adj_matrix, data_type="adjacency")

        # Check for change point
        if result["is_change"]:
            detected_changes.append(t)
            logger.info(f"Change detected at t={t}")

            # Print martingale values at change point
            if result.get("martingale_sum") is not None:
                logger.info(f"  Sum martingale: {result['martingale_sum']:.2f}")
                logger.info(f"  Avg martingale: {result['martingale_avg']:.2f}")
            else:
                logger.info(f"  Martingale: {result['martingale']:.2f}")

        pbar.update(1)

    pbar.close()

    # Get final state
    final_state = observed_stream.get_state()

    # Analyze detection performance
    if true_change_points and detected_changes:
        delays = []
        for true_cp in true_change_points:
            closest_detection = min(detected_changes, key=lambda x: abs(x - true_cp))
            delay = closest_detection - true_cp
            delays.append(delay)
            logger.info(
                f"Change point {true_cp}: detected at {closest_detection} (delay={delay})"
            )

        if delays:
            avg_delay = np.mean(delays)
            logger.info(f"Average detection delay: {avg_delay:.2f} time steps")

    # Compile results
    detection_results = {
        "feature_values": final_state["features_raw"],
        "martingale_values": final_state["martingale_values"],
        "detected_changes": detected_changes,
        "true_changes": true_change_points,
        "p_values": final_state["p_values"],
        "strangeness_values": final_state["strangeness_values"],
    }

    # Add multiview specific results if available
    if final_state["is_multiview"]:
        detection_results.update(
            {
                "martingales_sum": final_state["martingales_sum"],
                "martingales_avg": final_state["martingales_avg"],
                "individual_martingales": final_state["individual_martingales"],
            }
        )

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
    params.update(
        {
            "seq_len": 100,
            "n": 50,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 40,
        }
    )

    # Generate sequence
    result = generator.generate_sequence(params)
    graphs = result["graphs"]  # Already adjacency matrices
    true_change_points = result["change_points"]

    # 3. Run ObservedStream Detection
    logger.info("Running ObservedStream detection...")
    detection_results = test_observed_stream(
        graphs=graphs, true_change_points=true_change_points, params=DEFAULT_PARAMS
    )

    # 4. Visualize Results
    logger.info("Creating visualizations...")
    output_dir = f"results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create martingales dictionary for visualization
    martingales_dict = {
        "combined": {
            "martingales": detection_results["martingale_values"],
            "p_values": detection_results["p_values"],
            "strangeness": detection_results["strangeness_values"],
        }
    }

    # Add multiview specific results if available
    if "martingales_sum" in detection_results:
        martingales_dict["combined"].update(
            {
                "martingale_sum": detection_results["martingales_sum"],
                "martingale_avg": detection_results["martingales_avg"],
            }
        )

        # Add individual feature martingales if available
        if detection_results.get("individual_martingales"):
            feature_names = [
                "degree",
                "density",
                "clustering",
                "betweenness",
                "eigenvector",
                "closeness",
                "singular_value",
                "laplacian",
            ]
            for i, feature in enumerate(feature_names):
                martingales_dict[feature] = {
                    "martingales": [
                        m[i] for m in detection_results["individual_martingales"]
                    ],
                    "p_values": [1.0]
                    * len(detection_results["individual_martingales"]),  # Placeholder
                    "strangeness": [0.0]
                    * len(detection_results["individual_martingales"]),  # Placeholder
                }

    # Create visualizations using MartingaleVisualizer
    martingale_viz = MartingaleVisualizer(
        martingales=martingales_dict,
        change_points=true_change_points,
        threshold=DEFAULT_PARAMS["threshold"],
        epsilon=DEFAULT_PARAMS["epsilon"],
        output_dir=output_dir,
        prefix=f"{model_name}_",
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
