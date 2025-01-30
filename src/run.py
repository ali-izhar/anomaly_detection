# src/run.py

"""
Main script to run martingale-based change detection on network sequences.

Usage:
    python src/run.py <model_alias>
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.pipeline import MartingalePipeline
from src.configs.loader import get_config
from src.graph.generator import GraphGenerator

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def run_detection(
    model_alias: str,
    threshold: float = 60.0,
    epsilon: float = 0.7,
    batch_size: int = 1000,
    max_martingale: float = None,
    reset: bool = True,
    max_window: int = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run martingale detection on a network sequence.

    Args:
        model_alias: Short name of the network model to use
        threshold: Detection threshold for martingale
        epsilon: Sensitivity parameter for martingale
        batch_size: Batch size for multiview processing
        max_martingale: Early stopping threshold
        reset: Whether to reset after detection
        max_window: Maximum window size for strangeness computation
        random_state: Random seed

    Returns:
        Dictionary containing detection results
    """
    # 1. Setup
    model_name = get_full_model_name(model_alias)
    logger.info(f"Running detection on {model_name} network sequence...")

    # 2. Generate Network Sequence
    generator = GraphGenerator(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__

    # Set sequence parameters
    params.update(
        {
            "seq_len": 200,  # Length of sequence
            "n": 50,  # Number of nodes
            "min_changes": 2,  # Minimum number of change points
            "max_changes": 2,  # Maximum number of change points
            "min_segment": 50,  # Minimum segment length
        }
    )

    # Generate sequence
    logger.info(f"Generating {model_name} sequence with parameters: {params}")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]  # List of adjacency matrices
    true_change_points = result["change_points"]
    logger.info(f"Generated sequence with change points at: {true_change_points}")

    # 3. Initialize and run pipeline
    logger.info("Running martingale detection pipeline...")
    pipeline = MartingalePipeline(
        martingale_method="multiview",
        threshold=threshold,
        epsilon=epsilon,
        random_state=random_state,
        feature_set="all",
        batch_size=batch_size,
        max_martingale=max_martingale,
        reset=reset,
        max_window=max_window,
    )

    # Run pipeline on entire sequence
    pipeline_result = pipeline.run(
        data=graphs,
        data_type="adjacency",  # Specify that we're passing adjacency matrices
    )

    # 4. Analyze results
    logger.info("\nDetection Results:")
    logger.info(f"True change points: {true_change_points}")
    logger.info(f"Detected change points: {pipeline_result['change_points']}")

    # Print martingale statistics
    logger.info("\nMartingale Statistics:")
    logger.info(
        f"- Final sum martingale value: {pipeline_result['martingales_sum'][-1]:.2f}"
    )
    logger.info(
        f"- Final average martingale value: {pipeline_result['martingales_avg'][-1]:.2f}"
    )
    logger.info(
        f"- Maximum sum martingale value: {np.max(pipeline_result['martingales_sum']):.2f}"
    )
    logger.info(
        f"- Maximum average martingale value: {np.max(pipeline_result['martingales_avg']):.2f}"
    )

    # Calculate detection accuracy
    if true_change_points and pipeline_result["change_points"]:
        delays = []
        for true_cp in true_change_points:
            closest_detection = min(
                pipeline_result["change_points"], key=lambda x: abs(x - true_cp)
            )
            delay = closest_detection - true_cp
            delays.append(delay)
            logger.info(
                f"Change point {true_cp}: detected at {closest_detection} (delay={delay})"
            )

        if delays:
            avg_delay = np.mean(delays)
            logger.info(f"Average detection delay: {avg_delay:.2f} time steps")

    # 5. Return results
    return {
        "true_change_points": true_change_points,
        "detected_changes": pipeline_result["change_points"],
        "model_name": model_name,
        "params": params,
        "martingales_sum": pipeline_result["martingales_sum"],
        "martingales_avg": pipeline_result["martingales_avg"],
        "individual_martingales": pipeline_result["individual_martingales"],
        "p_values": pipeline_result["p_values"],
        "strangeness": pipeline_result["strangeness"],
        "features_raw": pipeline_result.get("features_raw"),
        "features_numeric": pipeline_result.get("features_numeric"),
    }


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
        "--threshold",
        type=float,
        default=60.0,
        help="Detection threshold (default: 60.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.7,
        help="Sensitivity parameter (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for multiview (default: 1000)",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=None,
        help="Maximum window size (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    results = run_detection(
        model_alias=args.model,
        threshold=args.threshold,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        max_window=args.max_window,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
