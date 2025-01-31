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
from src.predictor.factory import PredictorFactory

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
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
    prediction_horizon: int = 5,
) -> Dict[str, Any]:
    """
    ALGORITHM: Forecast-based Graph Structural Change Detection using Martingale

    INPUT:
    - Network sequence {G_t}_{t=1}^n (generated from model_alias)
    - Threshold λ (threshold parameter)
    - Epsilon ε (sensitivity parameter)
    - Batch size (for multiview processing)
    - Max window (for historical context)
    - Random state (for reproducibility)
    - Prediction horizon h (number of steps to predict ahead)

    OUTPUT:
    - Change points τ and detection statistics
    """

    # STEP 1: Setup and Initialization
    model_name = get_full_model_name(model_alias)
    logger.info(f"Running detection on {model_name} network sequence...")

    # Initialize predictor using factory
    predictor_config = {
        "n_history": max_window if max_window else 10,
        "adaptive": True,
        "enforce_connectivity": True,
        "binary": True,
        "spectral_reg": 0.4,
        "community_reg": 0.4,
        "n_communities": 2,
        "temporal_window": 10,
        "distribution_reg": 0.3,
    }
    predictor = PredictorFactory.create("adaptive", predictor_config)

    # STEP 2: Generate Network Sequence
    generator = GraphGenerator(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__
    # Override for testing
    params.update(
        {
            "n": 20,
            "seq_len": 50,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 20,
        }
    )

    # Generate sequence with ground truth change points
    logger.info(f"Generating {model_name} sequence with parameters: {params}")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]  # List of adjacency matrices
    true_change_points = result["change_points"]
    logger.info(f"Generated sequence with change points at: {true_change_points}")

    # STEP 3: Initialize Pipeline with Prediction
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

    # STEP 4: Run Detection Pipeline with Prediction
    predicted_graphs = []
    detection_results = []

    for t in range(len(graphs)):
        # Get current graph and history
        current = graphs[t]
        history_start = max(0, t - predictor.history_size)
        history = [{"adjacency": g} for g in graphs[history_start:t]]

        logger.debug(f"\n=== Timestep t={t} ===")
        logger.debug(f"Current graph shape: {current.shape}")
        logger.debug(f"History window: [{history_start}:{t}] (size={len(history)})")

        # Make predictions for next h steps
        if t >= predictor.history_size:
            predictions = predictor.predict(history, horizon=prediction_horizon)
            predicted_graphs.append(predictions)

            logger.debug(f"Made predictions at t={t}:")
            logger.debug(f"- Number of predictions: {len(predictions)}")
            logger.debug(
                f"- Each prediction shape: {predictions[0].shape if len(predictions) > 0 else 'N/A'}"
            )
            logger.debug(f"- Predicting for timesteps: t+1 to t+{prediction_horizon}")
            logger.debug(f"- Total predictions so far: {len(predicted_graphs)}")

            # Update predictor's state with actual observation
            predictor.update_state({"adjacency": current})
            logger.debug("Updated predictor state with current observation")

    # Run pipeline with both actual and predicted graphs
    pipeline_result = pipeline.run(
        data=graphs,
        data_type="adjacency",
        predicted_data=predicted_graphs,
    )

    # STEP 5: Analyze Results
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

    # STEP 6: Calculate Detection Accuracy
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

    # STEP 7: Return Complete Results
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
        "predicted_graphs": predicted_graphs,  # Add predictions to output
        "predictor_states": predictor.get_state(),  # Add final predictor state
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
