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
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import ChangePointDetector
from src.configs.loader import get_config
from src.graph.generator import GraphGenerator
from src.predictor.factory import PredictorFactory
from src.changepoint.visualizer import MartingaleVisualizer
from src.graph.visualizer import NetworkVisualizer
from src.graph.features import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.configs.plotting import FIGURE_DIMENSIONS as FD
from src.configs.plotting import TYPOGRAPHY as TYPO

# Setup logging with debug level
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
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
    betting_function: str = "power",
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
    - Betting function (strategy for martingale updates: 'power', 'exponential', 'mixture', 'constant', 'beta', 'kernel')

    OUTPUT:
    - Change points τ and detection statistics
    """

    # ========================================================================== #
    # =================== STEP 1: Setup and Initialization ===================== #
    # ========================================================================== #
    model_name = get_full_model_name(model_alias)
    logger.info(f"Starting detection process for model: {model_name}")

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
    logger.info(f"Initialized predictor with history_size={predictor.history_size}")
    logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 2: Generate Network Sequence ==================== #
    # ========================================================================== #
    generator = GraphGenerator(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__
    # Override for testing
    params.update(
        {
            "n": 50,
            "seq_len": 100,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 40,
        }
    )

    # Generate sequence with ground truth change points
    logger.info("Generating graph sequence...")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]  # List of adjacency matrices
    true_change_points = result["change_points"]

    # Log sequence dimensions
    logger.info("Graph Sequence Dimensions:")
    logger.info(f"- Sequence length (T): {len(graphs)}")
    logger.info(f"- Individual graph shape (NxN): {graphs[0].shape}")
    logger.info(f"- Memory footprint: {sum(g.nbytes for g in graphs) / 1024:.2f} KB")
    logger.info(f"- True change points: {true_change_points}")
    logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 3: Extract Features ============================ #
    # ========================================================================== #
    logger.info("Extracting features from graph sequence...")

    # Extract features from the graph sequence
    features_raw = []
    features_numeric = []

    # Process first graph to get feature dimensions
    first_graph = adjacency_to_graph(graphs[0])
    feature_extractor = NetworkFeatureExtractor()
    first_raw_features = feature_extractor.get_features(first_graph)
    first_numeric_features = feature_extractor.get_numeric_features(first_graph)

    logger.info("Feature Extraction Dimensions:")
    logger.info(f"- Number of raw feature types: {len(first_raw_features)}")
    logger.info(f"- Numeric feature vector length: {len(first_numeric_features)}")

    # Process all graphs
    for adj_matrix in graphs:
        graph = adjacency_to_graph(adj_matrix)
        raw_features = feature_extractor.get_features(graph)
        numeric_features = feature_extractor.get_numeric_features(graph)
        features_raw.append(raw_features)
        features_numeric.append(numeric_features)

    features_numeric = np.array(
        [
            [numeric_features[name] for name in first_numeric_features.keys()]
            for numeric_features in features_numeric
        ]
    )
    logger.info("Final Feature Sequence Dimensions:")
    logger.info(f"- Shape (TxF): {features_numeric.shape}")  # T timesteps × F features
    logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 4: Generate Predictions ========================= #
    # ========================================================================== #
    logger.info("Generating predictions...")
    predicted_graphs = []

    # Process predictions
    for t in range(len(graphs)):
        current = graphs[t]
        history_start = max(0, t - predictor.history_size)
        history = [{"adjacency": g} for g in graphs[history_start:t]]

        if t >= predictor.history_size:
            predictions = predictor.predict(history, horizon=prediction_horizon)
            predicted_graphs.append(predictions)

    # Process predictions if they exist
    predicted_features_numeric = None
    if predicted_graphs:
        logger.info("Prediction Sequence Dimensions:")
        logger.info(f"- Number of prediction timesteps (T'): {len(predicted_graphs)}")
        logger.info(f"- Predictions per timestep (H): {len(predicted_graphs[0])}")
        logger.info(
            f"- Individual prediction shape (NxN): {predicted_graphs[0][0].shape}"
        )

        # Process features from predictions
        predicted_features_raw = []
        predicted_features_numeric = []

        for predictions in predicted_graphs:
            timestep_features = []
            for pred_adj in predictions:
                graph = adjacency_to_graph(pred_adj)
                raw_features = feature_extractor.get_features(graph)
                numeric_features = feature_extractor.get_numeric_features(graph)
                timestep_features.append(
                    [numeric_features[name] for name in first_numeric_features.keys()]
                )
            predicted_features_numeric.append(timestep_features)

        predicted_features_numeric = np.array(predicted_features_numeric)
        logger.info("Predicted Features Dimensions:")
        logger.info(
            f"- Shape (T'xHxF): {predicted_features_numeric.shape}"
        )  # T' prediction timesteps × H horizon × F features
        logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 5: Initialize Detector ========================== #
    # ========================================================================== #
    logger.info("Initializing change point detector...")

    detector = ChangePointDetector(
        martingale_method="multiview",
        history_size=predictor.history_size,
        threshold=threshold,
        epsilon=epsilon,
        random_state=random_state,
        feature_set="all",
        batch_size=batch_size,
        max_martingale=max_martingale,
        reset=reset,
        max_window=max_window,
        betting_func=betting_function,
    )

    # Run detection with extracted features
    logger.info("Running detection with extracted features...")
    detection_result = detector.run(
        data=features_numeric,
        predicted_data=predicted_features_numeric,
    )

    # First check if detection_result exists
    if detection_result is None:
        logger.error("Detection result is None")
        return

    logger.info("-" * 50)
    logger.info("Change Point Detection Analysis:")

    # Get detections from both streams
    traditional_changes = detection_result["traditional_change_points"]
    horizon_changes = detection_result["horizon_change_points"]

    # Basic detection summary
    logger.info("\nDetection Summary:")
    logger.info(f"True change points: {true_change_points}")
    logger.info(f"Traditional martingale detections: {traditional_changes}")
    logger.info(f"Horizon martingale detections: {horizon_changes}")

    # Detailed detection analysis
    if detector.method == "single_view":
        traditional_martingales = detection_result["traditional_martingales"]
        horizon_martingales = detection_result["horizon_martingales"]

        logger.info("\nDetailed Detection Analysis:")
        logger.info("Traditional Martingale:")
        logger.info(f"- Number of detections: {len(traditional_changes)}")
        if traditional_changes:
            logger.info(f"- Detection times: {traditional_changes}")
            logger.info(
                f"- Martingale values at detection: {[traditional_martingales[i] for i in traditional_changes]}"
            )

        logger.info("\nHorizon Martingale:")
        logger.info(f"- Number of detections: {len(horizon_changes)}")
        if horizon_changes:
            logger.info(f"- Detection times: {horizon_changes}")
            logger.info(
                f"- Martingale values at detection: {[horizon_martingales[i] for i in horizon_changes]}"
            )
    else:  # multiview
        traditional_sum = detection_result["traditional_sum_martingales"]
        horizon_sum = detection_result["horizon_sum_martingales"]

        logger.info("\nDetailed Detection Analysis:")
        logger.info("Traditional Martingale:")
        logger.info(f"- Number of detections: {len(traditional_changes)}")
        if traditional_changes:
            logger.info(f"- Detection times: {traditional_changes}")
            logger.info(
                f"- Sum martingale values at detection: {[traditional_sum[i] for i in traditional_changes]}"
            )

        logger.info("\nHorizon Martingale:")
        logger.info(f"- Number of detections: {len(horizon_changes)}")
        if horizon_changes:
            logger.info(f"- Detection times: {horizon_changes}")
            logger.info(
                f"- Sum martingale values at detection: {[horizon_sum[i] for i in horizon_changes]}"
            )

    # Compare with true change points
    if true_change_points:
        logger.info("\nDelay Analysis:")
        logger.info(f"True change points: {true_change_points}")

        for tcp in true_change_points:
            # Traditional delays
            trad_delays = [d - tcp for d in traditional_changes if d >= tcp]
            if trad_delays:
                logger.info(
                    f"Traditional detection delay for change at t={tcp}: {min(trad_delays)} steps"
                )
            else:
                logger.info(f"Traditional martingale did not detect change at t={tcp}")

            # Prediction delays
            pred_delays = [d - tcp for d in horizon_changes if d >= tcp]
            if pred_delays:
                logger.info(
                    f"Horizon detection delay for change at t={tcp}: {min(pred_delays)} steps"
                )
            else:
                logger.info(f"Horizon martingale did not detect change at t={tcp}")

    # Return complete results
    results = {
        "true_change_points": true_change_points,
        "model_name": model_name,
        "params": params,
        "features_raw": features_raw,
        "features_numeric": features_numeric,
        "predicted_graphs": predicted_graphs,
        "predictor_states": predictor.get_state(),
    }

    # Add method-specific detection results
    if detector.method == "single_view":
        results.update(
            {
                "traditional_change_points": traditional_changes,
                "horizon_change_points": horizon_changes,
                "traditional_martingales": detection_result["traditional_martingales"],
                "horizon_martingales": detection_result["horizon_martingales"],
            }
        )
    else:  # multiview
        results.update(
            {
                "traditional_change_points": traditional_changes,
                "horizon_change_points": horizon_changes,
                "traditional_sum_martingales": detection_result[
                    "traditional_sum_martingales"
                ],
                "traditional_avg_martingales": detection_result[
                    "traditional_avg_martingales"
                ],
                "horizon_sum_martingales": detection_result["horizon_sum_martingales"],
                "horizon_avg_martingales": detection_result["horizon_avg_martingales"],
                "individual_traditional_martingales": detection_result[
                    "individual_traditional_martingales"
                ],
                "individual_horizon_martingales": detection_result[
                    "individual_horizon_martingales"
                ],
            }
        )

    return results


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
    parser.add_argument(
        "--betting-func",
        type=str,
        default="power",
        choices=["power", "exponential", "mixture", "constant", "beta", "kernel"],
        help="Betting function for martingale updates (default: power)",
    )

    args = parser.parse_args()
    results = run_detection(
        model_alias=args.model,
        threshold=args.threshold,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        max_window=args.max_window,
        random_state=args.seed,
        betting_function=args.betting_func,
    )


if __name__ == "__main__":
    main()
