# src/main.py

"""Main script for network forecasting."""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent))

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from predictor import (
    WeightedPredictor,
    HybridPredictor,
    Visualizer,
)
from config.graph_configs import GRAPH_CONFIGS

from typing import Dict, List, Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Define predictor mapping
PREDICTOR_MAP = {
    "weighted": WeightedPredictor,
    "hybrid": HybridPredictor,
}

# Define all available graph models
GRAPH_MODELS = {
    # Full names
    "barabasi_albert": "ba",
    "watts_strogatz": "ws",
    "erdos_renyi": "er",
    "stochastic_block_model": "sbm",
    "random_core_periphery": "rcp",
    "lfr_benchmark": "lfr",
    # Short aliases
    "ba": "ba",
    "ws": "ws",
    "er": "er",
    "sbm": "sbm",
    "rcp": "rcp",
    "lfr": "lfr",
}

# Define recommended predictors for each model
MODEL_PREDICTOR_RECOMMENDATIONS = {
    model: ["weighted", "hybrid"] for model in GRAPH_MODELS.values()
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Network prediction framework")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=list(GRAPH_MODELS.keys()),
        default="ba",
        help="Type of network model to use",
    )

    # Predictor selection
    parser.add_argument(
        "--predictor",
        type=str,
        choices=list(PREDICTOR_MAP.keys()),
        help="Type of predictor to use. If not specified, will use recommended predictor for the model.",
    )

    # Other parameters
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=50,
        help="Number of nodes in the network",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="Length of the network sequence",
    )
    parser.add_argument(
        "--min-changes",
        type=int,
        default=1,
        help="Minimum number of change points",
    )
    parser.add_argument(
        "--max-changes",
        type=int,
        default=3,
        help="Maximum number of change points",
    )
    parser.add_argument(
        "--min-segment",
        type=int,
        default=30,
        help="Minimum length between changes",
    )
    parser.add_argument(
        "--prediction-window",
        type=int,
        default=3,
        help="Number of steps to predict ahead",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=10,
        help="Minimum history length before making predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Convert model name to standard format
    args.model = GRAPH_MODELS[args.model]

    # If predictor not specified, use the recommended one
    if args.predictor is None:
        args.predictor = MODEL_PREDICTOR_RECOMMENDATIONS[args.model][0]
        logger.info(f"Using recommended predictor for {args.model}: {args.predictor}")
        logger.info(
            f"Alternative recommendation: {MODEL_PREDICTOR_RECOMMENDATIONS[args.model][1]}"
        )

    return args


def generate_network_series(
    config: Dict[str, Any], seed: int = None
) -> List[Dict[str, Any]]:
    """Generate a time series of evolving networks.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing model and parameters
    seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    List[Dict[str, Any]]
        List of network states over time
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize generator and feature extractor
    generator = GraphGenerator()
    feature_extractor = NetworkFeatureExtractor()

    # Generate sequence with model-specific parameters
    result = generator.generate_sequence(
        model=config["model"], params=config["params"], seed=seed
    )

    # Create a mapping of parameters for each time step
    param_map = {}
    current_params = result["parameters"][0]
    for i in range(config["params"].seq_len):
        # Update parameters at change points
        if i in result["change_points"]:
            current_params = result["parameters"][result["change_points"].index(i) + 1]
        param_map[i] = current_params

    # Convert to list of network states
    network_series = []
    for i, adj in enumerate(result["graphs"]):
        G = nx.from_numpy_array(adj)
        network_series.append(
            {
                "time": i,
                "adjacency": adj,
                "graph": G,
                "metrics": feature_extractor.get_all_metrics(G).__dict__,
                "params": param_map[i],
                "is_change_point": i in result["change_points"],
            }
        )

    return network_series


def analyze_prediction_phases(
    predictions: List[Dict[str, Any]],
    actual_series: List[Dict[str, Any]],
    min_history: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Analyze prediction accuracy across different phases."""
    # Initialize feature extractor
    feature_extractor = NetworkFeatureExtractor()

    # Check if we have predictions to analyze
    if not predictions or not actual_series:
        logger.warning("No predictions or actual series to analyze")
        return {"phases": {}, "error": "No data to analyze"}

    # Define phases based on available predictions
    total_predictions = len(predictions)
    if total_predictions < 3:
        phases = [(0, total_predictions, "All predictions")]
    else:
        third = total_predictions // 3
        phases = [
            (0, third, "First third"),
            (third, 2 * third, "Middle third"),
            (2 * third, None, "Last third"),
        ]

    # Store results
    results = {"phases": {}}

    # Calculate errors for each phase
    for start, end, phase_name in phases:
        logger.info(f"\nAnalyzing {phase_name}:")
        phase_predictions = predictions[start:end]
        phase_actuals = actual_series[
            min_history + start : min_history + (end if end else len(predictions))
        ]

        # Calculate average errors for this phase
        all_errors = []
        for pred, actual in zip(phase_predictions, phase_actuals):
            pred_metrics = feature_extractor.get_all_metrics(pred["graph"]).__dict__
            actual_metrics = feature_extractor.get_all_metrics(actual["graph"]).__dict__
            errors = calculate_error_metrics(actual_metrics, pred_metrics)
            all_errors.append(errors)

        if not all_errors:
            logger.warning(f"No errors calculated for {phase_name}")
            continue

        # Calculate average errors for each metric
        avg_errors = {
            metric: np.mean([e[metric] for e in all_errors])
            for metric in all_errors[0].keys()
        }

        # Store results
        results["phases"][phase_name] = {
            "start": start,
            "end": end,
            "errors": avg_errors,
        }

        # Print results
        for metric, error in avg_errors.items():
            logger.info(f"Average MAE for {metric}: {error:.3f}")

    # Save results to JSON
    with open(output_dir / "prediction_analysis.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def main():
    """Main execution function."""
    args = get_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/{args.model}_{args.predictor}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model-specific configuration
    config = GRAPH_CONFIGS[args.model](
        n=args.n_nodes,
        seq_len=args.seq_len,
        min_segment=args.min_segment,
        min_changes=args.min_changes,
        max_changes=args.max_changes,
    )

    # Initialize predictor with model-specific config
    if args.predictor == "hybrid":
        predictor = PREDICTOR_MAP[args.predictor](model_type=args.model, config=config)
    else:
        # For non-hybrid predictors (e.g., weighted)
        predictor = PREDICTOR_MAP[args.predictor]()

    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "parameters": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "model_config": {
                    "model": config["model"],
                    "params": {
                        k: v if isinstance(v, (int, float, str, bool)) else str(v)
                        for k, v in vars(config["params"]).items()
                    },
                },
            },
            f,
            indent=4,
        )

    # Parameters for prediction
    min_history = args.min_history
    prediction_window = args.prediction_window

    # Generate network time series
    print(f"Generating {args.model} network time series...")
    network_series = generate_network_series(config, seed=args.seed)

    # Perform rolling predictions
    print("Performing rolling predictions...")
    predictions = []
    feature_extractor = NetworkFeatureExtractor()

    for t in range(min_history, args.seq_len):
        # Get historical data up to current time t
        history = network_series[:t]

        # Generate prediction
        predicted_adjs = predictor.predict(history, horizon=prediction_window)

        # Store first prediction and its metrics
        pred_graph = nx.from_numpy_array(predicted_adjs[0])
        predictions.append(
            {
                "time": t,
                "adjacency": predicted_adjs[0],
                "graph": pred_graph,
                "metrics": feature_extractor.get_all_metrics(pred_graph).__dict__,
                "history_size": len(history),
            }
        )

    # Generate and save visualizations
    print("Generating visualizations...")
    visualizer = Visualizer()

    # 1. Metric evolution plot
    plt.figure(figsize=(12, 8))
    visualizer.plot_metric_evolution(
        network_series, predictions, min_history, model_type=args.model
    )
    plt.savefig(output_dir / "metric_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Comprehensive prediction dashboard
    plt.figure(figsize=(20, 15))
    visualizer.plot_prediction_dashboard(
        network_series,
        predictions,
        [min_history, len(predictions) // 2, -1],
        model_type=args.model,
    )
    plt.savefig(output_dir / "prediction_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Node degree evolution plot
    visualizer.plot_node_degree_evolution(
        network_series, output_path=output_dir / "node_degree_evolution.png"
    )

    # Analyze prediction accuracy
    print(f"\nPrediction Performance Summary for {args.model}:")
    print("-" * 50)
    analysis_results = analyze_prediction_phases(
        predictions, network_series, min_history, output_dir
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
