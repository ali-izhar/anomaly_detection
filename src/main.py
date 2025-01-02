"""Example script demonstrating network forecasting capabilities.

This script shows how to use the graph_forecasting package to:
1. Generate evolving network time series for different graph types
2. Predict future network states
3. Visualize and analyze prediction accuracy
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from predictor import WeightedPredictor, plot_metric_evolution
from config.graph_configs import GRAPH_CONFIGS

from typing import Dict, List, Any
import numpy as np
import networkx as nx


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

    # Generate sequence
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
) -> None:
    """Analyze prediction accuracy across different phases.

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        Prediction results
    actual_series : List[Dict[str, Any]]
        Actual network series
    min_history : int
        Minimum history used
    """
    # Initialize feature extractor
    feature_extractor = NetworkFeatureExtractor()

    # Define phases
    phases = [
        (0, 50, "First 50 predictions"),
        (50, 100, "Middle 50 predictions"),
        (100, None, "Last predictions"),
    ]

    # Calculate errors for each phase
    for start, end, phase_name in phases:
        print(f"\n{phase_name}:")
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

        # Print average errors for each metric
        avg_errors = {
            metric: np.mean([e[metric] for e in all_errors])
            for metric in all_errors[0].keys()
        }

        for metric, error in avg_errors.items():
            print(f"Average MAE for {metric}: {error:.3f}")


def main():
    """Run network forecasting example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate and analyze network time series"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="barabasi_albert",
        choices=list(GRAPH_CONFIGS.keys()),
        help="Type of network model to generate",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of nodes")
    parser.add_argument(
        "--seq_len", type=int, default=200, help="Length of time series"
    )
    parser.add_argument(
        "--min_segment", type=int, default=20, help="Minimum length between changes"
    )
    parser.add_argument(
        "--min_changes", type=int, default=2, help="Minimum number of changes"
    )
    parser.add_argument(
        "--max_changes", type=int, default=5, help="Maximum number of changes"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Get configuration for selected model
    config = GRAPH_CONFIGS[args.model](
        n=args.n,
        seq_len=args.seq_len,
        min_segment=args.min_segment,
        min_changes=args.min_changes,
        max_changes=args.max_changes,
    )

    # Parameters for prediction
    min_history = 3
    prediction_window = 10

    # Generate network time series
    print(f"Generating {args.model} network time series...")
    network_series = generate_network_series(config, seed=args.seed)

    # Create predictor
    predictor = WeightedPredictor(n_history=min_history)

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

    # Visualize results
    print("Visualizing metric evolution...")
    plot_metric_evolution(network_series, predictions, min_history)

    # Analyze prediction accuracy
    print(f"\nPrediction Performance Summary for {args.model}:")
    print("-" * 50)
    analyze_prediction_phases(predictions, network_series, min_history)


if __name__ == "__main__":
    main()
