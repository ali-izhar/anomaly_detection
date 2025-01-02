"""Example script demonstrating network forecasting capabilities.

This script shows how to use the graph_forecasting package to:
1. Generate evolving network time series
2. Predict future network states
3. Visualize and analyze prediction accuracy
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from graph.generator import GraphGenerator
from graph.params import BAParams
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from graph_forecasting.predictors import WeightedAveragePredictor
from graph_forecasting.visualization import plot_metric_evolution

from typing import Dict, List, Any
import numpy as np
import networkx as nx


def generate_network_series(
    total_steps: int = 200, seed: int = None
) -> List[Dict[str, Any]]:
    """Generate a time series of evolving networks.

    Parameters
    ----------
    total_steps : int, optional
        Number of time steps to generate, by default 200
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

    # Configure BA parameters with evolution and anomaly injection
    params = BAParams(
        # Required parameters
        n=100,  # Fixed number of nodes
        seq_len=total_steps,
        min_segment=20,  # Minimum length between anomalies
        min_changes=2,  # At least 2 anomalies
        max_changes=5,  # At most 5 anomalies
        m=3,  # Base number of edges per new node
        min_m=1,  # Minimum edges during anomalies
        max_m=6,  # Maximum edges during anomalies
        # Optional evolution parameters
        n_std=None,  # Keep node count fixed
        m_std=0.5,  # Allow edge count to evolve
    )

    # Generate sequence
    result = generator.generate_sequence(model="barabasi_albert", params=params)

    # Create a mapping of parameters for each time step
    param_map = {}
    current_params = result["parameters"][0]
    for i in range(total_steps):
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
                "metrics": feature_extractor.get_all_metrics(
                    G
                ).__dict__,  # Convert to dict for serialization
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
    # Parameters
    total_steps = 200
    min_history = 3
    prediction_window = 10

    # Generate network time series
    print("Generating network time series...")
    network_series = generate_network_series(total_steps=total_steps)

    # Create predictor
    predictor = WeightedAveragePredictor(n_history=3)

    # Perform rolling predictions
    print("Performing rolling predictions...")
    predictions = []
    feature_extractor = NetworkFeatureExtractor()

    for t in range(min_history, total_steps):
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
    print("\nPrediction Performance Summary:")
    print("-------------------------------")
    analyze_prediction_phases(predictions, network_series, min_history)


if __name__ == "__main__":
    main()
