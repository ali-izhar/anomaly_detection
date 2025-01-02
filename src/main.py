"""Example script demonstrating network forecasting capabilities.

This script shows how to use the graph_forecasting package to:
1. Generate evolving network time series
2. Predict future network states
3. Visualize and analyze prediction accuracy
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from graph_forecasting.generators import generate_evolving_ba_network
from graph_forecasting.predictors import WeightedAveragePredictor
from graph_forecasting.visualization import plot_metric_evolution
from graph_forecasting.metrics import get_network_metrics, calculate_error_metrics

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
    network_series = []
    for i in range(total_steps):
        network = generate_evolving_ba_network(
            N=100, m_mean=3, m_std=0.5, seed=None if seed is None else seed + i
        )
        network_series.append(network)
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
            pred_metrics = get_network_metrics(pred["graph"])
            actual_metrics = get_network_metrics(actual["graph"])
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
                "metrics": get_network_metrics(pred_graph),
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
