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
from predictor.weighted import WeightedPredictor
from predictor.hybrid import (
    BAPredictor,
    SBMPredictor,
    RCPPredictor,
    WSPredictor,
    ERPredictor,
)
from predictor.visualizer import Visualizer
from changepoint.detector import ChangePointDetector
from changepoint.threshold import CustomThresholdModel
from config.graph_configs import GRAPH_CONFIGS

from typing import Dict, List, Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Define predictor mapping
PREDICTOR_MAP = {
    "weighted": WeightedPredictor,
    "hybrid": {  # Map model types to their hybrid predictors
        "ba": BAPredictor,
        "ws": WSPredictor,
        "er": ERPredictor,
        "sbm": SBMPredictor,
        "rcp": RCPPredictor,
        "lfr": SBMPredictor,  # Use SBM predictor for LFR
    },
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
    "ba": ["hybrid", "weighted"],
    "ws": ["hybrid", "weighted"],
    "er": ["hybrid", "weighted"],
    "sbm": ["hybrid", "weighted"],
    "rcp": ["hybrid", "weighted"],
    "lfr": ["hybrid", "weighted"],
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
        choices=["weighted", "hybrid"],
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


def compute_network_features(graphs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Compute network features for martingale analysis."""
    feature_extractor = NetworkFeatureExtractor()
    features = {
        "degree": [],
        "clustering": [],
        "betweenness": [],
        "closeness": [],
    }

    for state in graphs:
        G = state["graph"]
        metrics = feature_extractor.get_all_metrics(G)
        features["degree"].append(metrics.avg_degree)
        features["clustering"].append(metrics.clustering)
        features["betweenness"].append(metrics.avg_betweenness)
        features["closeness"].append(metrics.avg_closeness)

    return {k: np.array(v) for k, v in features.items()}


def analyze_martingales(
    network_series: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    min_history: int,
    output_dir: Path,
) -> None:
    """Analyze and visualize martingales for actual and predicted networks."""
    # Extract features for actual and predicted networks
    actual_features = compute_network_features(
        network_series[min_history : min_history + len(predictions)]
    )
    pred_features = compute_network_features(predictions)

    # Initialize detector
    detector = ChangePointDetector()
    threshold = 30.0  # Can be made configurable
    epsilon = 0.8  # Can be made configurable

    # Compute martingales for actual networks
    actual_martingales = {"reset": {}, "cumulative": {}}
    for feature_name, feature_data in actual_features.items():
        # Reset martingales (with reset=True)
        reset_results = detector.detect_changes(
            data=feature_data.reshape(-1, 1),
            threshold=threshold,
            epsilon=epsilon,
            reset=True,
        )
        actual_martingales["reset"][feature_name] = {
            "martingales": reset_results["martingale_values"],
            "change_detected_instant": reset_results["change_points"],
        }

        # Cumulative martingales (with reset=False)
        cumul_results = detector.detect_changes(
            data=feature_data.reshape(-1, 1),
            threshold=threshold,
            epsilon=epsilon,
            reset=False,
        )
        actual_martingales["cumulative"][feature_name] = {
            "martingales": cumul_results["martingale_values"],
            "change_detected_instant": cumul_results["change_points"],
        }

    # Compute martingales for predicted networks
    pred_martingales = {"reset": {}, "cumulative": {}}
    for feature_name, feature_data in pred_features.items():
        # Reset martingales
        reset_results = detector.detect_changes(
            data=feature_data.reshape(-1, 1),
            threshold=threshold,
            epsilon=epsilon,
            reset=True,
        )
        pred_martingales["reset"][feature_name] = {
            "martingales": reset_results["martingale_values"],
            "change_detected_instant": reset_results["change_points"],
        }

        # Cumulative martingales
        cumul_results = detector.detect_changes(
            data=feature_data.reshape(-1, 1),
            threshold=threshold,
            epsilon=epsilon,
            reset=False,
        )
        pred_martingales["cumulative"][feature_name] = {
            "martingales": cumul_results["martingale_values"],
            "change_detected_instant": cumul_results["change_points"],
        }

    # Compute SHAP values for actual and predicted networks
    model = CustomThresholdModel(threshold=threshold)

    # For actual networks
    actual_feature_matrix = np.column_stack(
        [
            actual_martingales["reset"][feature]["martingales"]
            for feature in actual_features.keys()
        ]
    )
    actual_shap = model.compute_shap_values(
        X=actual_feature_matrix,
        change_points=sorted(
            list(
                set(
                    cp
                    for m in actual_martingales["reset"].values()
                    for cp in m["change_detected_instant"]
                )
            )
        ),
        sequence_length=len(actual_feature_matrix),
        window_size=5,
    )

    # For predicted networks
    pred_feature_matrix = np.column_stack(
        [
            pred_martingales["reset"][feature]["martingales"]
            for feature in pred_features.keys()
        ]
    )
    pred_shap = model.compute_shap_values(
        X=pred_feature_matrix,
        change_points=sorted(
            list(
                set(
                    cp
                    for m in pred_martingales["reset"].values()
                    for cp in m["change_detected_instant"]
                )
            )
        ),
        sequence_length=len(pred_feature_matrix),
        window_size=5,
    )

    # Create simplified dashboard using the Visualizer class
    visualizer = Visualizer()
    visualizer.create_martingale_comparison_dashboard(
        network_series=network_series,
        predictions=predictions,
        min_history=min_history,
        actual_martingales=actual_martingales,
        pred_martingales=pred_martingales,
        actual_shap=actual_shap,
        pred_shap=pred_shap,
        output_path=output_dir / "martingale_comparison_dashboard.png",
        threshold=threshold,
    )


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

    # Initialize predictor
    if args.predictor == "weighted":
        predictor = WeightedPredictor()
    else:  # hybrid
        # Get the appropriate hybrid predictor for the model type
        predictor_class = PREDICTOR_MAP["hybrid"][args.model]
        predictor = predictor_class(config=config)

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

    # 2. Node degree evolution plot
    visualizer.plot_node_degree_evolution(
        network_series, output_path=output_dir / "node_degree_evolution.png"
    )

    # 3. Performance extremes visualization
    plt.figure(figsize=(20, 15))
    visualizer.plot_performance_extremes(
        network_series[
            min_history : min_history + len(predictions)
        ],  # Align with predictions
        predictions,
        min_history=min_history,
        model_type=args.model,
    )
    plt.savefig(output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Analyze martingales
    print("Analyzing martingales...")
    analyze_martingales(network_series, predictions, min_history, output_dir)

    # Analyze prediction accuracy
    print(f"\nPrediction Performance Summary for {args.model}:")
    print("-" * 50)
    analysis_results = analyze_prediction_phases(
        predictions, network_series, min_history, output_dir
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
