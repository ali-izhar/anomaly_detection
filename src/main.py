# src/main.py

"""Main script for network forecasting with clear separation of data generation, actual processing, and forecasting."""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from predictor.weighted import WeightedPredictor
from predictor.hybrid import (
    BAPredictor,
    SBMPredictor,
    RCPPredictor,
    WSPredictor,
    ERPredictor,
)
from predictor.visualizer import Visualizer
from predictor.utils import (
    generate_network_series,
    generate_predictions,
    compute_network_features,
    analyze_prediction_phases,
    compute_martingales,
    compute_shap_values,
)
from changepoint.detector import ChangePointDetector
from config.graph_configs import GRAPH_CONFIGS

logger = logging.getLogger(__name__)

#######################
# Global Configurations
#######################

PREDICTOR_MAP = {
    "weighted": WeightedPredictor,
    "hybrid": {
        "ba": BAPredictor,
        "ws": WSPredictor,
        "er": ERPredictor,
        "sbm": SBMPredictor,
        "rcp": RCPPredictor,
        "lfr": SBMPredictor,
    },
}

GRAPH_MODELS = {
    "barabasi_albert": "ba",
    "watts_strogatz": "ws",
    "erdos_renyi": "er",
    "stochastic_block_model": "sbm",
    "random_core_periphery": "rcp",
    "lfr_benchmark": "lfr",
    "ba": "ba",
    "ws": "ws",
    "er": "er",
    "sbm": "sbm",
    "rcp": "rcp",
    "lfr": "lfr",
}

MODEL_PREDICTOR_RECOMMENDATIONS = {
    "ba": ["weighted", "hybrid"],
    "ws": ["weighted", "hybrid"],
    "er": ["weighted", "hybrid"],
    "sbm": ["weighted", "hybrid"],
    "rcp": ["weighted", "hybrid"],
    "lfr": ["weighted", "hybrid"],
}

THRESHOLD = 70.0
EPSILON = 0.6


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Network prediction framework")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=list(GRAPH_MODELS.keys()),
        default="ba",
        help="Type of network model",
    )
    parser.add_argument(
        "-p",
        "--predictor",
        type=str,
        choices=["weighted", "hybrid"],
        help="Type of predictor",
    )
    parser.add_argument("-n", "--n-nodes", type=int, default=50, help="Number of nodes")
    parser.add_argument(
        "-l", "--seq-len", type=int, default=100, help="Sequence length"
    )
    parser.add_argument("--min-changes", type=int, default=1, help="Min change points")
    parser.add_argument("--max-changes", type=int, default=3, help="Max change points")
    parser.add_argument(
        "-s", "--min-segment", type=int, default=50, help="Min segment length"
    )
    parser.add_argument(
        "-w", "--prediction-window", type=int, default=5, help="Prediction steps"
    )
    parser.add_argument(
        "-mh", "--min-history", type=int, default=10, help="Min history length"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    args.model = GRAPH_MODELS[args.model]

    if args.predictor is None:
        args.predictor = MODEL_PREDICTOR_RECOMMENDATIONS[args.model][0]
        logger.info(f"Using recommended predictor for {args.model}: {args.predictor}")

    return args


def generate_data(args, config):
    """Generate synthetic network data."""
    logger.info(f"Generating {args.model} network time series...")
    network_data = generate_network_series(config, seed=args.seed)

    # Extract ground truth information
    ground_truth = {
        "change_points": network_data["change_points"],
        "parameters": network_data["parameters"],
        "metadata": network_data["metadata"],
        "model": network_data["model"],
        "num_changes": network_data["num_changes"],
        "n": network_data["n"],
        "sequence_length": network_data["sequence_length"],
    }

    return network_data["graphs"], ground_truth


def compute_actual_metrics(graphs, min_history):
    """Compute actual network metrics without data leakage."""
    logger.info("Computing actual network metrics...")
    actual_features = compute_network_features(graphs[min_history:])

    detector = ChangePointDetector()
    actual_martingales = compute_martingales(
        actual_features, detector, threshold=THRESHOLD, epsilon=EPSILON
    )
    actual_shap = compute_shap_values(actual_martingales, actual_features)

    return actual_features, actual_martingales, actual_shap


def compute_forecast_metrics(graphs, predictor, args):
    """Compute forecasting metrics without data leakage."""
    logger.info("Performing rolling predictions...")
    predictions = generate_predictions(
        network_series=graphs,
        predictor=predictor,
        min_history=args.min_history,
        seq_len=args.seq_len,
        prediction_window=args.prediction_window,
    )

    pred_features = compute_network_features(predictions)

    detector = ChangePointDetector()
    pred_martingales = compute_martingales(
        pred_features, detector, threshold=THRESHOLD, epsilon=EPSILON
    )
    pred_shap = compute_shap_values(pred_martingales, pred_features)

    return predictions, pred_features, pred_martingales, pred_shap


def save_results(
    output_dir, args, config, ground_truth, actual_metrics, forecast_metrics
):
    """Save all results and configurations."""

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    # Convert ground truth to serializable format
    serializable_ground_truth = convert_to_serializable(ground_truth)

    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "parameters": {
                    k: convert_to_serializable(v) for k, v in vars(args).items()
                },
                "model_config": {
                    "model": config["model"],
                    "params": {
                        k: convert_to_serializable(v)
                        for k, v in vars(config["params"]).items()
                    },
                },
                "ground_truth": serializable_ground_truth,
            },
            f,
            indent=4,
        )

    # # Save metrics
    # np.save(output_dir / "actual_features.npy", actual_metrics[0])
    # np.save(output_dir / "actual_martingales.npy", actual_metrics[1])
    # np.save(output_dir / "actual_shap.npy", actual_metrics[2])

    # np.save(output_dir / "predictions.npy", forecast_metrics[0])
    # np.save(output_dir / "pred_features.npy", forecast_metrics[1])
    # np.save(output_dir / "pred_martingales.npy", forecast_metrics[2])
    # np.save(output_dir / "pred_shap.npy", forecast_metrics[3])


def generate_visualizations(
    output_dir,
    graphs,
    predictions,
    actual_metrics,
    forecast_metrics,
    args,
    ground_truth,
):
    """Generate and save visualizations."""
    logger.info("Generating visualizations...")
    visualizer = Visualizer()

    # 1. Metric evolution
    plt.figure(figsize=(12, 8))
    visualizer.plot_metric_evolution(
        graphs,
        predictions,
        args.min_history,
        model_type=args.model,
        change_points=ground_truth["change_points"],
    )
    plt.savefig(output_dir / "metric_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Node degree evolution
    visualizer.plot_node_degree_evolution(
        graphs,
        change_points=ground_truth["change_points"],
        output_path=output_dir / "node_degree_evolution.png",
    )

    # 3. Performance extremes
    plt.figure(figsize=(20, 15))
    visualizer.plot_performance_extremes(
        graphs[args.min_history : args.min_history + len(predictions)],
        predictions,
        min_history=args.min_history,
        model_type=args.model,
        change_points=ground_truth["change_points"],
    )
    plt.savefig(output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Martingale comparison dashboard
    visualizer.create_martingale_comparison_dashboard(
        network_series=graphs,
        actual_martingales=actual_metrics[1],
        pred_martingales=forecast_metrics[2],
        actual_shap=actual_metrics[2],
        pred_shap=forecast_metrics[3],
        output_path=output_dir / "martingale_comparison_dashboard.png",
        threshold=THRESHOLD,
        epsilon=EPSILON,
        change_points=ground_truth["change_points"],
        prediction_window=args.prediction_window,
    )


def main():
    """Main execution function with clear separation of data generation, processing, and forecasting."""
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/{args.model}_{args.predictor}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get configuration and initialize predictor
    config = GRAPH_CONFIGS[args.model](
        n=args.n_nodes,
        seq_len=args.seq_len,
        min_segment=args.min_segment,
        min_changes=args.min_changes,
        max_changes=args.max_changes,
    )

    predictor = (
        WeightedPredictor()
        if args.predictor == "weighted"
        else PREDICTOR_MAP["hybrid"][args.model](config=config)
    )

    # 1. Generate synthetic data
    graphs, ground_truth = generate_data(args, config)

    # 2. Compute actual metrics (without data leakage)
    actual_metrics = compute_actual_metrics(graphs, args.min_history)

    # 3. Compute forecast metrics (without data leakage)
    forecast_metrics = compute_forecast_metrics(graphs, predictor, args)

    # 4. Save results
    save_results(
        output_dir, args, config, ground_truth, actual_metrics, forecast_metrics
    )

    # 5. Generate visualizations
    generate_visualizations(
        output_dir,
        graphs,
        forecast_metrics[0],
        actual_metrics,
        forecast_metrics,
        args,
        ground_truth,
    )

    # 6. Analyze prediction accuracy
    print(f"\nPrediction Performance Summary for {args.model}:")
    print("-" * 50)
    analyze_prediction_phases(forecast_metrics[0], graphs, args.min_history, output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
