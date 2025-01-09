# src/main.py

"""Main script for network forecasting."""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt

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
        "-s", "--min-segment", type=int, default=30, help="Min segment length"
    )
    parser.add_argument(
        "-w", "--prediction-window", type=int, default=3, help="Prediction steps"
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


def main():
    """Main execution function."""
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

    # Generate and predict network series
    print(f"Generating {args.model} network time series...")
    network_series = generate_network_series(config, seed=args.seed)

    print("Performing rolling predictions...")
    predictions = generate_predictions(
        network_series=network_series,
        predictor=predictor,
        min_history=args.min_history,
        seq_len=args.seq_len,
        prediction_window=args.prediction_window,
    )

    # Compute features and martingales
    print("Computing features and martingales...")
    actual_features = compute_network_features(
        network_series[args.min_history : args.min_history + len(predictions)]
    )
    pred_features = compute_network_features(predictions)

    detector = ChangePointDetector()
    actual_martingales = compute_martingales(actual_features, detector)
    pred_martingales = compute_martingales(pred_features, detector)

    actual_shap = compute_shap_values(actual_martingales, actual_features)
    pred_shap = compute_shap_values(pred_martingales, pred_features)

    # Generate visualizations
    print("Generating visualizations...")
    visualizer = Visualizer()

    # 1. Metric evolution
    plt.figure(figsize=(12, 8))
    visualizer.plot_metric_evolution(
        network_series, predictions, args.min_history, model_type=args.model
    )
    plt.savefig(output_dir / "metric_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Node degree evolution
    visualizer.plot_node_degree_evolution(
        network_series, output_path=output_dir / "node_degree_evolution.png"
    )

    # 3. Performance extremes
    plt.figure(figsize=(20, 15))
    visualizer.plot_performance_extremes(
        network_series[args.min_history : args.min_history + len(predictions)],
        predictions,
        min_history=args.min_history,
        model_type=args.model,
    )
    plt.savefig(output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Martingale comparison dashboard
    visualizer.create_martingale_comparison_dashboard(
        network_series=network_series,
        predictions=predictions,
        min_history=args.min_history,
        actual_martingales=actual_martingales,
        pred_martingales=pred_martingales,
        actual_shap=actual_shap,
        pred_shap=pred_shap,
        output_path=output_dir / "martingale_comparison_dashboard.png",
        threshold=30.0,
    )

    # Analyze prediction accuracy
    print(f"\nPrediction Performance Summary for {args.model}:")
    print("-" * 50)
    analyze_prediction_phases(predictions, network_series, args.min_history, output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
