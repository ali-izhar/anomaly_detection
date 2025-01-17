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
    },
}

GRAPH_MODELS = {
    "barabasi_albert": "ba",
    "watts_strogatz": "ws",
    "erdos_renyi": "er",
    "stochastic_block_model": "sbm",
    "ba": "ba",
    "ws": "ws",
    "er": "er",
    "sbm": "sbm",
}

MODEL_PREDICTOR_RECOMMENDATIONS = {
    "ba": ["weighted", "hybrid"],
    "ws": ["weighted", "hybrid"],
    "er": ["weighted", "hybrid"],
    "sbm": ["weighted", "hybrid"],
}

THRESHOLD = 50.0
EPSILON = 0.7


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


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return {convert_to_serializable(item) for item in obj}
    elif isinstance(obj, Path):
        return str(obj)
    return obj


class ExperimentRunner:
    """Class to handle experiment execution with support for single and multiple runs."""

    def __init__(self, args, output_dir=None, seed=None):
        self.args = args
        self.seed = seed if seed is not None else args.seed
        self.args.seed = self.seed  # Update args seed with the new seed
        self.output_dir = output_dir or self._create_output_dir()
        self.config = self._get_config()

    def _create_output_dir(self):
        """Create and return output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(
            f"results/{self.args.model}_{self.args.predictor}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_config(self):
        """Get experiment configuration."""
        return GRAPH_CONFIGS[self.args.model](
            n=self.args.n_nodes,
            seq_len=self.args.seq_len,
            min_segment=self.args.min_segment,
            min_changes=self.args.min_changes,
            max_changes=self.args.max_changes,
        )

    def run_single_experiment(self):
        """Run a single experiment and return results."""
        # Generate data
        graphs, ground_truth = generate_data(self.args, self.config)

        # Initialize predictor
        predictor = (
            WeightedPredictor()
            if self.args.predictor == "weighted"
            else PREDICTOR_MAP["hybrid"][self.args.model](config=self.config)
        )

        # Compute metrics
        actual_metrics = compute_actual_metrics(graphs, self.args.min_history)
        forecast_metrics = compute_forecast_metrics(graphs, predictor, self.args)

        return {
            "graphs": graphs,
            "ground_truth": ground_truth,
            "actual_metrics": actual_metrics,
            "forecast_metrics": forecast_metrics,
        }

    def visualize_results(self, results):
        """Generate visualizations for the results."""
        visualizer = Visualizer()

        # 1. Metric evolution
        plt.figure(figsize=(12, 8))
        visualizer.plot_metric_evolution(
            results["graphs"],
            results["forecast_metrics"][0],  # predictions
            self.args.min_history,
            model_type=self.args.model,
            change_points=results["ground_truth"]["change_points"],
        )
        plt.savefig(
            self.output_dir / "metric_evolution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Performance extremes
        plt.figure(figsize=(20, 15))
        visualizer.plot_performance_extremes(
            results["graphs"][
                self.args.min_history : self.args.min_history
                + len(results["forecast_metrics"][0])
            ],
            results["forecast_metrics"][0],
            min_history=self.args.min_history,
            model_type=self.args.model,
            change_points=results["ground_truth"]["change_points"],
        )
        plt.savefig(
            self.output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Martingale comparison dashboard
        visualizer.create_martingale_comparison_dashboard(
            network_series=results["graphs"],
            actual_martingales=results["actual_metrics"][1],
            pred_martingales=results["forecast_metrics"][2],
            actual_shap=results["actual_metrics"][2],
            pred_shap=results["forecast_metrics"][3],
            output_path=self.output_dir / "martingale_comparison_dashboard.png",
            threshold=THRESHOLD,
            epsilon=EPSILON,
            change_points=results["ground_truth"]["change_points"],
            prediction_window=self.args.prediction_window,
        )

    def save_results(self, results):
        """Save experiment results and configuration."""
        # Convert all data to JSON serializable format
        serializable_data = {
            "model": self.args.model,
            "parameters": convert_to_serializable(vars(self.args)),
            "model_config": {
                "model": self.config["model"],
                "params": convert_to_serializable(vars(self.config["params"])),
            },
            "ground_truth": convert_to_serializable(results["ground_truth"]),
        }

        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(serializable_data, f, indent=4)

    def analyze_results(self, results):
        """Analyze prediction accuracy."""
        print(f"\nPrediction Performance Summary for {self.args.model}:")
        print("-" * 50)
        analyze_prediction_phases(
            results["forecast_metrics"][0],
            results["graphs"],
            self.args.min_history,
            self.output_dir,
        )


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
    parser.add_argument("--min-changes", type=int, default=2, help="Min change points")
    parser.add_argument("--max-changes", type=int, default=2, help="Max change points")
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


def main():
    """Main execution function with clear separation of data generation, processing, and forecasting."""
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    runner = ExperimentRunner(args)
    results = runner.run_single_experiment()
    runner.visualize_results(results)
    runner.save_results(results)
    runner.analyze_results(results)
    print(f"\nResults saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
