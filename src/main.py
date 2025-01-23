# src/main.py

"""Main script for network forecasting with clear separation of data generation, actual processing, and forecasting."""

import sys
from pathlib import Path
import argparse
import logging

sys.path.append(str(Path(__file__).parent.parent))

from src.runner import ExperimentRunner
from src.setup.config import ExperimentConfig
from src.setup.prediction import GRAPH_MODELS, MODEL_PREDICTOR_RECOMMENDATIONS
from config.graph_configs import GRAPH_CONFIGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Network prediction framework")

    # Model and predictor arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        choices=list(GRAPH_MODELS.keys()),
        default="ba",
        help="Type of network model",
    )
    model_group.add_argument(
        "--predictor",
        type=str,
        choices=["weighted", "hybrid"],
        help="Type of predictor",
    )

    # Network parameters
    network_group = parser.add_argument_group("Network Parameters")
    network_group.add_argument(
        "--nodes",
        type=int,
        default=50,
        help="Number of nodes",
    )
    network_group.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length",
    )
    network_group.add_argument(
        "--min-changes",
        type=int,
        default=2,
        help="Minimum number of change points",
    )
    network_group.add_argument(
        "--max-changes",
        type=int,
        default=2,
        help="Maximum number of change points",
    )
    network_group.add_argument(
        "--min-segment",
        type=int,
        default=50,
        help="Minimum segment length",
    )

    # Prediction parameters
    pred_group = parser.add_argument_group("Prediction Parameters")
    pred_group.add_argument(
        "--prediction-window",
        type=int,
        default=5,
        help="Number of steps to predict ahead",
    )
    pred_group.add_argument(
        "--min-history",
        type=int,
        default=10,
        help="Minimum history length required for prediction",
    )

    # Experiment parameters
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    exp_group.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of experiment runs",
    )
    exp_group.add_argument(
        "--save-individual",
        action="store_true",
        help="Save results from individual runs",
    )
    exp_group.add_argument(
        "--visualize-individual",
        action="store_true",
        help="Generate visualizations for individual runs",
    )

    args = parser.parse_args()
    args.model = GRAPH_MODELS[args.model]

    if args.predictor is None:
        args.predictor = MODEL_PREDICTOR_RECOMMENDATIONS[args.model][0]
        logger.info(f"Using recommended predictor for {args.model}: {args.predictor}")

    return args


def create_experiment_config(args) -> ExperimentConfig:
    """Create ExperimentConfig from command line arguments."""
    graph_config = GRAPH_CONFIGS[args.model](
        n=args.nodes,
        seq_len=args.sequence_length,
        min_segment=args.min_segment,
        min_changes=args.min_changes,
        max_changes=args.max_changes,
    )

    config = ExperimentConfig(
        model=args.model,
        params=graph_config["params"],
        min_history=args.min_history,
        prediction_window=args.prediction_window,
        n_runs=args.runs,
        save_individual=args.save_individual,
        visualize_individual=args.visualize_individual,
    )

    # Add predictor type as an attribute
    config.predictor_type = args.predictor

    return config


def main():
    """Main execution function."""
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    # Create experiment configuration
    config = create_experiment_config(args)

    # Create and run experiment
    runner = ExperimentRunner(config=config, seed=args.seed)
    results = runner.run()

    logger.info(f"Experiment completed. Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
