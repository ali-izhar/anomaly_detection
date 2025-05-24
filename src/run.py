#!/usr/bin/env python

"""Main entry point for running the detection pipeline."""

import argparse
import logging
import yaml
import sys
import copy
from typing import Dict, Any

from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection
from src.utils import print_analysis_report

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Apply command-line overrides to the configuration.

    Args:
        config: Original configuration dictionary
        args: Command-line arguments

    Returns:
        Updated configuration dictionary
    """
    # Make a deep copy to avoid modifying the original
    updated_config = copy.deepcopy(config)

    # Override n_trials
    if args.n_trials is not None:
        updated_config["trials"]["n_trials"] = args.n_trials
        logger.info(f"Overriding n_trials: {args.n_trials}")

    # Override enable_prediction
    if args.prediction is not None:
        updated_config["execution"]["enable_prediction"] = args.prediction
        logger.info(f"Overriding enable_prediction: {args.prediction}")

    # Override output directory
    if args.output_dir is not None:
        updated_config["output"]["directory"] = args.output_dir
        logger.info(f"Overriding output directory: {args.output_dir}")

    # Override network type
    if args.network is not None:
        updated_config["model"]["network"] = args.network
        logger.info(f"Overriding network type: {args.network}")

    # Override threshold
    if args.threshold is not None:
        updated_config["detection"]["threshold"] = args.threshold
        logger.info(f"Overriding threshold: {args.threshold}")

    # Override detection method
    if args.detection_method is not None:
        updated_config["detection"]["method"] = args.detection_method
        logger.info(f"Overriding detection method: {args.detection_method}")

    # Override betting function name
    if args.betting_func is not None:
        updated_config["detection"]["betting_func_config"]["name"] = args.betting_func
        logger.info(f"Overriding betting function: {args.betting_func}")

    # Override epsilon for power betting
    if args.epsilon is not None:
        if "power" not in updated_config["detection"]["betting_func_config"]:
            updated_config["detection"]["betting_func_config"]["power"] = {}
        updated_config["detection"]["betting_func_config"]["power"][
            "epsilon"
        ] = args.epsilon
        logger.info(f"Overriding power betting epsilon: {args.epsilon}")

    # Override beta parameters for beta betting
    if args.beta_a is not None:
        if "beta" not in updated_config["detection"]["betting_func_config"]:
            updated_config["detection"]["betting_func_config"]["beta"] = {}
        updated_config["detection"]["betting_func_config"]["beta"]["a"] = args.beta_a
        logger.info(f"Overriding beta betting alpha: {args.beta_a}")

    if args.beta_b is not None:
        if "beta" not in updated_config["detection"]["betting_func_config"]:
            updated_config["detection"]["betting_func_config"]["beta"] = {}
        updated_config["detection"]["betting_func_config"]["beta"]["b"] = args.beta_b
        logger.info(f"Overriding beta betting beta: {args.beta_b}")

    # Override mixture epsilons
    if args.mixture_epsilons is not None:
        if "mixture" not in updated_config["detection"]["betting_func_config"]:
            updated_config["detection"]["betting_func_config"]["mixture"] = {}
        # Parse comma-separated epsilon values
        epsilons = [float(x.strip()) for x in args.mixture_epsilons.split(",")]
        updated_config["detection"]["betting_func_config"]["mixture"][
            "epsilons"
        ] = epsilons
        logger.info(f"Overriding mixture betting epsilons: {epsilons}")

    # Override distance measure
    if args.distance is not None:
        updated_config["detection"]["distance"]["measure"] = args.distance
        logger.info(f"Overriding distance measure: {args.distance}")

    # Override reset_on_traditional
    if args.reset_on_traditional is not None:
        updated_config["detection"]["reset_on_traditional"] = args.reset_on_traditional
        logger.info(f"Overriding reset_on_traditional: {args.reset_on_traditional}")

    # Override dataset path for real datasets
    if args.dataset is not None:
        # For real datasets, we might need to adjust the model configuration
        updated_config["model"]["network"] = "real_dataset"
        updated_config["model"]["dataset_path"] = args.dataset
        logger.info(f"Overriding dataset path: {args.dataset}")

    # Override prediction horizon
    if args.prediction_horizon is not None:
        updated_config["detection"]["prediction_horizon"] = args.prediction_horizon
        logger.info(f"Overriding prediction horizon: {args.prediction_horizon}")

    # Override cooldown period
    if args.cooldown_period is not None:
        updated_config["detection"]["cooldown_period"] = args.cooldown_period
        logger.info(f"Overriding cooldown period: {args.cooldown_period}")

    return updated_config


def run_detection(
    config_file: str, log_level: str = "INFO", cli_args: argparse.Namespace = None
) -> Dict[str, Any]:
    """Run the change point detection pipeline.

    Args:
        config_file: Path to the configuration file
        log_level: Logging level
        cli_args: Command-line arguments for overriding config values

    Returns:
        Dict containing the detection results
    """
    setup_logging(log_level)
    logger.info(f"Using configuration file: {config_file}")

    try:
        # Load the base configuration
        config = load_config(config_file)

        # Apply command-line overrides if provided
        if cli_args:
            config = apply_cli_overrides(config, cli_args)

        # Initialize detector with the modified config
        detector = GraphChangeDetection(config_dict=config)
        results = detector.run()

        # Generate and print the detection analysis report
        print_analysis_report(results)

        return results

    except Exception as e:
        logger.error(f"Error running detection: {str(e)}")
        raise


def main() -> None:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the change point detection pipeline."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )

    # Add a new argument to visualize from Excel file
    parser.add_argument(
        "-e",
        "--excel-file",
        type=str,
        help="Path to Excel file with detection results for visualization only",
    )

    # Basic experiment parameters
    parser.add_argument(
        "-n",
        "--n-trials",
        type=int,
        help="Number of detection trials to run",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        type=lambda x: x.lower() == "true",
        choices=[True, False],
        help="Enable or disable prediction (true/false)",
    )
    parser.add_argument(
        "--network",
        "-net",
        type=str,
        choices=["sbm", "ba", "ws", "er", "real_dataset"],
        help="Network type (sbm: Stochastic Block Model, ba: Barabási–Albert, ws: Watts-Strogatz, er: Erdős–Rényi, real_dataset: Real dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to real dataset (when using --network real_dataset)",
    )
    parser.add_argument(
        "--threshold",
        "-l",
        type=float,
        help="Detection threshold value",
        dest="threshold",
    )

    # Detection method parameters
    parser.add_argument(
        "--detection-method",
        "-dm",
        type=str,
        choices=["martingale", "cusum", "ewma"],
        help="Detection method to use (martingale, cusum, ewma)",
        dest="detection_method",
    )

    # Betting function parameters (for martingale method)
    parser.add_argument(
        "--betting-func",
        "-bf",
        type=str,
        choices=["power", "exponential", "mixture", "constant", "beta", "kernel"],
        help="Betting function type",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Epsilon parameter for power betting function (e.g., 0.2, 0.5, 0.7, 0.9)",
    )
    parser.add_argument(
        "--beta-a",
        type=float,
        help="Alpha parameter for beta betting function",
    )
    parser.add_argument(
        "--beta-b",
        type=float,
        help="Beta parameter for beta betting function",
    )
    parser.add_argument(
        "--mixture-epsilons",
        type=str,
        help="Comma-separated epsilon values for mixture betting (e.g., '0.2,0.5,0.7,0.9')",
    )

    # Distance measure parameters
    parser.add_argument(
        "--distance",
        "-d",
        type=str,
        choices=[
            "euclidean",
            "mahalanobis",
            "manhattan",
            "minkowski",
            "cosine",
            "chebyshev",
        ],
        help="Distance measure for detection",
    )

    # Additional detection parameters
    parser.add_argument(
        "--reset-on-traditional",
        "-r",
        type=lambda x: x.lower() == "true",
        choices=[True, False],
        help="Reset on traditional change detection",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        help="Number of steps to predict ahead for horizon martingales",
    )
    parser.add_argument(
        "--cooldown-period",
        type=int,
        help="Minimum timesteps between consecutive detections",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for results and visualizations",
    )

    args = parser.parse_args()

    try:
        if args.config:
            run_detection(args.config, args.log_level, args)
        else:
            logger.error("Either --config or --excel-file must be provided")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
