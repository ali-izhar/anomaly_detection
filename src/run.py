#!/usr/bin/env python

"""Main entry point for running the detection pipeline."""

import argparse
import logging
import yaml
import sys
import copy
from typing import Dict, Any
import os

from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection
from src.utils import (
    print_analysis_report,
    plot_martingales_from_csv,
)

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

    # Override enable_visualization
    if args.visualize is not None:
        if "visualization" not in updated_config["output"]:
            updated_config["output"]["visualization"] = {}
        updated_config["output"]["visualization"]["enabled"] = args.visualize
        logger.info(f"Overriding visualization: {args.visualize}")

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

    # Override betting function name
    if args.betting_func is not None:
        updated_config["detection"]["betting_func_config"]["name"] = args.betting_func
        logger.info(f"Overriding betting function: {args.betting_func}")

    # Override distance measure
    if args.distance is not None:
        updated_config["detection"]["distance"]["measure"] = args.distance
        logger.info(f"Overriding distance measure: {args.distance}")

    # Override reset_on_traditional
    if args.reset_on_traditional is not None:
        updated_config["detection"]["reset_on_traditional"] = args.reset_on_traditional
        logger.info(f"Overriding reset_on_traditional: {args.reset_on_traditional}")

    return updated_config


def create_visualizations(results, config=None):
    """Create visualizations from detection results.

    Args:
        results: Dictionary containing detection results
        config: Optional configuration dictionary. If None, uses values from results['params']
    """
    if not results:
        logger.warning("No results provided for visualization")
        return

    if config is None and "params" in results:
        config = results["params"]

    if not config:
        logger.warning("No configuration provided for visualization")
        return

    try:
        # Get visualization parameters
        output_config = config.get("output", {})
        detection_config = config.get("detection", {})

        # Skip if visualization is disabled
        if not output_config.get("visualization", {}).get("enabled", True):
            logger.info("Visualization is disabled in configuration")
            return

        # Get output directory
        output_dir = output_config.get("directory", "results")

        # Get detection threshold
        threshold = detection_config.get("threshold", 50.0)

        # Try to find the Excel results file in the output directory
        excel_file = os.path.join(output_dir, "detection_results.xlsx")

        if os.path.exists(excel_file):
            # Use the new plotting function directly from the Excel file
            logger.info(f"Creating visualizations from Excel file: {excel_file}")
            plot_martingales_from_csv(
                csv_path=excel_file,
                sheet_name="Aggregate",
                output_dir=output_dir,
                threshold=threshold,
            )
            logger.info(f"Visualizations created in {output_dir}")
        else:
            logger.error(f"Excel file not found: {excel_file}")

    except Exception as e:
        logger.error(f"Visualization creation failed: {str(e)}")
        logger.error("Continuing without visualizations")
        import traceback

        logger.debug(traceback.format_exc())


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

        # Create visualizations after analysis
        create_visualizations(results)

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

    # Add new CLI parameters to override config values
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
        choices=["sbm", "ba", "ws", "er"],
        help="Network type (sbm: Stochastic Block Model, ba: Barabási–Albert, ws: Watts-Strogatz, er: Erdős–Rényi)",
    )
    parser.add_argument(
        "--threshold",
        "-l",
        type=float,
        help="Detection threshold value",
        dest="threshold",
    )
    parser.add_argument(
        "--betting-func",
        "-bf",
        type=str,
        choices=["power", "exponential", "mixture", "constant", "beta", "kernel"],
        help="Betting function type",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=str,
        choices=["euclidean", "mahalanobis", "manhattan", "minkowski", "cosine"],
        help="Distance measure for detection",
    )
    parser.add_argument(
        "--reset-on-traditional",
        "-r",
        type=lambda x: x.lower() == "true",
        choices=[True, False],
        help="Reset on traditional change detection",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        type=lambda x: x.lower() == "true",
        choices=[True, False],
        help="Enable or disable visualization (true/false)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for results and visualizations",
    )

    args = parser.parse_args()

    try:
        # If Excel file is provided, just visualize without running detection
        if args.excel_file:
            visualize_from_excel(args.excel_file, args.config, args.log_level)
        # Otherwise, run the full detection pipeline
        elif args.config:
            run_detection(args.config, args.log_level, args)
        else:
            logger.error("Either --config or --excel-file must be provided")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)


def visualize_from_excel(
    excel_file: str, config_file: str = None, log_level: str = "INFO"
) -> None:
    """Generate visualizations from an existing Excel file with detection results.

    Args:
        excel_file: Path to the Excel file with detection results
        config_file: Optional path to configuration file for visualization settings
        log_level: Logging level
    """
    setup_logging(log_level)
    logger.info(f"Visualizing from Excel file: {excel_file}")

    config = None
    if config_file:
        try:
            config = load_config(config_file)
            logger.info(f"Using configuration from {config_file}")
        except Exception as e:
            logger.warning(
                f"Failed to load config file: {str(e)}. Using default settings."
            )

    # Get output directory from Excel file path if not in config
    output_dir = os.path.dirname(excel_file)
    if not output_dir:
        output_dir = "results"

    try:
        # Configure visualization settings
        if config is None:
            config = {
                "output": {
                    "directory": output_dir,
                    "visualization": {"enabled": True, "skip_shap": False},
                    "prefix": "",
                },
                "detection": {
                    "threshold": 50.0,
                    "betting_func_config": {"name": "power"},
                },
                "model": {"type": "multiview"},
            }

        # Get threshold from config
        threshold = config.get("detection", {}).get("threshold", 50.0)

        # Use the new plotting function directly
        plot_martingales_from_csv(
            csv_path=excel_file,
            sheet_name="Aggregate",
            output_dir=output_dir,
            threshold=threshold,
        )

        logger.info(f"Visualizations created in {output_dir}")

    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
