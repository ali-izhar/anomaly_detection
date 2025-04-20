#!/usr/bin/env python

"""Entry point for running the graph change point detection pipeline.

This script handles command-line arguments and runs the detection pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path if needed
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection


logger = logging.getLogger(__name__)


def main(
    config_path,
    prediction=None,
    visualize=None,
    save_csv=None,
):
    """Run the detection pipeline with the given configuration.

    Args:
        config_path: Path to YAML configuration file
        prediction: Whether to generate and use predictions for detection.
                   If None, uses the value from the config file.
        visualize: Whether to create visualizations of the results.
                   If None, uses the value from the config file.
        save_csv: Whether to save results to CSV files.
                 If None, uses the value from the config file.

    Returns:
        Dictionary containing all results
    """
    try:
        # Initialize the pipeline
        pipeline = GraphChangeDetection(config_path)

        # Run the pipeline with specified options
        results = pipeline.run(
            prediction=prediction, visualize=visualize, save_csv=save_csv
        )

        logger.info("Pipeline execution completed successfully")
        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Run graph change detection pipeline")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument(
        "--no-prediction", action="store_true", help="Disable prediction in detection"
    )
    parser.add_argument(
        "--no-visualization", action="store_true", help="Disable result visualization"
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Disable CSV export of results"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Convert command-line flags to parameter values
    prediction = not args.no_prediction if args.no_prediction else None
    visualize = not args.no_visualization if args.no_visualization else None
    save_csv = not args.no_csv if args.no_csv else None

    # Run the pipeline
    try:
        main(
            args.config_path,
            prediction=prediction,
            visualize=visualize,
            save_csv=save_csv,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        sys.exit(1)
