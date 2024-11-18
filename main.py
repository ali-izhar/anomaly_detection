# main.py

#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path
from typing import Any

from src.utils.log_handling import setup_root_logger, get_logger
from config.config import load_config


def setup_paths() -> None:
    """Add project root to system path for imports."""
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


def run_experiment(experiment: str, config_path: str) -> None:
    """Run specified change detection experiment."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load experiment-specific config
        config = load_config(config_path)

        # Setup directories
        _setup_directories(config)

        # Setup logging for this experiment
        setup_root_logger(config, experiment)
        logger = get_logger(__name__)

        logger.info(f"Starting {experiment} experiment")

        if experiment == "linear":
            from tests.linear_models import main

            main(config_path)

        elif experiment == "reality":
            from tests.reality_mining_data import RealityMiningDataPipeline

            pipeline = RealityMiningDataPipeline(config_path)
            pipeline.run()

        elif experiment == "synthetic":
            from tests.synthetic_data import run_synthetic_pipeline

            run_synthetic_pipeline(config_path)

        else:
            raise ValueError(
                f"Unknown experiment: {experiment}. "
                "Must be one of: linear, reality, synthetic"
            )

        logger.info(f"Successfully completed {experiment} experiment")

    except Exception as e:
        logger.error(f"Error running {experiment} experiment: {str(e)}")
        raise


def _setup_directories(config: Any) -> None:
    """Create necessary directories based on the config."""
    Path(config.paths.output.dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging.file).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run change point detection experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "experiment",
        choices=["linear", "reality", "synthetic"],
        help=(
            "Experiment to run:\n"
            "  linear    - Linear model experiments\n"
            "  reality   - Reality Mining dataset analysis\n"
            "  synthetic - Synthetic data experiments"
        ),
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    try:
        # Parse arguments
        args = parse_args()

        # Setup environment
        setup_paths()

        # Run experiment (logging setup moved inside run_experiment)
        run_experiment(args.experiment, args.config)

    except Exception as e:
        # Basic logging for startup errors
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
