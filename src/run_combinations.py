#!/usr/bin/env python3
# src/run_combinations.py

"""
Script to run the GraphChangeDetection algorithm with multiple combinations of
betting functions and distance measures on the same data.
"""

import os
import sys
import yaml
import time
import logging
from pathlib import Path
from itertools import product
import copy
import random

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import main as run_algorithm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# set constant seed
random.seed(42)


def run_combinations(base_config_path, output_dir=None):
    """
    Run the algorithm with multiple combinations of betting functions and distance measures.

    Args:
        base_config_path: Path to the base configuration YAML file
        output_dir: Directory to store results (if None, will use timestamp)
    """
    # Define the combinations to test
    betting_functions = ["power", "mixture", "beta"]
    distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]

    # Load the base configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Create timestamped output directory if not provided
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            base_config["output"]["directory"], f"combinations_{timestamp}"
        )

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Track results
    results = {}

    # Run all combinations
    total_combinations = len(betting_functions) * len(distance_measures)
    current_combination = 0

    for betting_func, distance in product(betting_functions, distance_measures):
        current_combination += 1
        logger.info(
            f"Running combination {current_combination}/{total_combinations}: "
            f"betting_func={betting_func}, distance={distance}"
        )

        # Create a deep copy of the base configuration
        config = copy.deepcopy(base_config)

        # Modify the configuration for this combination
        config["detection"]["betting_func_config"]["name"] = betting_func
        config["detection"]["distance"]["measure"] = distance

        # Create a subfolder for this combination
        combination_dir = os.path.join(output_dir, f"{betting_func}_{distance}")
        config["output"]["directory"] = combination_dir
        os.makedirs(combination_dir, exist_ok=True)

        # Create a temporary configuration file
        temp_config_path = os.path.join(combination_dir, "config.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        # Run the algorithm with this configuration
        try:
            logger.info(
                f"Starting algorithm with betting_func={betting_func}, distance={distance}"
            )
            result = run_algorithm(temp_config_path)
            results[f"{betting_func}_{distance}"] = result
            logger.info(
                f"Successfully completed combination: betting_func={betting_func}, distance={distance}"
            )
        except Exception as e:
            logger.error(
                f"Error running combination betting_func={betting_func}, distance={distance}: {str(e)}"
            )

    # Log completion
    logger.info(f"Completed all {total_combinations} combinations")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run algorithm with multiple combinations"
    )
    parser.add_argument("config_path", help="Path to base configuration YAML file")
    parser.add_argument("--output-dir", help="Output directory for all results")

    args = parser.parse_args()

    run_combinations(args.config_path, args.output_dir)
