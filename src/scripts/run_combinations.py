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
import glob
import numpy as np
import pickle

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection
from src.changepoint.detector import ChangePointDetector, DetectorConfig
from src.algorithm import main as run_algorithm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# set constant seed
SEED = 77


def run_combinations(base_config_path, output_dir=None):
    """
    Run the algorithm with multiple combinations of betting functions and distance measures
    on the same data to ensure consistent change points.

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

    # Step 1: Generate the data once using a single GraphChangeDetection instance
    logger.info("Generating data once to use across all combinations...")

    # Create data directory
    data_dir = os.path.join(output_dir, "shared_data")
    os.makedirs(data_dir, exist_ok=True)

    # Initialize the algorithm with the base configuration
    algorithm = GraphChangeDetection(base_config_path)

    # Initialize the graph generator
    generator = algorithm._init_generator()

    # Generate the graph sequence with fixed random seed for reproducibility
    np.random.seed(SEED)  # Fixed seed for reproducible data generation
    random.seed(SEED)
    sequence_result = algorithm._generate_sequence(generator)

    # Extract graph data and true change points
    graphs = sequence_result["graphs"]
    true_change_points = sequence_result["change_points"]

    # Save the shared data
    data_path = os.path.join(data_dir, "graph_sequence.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({"graphs": graphs, "true_change_points": true_change_points}, f)

    # Extract features to be used by all detectors
    features_numeric, features_raw = algorithm._extract_features(graphs)

    # Save the features
    features_path = os.path.join(data_dir, "features.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(
            {"features_numeric": features_numeric, "features_raw": features_raw}, f
        )

    logger.info(f"Generated graph sequence with change points at: {true_change_points}")
    logger.info(f"Extracted features with shape: {features_numeric.shape}")
    logger.info(f"Shared data saved to: {data_dir}")

    # Step 2: Run all combinations using the same data
    total_combinations = len(betting_functions) * len(distance_measures)
    current_combination = 0

    for betting_func, distance in product(betting_functions, distance_measures):
        current_combination += 1
        logger.info(
            f"Running combination {current_combination}/{total_combinations}: "
            f"betting_func={betting_func}, distance={distance}"
        )

        # Create a subfolder for this combination
        combination_dir = os.path.join(output_dir, f"{betting_func}_{distance}")
        os.makedirs(combination_dir, exist_ok=True)

        # Create a deep copy of the base configuration for this combination
        config = copy.deepcopy(base_config)

        # Update the configuration
        config["detection"]["betting_func_config"]["name"] = betting_func
        config["detection"]["distance"]["measure"] = distance
        config["output"]["directory"] = combination_dir

        # Save the configuration
        config_path = os.path.join(combination_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Initialize a detector with the specific combination
        detector_config = DetectorConfig(
            method=config["model"]["type"],
            threshold=config["detection"]["threshold"],
            history_size=config["model"]["predictor"]["config"]["n_history"],
            batch_size=config["detection"]["batch_size"],
            reset=config["detection"]["reset"],
            max_window=config["detection"]["max_window"],
            betting_func_config={
                "name": betting_func,
                "params": config["detection"]["betting_func_config"].get(
                    betting_func, {}
                ),
            },
            distance_measure=distance,
            distance_p=config["detection"]["distance"]["p"],
            random_state=SEED,  # Use consistent random state
        )

        detector = ChangePointDetector(detector_config)

        try:
            # Run detection on the shared features
            detection_result = detector.run(data=features_numeric)

            if detection_result is None:
                raise RuntimeError(
                    f"Detection failed for combination {betting_func}_{distance}"
                )

            # Save results for this combination
            # Create algorithm instance for output management
            algorithm_instance = GraphChangeDetection(config_path)

            # Manually construct results for output
            trial_results = {
                "individual_trials": [detection_result],
                "aggregated": detection_result,  # For single trial, aggregated is the same
                "random_seeds": [SEED],
            }

            # Create output manager and export to CSV/Excel
            from src.utils.output_manager import OutputManager

            output_manager = OutputManager(combination_dir, config)
            output_manager.export_to_csv(
                detection_result,
                true_change_points,
                individual_trials=[detection_result],
            )

            # Store the results
            results[f"{betting_func}_{distance}"] = detection_result

            logger.info(
                f"Successfully completed combination: betting_func={betting_func}, distance={distance}"
            )

        except Exception as e:
            logger.error(
                f"Error running combination betting_func={betting_func}, distance={distance}: {str(e)}"
            )
            import traceback

            logger.error(traceback.format_exc())

    # Log completion
    logger.info(f"Completed all {total_combinations} combinations on the same dataset")
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
