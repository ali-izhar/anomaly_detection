#!/usr/bin/env python

"""
Script to run all experiments for the Tables III and IV.

This script:
1. Loads experiment configurations from YAML files
2. Runs each experiment with appropriate parameters
3. Saves results for further analysis
"""

import argparse
import copy
import logging
import os
import sys
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.run import run_detection
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base_config, exp_config):
    """Merge base configuration with experiment-specific configuration."""
    merged = copy.deepcopy(base_config)

    # Merge dictionaries recursively
    def merge_dict(d1, d2):
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                merge_dict(d1[k], v)
            else:
                d1[k] = v

    merge_dict(merged, exp_config)
    return merged


def setup_output_dir(config, experiment_name):
    """Create a dedicated output directory for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_output_dir = os.path.join(
        config["output"]["directory"], f"{experiment_name}_{timestamp}"
    )
    os.makedirs(exp_output_dir, exist_ok=True)

    # Update config with new output directory
    config["output"]["directory"] = exp_output_dir
    return config


def run_sensitivity_experiments(config_file):
    """Run all parameter sensitivity experiments from Table III."""
    config_data = load_config(config_file)
    base_config = config_data["base_config"]
    results = []

    logger.info(f"Starting sensitivity experiments from {config_file}")

    # Process each experiment group
    for group_name, experiments in config_data["experiments"].items():
        logger.info(f"Running {group_name} experiments")

        for exp in experiments:
            exp_name = exp.pop("name")
            logger.info(f"Running experiment: {exp_name}")

            # Merge base config with experiment config
            merged_config = merge_configs(base_config, exp)

            # Setup output directory
            merged_config = setup_output_dir(merged_config, exp_name)

            # Save the merged config for reproducibility
            config_path = os.path.join(
                merged_config["output"]["directory"], "config.yaml"
            )
            with open(config_path, "w") as f:
                yaml.dump(merged_config, f)

            # Run the experiment and collect results
            try:
                result = run_detection(config_dict=merged_config)

                # Extract key metrics for the sensitivity analysis
                metrics = {
                    "name": exp_name,
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                }

                results.append(metrics)
                logger.info(f"Experiment {exp_name} completed successfully")
            except Exception as e:
                logger.error(f"Error running experiment {exp_name}: {str(e)}")

    # Save the combined results
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(config_data["base_config"]["output"]["directory"])
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "sensitivity_results.csv"), index=False)

    logger.info(f"All sensitivity experiments completed. Results saved to {output_dir}")
    return results_df


def run_comparison_experiments(config_file):
    """Run all comparison experiments from Table IV."""
    config_data = load_config(config_file)
    base_config = config_data["base_config"]
    results = []

    logger.info(f"Starting comparison experiments from {config_file}")

    # Process main experiments
    for network_type, experiments in config_data["experiments"].items():
        logger.info(f"Running {network_type} experiments")

        for exp in experiments:
            exp_name = exp.pop("name")
            logger.info(f"Running experiment: {exp_name}")

            # Merge base config with experiment config
            merged_config = merge_configs(base_config, exp)

            # Setup output directory
            merged_config = setup_output_dir(merged_config, exp_name)

            # Save the merged config for reproducibility
            config_path = os.path.join(
                merged_config["output"]["directory"], "config.yaml"
            )
            with open(config_path, "w") as f:
                yaml.dump(merged_config, f)

            # Run the experiment and collect results
            try:
                result = run_detection(config_dict=merged_config)

                # Extract key metrics for the comparison
                metrics = {
                    "name": exp_name,
                    "network": network_type,
                    "change_scenario": merged_config.get("change_scenario", "unknown"),
                    "detection_method": merged_config["detection"]["method"],
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }

                results.append(metrics)
                logger.info(f"Experiment {exp_name} completed successfully")
            except Exception as e:
                logger.error(f"Error running experiment {exp_name}: {str(e)}")

    # Process threshold experiments
    for exp in config_data.get("threshold_experiments", []):
        exp_name = exp.pop("name")
        logger.info(f"Running threshold experiment: {exp_name}")

        # Merge base config with experiment config
        merged_config = merge_configs(base_config, exp)

        # Setup output directory
        merged_config = setup_output_dir(merged_config, exp_name)

        # Run the experiment and collect results
        try:
            result = run_detection(config_dict=merged_config)

            # Extract key metrics
            metrics = {
                "name": exp_name,
                "network": exp.get("network", "unknown"),
                "threshold": merged_config["detection"]["threshold"],
                "TPR": result.get("detection_rate", 0),
                "FPR": result.get("false_positive_rate", 0),
                "ADD": result.get("average_detection_delay", 0),
            }

            results.append(metrics)
            logger.info(f"Threshold experiment {exp_name} completed successfully")
        except Exception as e:
            logger.error(f"Error running threshold experiment {exp_name}: {str(e)}")

    # Save the combined results
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(config_data["base_config"]["output"]["directory"])
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)

    logger.info(f"All comparison experiments completed. Results saved to {output_dir}")
    return results_df


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run change point detection experiments"
    )
    parser.add_argument(
        "-s",
        "--sensitivity",
        action="store_true",
        help="Run sensitivity experiments (Table III)",
    )
    parser.add_argument(
        "-c",
        "--comparison",
        action="store_true",
        help="Run comparison experiments (Table IV)",
    )
    parser.add_argument("-a", "--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Run requested experiments
    if args.all or args.sensitivity:
        logger.info("Running sensitivity experiments (Table III)")
        sensitivity_config = os.path.join(
            "src", "configs", "table_iii_sensitivity.yaml"
        )
        run_sensitivity_experiments(sensitivity_config)

    if args.all or args.comparison:
        logger.info("Running comparison experiments (Table IV)")
        comparison_config = os.path.join("src", "configs", "table_iv_comparison.yaml")
        run_comparison_experiments(comparison_config)

    if not (args.all or args.sensitivity or args.comparison):
        parser.print_help()


if __name__ == "__main__":
    main()
