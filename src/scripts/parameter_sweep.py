#!/usr/bin/env python3
"""Parameter sweep script for graph change detection framework.

This script performs a systematic evaluation of the graph change detection algorithm
across different parameter combinations for each network type.
"""

import argparse
import copy
import itertools
import logging
import os
import sys
import time
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random

# Add parent directory to path to allow imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection
from src.utils.metrics_utils import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_config(base_config, param_dict):
    """Update configuration with parameter values."""
    config = copy.deepcopy(base_config)

    # Apply parameter values
    if "threshold" in param_dict:
        config["detection"]["threshold"] = param_dict["threshold"]

    if "window" in param_dict:
        config["model"]["predictor"]["config"]["n_history"] = param_dict["window"]

    if "horizon" in param_dict:
        config["detection"]["prediction_horizon"] = param_dict["horizon"]

    if "epsilon" in param_dict:
        if config["detection"]["betting_func_config"]["name"] == "power":
            config["detection"]["betting_func_config"]["power"]["epsilon"] = param_dict[
                "epsilon"
            ]
        elif config["detection"]["betting_func_config"]["name"] == "mixture":
            # For mixture, update the middle epsilon and adjust others accordingly
            mid_val = param_dict["epsilon"]
            config["detection"]["betting_func_config"]["mixture"]["epsilons"] = [
                max(0.1, mid_val - 0.1),
                mid_val,
                min(0.9, mid_val + 0.1),
            ]

    if "betting_func" in param_dict:
        config["detection"]["betting_func_config"]["name"] = param_dict["betting_func"]

    if "distance" in param_dict:
        config["detection"]["distance"]["measure"] = param_dict["distance"]

    if "network" in param_dict:
        config["model"]["network"] = param_dict["network"]

    return config


def run_parameter_sweep(base_config_path, output_dir, networks=None, n_trials=5):
    """Run parameter sweep across specified parameter combinations.

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save results
        networks: List of network types to evaluate (default: all)
        n_trials: Number of trials for each parameter combination
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base configuration
    base_config = load_config(base_config_path)

    # Set number of trials
    base_config["trials"]["n_trials"] = n_trials

    # Define parameter ranges to sweep
    param_ranges = {
        "network": networks or ["sbm", "ba", "er", "ws"],
        "threshold": [20, 50, 70, 100],
        "window": [5, 10],
        "horizon": [1, 3, 5, 10],
        "epsilon": [0.2, 0.5, 0.7, 0.9],
        "betting_func": ["power", "mixture", "beta"],
        "distance": ["euclidean", "mahalanobis", "cosine", "chebyshev"],
    }

    # Store all results
    all_results = []

    # Log parameter space size
    total_combinations = (
        len(param_ranges["network"])
        * len(param_ranges["threshold"])
        * len(param_ranges["window"])
        * len(param_ranges["horizon"])
        * len(param_ranges["epsilon"])
        * len(param_ranges["betting_func"])
        * len(param_ranges["distance"])
    )

    logger.info(f"Parameter space contains {total_combinations} combinations")

    # Generate all possible parameter combinations (full factorial design)
    param_combinations = list(
        itertools.product(
            param_ranges["network"],
            param_ranges["threshold"],
            param_ranges["window"],
            param_ranges["horizon"],
            param_ranges["epsilon"],
            param_ranges["betting_func"],
            param_ranges["distance"],
        )
    )

    logger.info(
        f"Running exhaustive sweep with {len(param_combinations)} parameter combinations"
    )
    logger.info(f"This will test all possible combinations of parameters")

    # Run parameter sweep
    for combo in tqdm(param_combinations, desc="Parameter combinations"):
        network, threshold, window, horizon, epsilon, betting_func, distance = combo

        # Create parameter dictionary
        param_dict = {
            "network": network,
            "threshold": threshold,
            "window": window,
            "horizon": horizon,
            "epsilon": epsilon,
            "betting_func": betting_func,
            "distance": distance,
        }

        # Update configuration
        config = update_config(base_config, param_dict)

        # Set a unique output directory for this combination
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        param_str = f"{network}_t{threshold}_w{window}_h{horizon}_e{epsilon}_{betting_func}_{distance}"
        config["output"]["directory"] = os.path.join(
            output_dir, f"{param_str}_{timestamp}"
        )

        try:
            # Run detection
            detector = GraphChangeDetection(config_dict=config)
            result = detector.run()

            # Calculate metrics
            if result and "aggregated" in result:
                metrics = calculate_performance_metrics(
                    result["aggregated"], result["true_change_points"]
                )

                # Store results with parameters
                result_entry = {**param_dict, **metrics, "timestamp": timestamp}

                all_results.append(result_entry)

                # Log key metrics
                logger.info(
                    f"Network: {network}, Threshold: {threshold}, "
                    f"TPR: {metrics.get('tpr', 'N/A'):.3f}, "
                    f"FPR: {metrics.get('fpr', 'N/A'):.3f}, "
                    f"Delay: {metrics.get('avg_delay', 'N/A'):.3f}"
                )

        except Exception as e:
            logger.error(f"Error running combination {param_str}: {str(e)}")

    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "parameter_sweep_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved all results to {results_path}")

    # Generate summary report
    generate_summary_report(results_df, output_dir)

    return results_df


def calculate_performance_metrics(detection_result, true_change_points):
    """Calculate performance metrics from detection results."""
    if not detection_result:
        return {"tpr": 0, "fpr": 0, "avg_delay": 0, "auc": 0}

    # Get detected change points
    trad_cps = detection_result.get("traditional_change_points", [])
    horizon_cps = detection_result.get("horizon_change_points", [])

    # Total time steps
    total_steps = detection_result.get("total_steps", 0)

    # Calculate metrics
    metrics = {}

    # For traditional martingale
    trad_metrics = calculate_metrics(
        trad_cps,
        true_change_points,
        total_steps,
        max_delay=15,  # Consider detection within 15 steps as true positive
    )

    # Add traditional metrics
    metrics.update(
        {
            "tpr": trad_metrics["tpr"],
            "fpr": trad_metrics["fpr"],
            "avg_delay": trad_metrics["avg_delay"],
            "auc": trad_metrics.get("auc", 0),
        }
    )

    # For horizon martingale (if available)
    if horizon_cps:
        horizon_metrics = calculate_metrics(
            horizon_cps, true_change_points, total_steps, max_delay=15
        )

        # Add horizon-specific metrics
        metrics.update(
            {
                "horizon_tpr": horizon_metrics["tpr"],
                "horizon_fpr": horizon_metrics["fpr"],
                "horizon_avg_delay": horizon_metrics["avg_delay"],
                "horizon_auc": horizon_metrics.get("auc", 0),
            }
        )

    return metrics


def generate_summary_report(results_df, output_dir):
    """Generate summary report with key findings from parameter sweep."""
    if results_df.empty:
        logger.warning("No results to generate summary report")
        return

    report_path = os.path.join(output_dir, "parameter_sweep_summary.txt")

    with open(report_path, "w") as f:
        f.write("PARAMETER SWEEP SUMMARY REPORT\n")
        f.write("=============================\n\n")

        # Overall best parameter combinations
        f.write("Best Parameter Combinations (by TPR):\n")
        best_by_tpr = results_df.sort_values("tpr", ascending=False).head(5)
        f.write(best_by_tpr.to_string(index=False) + "\n\n")

        f.write("Best Parameter Combinations (by Detection Delay):\n")
        best_by_delay = results_df.sort_values("avg_delay").head(5)
        f.write(best_by_delay.to_string(index=False) + "\n\n")

        # Network-specific analysis
        f.write("Network-Specific Analysis:\n")
        for network in results_df["network"].unique():
            network_df = results_df[results_df["network"] == network]

            f.write(f"\n{network.upper()} Network:\n")
            f.write(f"  Avg TPR: {network_df['tpr'].mean():.3f}\n")
            f.write(f"  Avg FPR: {network_df['fpr'].mean():.3f}\n")
            f.write(f"  Avg Detection Delay: {network_df['avg_delay'].mean():.3f}\n")

            # Best parameters for this network
            best_config = network_df.loc[network_df["tpr"].idxmax()]
            f.write(
                f"  Best Parameters: Threshold={best_config['threshold']}, "
                f"Window={best_config['window']}, Horizon={best_config['horizon']}, "
                f"Epsilon={best_config['epsilon']}, "
                f"Betting={best_config['betting_func']}, "
                f"Distance={best_config['distance']}\n"
            )

        # Parameter-specific analysis
        for param in [
            "threshold",
            "window",
            "horizon",
            "epsilon",
            "betting_func",
            "distance",
        ]:
            f.write(f"\nEffect of {param}:\n")
            param_effect = (
                results_df.groupby(param)[["tpr", "fpr", "avg_delay"]]
                .mean()
                .reset_index()
            )
            f.write(param_effect.to_string(index=False) + "\n")

    logger.info(f"Generated summary report at {report_path}")

    # Also generate more detailed CSV summaries
    param_summaries = {}
    for param in [
        "threshold",
        "window",
        "horizon",
        "epsilon",
        "betting_func",
        "distance",
    ]:
        param_summary = (
            results_df.groupby([param, "network"])[["tpr", "fpr", "avg_delay"]]
            .mean()
            .reset_index()
        )
        param_summary_path = os.path.join(output_dir, f"{param}_effect_by_network.csv")
        param_summary.to_csv(param_summary_path, index=False)
        param_summaries[param] = param_summary

    return param_summaries


def main():
    """Main entry point for parameter sweep."""
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for graph change detection."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/configs/algorithm.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/parameter_sweep",
        help="Output directory for results",
    )
    parser.add_argument(
        "-n",
        "--networks",
        type=str,
        nargs="+",
        choices=["sbm", "ba", "er", "ws"],
        default=["sbm", "ba", "er", "ws"],
        help="Network types to evaluate",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=5,
        help="Number of trials per parameter combination",
    )

    args = parser.parse_args()

    logger.info("Starting parameter sweep")
    logger.info(f"Base config: {args.config}")
    logger.info(f"Networks: {args.networks}")
    logger.info(f"Trials per combination: {args.trials}")

    # Run parameter sweep
    results = run_parameter_sweep(
        args.config, args.output, networks=args.networks, n_trials=args.trials
    )

    logger.info("Parameter sweep completed")


if __name__ == "__main__":
    main()
