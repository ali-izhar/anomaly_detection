#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis Runner for Network Change Detection

This script conducts parameter sensitivity sweeps across synthetic networks,
saving only the raw detection_results.xlsx files. Metrics will be computed later.
"""

import argparse
import logging
import os
import sys
import yaml
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_base_config() -> Dict[str, Any]:
    """Return the base configuration for all experiments."""
    return {
        "execution": {
            "enable_prediction": True,
            "enable_visualization": False,
            "save_csv": True,
        },
        "trials": {
            "n_trials": 1,
            "random_seeds": [42],
        },
        "detection": {
            "method": "martingale",
            "threshold": 50.0,
            "batch_size": 1000,
            "reset": True,
            "reset_on_traditional": True,
            "max_window": None,
            "prediction_horizon": 5,
            "enable_pvalue_dampening": False,
            "cooldown_period": 30,
        },
        "features": [
            "mean_degree",
            "density",
            "mean_clustering",
            "mean_betweenness",
            "mean_eigenvector",
            "mean_closeness",
            "max_singular_value",
            "min_nonzero_laplacian",
        ],
        "model": {
            "type": "multiview",
            "predictor": {
                "type": "graph",
                "config": {
                    "alpha": 0.8,
                    "gamma": 0.5,
                    "beta_init": 0.5,
                    "enforce_connectivity": True,
                    "threshold": 0.5,
                    "n_history": 10,
                },
            },
        },
        "output": {
            "save_predictions": True,
            "save_features": True,
            "save_martingales": True,
            "save_results": True,
            "save_detection_data": True,
            "results_filename": "detection_results.xlsx",
        },
    }


def create_experiment_config(network: str, **params) -> Dict[str, Any]:
    """Create experiment configuration for given parameters."""
    config = get_base_config()

    # Set network
    config["model"]["network"] = network

    # Set betting function
    if "epsilon" in params:
        config["detection"]["betting_func_config"] = {
            "name": "power",
            "power": {"epsilon": params["epsilon"]},
        }
    elif "beta_a" in params:
        config["detection"]["betting_func_config"] = {
            "name": "beta",
            "beta": {"a": params["beta_a"], "b": 1.5},
        }
    elif "mixture" in params and params["mixture"]:
        config["detection"]["betting_func_config"] = {
            "name": "mixture",
            "mixture": {"epsilons": [0.7, 0.8, 0.9]},
        }

    # Set distance measure
    if "distance" in params:
        config["detection"]["distance"] = {"measure": params["distance"], "p": 2.0}

    # Set threshold
    if "threshold" in params:
        config["detection"]["threshold"] = params["threshold"]

    return config


def run_single_experiment(
    exp_name: str, config: Dict[str, Any], output_dir: str
) -> Dict[str, Any]:
    """Run a single experiment and save results."""
    start_time = datetime.now()

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Set output directory in config
        config["output"]["directory"] = output_dir

        # Save config
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run experiment
        detector = GraphChangeDetection(config_dict=config)
        result = detector.run()

        # Verify detection results file exists
        detection_file = os.path.join(output_dir, "detection_results.xlsx")
        if not os.path.exists(detection_file):
            raise FileNotFoundError(f"Detection results file missing: {detection_file}")

        # Verify required files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file missing: {config_path}")
        if not os.path.exists(detection_file):
            raise FileNotFoundError(f"Detection file missing: {detection_file}")

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"✓ {exp_name} completed successfully ({duration:.1f}s)")

        return {
            "success": True,
            "exp_name": exp_name,
            "duration": duration,
            "output_dir": output_dir,
        }

    except Exception as e:
        # Clean up on failure
        if os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)

        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"✗ {exp_name} failed after {duration:.1f}s: {str(e)}"
        logger.error(error_msg)

        return {
            "success": False,
            "exp_name": exp_name,
            "error": str(e),
            "duration": duration,
        }


def generate_sensitivity_experiments() -> List[Dict[str, Any]]:
    """Generate all sensitivity analysis experiment configurations."""
    experiments = []
    base_output = "results/sensitivity_analysis"

    networks = ["sbm", "er", "ba", "ws"]
    distances = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
    epsilons = [0.2, 0.5, 0.7, 0.9]
    beta_as = [0.3, 0.5, 0.7]
    thresholds = [20.0, 50.0, 100.0]

    # Power betting experiments
    for network in networks:
        for distance in distances:
            for epsilon in epsilons:
                exp_name = f"{network}_power_{epsilon}_{distance}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                output_dir = f"{base_output}/{exp_name}_{timestamp}"

                config = create_experiment_config(
                    network=network, epsilon=epsilon, distance=distance
                )

                experiments.append(
                    {
                        "exp_name": exp_name,
                        "config": config,
                        "output_dir": output_dir,
                        "group": "power_betting",
                    }
                )

    # Mixture betting experiments
    for network in networks:
        for distance in distances:
            exp_name = f"{network}_mixture_{distance}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_dir = f"{base_output}/{exp_name}_{timestamp}"

            config = create_experiment_config(
                network=network, mixture=True, distance=distance
            )

            experiments.append(
                {
                    "exp_name": exp_name,
                    "config": config,
                    "output_dir": output_dir,
                    "group": "mixture_betting",
                }
            )

    # Beta betting experiments
    for network in networks:
        for distance in distances:
            for beta_a in beta_as:
                exp_name = f"{network}_beta_{beta_a}_{distance}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                output_dir = f"{base_output}/{exp_name}_{timestamp}"

                config = create_experiment_config(
                    network=network, beta_a=beta_a, distance=distance
                )

                experiments.append(
                    {
                        "exp_name": exp_name,
                        "config": config,
                        "output_dir": output_dir,
                        "group": "beta_betting",
                    }
                )

    # Threshold experiments
    for network in networks:
        for distance in distances:
            for threshold in thresholds:
                exp_name = f"{network}_threshold_{int(threshold)}_{distance}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                output_dir = f"{base_output}/{exp_name}_{timestamp}"

                config = create_experiment_config(
                    network=network,
                    epsilon=0.5,  # Default
                    distance=distance,
                    threshold=threshold,
                )

                experiments.append(
                    {
                        "exp_name": exp_name,
                        "config": config,
                        "output_dir": output_dir,
                        "group": "threshold_analysis",
                    }
                )

    return experiments


def run_experiment_batch(
    experiments: List[Dict[str, Any]], max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Run a batch of experiments in parallel."""
    logger.info(f"Starting {len(experiments)} experiments with {max_workers} workers")

    results = []
    completed = 0
    failed = 0
    start_time = datetime.now()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                run_single_experiment, exp["exp_name"], exp["config"], exp["output_dir"]
            ): exp
            for exp in experiments
        }

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_exp)):
            try:
                result = future.result()
                results.append(result)

                if result["success"]:
                    completed += 1
                else:
                    failed += 1

                # Progress update
                elapsed = (datetime.now() - start_time).total_seconds() / 60.0
                remaining = (
                    (elapsed / (i + 1)) * (len(experiments) - i - 1) if i > 0 else 0
                )

                logger.info(
                    f"[{i+1}/{len(experiments)}] {result['exp_name']} - "
                    f"Completed: {completed}, Failed: {failed}, "
                    f"Elapsed: {elapsed:.1f}min, Remaining: ~{remaining:.1f}min"
                )

            except Exception as e:
                failed += 1
                logger.error(f"Exception in experiment: {str(e)}")
                results.append({"success": False, "error": str(e)})

    # Summary
    total_time = (datetime.now() - start_time).total_seconds() / 60.0
    logger.info(
        f"Batch completed: {completed} succeeded, {failed} failed, {total_time:.1f} minutes total"
    )

    return results


def validate_results(base_dir: str) -> Dict[str, int]:
    """Validate experiment results in directory."""
    if not os.path.exists(base_dir):
        return {"total_dirs": 0, "valid_dirs": 0, "config_only": 0, "invalid": 0}

    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    valid = 0
    config_only = 0
    invalid = 0

    for dirname in dirs:
        dir_path = os.path.join(base_dir, dirname)
        has_config = os.path.exists(os.path.join(dir_path, "config.yaml"))
        has_results = os.path.exists(os.path.join(dir_path, "detection_results.xlsx"))

        if has_config and has_results:
            valid += 1
        elif has_config and not has_results:
            config_only += 1
        else:
            invalid += 1

    return {
        "total_dirs": len(dirs),
        "valid_dirs": valid,
        "config_only": config_only,
        "invalid": invalid,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis experiments"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean results directory first"
    )

    args = parser.parse_args()

    if args.clean:
        import shutil

        results_dir = "results/sensitivity_analysis"
        if os.path.exists(results_dir):
            logger.info(f"Cleaning {results_dir}")
            shutil.rmtree(results_dir)

    logger.info("=" * 70)
    logger.info("RUNNING PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    experiments = generate_sensitivity_experiments()
    logger.info(f"Generated {len(experiments)} sensitivity experiments")

    # Expected counts
    expected_power = 4 * 4 * 4  # networks * distances * epsilons = 64
    expected_mixture = 4 * 4  # networks * distances = 16
    expected_beta = 4 * 4 * 3  # networks * distances * beta_as = 48
    expected_threshold = 4 * 4 * 3  # networks * distances * thresholds = 48
    expected_total = (
        expected_power + expected_mixture + expected_beta + expected_threshold
    )

    logger.info(
        f"Expected: {expected_total} total ({expected_power} power + {expected_mixture} mixture + {expected_beta} beta + {expected_threshold} threshold)"
    )

    if len(experiments) != expected_total:
        logger.error(
            f"Experiment count mismatch! Expected {expected_total}, got {len(experiments)}"
        )
        return

    results = run_experiment_batch(experiments, args.workers)

    # Validate results
    validation = validate_results("results/sensitivity_analysis")
    logger.info(
        f"Validation: {validation['valid_dirs']}/{validation['total_dirs']} valid directories"
    )

    if validation["valid_dirs"] == expected_total:
        logger.info("✅ SUCCESS: All experiments completed successfully!")
    else:
        logger.warning(
            f"⚠️  Only {validation['valid_dirs']}/{expected_total} experiments completed successfully"
        )


if __name__ == "__main__":
    main()
