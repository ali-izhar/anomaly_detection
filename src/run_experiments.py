#!/usr/bin/env python

"""
Script to run all experiments for the Tables III and IV using parameter sweeping.

This script:
1. Defines parameter spaces for sensitivity and comparison experiments
2. Generates all parameter combinations
3. Runs each experiment configuration
4. Collects and aggregates results for analysis
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import GraphChangeDetection

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


def setup_output_dir(base_dir: str, experiment_name: str) -> str:
    """Create a dedicated output directory for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_output_dir, exist_ok=True)
    return exp_output_dir


def get_base_config() -> Dict[str, Any]:
    """Return the base configuration for all experiments."""
    return {
        "execution": {
            "enable_prediction": True,
            "enable_visualization": False,
            "save_csv": True,
        },
        "trials": {
            "n_trials": 10,
            "random_seeds": [42, 142, 242, 342, 442, 542, 642, 742, 842, 1042],
        },
        "detection": {
            "method": "martingale",
            "batch_size": 1000,
            "reset": True,
            "reset_on_traditional": True,
            "max_window": None,
            "prediction_horizon": 5,
            "enable_pvalue_dampening": True,
            "cooldown_period": 30,
            "distance": {"measure": "euclidean", "p": 2.0},  # Adding default p value
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
                    "n_history": 10,
                    "enforce_connectivity": True,
                    "threshold": 0.5,
                },
            },
        },
        "output": {
            "save_predictions": True,
            "save_features": True,
            "save_martingales": True,
        },
    }


def run_sensitivity_experiments() -> pd.DataFrame:
    """Run parameter sensitivity experiments for Table III."""
    # Base output directory
    base_output_dir = "results/sensitivity_analysis"
    os.makedirs(base_output_dir, exist_ok=True)

    # Parameter spaces for the sweep
    network_types = ["sbm", "er", "ba", "nws"]
    distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
    threshold_values = [20.0, 50.0, 100.0]
    epsilon_values = [0.2, 0.5, 0.7, 0.9]
    a_values = [0.3, 0.5, 0.7]

    # Calculate and print total number of experiments
    total_power_exps = len(network_types) * len(epsilon_values) * len(distance_measures)
    total_mixture_exps = len(network_types) * len(distance_measures)
    total_beta_exps = len(network_types) * len(a_values) * len(distance_measures)
    total_threshold_exps = (
        len(network_types) * len(threshold_values) * len(distance_measures)
    )
    total_experiments = (
        total_power_exps + total_mixture_exps + total_beta_exps + total_threshold_exps
    )

    logger.info(f"===== SENSITIVITY EXPERIMENTS =====")
    logger.info(f"Total experiments to run: {total_experiments}")
    logger.info(f"- Power betting: {total_power_exps} experiments")
    logger.info(f"- Mixture betting: {total_mixture_exps} experiments")
    logger.info(f"- Beta betting: {total_beta_exps} experiments")
    logger.info(f"- Threshold analysis: {total_threshold_exps} experiments")
    logger.info(f"===================================\n")

    all_results = []
    current_exp = 0

    # Run Power Betting Function experiments
    logger.info("Running Power Betting Function experiments")

    for network in network_types:
        for epsilon in epsilon_values:
            for distance in distance_measures:
                current_exp += 1
                exp_name = f"{network}_power_{epsilon}_{distance}"
                logger.info(
                    f"[{current_exp}/{total_experiments}] Running experiment: {exp_name}"
                )

                config = get_base_config()
                config["model"]["network"] = network
                config["detection"]["method"] = "martingale"
                config["detection"]["threshold"] = 50.0  # Default threshold
                config["detection"]["betting_func_config"] = {
                    "name": "power",
                    "power": {"epsilon": epsilon},
                }
                config["detection"]["distance"] = {"measure": distance, "p": 2.0}

                # Set up output directory
                output_dir = setup_output_dir(base_output_dir, exp_name)
                config["output"]["directory"] = output_dir

                # Save the config for reproducibility
                with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                    yaml.dump(config, f)

                # Run the experiment and collect results
                try:
                    # Initialize detector with the config dictionary
                    detector = GraphChangeDetection(config_dict=config)
                    result = detector.run()

                    metrics = {
                        "name": exp_name,
                        "group": "power_betting",
                        "network": network,
                        "epsilon": epsilon,
                        "distance": distance,
                        "threshold": 50.0,
                        "TPR": result.get("detection_rate", 0),
                        "FPR": result.get("false_positive_rate", 0),
                        "ADD": result.get("average_detection_delay", 0),
                        "TPR_horizon": result.get("horizon_detection_rate", 0),
                        "ADD_horizon": result.get("horizon_average_delay", 0),
                        "delay_reduction": result.get("delay_reduction_percent", 0),
                    }
                    all_results.append(metrics)
                    logger.info(f"Experiment {exp_name} completed successfully")
                except Exception as e:
                    logger.error(f"Error running experiment {exp_name}: {str(e)}")

    # Run Mixture Betting Function experiments
    logger.info("Running Mixture Betting Function experiments")

    for network in network_types:
        for distance in distance_measures:
            current_exp += 1
            exp_name = f"{network}_mixture_{distance}"
            logger.info(
                f"[{current_exp}/{total_experiments}] Running experiment: {exp_name}"
            )

            config = get_base_config()
            config["model"]["network"] = network
            config["detection"]["method"] = "martingale"
            config["detection"]["threshold"] = 50.0
            config["detection"]["betting_func_config"] = {
                "name": "mixture",
                "mixture": {"epsilons": [0.7, 0.8, 0.9]},
            }
            config["detection"]["distance"] = {"measure": distance, "p": 2.0}

            # Set up output directory
            output_dir = setup_output_dir(base_output_dir, exp_name)
            config["output"]["directory"] = output_dir

            # Save the config for reproducibility
            with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)

            # Run the experiment and collect results
            try:
                # Initialize detector with the config dictionary
                detector = GraphChangeDetection(config_dict=config)
                result = detector.run()

                metrics = {
                    "name": exp_name,
                    "group": "mixture_betting",
                    "network": network,
                    "distance": distance,
                    "threshold": 50.0,
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }
                all_results.append(metrics)
                logger.info(f"Experiment {exp_name} completed successfully")
            except Exception as e:
                logger.error(f"Error running experiment {exp_name}: {str(e)}")

    # Run Beta Betting Function experiments
    logger.info("Running Beta Betting Function experiments")

    for network in network_types:
        for a in a_values:
            for distance in distance_measures:
                current_exp += 1
                exp_name = f"{network}_beta_{a}_{distance}"
                logger.info(
                    f"[{current_exp}/{total_experiments}] Running experiment: {exp_name}"
                )

                config = get_base_config()
                config["model"]["network"] = network
                config["detection"]["method"] = "martingale"
                config["detection"]["threshold"] = 50.0
                config["detection"]["betting_func_config"] = {
                    "name": "beta",
                    "beta": {"a": a, "b": 1.5},
                }
                config["detection"]["distance"] = {"measure": distance, "p": 2.0}

                # Set up output directory
                output_dir = setup_output_dir(base_output_dir, exp_name)
                config["output"]["directory"] = output_dir

                # Save the config for reproducibility
                with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                    yaml.dump(config, f)

                # Run the experiment and collect results
                try:
                    # Initialize detector with the config dictionary
                    detector = GraphChangeDetection(config_dict=config)
                    result = detector.run()

                    metrics = {
                        "name": exp_name,
                        "group": "beta_betting",
                        "network": network,
                        "a": a,
                        "distance": distance,
                        "threshold": 50.0,
                        "TPR": result.get("detection_rate", 0),
                        "FPR": result.get("false_positive_rate", 0),
                        "ADD": result.get("average_detection_delay", 0),
                        "TPR_horizon": result.get("horizon_detection_rate", 0),
                        "ADD_horizon": result.get("horizon_average_delay", 0),
                        "delay_reduction": result.get("delay_reduction_percent", 0),
                    }
                    all_results.append(metrics)
                    logger.info(f"Experiment {exp_name} completed successfully")
                except Exception as e:
                    logger.error(f"Error running experiment {exp_name}: {str(e)}")

    # Run Threshold experiments
    logger.info("Running Threshold experiments")

    for network in network_types:
        for threshold in threshold_values:
            for distance in distance_measures:
                current_exp += 1
                # Use power betting with epsilon 0.5 as our "middle ground" default
                exp_name = f"{network}_threshold_{int(threshold)}_{distance}"
                logger.info(
                    f"[{current_exp}/{total_experiments}] Running threshold experiment: {exp_name}"
                )

                config = get_base_config()
                config["model"]["network"] = network
                config["detection"]["method"] = "martingale"
                config["detection"]["threshold"] = threshold
                config["detection"]["betting_func_config"] = {
                    "name": "power",
                    "power": {"epsilon": 0.5},  # Default middle value
                }
                config["detection"]["distance"] = {"measure": distance, "p": 2.0}

                # Set up output directory
                output_dir = setup_output_dir(base_output_dir, exp_name)
                config["output"]["directory"] = output_dir

                # Save the config for reproducibility
                with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                    yaml.dump(config, f)

                # Run the experiment and collect results
                try:
                    # Initialize detector with the config dictionary
                    detector = GraphChangeDetection(config_dict=config)
                    result = detector.run()

                    metrics = {
                        "name": exp_name,
                        "group": "thresholds",
                        "network": network,
                        "distance": distance,
                        "threshold": threshold,
                        "TPR": result.get("detection_rate", 0),
                        "FPR": result.get("false_positive_rate", 0),
                        "ADD": result.get("average_detection_delay", 0),
                        "TPR_horizon": result.get("horizon_detection_rate", 0),
                        "ADD_horizon": result.get("horizon_average_delay", 0),
                        "delay_reduction": result.get("delay_reduction_percent", 0),
                    }
                    all_results.append(metrics)
                    logger.info(
                        f"Threshold experiment {exp_name} completed successfully"
                    )
                except Exception as e:
                    logger.error(
                        f"Error running threshold experiment {exp_name}: {str(e)}"
                    )

    # Save combined results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(base_output_dir, "sensitivity_results.csv"), index=False
    )

    logger.info(
        f"All sensitivity experiments completed. Results saved to {base_output_dir}"
    )
    return results_df


def run_comparison_experiments() -> pd.DataFrame:
    """Run comparison experiments for Table IV."""
    # Base output directory
    base_output_dir = "results/method_comparison"
    os.makedirs(base_output_dir, exist_ok=True)

    # Define change scenarios per network type
    change_scenarios = {
        "sbm": ["community_merge", "density_change", "mixed_changes"],
        "er": ["density_increase", "density_decrease"],
        "ba": ["parameter_shift", "hub_addition"],
        "nws": ["rewiring_increase", "k_parameter_shift"],
    }

    # Detection methods to compare
    detection_methods = ["martingale", "cusum", "ewma"]

    # Distance measures to use
    distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]

    # Calculate total number of parameter determination experiments
    total_param_tests = 0
    for network in change_scenarios:
        # Distance measure tests
        distance_tests = len(distance_measures)
        # Epsilon tests with best distance
        epsilon_tests = 4  # [0.2, 0.5, 0.7, 0.9]
        total_param_tests += distance_tests + epsilon_tests

    # Calculate total number of comparison experiments
    total_comparison_exps = 0
    for network, scenarios in change_scenarios.items():
        total_comparison_exps += len(scenarios) * len(detection_methods)

    # Threshold experiments
    threshold_values = [20.0, 50.0, 100.0]
    total_threshold_exps = len(change_scenarios) * len(threshold_values)

    # Total experiments
    total_experiments = total_param_tests + total_comparison_exps + total_threshold_exps

    logger.info(f"===== COMPARISON EXPERIMENTS =====")
    logger.info(f"Total experiments to run: {total_experiments}")
    logger.info(f"- Parameter determination: {total_param_tests} experiments")
    logger.info(f"- Method comparison: {total_comparison_exps} experiments")
    logger.info(f"- Threshold analysis: {total_threshold_exps} experiments")
    logger.info(f"===================================\n")

    # Run all experiment combinations
    all_results = []
    current_exp = 0

    # First, determine the best parameters for each network type
    # We'll run some initial experiments with martingale detection
    logger.info("Running preliminary parameter determination experiments")

    best_params = {}
    for network in change_scenarios.keys():
        # Use the first scenario as our test bed
        scenario = change_scenarios[network][0]

        # Test all distance measures with default parameters
        best_tpr = 0
        best_distance = "euclidean"  # Default

        for distance in distance_measures:
            current_exp += 1
            exp_name = f"{network}_{scenario}_param_test_{distance}"
            logger.info(
                f"[{current_exp}/{total_experiments}] Running parameter test: {exp_name}"
            )

            config = get_base_config()
            config["model"]["network"] = network
            config["change_scenario"] = scenario
            config["detection"]["method"] = "martingale"
            config["detection"]["threshold"] = 50.0
            config["detection"]["betting_func_config"] = {
                "name": "power",
                "power": {"epsilon": 0.5},  # Default middle value
            }
            config["detection"]["distance"] = {"measure": distance, "p": 2.0}

            # Set up output directory
            output_dir = setup_output_dir(base_output_dir, exp_name)
            config["output"]["directory"] = output_dir

            # Save the config for reproducibility
            with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)

            # Run the test
            try:
                # Initialize detector with the config dictionary
                detector = GraphChangeDetection(config_dict=config)
                result = detector.run()

                tpr = result.get("detection_rate", 0)

                if tpr > best_tpr:
                    best_tpr = tpr
                    best_distance = distance

                logger.info(f"Parameter test {exp_name} completed: TPR = {tpr}")
            except Exception as e:
                logger.error(f"Error running parameter test {exp_name}: {str(e)}")

        # Test different epsilon values with the best distance
        best_epsilon = 0.5  # Default
        best_tpr = 0

        for epsilon in [0.2, 0.5, 0.7, 0.9]:
            current_exp += 1
            exp_name = f"{network}_{scenario}_param_test_eps_{epsilon}"
            logger.info(
                f"[{current_exp}/{total_experiments}] Running parameter test: {exp_name}"
            )

            config = get_base_config()
            config["model"]["network"] = network
            config["change_scenario"] = scenario
            config["detection"]["method"] = "martingale"
            config["detection"]["threshold"] = 50.0
            config["detection"]["betting_func_config"] = {
                "name": "power",
                "power": {"epsilon": epsilon},
            }
            config["detection"]["distance"] = {"measure": best_distance, "p": 2.0}

            # Set up output directory
            output_dir = setup_output_dir(base_output_dir, exp_name)
            config["output"]["directory"] = output_dir

            # Save the config for reproducibility
            with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)

            # Run the test
            try:
                # Initialize detector with the config dictionary
                detector = GraphChangeDetection(config_dict=config)
                result = detector.run()

                tpr = result.get("detection_rate", 0)

                if tpr > best_tpr:
                    best_tpr = tpr
                    best_epsilon = epsilon

                logger.info(f"Parameter test {exp_name} completed: TPR = {tpr}")
            except Exception as e:
                logger.error(f"Error running parameter test {exp_name}: {str(e)}")

        # Store best parameters for this network
        best_params[network] = {"distance": best_distance, "epsilon": best_epsilon}

        logger.info(
            f"Best parameters for {network}: distance={best_distance}, epsilon={best_epsilon}"
        )

    # Now run the actual comparison experiments with the best parameters
    logger.info("Running method comparison experiments with determined parameters")

    for network, scenarios in change_scenarios.items():
        logger.info(f"Running {network} experiments")

        # Use the best parameters determined earlier
        best_distance = best_params[network]["distance"]
        best_epsilon = best_params[network]["epsilon"]

        for scenario in scenarios:
            for method in detection_methods:
                current_exp += 1
                exp_name = f"{network}_{scenario}_{method}"
                logger.info(
                    f"[{current_exp}/{total_experiments}] Running experiment: {exp_name}"
                )

                # Create experiment configuration
                config = get_base_config()
                config["model"]["network"] = network
                config["change_scenario"] = scenario
                config["detection"]["method"] = method

                # Configure method-specific parameters
                if method == "martingale":
                    # Enable prediction for martingale method
                    config["execution"]["enable_prediction"] = True
                    config["detection"]["betting_func_config"] = {
                        "name": "power",
                        "power": {"epsilon": best_epsilon},
                    }
                elif method == "cusum":
                    # Disable prediction for CUSUM method
                    config["execution"]["enable_prediction"] = False
                    config["detection"]["cusum"] = {
                        "startup_period": 40,
                        "fixed_threshold": True,
                        "k": 0.25,
                        "h": 8.0,
                        "enable_adaptive": True,
                    }
                elif method == "ewma":
                    # Disable prediction for EWMA method
                    config["execution"]["enable_prediction"] = False
                    config["detection"]["ewma"] = {
                        "lambda": 0.15,
                        "L": 4.5,
                        "startup_period": 50,
                        "use_var_adjust": True,
                        "robust": True,
                    }

                # Set common parameters
                config["detection"]["threshold"] = 50.0
                config["detection"]["distance"] = {"measure": best_distance, "p": 2.0}

                # Set up output directory
                output_dir = setup_output_dir(base_output_dir, exp_name)
                config["output"]["directory"] = output_dir

                # Save the config for reproducibility
                with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                    yaml.dump(config, f)

                # Run the experiment and collect results
                try:
                    # Initialize detector with the config dictionary
                    detector = GraphChangeDetection(config_dict=config)
                    result = detector.run()

                    # Extract key metrics for the comparison
                    metrics = {
                        "name": exp_name,
                        "network": network,
                        "change_scenario": scenario,
                        "detection_method": method,
                        "distance": best_distance,
                        "epsilon": best_epsilon if method == "martingale" else None,
                        "TPR": result.get("detection_rate", 0),
                        "FPR": result.get("false_positive_rate", 0),
                        "ADD": result.get("average_detection_delay", 0),
                    }

                    # Add horizon-specific metrics for martingale method
                    if method == "martingale":
                        metrics.update(
                            {
                                "TPR_horizon": result.get("horizon_detection_rate", 0),
                                "ADD_horizon": result.get("horizon_average_delay", 0),
                                "delay_reduction": result.get(
                                    "delay_reduction_percent", 0
                                ),
                            }
                        )

                    all_results.append(metrics)
                    logger.info(f"Experiment {exp_name} completed successfully")
                except Exception as e:
                    logger.error(f"Error running experiment {exp_name}: {str(e)}")

    # Generate threshold experiments
    logger.info("Running threshold experiments")

    for network, scenarios in change_scenarios.items():
        # Use the first scenario for threshold experiments
        scenario = scenarios[0]

        # Use the best parameters determined earlier
        best_distance = best_params[network]["distance"]
        best_epsilon = best_params[network]["epsilon"]

        for threshold in threshold_values:
            current_exp += 1
            exp_name = f"{network}_threshold_{int(threshold)}"
            logger.info(
                f"[{current_exp}/{total_experiments}] Running threshold experiment: {exp_name}"
            )

            # Create experiment configuration
            config = get_base_config()
            config["model"]["network"] = network
            config["change_scenario"] = scenario
            config["detection"]["method"] = "martingale"
            config["detection"]["threshold"] = threshold
            config["detection"]["betting_func_config"] = {
                "name": "power",
                "power": {"epsilon": best_epsilon},
            }
            config["detection"]["distance"] = {"measure": best_distance, "p": 2.0}

            # Set up output directory
            output_dir = setup_output_dir(base_output_dir, exp_name)
            config["output"]["directory"] = output_dir

            # Save the config for reproducibility
            with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)

            # Run the experiment and collect results
            try:
                # Initialize detector with the config dictionary
                detector = GraphChangeDetection(config_dict=config)
                result = detector.run()

                # Extract key metrics
                metrics = {
                    "name": exp_name,
                    "network": network,
                    "change_scenario": scenario,
                    "threshold": threshold,
                    "distance": best_distance,
                    "epsilon": best_epsilon,
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }

                all_results.append(metrics)
                logger.info(f"Threshold experiment {exp_name} completed successfully")
            except Exception as e:
                logger.error(f"Error running threshold experiment {exp_name}: {str(e)}")

    # Save combined results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(base_output_dir, "comparison_results.csv"), index=False
    )

    logger.info(
        f"All comparison experiments completed. Results saved to {base_output_dir}"
    )
    return results_df


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run change point detection experiments using parameter sweeping"
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
        run_sensitivity_experiments()

    if args.all or args.comparison:
        logger.info("Running comparison experiments (Table IV)")
        run_comparison_experiments()

    if not (args.all or args.sensitivity or args.comparison):
        parser.print_help()


if __name__ == "__main__":
    main()
