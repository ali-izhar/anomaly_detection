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
import concurrent.futures
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import traceback
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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


def save_results_robust(
    results_df: pd.DataFrame, base_output_dir: str, filename_prefix: str
) -> None:
    """Robustly save results to both CSV and Excel formats with error handling."""
    try:
        # Ensure output directory exists
        os.makedirs(base_output_dir, exist_ok=True)

        # Save as CSV (fallback)
        csv_path = os.path.join(base_output_dir, f"{filename_prefix}.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to CSV: {csv_path}")

        # Save as Excel (preferred)
        try:
            excel_path = os.path.join(base_output_dir, f"{filename_prefix}.xlsx")
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="Results", index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets["Results"]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            logger.info(f"Results saved to Excel: {excel_path}")
        except ImportError:
            logger.warning(
                "openpyxl not available, only CSV file saved. Install with: pip install openpyxl"
            )
        except Exception as e:
            logger.error(
                f"Error saving Excel file: {str(e)}, CSV file available as backup"
            )

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def standardize_metrics(
    metrics: Dict[str, Any], experiment_type: str
) -> Dict[str, Any]:
    """Standardize metrics dictionary to ensure consistent structure across all experiment types."""
    standardized = {
        "name": metrics.get("name", "unknown"),
        "experiment_type": experiment_type,
        "network": metrics.get("network", ""),
        "TPR": float(metrics.get("TPR", metrics.get("tpr", 0))),
        "FPR": float(metrics.get("FPR", 0)),
        "ADD": float(metrics.get("ADD", 0)),
        "TPR_horizon": float(metrics.get("TPR_horizon", 0)),
        "ADD_horizon": float(metrics.get("ADD_horizon", 0)),
        "delay_reduction": float(metrics.get("delay_reduction", 0)),
    }

    # Add experiment-specific fields while maintaining structure
    for key, value in metrics.items():
        if key not in standardized:
            standardized[key] = value

    return standardized


def get_base_config() -> Dict[str, Any]:
    """Return the base configuration for all experiments."""
    return {
        "execution": {
            "enable_prediction": True,
            "enable_visualization": False,
            "save_csv": True,
        },
        "trials": {
            "n_trials": 5,
            "random_seeds": [42, 142, 242, 342, 442],
        },
        "detection": {
            "method": "martingale",
            "batch_size": 1000,
            "reset": True,
            "reset_on_traditional": True,
            "max_window": None,
            "prediction_horizon": 5,
            "enable_pvalue_dampening": False,
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
            "directory": "results/sensitivity_analysis",
            "save_predictions": True,
            "save_features": True,
            "save_martingales": True,
        },
    }


def run_experiment(exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment with the given configuration.

    Args:
        exp_config: Dictionary containing:
            - config: The experiment configuration
            - exp_name: The experiment name
            - output_dir: Path to save results
            - base_output_dir: Base output directory

    Returns:
        A dictionary with the experiment results and metadata
    """
    config = exp_config["config"]
    exp_name = exp_config["exp_name"]
    output_dir = exp_config["output_dir"]

    # Log that we're starting this experiment
    start_time = datetime.now()
    formatted_time = start_time.strftime("%H:%M:%S")
    print(f"[{formatted_time}] Starting experiment: {exp_name}")

    # Save the config for reproducibility
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    try:
        # Initialize detector with the config dictionary
        detector = GraphChangeDetection(config_dict=config)
        result = detector.run()

        # Calculate experiment duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Build metrics dict based on experiment type
        if "group" in exp_config:
            # Sensitivity experiment
            if exp_config["group"] == "power_betting":
                metrics = {
                    "name": exp_name,
                    "group": "power_betting",
                    "network": exp_config["network"],
                    "epsilon": exp_config["epsilon"],
                    "distance": exp_config["distance"],
                    "threshold": config["detection"]["threshold"],
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }
            elif exp_config["group"] == "mixture_betting":
                metrics = {
                    "name": exp_name,
                    "group": "mixture_betting",
                    "network": exp_config["network"],
                    "distance": exp_config["distance"],
                    "threshold": config["detection"]["threshold"],
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }
            elif exp_config["group"] == "beta_betting":
                metrics = {
                    "name": exp_name,
                    "group": "beta_betting",
                    "network": exp_config["network"],
                    "a": exp_config["a"],
                    "distance": exp_config["distance"],
                    "threshold": config["detection"]["threshold"],
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }
            elif exp_config["group"] == "thresholds":
                metrics = {
                    "name": exp_name,
                    "group": "thresholds",
                    "network": exp_config["network"],
                    "distance": exp_config["distance"],
                    "threshold": config["detection"]["threshold"],
                    "TPR": result.get("detection_rate", 0),
                    "FPR": result.get("false_positive_rate", 0),
                    "ADD": result.get("average_detection_delay", 0),
                    "TPR_horizon": result.get("horizon_detection_rate", 0),
                    "ADD_horizon": result.get("horizon_average_delay", 0),
                    "delay_reduction": result.get("delay_reduction_percent", 0),
                }
        elif "parameter_test" in exp_config:
            # Parameter determination experiments
            metrics = {
                "name": exp_name,
                "tpr": result.get("detection_rate", 0),
                "parameter": exp_config["parameter"],
                "parameter_value": exp_config["parameter_value"],
                "network": exp_config["network"],
                "scenario": exp_config["scenario"],
            }
        elif "comparison" in exp_config:
            # Comparison experiments
            metrics = {
                "name": exp_name,
                "network": exp_config["network"],
                "change_scenario": exp_config["scenario"],
                "detection_method": exp_config["method"],
                "distance": exp_config["distance"],
                "epsilon": exp_config.get("epsilon"),
                "TPR": result.get("detection_rate", 0),
                "FPR": result.get("false_positive_rate", 0),
                "ADD": result.get("average_detection_delay", 0),
            }

            # Add horizon-specific metrics for martingale method
            if exp_config["method"] == "martingale":
                metrics.update(
                    {
                        "TPR_horizon": result.get("horizon_detection_rate", 0),
                        "ADD_horizon": result.get("horizon_average_delay", 0),
                        "delay_reduction": result.get("delay_reduction_percent", 0),
                    }
                )
        elif "threshold_experiment" in exp_config:
            # Threshold experiments
            metrics = {
                "name": exp_name,
                "network": exp_config["network"],
                "change_scenario": exp_config["scenario"],
                "threshold": exp_config["threshold"],
                "distance": exp_config["distance"],
                "epsilon": exp_config["epsilon"],
                "TPR": result.get("detection_rate", 0),
                "FPR": result.get("false_positive_rate", 0),
                "ADD": result.get("average_detection_delay", 0),
                "TPR_horizon": result.get("horizon_detection_rate", 0),
                "ADD_horizon": result.get("horizon_average_delay", 0),
                "delay_reduction": result.get("delay_reduction_percent", 0),
            }

            # Standardize metrics based on experiment type
        experiment_type = "unknown"
        if "group" in exp_config:
            experiment_type = "sensitivity"
        elif "parameter_test" in exp_config:
            experiment_type = "parameter_test"
        elif "comparison" in exp_config:
            experiment_type = "comparison"
        elif "threshold_experiment" in exp_config:
            experiment_type = "threshold"

        standardized_metrics = standardize_metrics(metrics, experiment_type)

        # Save individual experiment results to its own folder (alongside config.yaml)
        try:
            individual_results_df = pd.DataFrame([standardized_metrics])
            save_results_robust(
                individual_results_df, output_dir, f"{exp_name}_results"
            )

            # Create a summary file for this experiment
            summary_content = f"""# Experiment: {exp_name}

## Files in this directory:
- `config.yaml`: Complete configuration used for this experiment
- `{exp_name}_results.xlsx`: Results in Excel format
- `{exp_name}_results.csv`: Results in CSV format (backup)
- `experiment_summary.txt`: This summary file

## Experiment Details:
- **Experiment Name**: {exp_name}
- **Experiment Type**: {experiment_type}
- **Network**: {standardized_metrics.get('network', 'N/A')}
- **Completion Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {duration:.2f} seconds

## Key Results:
- **TPR (True Positive Rate)**: {standardized_metrics.get('TPR', 0):.4f}
- **FPR (False Positive Rate)**: {standardized_metrics.get('FPR', 0):.4f}
- **ADD (Average Detection Delay)**: {standardized_metrics.get('ADD', 0):.4f}
- **TPR Horizon**: {standardized_metrics.get('TPR_horizon', 0):.4f}
- **ADD Horizon**: {standardized_metrics.get('ADD_horizon', 0):.4f}
- **Delay Reduction**: {standardized_metrics.get('delay_reduction', 0):.4f}%

## Reproducibility:
To reproduce this experiment, use the `config.yaml` file in this directory with the main experiment runner.
"""

            with open(os.path.join(output_dir, "experiment_summary.txt"), "w") as f:
                f.write(summary_content)

            logger.info(f"Individual results saved to: {output_dir}")
        except Exception as e:
            logger.warning(
                f"Could not save individual results for {exp_name}: {str(e)}"
            )

        logger.info(f"Experiment {exp_name} completed successfully")
        return {"success": True, "metrics": standardized_metrics, "exp_name": exp_name}
    except Exception as e:
        error_details = (
            f"Error running experiment {exp_name}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_details)
        return {
            "success": False,
            "error": str(e),
            "exp_name": exp_name,
            "error_details": error_details,
        }


def run_sensitivity_experiments(max_workers: int = 4) -> pd.DataFrame:
    """Run parameter sensitivity experiments for Table III.

    Args:
        max_workers: Maximum number of parallel workers

    Returns:
        DataFrame with experiment results
    """
    # Base output directory
    base_output_dir = "results/sensitivity_analysis"
    os.makedirs(base_output_dir, exist_ok=True)

    # Parameter spaces for the sweep
    network_types = ["sbm", "er", "ba", "ws"]
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
    logger.info(f"- Using {max_workers} parallel workers")
    logger.info(f"===================================\n")

    # Prepare experiment configurations
    all_experiments = []
    current_exp = 0

    # Power Betting Function experiments
    logger.info("Preparing Power Betting Function experiments")
    for network in network_types:
        for epsilon in epsilon_values:
            for distance in distance_measures:
                current_exp += 1
                exp_name = f"{network}_power_{epsilon}_{distance}"

                config = get_base_config()
                config["model"]["network"] = network
                config["detection"]["method"] = "martingale"
                config["detection"]["threshold"] = 50.0  # Default threshold
                config["detection"]["betting_func_config"] = {
                    "name": "power",
                    "power": {"epsilon": epsilon},
                }
                config["detection"]["distance"] = {"measure": distance, "p": 2.0}

                output_dir = setup_output_dir(base_output_dir, exp_name)

                # Create experiment config
                exp_config = {
                    "config": config,
                    "exp_name": exp_name,
                    "output_dir": output_dir,
                    "group": "power_betting",
                    "network": network,
                    "epsilon": epsilon,
                    "distance": distance,
                }

                all_experiments.append(exp_config)

    # Mixture Betting Function experiments
    logger.info("Preparing Mixture Betting Function experiments")
    for network in network_types:
        for distance in distance_measures:
            current_exp += 1
            exp_name = f"{network}_mixture_{distance}"

            config = get_base_config()
            config["model"]["network"] = network
            config["detection"]["method"] = "martingale"
            config["detection"]["threshold"] = 50.0
            config["detection"]["betting_func_config"] = {
                "name": "mixture",
                "mixture": {"epsilons": [0.7, 0.8, 0.9]},
            }
            config["detection"]["distance"] = {"measure": distance, "p": 2.0}

            output_dir = setup_output_dir(base_output_dir, exp_name)

            # Create experiment config
            exp_config = {
                "config": config,
                "exp_name": exp_name,
                "output_dir": output_dir,
                "group": "mixture_betting",
                "network": network,
                "distance": distance,
            }

            all_experiments.append(exp_config)

    # Beta Betting Function experiments
    logger.info("Preparing Beta Betting Function experiments")
    for network in network_types:
        for a in a_values:
            for distance in distance_measures:
                current_exp += 1
                exp_name = f"{network}_beta_{a}_{distance}"

                config = get_base_config()
                config["model"]["network"] = network
                config["detection"]["method"] = "martingale"
                config["detection"]["threshold"] = 50.0
                config["detection"]["betting_func_config"] = {
                    "name": "beta",
                    "beta": {"a": a, "b": 1.5},
                }
                config["detection"]["distance"] = {"measure": distance, "p": 2.0}

                output_dir = setup_output_dir(base_output_dir, exp_name)

                # Create experiment config
                exp_config = {
                    "config": config,
                    "exp_name": exp_name,
                    "output_dir": output_dir,
                    "group": "beta_betting",
                    "network": network,
                    "a": a,
                    "distance": distance,
                }

                all_experiments.append(exp_config)

    # Threshold experiments
    logger.info("Preparing Threshold experiments")
    for network in network_types:
        for threshold in threshold_values:
            for distance in distance_measures:
                current_exp += 1
                exp_name = f"{network}_threshold_{int(threshold)}_{distance}"

                config = get_base_config()
                config["model"]["network"] = network
                config["detection"]["method"] = "martingale"
                config["detection"]["threshold"] = threshold
                config["detection"]["betting_func_config"] = {
                    "name": "power",
                    "power": {"epsilon": 0.5},  # Default middle value
                }
                config["detection"]["distance"] = {"measure": distance, "p": 2.0}

                output_dir = setup_output_dir(base_output_dir, exp_name)

                # Create experiment config
                exp_config = {
                    "config": config,
                    "exp_name": exp_name,
                    "output_dir": output_dir,
                    "group": "thresholds",
                    "network": network,
                    "distance": distance,
                }

                all_experiments.append(exp_config)

    # Run experiments in parallel
    logger.info(
        f"Running {len(all_experiments)} sensitivity experiments in parallel with {max_workers} workers"
    )
    all_results = []

    # Log start time
    start_time_all = datetime.now()
    logger.info(f"Starting experiments at {start_time_all.strftime('%H:%M:%S')}")

    completed = 0
    failed = 0

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(run_experiment, exp): exp for exp in all_experiments
        }

        logger.info(f"Submitted {len(future_to_exp)} experiments to the worker pool")

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_exp)):
            exp = future_to_exp[future]
            current_time = datetime.now()
            elapsed = (current_time - start_time_all).total_seconds() / 60.0  # minutes

            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    all_results.append(result["metrics"])
                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(all_experiments) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.info(
                        f"[{i+1}/{len(all_experiments)}] Experiment {result['exp_name']} completed successfully "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
                else:
                    failed += 1

                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(all_experiments) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.error(
                        f"[{i+1}/{len(all_experiments)}] Error in experiment {result['exp_name']}: {result['error']} "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
            except Exception as e:
                failed += 1

                # Calculate remaining time with safeguard against division by zero
                remaining_min = 0
                if completed + failed > 0:
                    avg_time_per_exp = elapsed / (completed + failed)
                    remaining_exps = len(all_experiments) - (completed + failed)
                    remaining_min = avg_time_per_exp * remaining_exps

                logger.error(
                    f"[{i+1}/{len(all_experiments)}] Exception in experiment {exp['exp_name']}: {str(e)} "
                    f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                    f"~{remaining_min:.1f} min remaining)"
                )

    # Save combined results with robust error handling
    try:
        if not all_results:
            logger.warning("No successful results to save")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # Add summary statistics
        total_experiments = len(all_experiments)
        successful_experiments = completed
        failed_experiments = failed
        success_rate = (
            (successful_experiments / total_experiments) * 100
            if total_experiments > 0
            else 0
        )

        logger.info(f"===== SENSITIVITY EXPERIMENTS SUMMARY =====")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"Successful: {successful_experiments}")
        logger.info(f"Failed: {failed_experiments}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"============================================")

        # Save results using robust function
        save_results_robust(results_df, base_output_dir, "sensitivity_results")

        # Save experiment summary
        summary_data = {
            "total_experiments": [total_experiments],
            "successful_experiments": [successful_experiments],
            "failed_experiments": [failed_experiments],
            "success_rate_percent": [success_rate],
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        }
        summary_df = pd.DataFrame(summary_data)
        save_results_robust(summary_df, base_output_dir, "sensitivity_summary")

        logger.info(
            f"All sensitivity experiments completed. Results saved to {base_output_dir}"
        )
        return results_df

    except Exception as e:
        logger.error(f"Error saving sensitivity results: {str(e)}")
        logger.error(traceback.format_exc())
        # Still try to return whatever results we have
        if all_results:
            return pd.DataFrame(all_results)
        else:
            return pd.DataFrame()


def run_comparison_experiments(max_workers: int = 4) -> pd.DataFrame:
    """Run comparison experiments for Table IV.

    Args:
        max_workers: Maximum number of parallel workers

    Returns:
        DataFrame with experiment results
    """
    # Base output directory
    base_output_dir = "results/method_comparison"
    os.makedirs(base_output_dir, exist_ok=True)

    # Define change scenarios per network type
    change_scenarios = {
        "sbm": ["community_merge", "density_change", "mixed_changes"],
        "er": ["density_increase", "density_decrease"],
        "ba": ["parameter_shift", "hub_addition"],
        "ws": ["rewiring_increase", "k_parameter_shift"],
    }

    # Detection methods to compare
    detection_methods = ["martingale", "cusum", "ewma"]

    # Distance measures to use
    distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]

    # Threshold values for experiments
    threshold_values = [20.0, 50.0, 100.0]

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
    total_threshold_exps = len(change_scenarios) * len(threshold_values)

    # Total experiments
    total_experiments = total_param_tests + total_comparison_exps + total_threshold_exps

    logger.info(f"===== COMPARISON EXPERIMENTS =====")
    logger.info(f"Total experiments to run: {total_experiments}")
    logger.info(f"- Parameter determination: {total_param_tests} experiments")
    logger.info(f"- Method comparison: {total_comparison_exps} experiments")
    logger.info(f"- Threshold analysis: {total_threshold_exps} experiments")
    logger.info(f"- Using {max_workers} parallel workers")
    logger.info(f"===================================\n")

    # First, determine the best parameters for each network type
    # These need to be run sequentially as later experiments depend on the results
    logger.info("Running preliminary parameter determination experiments")

    # Run parameter determination experiments (these need to be sequential)
    best_params = {}
    param_exp_configs = []

    # Create experiment configs for distance testing
    for network in change_scenarios.keys():
        # Use the first scenario as our test bed
        scenario = change_scenarios[network][0]

        for distance in distance_measures:
            exp_name = f"{network}_{scenario}_param_test_{distance}"

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

            output_dir = setup_output_dir(base_output_dir, exp_name)

            # Create experiment config
            exp_config = {
                "config": config,
                "exp_name": exp_name,
                "output_dir": output_dir,
                "parameter_test": True,
                "parameter": "distance",
                "parameter_value": distance,
                "network": network,
                "scenario": scenario,
            }

            param_exp_configs.append(exp_config)

    # Run distance parameter tests in parallel
    logger.info(
        f"Running {len(param_exp_configs)} distance parameter tests in parallel"
    )
    distance_results = {}

    # Log start time
    start_time_all = datetime.now()
    logger.info(
        f"Starting distance parameter tests at {start_time_all.strftime('%H:%M:%S')}"
    )

    completed = 0
    failed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(run_experiment, exp): exp for exp in param_exp_configs
        }

        logger.info(
            f"Submitted {len(future_to_exp)} parameter tests to the worker pool"
        )

        for i, future in enumerate(concurrent.futures.as_completed(future_to_exp)):
            exp = future_to_exp[future]
            current_time = datetime.now()
            elapsed = (current_time - start_time_all).total_seconds() / 60.0  # minutes

            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    # Store results by network for parameter selection
                    network = exp["network"]
                    if network not in distance_results:
                        distance_results[network] = []

                    distance_results[network].append(
                        {
                            "distance": exp["parameter_value"],
                            "tpr": result["metrics"]["tpr"],
                        }
                    )

                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(param_exp_configs) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.info(
                        f"[{i+1}/{len(param_exp_configs)}] Parameter test {result['exp_name']} completed: TPR = {result['metrics']['tpr']} "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
                else:
                    failed += 1

                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(param_exp_configs) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.error(
                        f"[{i+1}/{len(param_exp_configs)}] Error in parameter test {result['exp_name']}: {result['error']} "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
            except Exception as e:
                failed += 1

                # Calculate remaining time with safeguard against division by zero
                remaining_min = 0
                if completed + failed > 0:
                    avg_time_per_exp = elapsed / (completed + failed)
                    remaining_exps = len(param_exp_configs) - (completed + failed)
                    remaining_min = avg_time_per_exp * remaining_exps

                logger.error(
                    f"[{i+1}/{len(param_exp_configs)}] Exception in parameter test {exp['exp_name']}: {str(e)} "
                    f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                    f"~{remaining_min:.1f} min remaining)"
                )

    # Find best distance for each network
    best_distances = {}
    for network, results in distance_results.items():
        best_distance_result = max(results, key=lambda x: x["tpr"])
        best_distances[network] = best_distance_result["distance"]
        logger.info(f"Best distance for {network}: {best_distances[network]}")

    # Create epsilon test configs
    epsilon_exp_configs = []
    for network in change_scenarios.keys():
        # Use the first scenario as our test bed
        scenario = change_scenarios[network][0]
        best_distance = best_distances.get(network, "euclidean")

        for epsilon in [0.2, 0.5, 0.7, 0.9]:
            exp_name = f"{network}_{scenario}_param_test_eps_{epsilon}"

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

            output_dir = setup_output_dir(base_output_dir, exp_name)

            # Create experiment config
            exp_config = {
                "config": config,
                "exp_name": exp_name,
                "output_dir": output_dir,
                "parameter_test": True,
                "parameter": "epsilon",
                "parameter_value": epsilon,
                "network": network,
                "scenario": scenario,
            }

            epsilon_exp_configs.append(exp_config)

    # Run epsilon parameter tests in parallel
    logger.info(
        f"Running {len(epsilon_exp_configs)} epsilon parameter tests in parallel"
    )
    epsilon_results = {}

    # Log start time
    start_time_all = datetime.now()
    logger.info(
        f"Starting epsilon parameter tests at {start_time_all.strftime('%H:%M:%S')}"
    )

    completed = 0
    failed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(run_experiment, exp): exp for exp in epsilon_exp_configs
        }

        logger.info(f"Submitted {len(future_to_exp)} epsilon tests to the worker pool")

        for i, future in enumerate(concurrent.futures.as_completed(future_to_exp)):
            exp = future_to_exp[future]
            current_time = datetime.now()
            elapsed = (current_time - start_time_all).total_seconds() / 60.0  # minutes

            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    # Store results by network for parameter selection
                    network = exp["network"]
                    if network not in epsilon_results:
                        epsilon_results[network] = []

                    epsilon_results[network].append(
                        {
                            "epsilon": exp["parameter_value"],
                            "tpr": result["metrics"]["tpr"],
                        }
                    )

                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(epsilon_exp_configs) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.info(
                        f"[{i+1}/{len(epsilon_exp_configs)}] Parameter test {result['exp_name']} completed: TPR = {result['metrics']['tpr']} "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
                else:
                    failed += 1

                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(epsilon_exp_configs) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.error(
                        f"[{i+1}/{len(epsilon_exp_configs)}] Error in parameter test {result['exp_name']}: {result['error']} "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
            except Exception as e:
                failed += 1

                # Calculate remaining time with safeguard against division by zero
                remaining_min = 0
                if completed + failed > 0:
                    avg_time_per_exp = elapsed / (completed + failed)
                    remaining_exps = len(epsilon_exp_configs) - (completed + failed)
                    remaining_min = avg_time_per_exp * remaining_exps

                logger.error(
                    f"[{i+1}/{len(epsilon_exp_configs)}] Exception in parameter test {exp['exp_name']}: {str(e)} "
                    f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                    f"~{remaining_min:.1f} min remaining)"
                )

    # Find best epsilon for each network
    best_epsilons = {}
    for network, results in epsilon_results.items():
        best_epsilon_result = max(results, key=lambda x: x["tpr"])
        best_epsilons[network] = best_epsilon_result["epsilon"]
        logger.info(f"Best epsilon for {network}: {best_epsilons[network]}")

    # Store best parameters
    for network in change_scenarios.keys():
        best_params[network] = {
            "distance": best_distances.get(network, "euclidean"),
            "epsilon": best_epsilons.get(network, 0.5),
        }
        logger.info(
            f"Best parameters for {network}: distance={best_params[network]['distance']}, epsilon={best_params[network]['epsilon']}"
        )

    # Now run the actual comparison experiments with the best parameters
    logger.info("Preparing method comparison experiments with determined parameters")

    # Create experiment configs for comparison tests
    comparison_exp_configs = []

    for network, scenarios in change_scenarios.items():
        # Use the best parameters determined earlier
        best_distance = best_params[network]["distance"]
        best_epsilon = best_params[network]["epsilon"]

        for scenario in scenarios:
            for method in detection_methods:
                exp_name = f"{network}_{scenario}_{method}"

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

                output_dir = setup_output_dir(base_output_dir, exp_name)

                # Create experiment config
                exp_config = {
                    "config": config,
                    "exp_name": exp_name,
                    "output_dir": output_dir,
                    "comparison": True,
                    "network": network,
                    "scenario": scenario,
                    "method": method,
                    "distance": best_distance,
                    "epsilon": best_epsilon if method == "martingale" else None,
                }

                comparison_exp_configs.append(exp_config)

    # Create threshold experiment configs
    threshold_exp_configs = []

    for network, scenarios in change_scenarios.items():
        # Use the first scenario for threshold experiments
        scenario = scenarios[0]

        # Use the best parameters determined earlier
        best_distance = best_params[network]["distance"]
        best_epsilon = best_params[network]["epsilon"]

        for threshold in threshold_values:
            exp_name = f"{network}_threshold_{int(threshold)}"

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

            output_dir = setup_output_dir(base_output_dir, exp_name)

            # Create experiment config
            exp_config = {
                "config": config,
                "exp_name": exp_name,
                "output_dir": output_dir,
                "threshold_experiment": True,
                "network": network,
                "scenario": scenario,
                "threshold": threshold,
                "distance": best_distance,
                "epsilon": best_epsilon,
            }

            threshold_exp_configs.append(exp_config)

    # Combine all remaining experiment configs
    all_exp_configs = comparison_exp_configs + threshold_exp_configs

    # Run all experiments in parallel
    logger.info(
        f"Running {len(all_exp_configs)} comparison and threshold experiments in parallel"
    )
    all_results = []

    # Log start time
    start_time_all = datetime.now()
    logger.info(
        f"Starting comparison experiments at {start_time_all.strftime('%H:%M:%S')}"
    )

    completed = 0
    failed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(run_experiment, exp): exp for exp in all_exp_configs
        }

        logger.info(
            f"Submitted {len(future_to_exp)} comparison tests to the worker pool"
        )

        for i, future in enumerate(concurrent.futures.as_completed(future_to_exp)):
            exp = future_to_exp[future]
            current_time = datetime.now()
            elapsed = (current_time - start_time_all).total_seconds() / 60.0  # minutes

            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    all_results.append(result["metrics"])
                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(all_exp_configs) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.info(
                        f"[{i+1}/{len(all_exp_configs)}] Experiment {result['exp_name']} completed successfully "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
                else:
                    failed += 1

                    # Calculate remaining time with safeguard against division by zero
                    remaining_min = 0
                    if completed + failed > 0:
                        avg_time_per_exp = elapsed / (completed + failed)
                        remaining_exps = len(all_exp_configs) - (completed + failed)
                        remaining_min = avg_time_per_exp * remaining_exps

                    logger.error(
                        f"[{i+1}/{len(all_exp_configs)}] Error in experiment {result['exp_name']}: {result['error']} "
                        f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                        f"~{remaining_min:.1f} min remaining)"
                    )
            except Exception as e:
                failed += 1

                # Calculate remaining time with safeguard against division by zero
                remaining_min = 0
                if completed + failed > 0:
                    avg_time_per_exp = elapsed / (completed + failed)
                    remaining_exps = len(all_exp_configs) - (completed + failed)
                    remaining_min = avg_time_per_exp * remaining_exps

                logger.error(
                    f"[{i+1}/{len(all_exp_configs)}] Exception in experiment {exp['exp_name']}: {str(e)} "
                    f"({completed} done, {failed} failed, {elapsed:.1f} min elapsed, "
                    f"~{remaining_min:.1f} min remaining)"
                )

    # Add parameter determination results for completeness
    for network in change_scenarios.keys():
        for dist_result in distance_results.get(network, []):
            all_results.append(
                {
                    "name": f"{network}_dist_{dist_result['distance']}",
                    "network": network,
                    "parameter": "distance",
                    "value": dist_result["distance"],
                    "tpr": dist_result["tpr"],
                }
            )

        for eps_result in epsilon_results.get(network, []):
            all_results.append(
                {
                    "name": f"{network}_eps_{eps_result['epsilon']}",
                    "network": network,
                    "parameter": "epsilon",
                    "value": eps_result["epsilon"],
                    "tpr": eps_result["tpr"],
                }
            )

    # Save combined results with robust error handling
    try:
        if not all_results:
            logger.warning("No successful comparison results to save")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # Calculate totals for summary
        param_total = len(param_exp_configs) + len(epsilon_exp_configs)
        comparison_total = len(comparison_exp_configs)
        threshold_total = len(threshold_exp_configs)
        total_experiments = param_total + comparison_total + threshold_total

        logger.info(f"===== COMPARISON EXPERIMENTS SUMMARY =====")
        logger.info(f"Parameter determination experiments: {param_total}")
        logger.info(f"Method comparison experiments: {comparison_total}")
        logger.info(f"Threshold experiments: {threshold_total}")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"Successful results: {len(all_results)}")
        logger.info(f"==========================================")

        # Save results using robust function
        save_results_robust(results_df, base_output_dir, "comparison_results")

        # Save separate files for different experiment types for easier analysis
        try:
            # Separate parameter determination results
            param_results = [
                r for r in all_results if r.get("experiment_type") == "parameter_test"
            ]
            if param_results:
                param_df = pd.DataFrame(param_results)
                save_results_robust(
                    param_df, base_output_dir, "parameter_determination_results"
                )

            # Separate comparison results
            comparison_results = [
                r for r in all_results if r.get("experiment_type") == "comparison"
            ]
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                save_results_robust(
                    comparison_df, base_output_dir, "method_comparison_results"
                )

            # Separate threshold results
            threshold_results = [
                r for r in all_results if r.get("experiment_type") == "threshold"
            ]
            if threshold_results:
                threshold_df = pd.DataFrame(threshold_results)
                save_results_robust(
                    threshold_df, base_output_dir, "threshold_analysis_results"
                )

        except Exception as e:
            logger.warning(f"Could not create separate result files: {str(e)}")

        # Save experiment summary
        summary_data = {
            "parameter_experiments": [param_total],
            "comparison_experiments": [comparison_total],
            "threshold_experiments": [threshold_total],
            "total_experiments": [total_experiments],
            "successful_results": [len(all_results)],
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        }
        summary_df = pd.DataFrame(summary_data)
        save_results_robust(summary_df, base_output_dir, "comparison_summary")

        logger.info(
            f"All comparison experiments completed. Results saved to {base_output_dir}"
        )
        return results_df

    except Exception as e:
        logger.error(f"Error saving comparison results: {str(e)}")
        logger.error(traceback.format_exc())
        # Still try to return whatever results we have
        if all_results:
            return pd.DataFrame(all_results)
        else:
            return pd.DataFrame()


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
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to use",
    )
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

    # Track overall results
    overall_start_time = datetime.now()
    all_experiment_results = []
    experiment_summaries = []

    # Run requested experiments
    try:
        if args.all or args.sensitivity:
            logger.info("=" * 60)
            logger.info("STARTING SENSITIVITY EXPERIMENTS (Table III)")
            logger.info("=" * 60)

            try:
                sensitivity_df = run_sensitivity_experiments(max_workers=args.workers)
                if not sensitivity_df.empty:
                    all_experiment_results.append(("sensitivity", sensitivity_df))
                    experiment_summaries.append(
                        {
                            "experiment_type": "sensitivity",
                            "total_results": len(sensitivity_df),
                            "status": "completed",
                        }
                    )
                else:
                    experiment_summaries.append(
                        {
                            "experiment_type": "sensitivity",
                            "total_results": 0,
                            "status": "failed_no_results",
                        }
                    )
            except Exception as e:
                logger.error(f"Sensitivity experiments failed: {str(e)}")
                experiment_summaries.append(
                    {
                        "experiment_type": "sensitivity",
                        "total_results": 0,
                        "status": "failed_with_error",
                        "error": str(e),
                    }
                )

        if args.all or args.comparison:
            logger.info("=" * 60)
            logger.info("STARTING COMPARISON EXPERIMENTS (Table IV)")
            logger.info("=" * 60)

            try:
                comparison_df = run_comparison_experiments(max_workers=args.workers)
                if not comparison_df.empty:
                    all_experiment_results.append(("comparison", comparison_df))
                    experiment_summaries.append(
                        {
                            "experiment_type": "comparison",
                            "total_results": len(comparison_df),
                            "status": "completed",
                        }
                    )
                else:
                    experiment_summaries.append(
                        {
                            "experiment_type": "comparison",
                            "total_results": 0,
                            "status": "failed_no_results",
                        }
                    )
            except Exception as e:
                logger.error(f"Comparison experiments failed: {str(e)}")
                experiment_summaries.append(
                    {
                        "experiment_type": "comparison",
                        "total_results": 0,
                        "status": "failed_with_error",
                        "error": str(e),
                    }
                )

        # Create consolidated results file if we have any results
        if all_experiment_results:
            try:
                logger.info("=" * 60)
                logger.info("CREATING CONSOLIDATED RESULTS")
                logger.info("=" * 60)

                consolidated_output_dir = "results/consolidated"
                os.makedirs(consolidated_output_dir, exist_ok=True)

                # Create consolidated DataFrame
                all_dfs = []
                for exp_type, df in all_experiment_results:
                    df_copy = df.copy()
                    df_copy["source_experiment"] = exp_type
                    all_dfs.append(df_copy)

                if all_dfs:
                    consolidated_df = pd.concat(all_dfs, ignore_index=True, sort=False)
                    save_results_robust(
                        consolidated_df,
                        consolidated_output_dir,
                        "all_experiments_consolidated",
                    )

                    logger.info(
                        f"Consolidated results saved with {len(consolidated_df)} total experiment results"
                    )

                # Save overall summary
                overall_end_time = datetime.now()
                overall_duration = (
                    overall_end_time - overall_start_time
                ).total_seconds() / 60.0

                summary_df = pd.DataFrame(experiment_summaries)
                summary_df["overall_duration_minutes"] = overall_duration
                summary_df["overall_start_time"] = overall_start_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                summary_df["overall_end_time"] = overall_end_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                save_results_robust(
                    summary_df, consolidated_output_dir, "overall_experiment_summary"
                )

                logger.info("=" * 60)
                logger.info("ALL EXPERIMENTS COMPLETED")
                logger.info(f"Total duration: {overall_duration:.1f} minutes")
                logger.info(f"Results saved in: {consolidated_output_dir}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"Error creating consolidated results: {str(e)}")
                logger.error(traceback.format_exc())

        if not (args.all or args.sensitivity or args.comparison):
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("Experiments interrupted by user")
        # Still try to save any partial results
        if all_experiment_results:
            try:
                logger.info("Saving partial results before exit...")
                consolidated_output_dir = "results/consolidated_partial"
                os.makedirs(consolidated_output_dir, exist_ok=True)

                all_dfs = []
                for exp_type, df in all_experiment_results:
                    df_copy = df.copy()
                    df_copy["source_experiment"] = exp_type
                    all_dfs.append(df_copy)

                if all_dfs:
                    consolidated_df = pd.concat(all_dfs, ignore_index=True, sort=False)
                    save_results_robust(
                        consolidated_df, consolidated_output_dir, "partial_results"
                    )
                    logger.info(f"Partial results saved to {consolidated_output_dir}")
            except Exception as e:
                logger.error(f"Could not save partial results: {str(e)}")

        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
