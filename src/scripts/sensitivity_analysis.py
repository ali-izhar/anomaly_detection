#!/usr/bin/env python3
"""
Sensitivity Analysis Results Processor

This script analyzes experiment results from the parameter sensitivity sweep,
calculating TPR, FPR, and ADD metrics from detection Excel files and grouping
by network type and parameter combinations.

Key criteria:
- True positive: Detection within 30 timesteps AFTER actual changepoint
- False positive: Detection not matching any actual changepoint within criteria
- ADD: Average detection delay for true positives only
"""

import os
import pandas as pd
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_experiment_directories(base_dir: str) -> List[Path]:
    """Find all experiment directories containing config.yaml and detection_results.xlsx"""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.error(f"Base directory doesn't exist: {base_dir}")
        return []
    
    experiment_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            config_file = item / "config.yaml"
            results_file = item / "detection_results.xlsx"
            
            if config_file.exists() and results_file.exists():
                experiment_dirs.append(item)
            else:
                logger.warning(f"Incomplete experiment: {item.name}")
                if not config_file.exists():
                    logger.warning(f"  Missing: config.yaml")
                if not results_file.exists():
                    logger.warning(f"  Missing: detection_results.xlsx")
    
    logger.info(f"Found {len(experiment_dirs)} complete experiments")
    return experiment_dirs


def extract_parameters_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key parameters from experiment configuration"""
    params = {}

    # Network type
    params["network"] = config.get("model", {}).get("network", "unknown")

    # Betting function configuration
    betting_config = config.get("detection", {}).get("betting_func_config", {})
    params["betting_type"] = betting_config.get("name", "unknown")

    if params["betting_type"] == "power":
        power_config = betting_config.get("power", {})
        params["epsilon"] = power_config.get("epsilon")
    elif params["betting_type"] == "beta":
        beta_config = betting_config.get("beta", {})
        params["beta_a"] = beta_config.get("a")
        params["beta_b"] = beta_config.get("b")
    elif params["betting_type"] == "mixture":
        mixture_config = betting_config.get("mixture", {})
        params["mixture_epsilons"] = mixture_config.get("epsilons", [])

    # Distance measure
    distance_config = config.get("detection", {}).get("distance", {})
    params["distance"] = distance_config.get("measure", "unknown")

    # Threshold
    params["threshold"] = config.get("detection", {}).get("threshold")

    # Number of trials
    params["n_trials"] = config.get("trials", {}).get("n_trials", 1)

    return params


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return {}


def extract_actual_changepoints(metadata_df: pd.DataFrame) -> List[int]:
    """Extract actual changepoint locations from metadata"""
    if "change_point" in metadata_df.columns:
        changepoints = metadata_df["change_point"].dropna().unique().tolist()
        return [int(cp) for cp in changepoints if pd.notna(cp)]
    
    # Try other possible column names
    possible_columns = ["changepoint", "cp", "actual_cp", "true_cp"]
    for col in possible_columns:
        if col in metadata_df.columns:
            changepoints = metadata_df[col].dropna().unique().tolist()
            return [int(cp) for cp in changepoints if pd.notna(cp)]
    
    logger.error("Could not find actual changepoints in metadata")
    return []


def calculate_metrics(
    details_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calculate TPR, FPR, and ADD based on detection details.
    
    Criteria:
    - True Positive: Detection within 30 timesteps AFTER actual changepoint
    - False Positive: Detection not matching any actual changepoint within criteria
    - ADD: From metadata sheet for actual detected changepoints
    - Handle missed detections: When no detections exist but changepoints do
    """
    # Extract actual changepoints
    actual_changepoints = extract_actual_changepoints(metadata_df)

    if not actual_changepoints:
        logger.error("No actual changepoints found")
        return {}

    # Handle case where no detections occurred at all (complete missed detection)
    if details_df.empty:
        logger.warning(
            "No detection data found - treating as complete missed detection"
        )
        # Assume both Traditional and Horizon detection types were attempted
        n_trials = (
            metadata_df["change_point"].notna().sum()
            if "change_point" in metadata_df.columns
            else 1
        )
        total_actual_changes = len(actual_changepoints) * max(n_trials, 1)

        # Create metrics for missed detection case
        missed_detection_metrics = {
            "TPR": 0.0,
            "FPR": 0.0,  # No false positives if no detections
            "ADD": float("inf"),  # Infinite delay since no detection occurred
            "true_positives": 0,
            "false_positives": 0,
            "total_actual_changes": total_actual_changes,
            "detection_delays": [],
        }

        results = {
            "Traditional": missed_detection_metrics.copy(),
            "Horizon": missed_detection_metrics.copy(),
        }

        logger.info(
            f"Traditional - TPR: 0.000, FPR: 0.000000, ADD: inf (missed detection)"
        )
        logger.info(f"Horizon - TPR: 0.000, FPR: 0.000000, ADD: inf (missed detection)")

        return results

    # Group detections by type
    detection_types = (
        details_df["Type"].unique() if "Type" in details_df.columns else []
    )
    trials = details_df["Trial"].unique() if "Trial" in details_df.columns else []
    n_trials = len(trials)

    # Handle case where DataFrame exists but has no valid detection types
    if len(detection_types) == 0 or all(pd.isna(detection_types)):
        logger.warning(
            "No valid detection types found - treating as complete missed detection"
        )
        # Try to infer number of trials from the data
        if n_trials == 0:
            n_trials = (
                metadata_df["change_point"].notna().sum()
                if "change_point" in metadata_df.columns
                else 1
            )
        total_actual_changes = len(actual_changepoints) * max(n_trials, 1)

        # Create metrics for missed detection case
        missed_detection_metrics = {
            "TPR": 0.0,
            "FPR": 0.0,  # No false positives if no detections
            "ADD": float("inf"),  # Infinite delay since no detection occurred
            "true_positives": 0,
            "false_positives": 0,
            "total_actual_changes": total_actual_changes,
            "detection_delays": [],
        }

        results = {
            "Traditional": missed_detection_metrics.copy(),
            "Horizon": missed_detection_metrics.copy(),
        }

        logger.info(
            f"Traditional - TPR: 0.000, FPR: 0.000000, ADD: inf (missed detection)"
        )
        logger.info(f"Horizon - TPR: 0.000, FPR: 0.000000, ADD: inf (missed detection)")

        return results

    logger.info(f"Processing {n_trials} trials with detection types: {detection_types}")
    logger.info(f"Actual changepoints: {actual_changepoints}")
    
    results = {}
    
    for det_type in detection_types:
        type_data = details_df[details_df["Type"] == det_type].copy()
        
        true_positives = 0
        false_positives = 0
        total_actual_changes = len(actual_changepoints) * n_trials
        detection_delays = []
        
        # Process each detection
        for _, detection in type_data.iterrows():
            detection_index = detection["Detection Index"]
            nearest_cp = detection["Nearest True CP"]
            distance_to_cp = detection["Distance to CP"]
            is_within_30 = detection.get(
                "Is Within 20 Steps", False
            )  # May be named differently

            # Check if this is a true positive based on distance criteria
            if pd.notna(nearest_cp) and pd.notna(distance_to_cp):
                distance_to_cp = int(distance_to_cp)
                
                # True positive: within 30 steps AFTER the changepoint
                if distance_to_cp >= 0 and distance_to_cp <= 30:
                    true_positives += 1
                    detection_delays.append(distance_to_cp)
                    logger.debug(
                        f"TP: {det_type} detection at {detection_index}, CP at {nearest_cp}, delay {distance_to_cp}"
                    )
                else:
                    false_positives += 1
                    logger.debug(
                        f"FP: {det_type} detection at {detection_index}, CP at {nearest_cp}, distance {distance_to_cp}"
                    )
            else:
                false_positives += 1
                logger.debug(
                    f"FP: {det_type} detection at {detection_index}, no valid CP match"
                )
        
        # Calculate metrics
        tpr = true_positives / total_actual_changes if total_actual_changes > 0 else 0.0
        
        # For FPR calculation - use total possible false positive opportunities
        # Assuming 200 timesteps total minus actual changepoints
        total_timesteps = 200
        total_non_change_points = (
            total_timesteps - len(actual_changepoints)
        ) * n_trials
        fpr = (
            false_positives / total_non_change_points
            if total_non_change_points > 0
            else 0.0
        )

        # Average detection delay for true positives
        add = np.mean(detection_delays) if detection_delays else float("inf")

        # Try to get ADD from metadata if available
        if not detection_delays and det_type in ["Traditional", "Horizon"]:
            add_col = f"{det_type.lower()}_avg_delay"
            if add_col in metadata_df.columns:
                metadata_add = metadata_df[add_col].dropna()
                if len(metadata_add) > 0:
                    add = np.mean(metadata_add)
        
        results[det_type] = {
            "TPR": tpr,
            "FPR": fpr,
            "ADD": add if add != float("inf") else None,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "total_actual_changes": total_actual_changes,
            "detection_delays": detection_delays,
        }

        add_str = f"{add:.2f}" if add != float("inf") else "inf"
        logger.info(f"{det_type} - TPR: {tpr:.3f}, FPR: {fpr:.6f}, ADD: {add_str}")
    
    return results


def analyze_detection_results(excel_path: Path) -> Dict[str, Any]:
    """Analyze detection results from Excel file"""
    try:
        # Read the Excel file
        excel_data = pd.ExcelFile(excel_path)
        logger.debug(f"Available sheets: {excel_data.sheet_names}")

        # Read ChangePointMetadata sheet
        metadata_df = pd.read_excel(excel_path, sheet_name="ChangePointMetadata")
        logger.debug(f"ChangePointMetadata shape: {metadata_df.shape}")

        # Read Detection Details sheet
        details_df = pd.read_excel(excel_path, sheet_name="Detection Details")
        logger.debug(f"Detection Details shape: {details_df.shape}")

        # Calculate metrics
        metrics = calculate_metrics(details_df, metadata_df)

        return {
            "metrics": metrics,
            "actual_changepoints": extract_actual_changepoints(metadata_df),
            "raw_data": {
                "metadata": metadata_df.to_dict("records"),
                "details": details_df.to_dict("records"),
            },
        }

    except Exception as e:
        logger.error(f"Failed to analyze {excel_path}: {e}")
        return {}


def group_experiments_by_parameters(
    all_results: Dict[str, Any],
) -> Dict[str, Dict[str, List[Dict]]]:
    """Group experiments by network type and parameter combinations"""
    grouped = defaultdict(lambda: defaultdict(list))

    for exp_dir, exp_data in all_results.items():
        if "config" not in exp_data or "results" not in exp_data:
            continue
            
        params = exp_data["config"]
        results = exp_data["results"]

        if "metrics" not in results:
            continue
            
        network = params.get("network", "unknown").upper()

        # Create parameter key based on betting type and specific parameters
        param_key = create_parameter_key(params)

        if param_key:
            grouped[network][param_key].append(
                {"params": params, "metrics": results["metrics"], "exp_dir": exp_dir}
            )

    return grouped


def create_parameter_key(params: Dict[str, Any]) -> Optional[str]:
    """Create a parameter key for grouping experiments"""
    betting_type = params.get("betting_type", "unknown")
    distance = params.get("distance", "unknown")
    threshold = params.get("threshold", 50.0)

    if betting_type == "power":
        epsilon = params.get("epsilon")
        if epsilon is not None:
            return f"power_eps_{epsilon}_{distance}"
    elif betting_type == "beta":
        beta_a = params.get("beta_a")
        if beta_a is not None:
            return f"beta_a_{beta_a}_{distance}"
    elif betting_type == "mixture":
        return f"mixture_{distance}"

    # Also group by threshold
    return f"threshold_{int(threshold)}_{distance}"


def debug_parameter_coverage(grouped_results: Dict[str, Dict[str, List[Dict]]]):
    """Debug function to show parameter coverage for each network"""
    print("\n" + "=" * 60)
    print("PARAMETER COVERAGE ANALYSIS")
    print("=" * 60)

    for network, network_data in grouped_results.items():
        print(f"\n{network} Network:")
        print(f"  Total parameter combinations: {len(network_data)}")

        # Check power betting coverage
        power_coverage = {0.2: [], 0.5: [], 0.7: [], 0.9: []}
        for param_key in network_data.keys():
            for epsilon in [0.2, 0.5, 0.7, 0.9]:
                if f"power_eps_{epsilon}" in param_key:
                    power_coverage[epsilon].append(param_key)

        print("  Power betting coverage:")
        for epsilon, keys in power_coverage.items():
            print(f"    ε={epsilon}: {len(keys)} combinations")
            if not keys:
                print(f"      → Missing!")
            else:
                # Check which of these actually have data
                with_data = 0
                for key in keys:
                    if network_data[key]:  # Check if experiments list is not empty
                        for exp in network_data[key]:
                            if exp["metrics"]:  # Check if metrics exist
                                with_data += 1
                                break
                if with_data < len(keys):
                    print(
                        f"      → {len(keys) - with_data} combinations have no detection data"
                    )

        # Check beta betting coverage
        beta_coverage = {0.3: [], 0.5: [], 0.7: []}
        for param_key in network_data.keys():
            for beta_a in [0.3, 0.5, 0.7]:
                if f"beta_a_{beta_a}" in param_key:
                    beta_coverage[beta_a].append(param_key)

        print("  Beta betting coverage:")
        for beta_a, keys in beta_coverage.items():
            print(f"    α={beta_a}: {len(keys)} combinations")
            if not keys:
                print(f"      → Missing!")

        # Check threshold distribution
        threshold_distribution = {20: 0, 50: 0, 100: 0}
        for param_key, experiments in network_data.items():
            for exp in experiments:
                threshold = exp["params"].get("threshold", 50.0)
                if int(threshold) in threshold_distribution:
                    threshold_distribution[int(threshold)] += 1

        print("  Threshold distribution:")
        for threshold, count in threshold_distribution.items():
            print(f"    {threshold}: {count} experiments")

        # Show all parameter keys for this network
        print("  All parameter keys:")
        for key in sorted(network_data.keys()):
            print(f"    {key}")


def create_sensitivity_table(
    grouped_results: Dict[str, Dict[str, List[Dict]]],
) -> pd.DataFrame:
    """Create the sensitivity analysis table matching the screenshot format"""
    table_data = []

    networks = ["SBM", "ER", "BA", "WS"]  # Using WS instead of NWS
    metrics = ["TPR", "FPR", "ADD"]

    for network in networks:
        # Find network data (case insensitive)
        network_data = None
        for net_key in grouped_results.keys():
            if net_key.upper() == network or net_key.lower() == network.lower():
                network_data = grouped_results[net_key]
                break

        if network_data is None:
            logger.warning(f"No data found for network: {network}")
            # Create empty rows for this network
            for metric in metrics:
                row_data = {"Network": network, "Metric": metric}
                table_data.append(row_data)
            continue
            
        # Process each metric type
        for metric in metrics:
            row_data = {"Network": network, "Metric": metric}

            # Aggregate metrics by parameter categories

            # 1. Power betting parameters (aggregate across distances)
            for epsilon in [0.2, 0.5, 0.7, 0.9]:
                power_values = []
                matching_keys = []
                for param_key, experiments in network_data.items():
                    if f"power_eps_{epsilon}" in param_key:
                        matching_keys.append(param_key)
                        for exp in experiments:
                            for det_type in ["Traditional", "Horizon"]:
                                if det_type in exp["metrics"]:
                                    val = exp["metrics"][det_type].get(metric)
                                    if val is not None and val != float("inf"):
                                        power_values.append(val)

                if power_values:
                    row_data[f"ε={epsilon}"] = np.mean(power_values)

            # 2. Mixture betting (aggregate across distances)
            mixture_values = []
            for param_key, experiments in network_data.items():
                if "mixture" in param_key:
                    for exp in experiments:
                        for det_type in ["Traditional", "Horizon"]:
                            if det_type in exp["metrics"]:
                                val = exp["metrics"][det_type].get(metric)
                                if val is not None and val != float("inf"):
                                    mixture_values.append(val)

            if mixture_values:
                row_data["Mixture"] = np.mean(mixture_values)

            # 3. Beta betting parameters (aggregate across distances)
            for beta_a in [0.3, 0.5, 0.7]:
                beta_values = []
                for param_key, experiments in network_data.items():
                    if f"beta_a_{beta_a}" in param_key:
                        for exp in experiments:
                            for det_type in ["Traditional", "Horizon"]:
                                if det_type in exp["metrics"]:
                                    val = exp["metrics"][det_type].get(metric)
                                    if val is not None and val != float("inf"):
                                        beta_values.append(val)

                if beta_values:
                    row_data[f"α={beta_a}"] = np.mean(beta_values)

            # 4. Distance measures (aggregate across betting functions)
            distance_map = {
                "euclidean": "Euc.",
                "mahalanobis": "Mah.",
                "cosine": "Cos.",
                "chebyshev": "Cheb.",
            }

            for distance, col_name in distance_map.items():
                distance_values = []
                for param_key, experiments in network_data.items():
                    if distance in param_key:
                        for exp in experiments:
                            for det_type in ["Traditional", "Horizon"]:
                                if det_type in exp["metrics"]:
                                    val = exp["metrics"][det_type].get(metric)
                                    if val is not None and val != float("inf"):
                                        distance_values.append(val)

                if distance_values:
                    row_data[col_name] = np.mean(distance_values)

            # 5. Threshold values (look for experiments with specific threshold values)
            for threshold in [20, 50, 100]:
                threshold_values = []
                for param_key, experiments in network_data.items():
                    for exp in experiments:
                        # Check if this experiment has the specific threshold value
                        exp_threshold = exp["params"].get("threshold", 50.0)
                        if int(exp_threshold) == threshold:
                            for det_type in ["Traditional", "Horizon"]:
                                if det_type in exp["metrics"]:
                                    val = exp["metrics"][det_type].get(metric)
                                    if val is not None and val != float("inf"):
                                        threshold_values.append(val)

                if threshold_values:
                    row_data[str(threshold)] = np.mean(threshold_values)

            table_data.append(row_data)

    return pd.DataFrame(table_data)


def map_param_to_column(param_key: str) -> Optional[str]:
    """Map parameter key to column name in the output table"""
    if "power_eps_0.2" in param_key:
        return "ε=0.2"
    elif "power_eps_0.5" in param_key:
        return "ε=0.5"
    elif "power_eps_0.7" in param_key:
        return "ε=0.7"
    elif "power_eps_0.9" in param_key:
        return "ε=0.9"
    elif "mixture" in param_key:
        return "Mixture"
    elif "beta_a_0.3" in param_key:
        return "α=0.3"
    elif "beta_a_0.5" in param_key:
        return "α=0.5"
    elif "beta_a_0.7" in param_key:
        return "α=0.7"
    elif "euclidean" in param_key:
        return "Euc."
    elif "mahalanobis" in param_key:
        return "Mah."
    elif "cosine" in param_key:
        return "Cos."
    elif "chebyshev" in param_key:
        return "Cheb."
    elif "threshold_20" in param_key:
        return "20"
    elif "threshold_50" in param_key:
        return "50"
    elif "threshold_100" in param_key:
        return "100"

    return None


def analyze_all_experiments(
    base_dir: str = "results/sensitivity_analysis_5",
) -> Dict[str, Any]:
    """Analyze all experiments and return results"""
    experiment_dirs = find_experiment_directories(base_dir)
    
    if not experiment_dirs:
        logger.error("No experiments found")
        return {}
    
    all_results = {}
    
    for exp_dir in experiment_dirs:
        logger.info(f"Analyzing {exp_dir.name}")
        
        # Load config
        config = load_experiment_config(exp_dir / "config.yaml")
        if not config:
            continue

        # Extract parameters
        params = extract_parameters_from_config(config)
        
        # Analyze detection results
        results = analyze_detection_results(exp_dir / "detection_results.xlsx")
        
        if results:
            all_results[exp_dir.name] = {"config": params, "results": results}
        else:
            logger.warning(f"Failed to analyze {exp_dir.name}")
    
    logger.info(f"Successfully processed {len(all_results)} experiments")
    return all_results


def format_latex_table(sensitivity_table: pd.DataFrame) -> str:
    """Generate LaTeX table from sensitivity analysis results"""
    latex_rows = []

    # Define column mappings for each section
    power_columns = ["ε=0.2", "ε=0.5", "ε=0.7", "ε=0.9", "Mixture"]
    beta_columns = ["α=0.3", "α=0.5", "α=0.7"]
    distance_columns = ["Euc.", "Mah.", "Cos.", "Cheb."]
    threshold_columns = ["20", "50", "100"]

    def format_value(val, metric):
        """Format value with appropriate precision"""
        if pd.isna(val) or val == float("inf"):
            return "--"
        if metric == "FPR":
            return f"{val:.3f}"
        elif metric == "ADD":
            return f"{val:.1f}"
        else:  # TPR
            return f"{val:.2f}"

    def format_bold_value(val, metric):
        """Format bold value with appropriate precision"""
        if pd.isna(val) or val == float("inf"):
            return "--"
        if metric == "FPR":
            return f"\\textbf{{{val:.3f}}}"
        elif metric == "ADD":
            return f"\\textbf{{{val:.1f}}}"
        else:  # TPR
            return f"\\textbf{{{val:.2f}}}"

    # Process each network
    networks = ["SBM", "ER", "BA", "WS"]
    metrics = ["TPR", "FPR", "ADD"]

    for network in networks:
        network_data = sensitivity_table[sensitivity_table["Network"] == network]

        if network_data.empty:
            logger.warning(f"No data found for network: {network}")
            continue

        for i, metric in enumerate(metrics):
            metric_data = network_data[network_data["Metric"] == metric]

            if metric_data.empty:
                continue

            row = metric_data.iloc[
                0
            ]  # Should only be one row per network-metric combination
            row_parts = []

            # Add network name (only for first metric)
            if i == 0:
                row_parts.append(f"\\multirow{{3}}{{*}}{{{network}}}")
            else:
                row_parts.append("")

            row_parts.append(metric)

            # Power betting values (ε + Mixture)
            power_values = []
            for col in power_columns:
                if col in row and pd.notna(row[col]):
                    power_values.append(row[col])
                else:
                    power_values.append(None)

            # Find best value among power betting section
            valid_power_values = [
                v
                for v in power_values
                if v is not None and not pd.isna(v) and v != float("inf")
            ]
            if valid_power_values:
                if metric == "TPR":
                    best_power = max(valid_power_values)
                else:  # FPR and ADD - lower is better
                    best_power = min(valid_power_values)
            else:
                best_power = None

            # Format power values - bold only the single best in this section
            for val in power_values:
                if (
                    val is not None
                    and pd.notna(val)
                    and val != float("inf")
                    and best_power is not None
                    and abs(val - best_power) < 0.001
                ):
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Beta betting values (α)
            beta_values = []
            for col in beta_columns:
                if col in row and pd.notna(row[col]):
                    beta_values.append(row[col])
                else:
                    beta_values.append(None)

            # Find best value among beta betting section
            valid_beta_values = [
                v
                for v in beta_values
                if v is not None and not pd.isna(v) and v != float("inf")
            ]
            if valid_beta_values:
                if metric == "TPR":
                    best_beta = max(valid_beta_values)
                else:  # FPR and ADD - lower is better
                    best_beta = min(valid_beta_values)
            else:
                best_beta = None

            # Format beta values - bold only the single best in this section
            for val in beta_values:
                if (
                    val is not None
                    and pd.notna(val)
                    and val != float("inf")
                    and best_beta is not None
                    and abs(val - best_beta) < 0.001
                ):
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Distance measure values
            distance_values = []
            for col in distance_columns:
                if col in row and pd.notna(row[col]):
                    distance_values.append(row[col])
                else:
                    distance_values.append(None)

            # Find best distance value within distance measures section
            valid_distance_values = [
                v
                for v in distance_values
                if v is not None and not pd.isna(v) and v != float("inf")
            ]
            if valid_distance_values:
                if metric == "TPR":
                    best_distance = max(valid_distance_values)
                else:  # FPR and ADD - lower is better
                    best_distance = min(valid_distance_values)
            else:
                best_distance = None

            # Format distance values - bold only the single best in this section
            for val in distance_values:
                if (
                    val is not None
                    and pd.notna(val)
                    and val != float("inf")
                    and best_distance is not None
                    and abs(val - best_distance) < 0.001
                ):
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Threshold values
            threshold_values = []
            for col in threshold_columns:
                if col in row and pd.notna(row[col]):
                    threshold_values.append(row[col])
                else:
                    threshold_values.append(None)

            # Find best threshold value within threshold section
            valid_threshold_values = [
                v
                for v in threshold_values
                if v is not None and not pd.isna(v) and v != float("inf")
            ]
            if valid_threshold_values:
                if metric == "TPR":
                    best_threshold = max(valid_threshold_values)
                else:  # FPR and ADD - lower is better
                    best_threshold = min(valid_threshold_values)
            else:
                best_threshold = None

            # Format threshold values - bold only the single best in this section
            for val in threshold_values:
                if (
                    val is not None
                    and pd.notna(val)
                    and val != float("inf")
                    and best_threshold is not None
                    and abs(val - best_threshold) < 0.001
                ):
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Join row parts
            latex_row = " & ".join(row_parts) + " \\\\"
            latex_rows.append(latex_row)

        # Add horizontal line after each network (except the last one)
        if network != networks[-1]:
            latex_rows.append("\\hline")

    # Create complete LaTeX table
    latex_table = (
        """\\begin{table}[htbp]
\\centering
\\caption{Parameter sensitivity analysis results across network types}
\\label{tab:parameter_sensitivity}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{|c|c|cccc|c|ccc|cccc|ccc|}
\\hline
\\multirow{2}{*}{Network} & \\multirow{2}{*}{Metric} & \\multicolumn{5}{c|}{Power \\& Mixture Betting} & \\multicolumn{3}{c|}{Beta Betting} & \\multicolumn{4}{c|}{Distance Measures} & \\multicolumn{3}{c|}{Thresholds} \\\\
\\cline{3-16}
& & $\\varepsilon=0.2$ & $\\varepsilon=0.5$ & $\\varepsilon=0.7$ & $\\varepsilon=0.9$ & Mix & $\\alpha=0.3$ & $\\alpha=0.5$ & $\\alpha=0.7$ & Euc. & Mah. & Cos. & Cheb. & 20 & 50 & 100 \\\\
\\hline
"""
        + "\n".join(latex_rows)
        + """
\\hline
\\end{tabular}%
}
\\end{table}"""
    )

    return latex_table


def save_results(
    all_results: Dict[str, Any],
    grouped_results: Dict,
    sensitivity_table: pd.DataFrame,
    output_dir: str = "results/sensitivity_analysis_5_summary",
):
    """Save all results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    detailed_data = []
    for exp_name, exp_data in all_results.items():
        config = exp_data.get("config", {})
        results = exp_data.get("results", {})

        if "metrics" in results:
            for det_type, metrics in results["metrics"].items():
                row = {
                    "experiment": exp_name,
                    "network": config.get("network", "unknown"),
                    "betting_type": config.get("betting_type", "unknown"),
                    "detection_type": det_type,
                    "TPR": metrics.get("TPR"),
                    "FPR": metrics.get("FPR"),
                    "ADD": metrics.get("ADD"),
                    "true_positives": metrics.get("true_positives"),
                    "false_positives": metrics.get("false_positives"),
                    "total_actual_changes": metrics.get("total_actual_changes"),
                }

                # Add parameter-specific columns
                for key, value in config.items():
                    if key not in row:
                        row[key] = value

                detailed_data.append(row)

    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = os.path.join(output_dir, "detailed_results.csv")
    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed results to {detailed_path}")

    # Save sensitivity table
    table_path = os.path.join(output_dir, "sensitivity_table.csv")
    sensitivity_table.to_csv(table_path, index=False)
    logger.info(f"Saved sensitivity table to {table_path}")

    # Generate and save LaTeX table
    latex_table = format_latex_table(sensitivity_table)
    latex_path = os.path.join(output_dir, "sensitivity_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    logger.info(f"Saved LaTeX table to {latex_path}")

    # Save raw results as JSON
    json_path = os.path.join(output_dir, "raw_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved raw results to {json_path}")

    return detailed_path, table_path, json_path, latex_path


def main():
    """Main entry point"""
    logger.info("Starting sensitivity analysis")
    
    # Check if results directory exists
    results_dir = "results/sensitivity_analysis_5"
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Analyze all experiments
    all_results = analyze_all_experiments(results_dir)
    
    if not all_results:
        logger.error("No results to analyze")
        return

    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments analyzed: {len(all_results)}")
        
    # Group results by parameters
    grouped_results = group_experiments_by_parameters(all_results)
    print(f"Networks found: {list(grouped_results.keys())}")

    for network, params in grouped_results.items():
        print(f"{network}: {len(params)} parameter combinations")
        print(f"  Parameter keys: {list(params.keys())}")

    # Debug parameter coverage
    debug_parameter_coverage(grouped_results)

    # Create sensitivity table
    sensitivity_table = create_sensitivity_table(grouped_results)
    print(f"Sensitivity table shape: {sensitivity_table.shape}")

    # Save results
    detailed_path, table_path, json_path, latex_path = save_results(
        all_results, grouped_results, sensitivity_table
    )

    print(f"\nFiles saved:")
    print(f"- Detailed results: {detailed_path}")
    print(f"- Sensitivity table: {table_path}")
    print(f"- LaTeX table: {latex_path}")
    print(f"- Raw results: {json_path}")

    # Display a preview of the sensitivity table
    print(f"\n{'='*60}")
    print("SENSITIVITY TABLE PREVIEW")
    print(f"{'='*60}")
    print(sensitivity_table.head(10).to_string(index=False))

    # Display LaTeX table preview
    print(f"\n{'='*60}")
    print("LATEX TABLE PREVIEW")
    print(f"{'='*60}")
    latex_table = format_latex_table(sensitivity_table)
    # Show just the data rows (not the full table structure)
    latex_lines = latex_table.split("\n")
    data_lines = [
        line
        for line in latex_lines
        if line.strip() and not line.startswith("\\") and "&" in line
    ]
    for line in data_lines[:12]:  # Show first 12 data rows
        print(line)
    print("...")
    print(f"\nFull LaTeX table saved to: {latex_path}")
    print("Copy this file content into your LaTeX document.")


if __name__ == "__main__":
    main()
