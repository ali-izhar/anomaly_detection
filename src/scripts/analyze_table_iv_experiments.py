#!/usr/bin/env python3
"""
Table IV Experiments Analysis Script

This script analyzes the structure of Table IV experiments that compare
Traditional Martingale, Horizon Martingale, CUSUM, and EWMA methods
across different network types and scenarios.

Each experiment folder contains:
- config.yaml: Experiment configuration
- detection_results.xlsx: Detection results with multiple sheets
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def parse_experiment_folder_name(folder_name: str) -> Dict[str, str]:
    """
    Parse Table IV experiment folder name to extract parameters.

    Format: {network}_{scenario}_{method}_{timestamp}_pid{pid}_{hash}

    Args:
        folder_name: Name of the experiment folder

    Returns:
        Dictionary with parsed parameters
    """
    parts = folder_name.split("_")

    # Find the method by looking for known method names
    methods = ["traditional", "horizon", "cusum", "ewma"]
    method_idx = None
    method = None

    for i, part in enumerate(parts):
        if part in methods:
            if (
                part == "traditional"
                and i + 1 < len(parts)
                and parts[i + 1] == "martingale"
            ):
                method = "traditional_martingale"
                method_idx = i
                break
            elif (
                part == "horizon"
                and i + 1 < len(parts)
                and parts[i + 1] == "martingale"
            ):
                method = "horizon_martingale"
                method_idx = i
                break
            elif part in ["cusum", "ewma"]:
                method = part
                method_idx = i
                break

    if method_idx is None:
        return {"error": f"Could not parse method from {folder_name}"}

    # Network and scenario are everything before the method
    network_scenario_parts = parts[:method_idx]

    # Network is the first part
    network = network_scenario_parts[0] if network_scenario_parts else "unknown"

    # Scenario is the remaining parts joined
    scenario = (
        "_".join(network_scenario_parts[1:])
        if len(network_scenario_parts) > 1
        else "unknown"
    )

    return {
        "network": network,
        "scenario": scenario,
        "method": method,
        "full_name": folder_name,
    }


def read_experiment_config(folder_path: str) -> Optional[Dict[str, Any]]:
    """
    Read experiment configuration from config.yaml.

    Args:
        folder_path: Path to the experiment folder

    Returns:
        Configuration dictionary or None if failed
    """
    config_path = os.path.join(folder_path, "config.yaml")

    if not os.path.exists(config_path):
        print(f"Warning: config.yaml not found in {folder_path}")
        return None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error reading config from {folder_path}: {e}")
        return None


def read_experiment_results(
    folder_path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Read experiment results from detection_results.xlsx.

    Args:
        folder_path: Path to the experiment folder

    Returns:
        Tuple of (change_point_metadata, detection_details, raw_detection_data)
    """
    results_path = os.path.join(folder_path, "detection_results.xlsx")

    if not os.path.exists(results_path):
        print(f"Warning: detection_results.xlsx not found in {folder_path}")
        return None, None, None

    try:
        # Read all sheets to see what's available
        excel_file = pd.ExcelFile(results_path)

        change_point_metadata = None
        detection_details = None
        raw_detection_data = None

        # Read ChangePointMetadata sheet
        if "ChangePointMetadata" in excel_file.sheet_names:
            change_point_metadata = pd.read_excel(
                results_path, sheet_name="ChangePointMetadata"
            )

        # Read Detection Details sheet
        if "Detection Details" in excel_file.sheet_names:
            detection_details = pd.read_excel(
                results_path, sheet_name="Detection Details"
            )

        # Read the first sheet as raw detection data
        raw_detection_data = pd.read_excel(results_path, sheet_name=0)

        return change_point_metadata, detection_details, raw_detection_data

    except Exception as e:
        print(f"Error reading results from {folder_path}: {e}")
        return None, None, None


def compute_performance_metrics(
    change_points_df: pd.DataFrame,
    detection_details_df: pd.DataFrame,
    method_type: str = "Traditional",
    n_trials: int = 10,
    detection_window: int = 40,
) -> Dict[str, float]:
    """
    Compute TPR, FPR, and ADD for Table IV experiments.

    Args:
        change_points_df: DataFrame with change point metadata
        detection_details_df: DataFrame with detection details
        method_type: "Traditional", "Horizon", or "Single" for non-martingale methods
        n_trials: Number of trials (default: 10)
        detection_window: Window size for valid detections after CP (default: 40)

    Returns:
        Dictionary with performance metrics
    """
    if change_points_df is None or detection_details_df is None:
        return {"TPR": 0.0, "FPR": 0.0, "ADD": float("inf"), "error": "Missing data"}

    # Extract change points
    change_points = sorted(change_points_df["change_point"].tolist())

    # Filter detections by method type (Traditional or Horizon) for martingale methods
    # For single methods (CUSUM, EWMA), use all detections
    if (
        method_type in ["Traditional", "Horizon"]
        and "Type" in detection_details_df.columns
    ):
        method_detections = detection_details_df[
            detection_details_df["Type"] == method_type
        ]
    else:
        # For single methods or when Type column doesn't exist, use all detections
        method_detections = detection_details_df

    # For CUSUM/EWMA, we can use the detection data but need to apply correct window logic
    # Detection window should be CP to CP+40 (within 40 steps AFTER the change point)
    detected_change_points = set()
    detection_delays = []
    false_positive_count = 0

    for _, detection in method_detections.iterrows():
        detection_idx = detection["Detection Index"]
        is_true_positive = False

        # Check if this detection is within the valid window of any change point
        for cp in change_points:
            # Valid detection window: CP to CP+40 (after the change point)
            if cp <= detection_idx <= cp + detection_window:
                detected_change_points.add(cp)
                delay = detection_idx - cp  # Delay is always positive (after CP)
                detection_delays.append(delay)
                is_true_positive = True
                break

        if not is_true_positive:
            false_positive_count += 1

    # Calculate TPR: detected change points / total change points
    total_change_points = len(change_points) * n_trials
    # Each detected change point counts across all trials where it was detected
    total_detected_change_points = len(detected_change_points) * n_trials

    TPR = (
        total_detected_change_points / total_change_points
        if total_change_points > 0
        else 0.0
    )

    # Calculate FPR as false positives per time unit
    # Total observation time = time span * number of trials
    # Subtract the valid detection windows from total time to get false positive opportunities
    total_time_span = 200  # Assuming 200 time steps based on the data
    valid_detection_time = (
        len(change_points) * detection_window
    )  # Total valid detection time per trial
    false_positive_opportunities_per_trial = total_time_span - valid_detection_time
    total_false_positive_opportunities = (
        false_positive_opportunities_per_trial * n_trials
    )

    FPR = (
        false_positive_count / total_false_positive_opportunities
        if total_false_positive_opportunities > 0
        else 0.0
    )

    # Calculate ADD from detection delays (only for true positives)
    ADD = np.mean(detection_delays) if detection_delays else float("inf")

    # Try to get pre-calculated ADD from ChangePointMetadata if available
    if (
        method_type == "Traditional"
        and "traditional_avg_delay" in change_points_df.columns
    ):
        precalc_add = change_points_df["traditional_avg_delay"].mean()
        if not np.isnan(precalc_add) and precalc_add > 0:
            ADD = precalc_add
    elif method_type == "Horizon" and "horizon_avg_delay" in change_points_df.columns:
        precalc_add = change_points_df["horizon_avg_delay"].mean()
        if not np.isnan(precalc_add) and precalc_add > 0:
            ADD = precalc_add

    return {
        "TPR": TPR,
        "FPR": FPR,
        "ADD": ADD,
        "total_change_points": total_change_points,
        "detected_change_points": total_detected_change_points,
        "total_false_positives": false_positive_count,
        "n_trials": n_trials,
        "method_type": method_type,
        "detection_delays": detection_delays,
    }


def analyze_single_experiment(
    folder_path: str, detection_window: int = 40
) -> Dict[str, Any]:
    """
    Analyze a single Table IV experiment folder.

    Args:
        folder_path: Path to the experiment folder
        detection_window: Window size for valid detections after CP (default: 40)

    Returns:
        Dictionary with experiment analysis
    """
    folder_name = os.path.basename(folder_path)
    parsed_params = parse_experiment_folder_name(folder_name)

    if "error" in parsed_params:
        return {
            "folder_name": folder_name,
            "error": parsed_params["error"],
            "status": "parse_error",
        }

    # Skip traditional_martingale folders since horizon_martingale folders contain both
    if parsed_params["method"] == "traditional_martingale":
        return {
            "folder_name": folder_name,
            "network": parsed_params["network"],
            "scenario": parsed_params["scenario"],
            "method": parsed_params["method"],
            "status": "skipped",
            "reason": "Redundant - data available in horizon_martingale folder",
        }

    # Read configuration
    config = read_experiment_config(folder_path)

    # Read results
    change_points, detection_details, raw_data = read_experiment_results(folder_path)

    # Basic analysis
    analysis = {
        "folder_name": folder_name,
        "network": parsed_params["network"],
        "scenario": parsed_params["scenario"],
        "method": parsed_params["method"],
        "status": "unknown",
    }

    # Analyze configuration
    if config:
        analysis["config_present"] = True
        analysis["detection_method"] = config.get("detection", {}).get(
            "method", "unknown"
        )
        analysis["trials"] = config.get("trials", {}).get("n_trials", 0)
        analysis["threshold"] = config.get("detection", {}).get("threshold", 0)
        analysis["betting_function"] = (
            config.get("detection", {})
            .get("betting_func_config", {})
            .get("name", "unknown")
        )
        analysis["distance_measure"] = (
            config.get("detection", {}).get("distance", {}).get("measure", "unknown")
        )
        analysis["enable_prediction"] = config.get("execution", {}).get(
            "enable_prediction", False
        )
    else:
        analysis["config_present"] = False

    # Analyze results
    if change_points is not None:
        analysis["change_points_present"] = True
        analysis["num_change_points"] = len(change_points)
        if "change_point" in change_points.columns:
            analysis["change_point_locations"] = sorted(
                change_points["change_point"].tolist()
            )
    else:
        analysis["change_points_present"] = False

    if detection_details is not None:
        analysis["detection_details_present"] = True
        analysis["num_detections"] = len(detection_details)
        if "Trial" in detection_details.columns:
            analysis["trials_with_detections"] = sorted(
                detection_details["Trial"].unique().tolist()
            )

        # Check if this has both Traditional and Horizon data
        if "Type" in detection_details.columns:
            types = detection_details["Type"].unique()
            analysis["detection_types"] = list(types)
            analysis["has_both_methods"] = "Traditional" in types and "Horizon" in types
        else:
            analysis["detection_types"] = ["Unknown"]
            analysis["has_both_methods"] = False
    else:
        analysis["detection_details_present"] = False

    if raw_data is not None:
        analysis["raw_data_present"] = True
        analysis["raw_data_rows"] = len(raw_data)
        analysis["raw_data_columns"] = list(raw_data.columns)
    else:
        analysis["raw_data_present"] = False

    # Compute performance metrics
    if change_points is not None and detection_details is not None:
        n_trials = analysis.get("trials", 10)

        # For horizon_martingale folders, compute metrics for both Traditional and Horizon
        if parsed_params["method"] == "horizon_martingale" and analysis.get(
            "has_both_methods", False
        ):
            # Traditional metrics
            trad_metrics = compute_performance_metrics(
                change_points,
                detection_details,
                "Traditional",
                n_trials,
                detection_window,
            )
            for key, value in trad_metrics.items():
                analysis[f"traditional_{key}"] = value

            # Horizon metrics
            horizon_metrics = compute_performance_metrics(
                change_points, detection_details, "Horizon", n_trials, detection_window
            )
            for key, value in horizon_metrics.items():
                analysis[f"horizon_{key}"] = value

        # For other methods (cusum, ewma), compute single metrics
        else:
            metrics = compute_performance_metrics(
                change_points, detection_details, "Single", n_trials, detection_window
            )
            analysis.update(metrics)

    # Determine overall status
    if (
        config
        and change_points is not None
        and detection_details is not None
        and raw_data is not None
    ):
        analysis["status"] = "complete"
    elif config and raw_data is not None:
        analysis["status"] = "partial"
    else:
        analysis["status"] = "incomplete"

    return analysis


def analyze_all_plot_using_best_params(
    results_dir: str, detection_window: int = 40
) -> List[Dict[str, Any]]:
    """
    Analyze all Table IV experiment folders.

    Args:
        results_dir: Path to the results directory containing experiment folders
        detection_window: Window size for valid detections after CP (default: 40)

    Returns:
        List of experiment analyses
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return []

    # Get all experiment folders
    experiment_folders = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    print(f"Found {len(experiment_folders)} experiment folders")
    print(f"Using detection window: {detection_window} steps after change point")

    analyses = []

    for i, folder_name in enumerate(experiment_folders):
        folder_path = os.path.join(results_dir, folder_name)
        print(f"Analyzing [{i+1}/{len(experiment_folders)}]: {folder_name}")

        analysis = analyze_single_experiment(folder_path, detection_window)
        analyses.append(analysis)

    return analyses


def create_table_iv_data(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create Table IV data from experiment analyses.

    Args:
        analyses: List of experiment analyses

    Returns:
        DataFrame with Table IV format
    """
    # Filter successful analyses (exclude skipped traditional_martingale folders)
    valid_analyses = [
        a
        for a in analyses
        if a.get("status") == "complete" and a.get("method") != "traditional_martingale"
    ]

    # Group by network and scenario
    grouped_data = {}

    for analysis in valid_analyses:
        network = analysis["network"]
        scenario = analysis["scenario"]
        method = analysis["method"]

        key = f"{network}_{scenario}"
        if key not in grouped_data:
            grouped_data[key] = {}

        # Handle horizon_martingale folders that contain both Traditional and Horizon data
        if method == "horizon_martingale" and analysis.get("has_both_methods", False):
            # Extract Traditional metrics
            if "traditional_TPR" in analysis:
                grouped_data[key]["traditional_martingale"] = {
                    "TPR": analysis["traditional_TPR"],
                    "FPR": analysis["traditional_FPR"],
                    "ADD": analysis["traditional_ADD"],
                }

            # Extract Horizon metrics
            if "horizon_TPR" in analysis:
                grouped_data[key]["horizon_martingale"] = {
                    "TPR": analysis["horizon_TPR"],
                    "FPR": analysis["horizon_FPR"],
                    "ADD": analysis["horizon_ADD"],
                }

        # Handle other methods (cusum, ewma)
        else:
            if "TPR" in analysis:
                grouped_data[key][method] = {
                    "TPR": analysis["TPR"],
                    "FPR": analysis["FPR"],
                    "ADD": analysis["ADD"],
                }

    # Create table rows
    table_rows = []

    # Define network order and scenario names
    network_scenarios = [
        ("sbm", "community_merge", "Community Merge"),
        ("sbm", "density_change", "Density Change"),
        ("sbm", "mixed_changes", "Mixed Changes"),
        ("er", "density_increase", "Density Increase"),
        ("er", "density_decrease", "Density Decrease"),
        ("ba", "parameter_shift", "Parameter Shift"),
        ("ba", "hub_addition", "Hub Addition"),
        ("ws", "rewiring_increase", "Rewiring Increase"),
        ("ws", "k_parameter_shift", "k Parameter Shift"),
    ]

    for network, scenario, display_name in network_scenarios:
        key = f"{network}_{scenario}"

        if key in grouped_data:
            data = grouped_data[key]

            # Get metrics for each method
            traditional = data.get(
                "traditional_martingale", {"TPR": 0, "FPR": 0, "ADD": float("inf")}
            )
            horizon = data.get(
                "horizon_martingale", {"TPR": 0, "FPR": 0, "ADD": float("inf")}
            )
            cusum = data.get("cusum", {"TPR": 0, "FPR": 0, "ADD": float("inf")})
            ewma = data.get("ewma", {"TPR": 0, "FPR": 0, "ADD": float("inf")})

            # Calculate improvements (Horizon vs Traditional)
            delay_improvement = 0.0
            tpr_improvement = 0.0

            if traditional["ADD"] != float("inf") and horizon["ADD"] != float("inf"):
                delay_improvement = (
                    (traditional["ADD"] - horizon["ADD"]) / traditional["ADD"]
                ) * 100

            if traditional["TPR"] > 0:
                tpr_improvement = (
                    (horizon["TPR"] - traditional["TPR"]) / traditional["TPR"]
                ) * 100

            row = {
                "Network Type": network.upper(),
                "Scenario": display_name,
                "Traditional_TPR": traditional["TPR"],
                "Traditional_FPR": traditional["FPR"],
                "Traditional_ADD": traditional["ADD"],
                "Horizon_TPR": horizon["TPR"],
                "Horizon_FPR": horizon["FPR"],
                "Horizon_ADD": horizon["ADD"],
                "CUSUM_TPR": cusum["TPR"],
                "CUSUM_FPR": cusum["FPR"],
                "CUSUM_ADD": cusum["ADD"],
                "EWMA_TPR": ewma["TPR"],
                "EWMA_FPR": ewma["FPR"],
                "EWMA_ADD": ewma["ADD"],
                "Delay_Improvement": delay_improvement,
                "TPR_Improvement": tpr_improvement,
            }

            table_rows.append(row)

    return pd.DataFrame(table_rows)


def format_table_iv_for_display(df: pd.DataFrame) -> str:
    """
    Format Table IV data for display matching the reference image.

    Args:
        df: DataFrame with Table IV data

    Returns:
        Formatted string for display
    """
    output = []
    output.append("TABLE IV")
    output.append(
        "THE TRADITIONAL AND HORIZON MARTINGALE RESULTS USE POWER BETTING WITH OPTIMAL ε VALUES AND DISTANCE METRICS FOR EACH"
    )
    output.append(
        "NETWORK TYPE. DETECTION DELAY REDUCTION (↓) AND TPR IMPROVEMENT (↑) COMPARE HORIZON MARTINGALE TO TRADITIONAL MARTINGALE."
    )
    output.append("")

    # Header
    header = f"{'Network Type':<15} {'TPR':<6} {'FPR':<6} {'ADD':<6} {'TPR':<6} {'FPR':<6} {'ADD':<6} {'TPR':<6} {'FPR':<6} {'ADD':<6} {'TPR':<6} {'FPR':<6} {'ADD':<6} {'Improvements':<12}"
    output.append(header)

    subheader = f"{'':<15} {'Traditional Martingale':<18} {'Horizon Martingale':<18} {'CUSUM':<18} {'EWMA':<18} {'Delay ↓':<8} {'TPR ↑':<6}"
    output.append(subheader)
    output.append("-" * 120)

    # Group by network type
    current_network = None

    for _, row in df.iterrows():
        network = row["Network Type"]
        scenario = row["Scenario"]

        # Network type header
        if network != current_network:
            if current_network is not None:
                output.append("")  # Blank line between networks

            network_name = {
                "SBM": "Stochastic Block Model (SBM)",
                "ER": "Erdős-Rényi (ER)",
                "BA": "Barabási-Albert (BA)",
                "WS": "Newman-Watts-Strogatz (WS)",
            }.get(network, network)

            output.append(f"{network_name}")
            current_network = network

        # Format values
        def format_val(val, metric_type):
            if metric_type == "ADD" and (val == float("inf") or val > 999):
                return "∞"
            elif metric_type in ["TPR", "FPR"]:
                return f"{val:.2f}"
            elif metric_type == "ADD":
                return f"{val:.1f}"
            elif metric_type == "improvement":
                return f"{val:.1f}%"
            else:
                return f"{val:.3f}"

        # Data row
        line = f"{scenario:<15} "
        line += f"{format_val(row['Traditional_TPR'], 'TPR'):<6} "
        line += f"{format_val(row['Traditional_FPR'], 'FPR'):<6} "
        line += f"{format_val(row['Traditional_ADD'], 'ADD'):<6} "
        line += f"{format_val(row['Horizon_TPR'], 'TPR'):<6} "
        line += f"{format_val(row['Horizon_FPR'], 'FPR'):<6} "
        line += f"{format_val(row['Horizon_ADD'], 'ADD'):<6} "
        line += f"{format_val(row['CUSUM_TPR'], 'TPR'):<6} "
        line += f"{format_val(row['CUSUM_FPR'], 'FPR'):<6} "
        line += f"{format_val(row['CUSUM_ADD'], 'ADD'):<6} "
        line += f"{format_val(row['EWMA_TPR'], 'TPR'):<6} "
        line += f"{format_val(row['EWMA_FPR'], 'FPR'):<6} "
        line += f"{format_val(row['EWMA_ADD'], 'ADD'):<6} "
        line += f"{format_val(row['Delay_Improvement'], 'improvement'):<8} "
        line += f"{format_val(row['TPR_Improvement'], 'improvement'):<6}"

        output.append(line)

    return "\n".join(output)


def format_table_iv_for_latex(df: pd.DataFrame) -> str:
    """
    Format Table IV data for LaTeX output.

    Args:
        df: DataFrame with Table IV data

    Returns:
        LaTeX table string
    """
    latex_lines = []

    # Table header
    latex_lines.append("\\begin{table*}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append(
        "\\caption{The Traditional and Horizon martingale results use Power betting with optimal $\\epsilon$ values and distance metrics for each network type. Detection delay reduction $(\\downarrow)$ and TPR improvement $(\\uparrow)$ compare Horizon martingale to Traditional martingale.}"
    )
    latex_lines.append("\\label{tab:comprehensive_results}")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}")
    latex_lines.append("\\hline")

    # Column headers
    latex_lines.append(
        "\\multirow{2}{*}{\\textbf{Network Type}} & \\multicolumn{3}{c|}{\\textbf{Traditional Martingale}} & \\multicolumn{3}{c|}{\\textbf{Horizon Martingale}} & \\multicolumn{3}{c|}{\\textbf{CUSUM}} & \\multicolumn{3}{c|}{\\textbf{EWMA}} & \\multicolumn{2}{c|}{\\textbf{Improvements}} \\\\"
    )
    latex_lines.append("\\cline{2-15}")
    latex_lines.append(
        " & \\textbf{TPR} & \\textbf{FPR} & \\textbf{ADD} & \\textbf{TPR} & \\textbf{FPR} & \\textbf{ADD} & \\textbf{TPR} & \\textbf{FPR} & \\textbf{ADD} & \\textbf{TPR} & \\textbf{FPR} & \\textbf{ADD} & \\textbf{Delay $\\downarrow$} & \\textbf{TPR $\\uparrow$} \\\\"
    )
    latex_lines.append("\\hline")

    # Helper function to format values
    def format_latex_val(val, metric_type):
        if metric_type == "ADD" and (val == float("inf") or val > 999):
            return "$\\infty$"
        elif metric_type in ["TPR", "FPR"]:
            return f"{val:.3f}"
        elif metric_type == "ADD":
            return f"{val:.1f}" if val < 100 else f"{val:.0f}"
        elif metric_type == "improvement":
            return f"{val:.1f}\\%"
        else:
            return f"{val:.3f}"

    # Group by network type
    current_network = None
    network_name_map = {
        "SBM": "Stochastic Block Model (SBM)",
        "ER": "Erdős-Rényi (ER)",
        "BA": "Barabási-Albert (BA)",
        "WS": "Newman-Watts-Strogatz (NWS)",
    }

    for _, row in df.iterrows():
        network = row["Network Type"]
        scenario = row["Scenario"]

        # Network type header
        if network != current_network:
            if current_network is not None:
                latex_lines.append("\\hline")  # Separator between networks

            network_display = network_name_map.get(network, network)
            latex_lines.append(
                f"\\multicolumn{{15}}{{|c|}}{{\\textbf{{{network_display}}}}} \\\\"
            )
            latex_lines.append("\\hline")
            current_network = network

        # Data row
        line_parts = [scenario]

        # Traditional Martingale
        line_parts.append(format_latex_val(row["Traditional_TPR"], "TPR"))
        line_parts.append(format_latex_val(row["Traditional_FPR"], "FPR"))
        line_parts.append(format_latex_val(row["Traditional_ADD"], "ADD"))

        # Horizon Martingale
        line_parts.append(format_latex_val(row["Horizon_TPR"], "TPR"))
        line_parts.append(format_latex_val(row["Horizon_FPR"], "FPR"))
        line_parts.append(format_latex_val(row["Horizon_ADD"], "ADD"))

        # CUSUM
        line_parts.append(format_latex_val(row["CUSUM_TPR"], "TPR"))
        line_parts.append(format_latex_val(row["CUSUM_FPR"], "FPR"))
        line_parts.append(format_latex_val(row["CUSUM_ADD"], "ADD"))

        # EWMA
        line_parts.append(format_latex_val(row["EWMA_TPR"], "TPR"))
        line_parts.append(format_latex_val(row["EWMA_FPR"], "FPR"))
        line_parts.append(format_latex_val(row["EWMA_ADD"], "ADD"))

        # Improvements
        line_parts.append(format_latex_val(row["Delay_Improvement"], "improvement"))
        line_parts.append(format_latex_val(row["TPR_Improvement"], "improvement"))

        latex_lines.append(" & ".join(line_parts) + " \\\\")

    # Table footer
    latex_lines.append("\\hline")
    latex_lines.append("\\multicolumn{15}{|c|}{\\textbf{MIT Reality Dataset}} \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)


def save_table_iv_results(df: pd.DataFrame, output_path: str) -> None:
    """Save Table IV results to Excel file and LaTeX file."""
    try:
        # Save Excel file
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Save formatted table
            df.to_excel(writer, sheet_name="Table IV Results", index=False)

        print(f"✓ Table IV results saved to: {output_path}")

        # Save LaTeX file
        latex_output_path = output_path.replace(".xlsx", ".tex")
        latex_table = format_table_iv_for_latex(df)

        with open(latex_output_path, "w", encoding="utf-8") as f:
            f.write(latex_table)

        print(f"✓ LaTeX table saved to: {latex_output_path}")

    except Exception as e:
        print(f"Error saving Table IV results: {e}")


def print_structure_summary(analyses: List[Dict[str, Any]]) -> None:
    """Print a summary of the Table IV experiment structure."""
    print("\n" + "=" * 80)
    print("TABLE IV EXPERIMENTS STRUCTURE SUMMARY")
    print("=" * 80)

    total_experiments = len(analyses)
    complete_experiments = len([a for a in analyses if a.get("status") == "complete"])
    partial_experiments = len([a for a in analyses if a.get("status") == "partial"])
    incomplete_experiments = len(
        [a for a in analyses if a.get("status") == "incomplete"]
    )
    error_experiments = len([a for a in analyses if a.get("status") == "parse_error"])

    print(f"Total experiments: {total_experiments}")
    print(f"Complete: {complete_experiments}")
    print(f"Partial: {partial_experiments}")
    print(f"Incomplete: {incomplete_experiments}")
    print(f"Parse errors: {error_experiments}")

    # Group by network type
    print(f"\nBy Network Type:")
    networks = {}
    for analysis in analyses:
        if analysis.get("status") != "parse_error":
            network = analysis.get("network", "unknown")
            if network not in networks:
                networks[network] = []
            networks[network].append(analysis)

    for network, network_analyses in networks.items():
        print(f"  {network}: {len(network_analyses)} experiments")

        # Group by scenario within network
        scenarios = {}
        for analysis in network_analyses:
            scenario = analysis.get("scenario", "unknown")
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(analysis)

        for scenario, scenario_analyses in scenarios.items():
            print(f"    {scenario}: {len(scenario_analyses)} experiments")

            # Show methods for this scenario
            methods = [a.get("method", "unknown") for a in scenario_analyses]
            print(f"      Methods: {', '.join(sorted(set(methods)))}")

    # Method distribution
    print(f"\nBy Method:")
    methods = {}
    for analysis in analyses:
        if analysis.get("status") != "parse_error":
            method = analysis.get("method", "unknown")
            if method not in methods:
                methods[method] = 0
            methods[method] += 1

    for method, count in sorted(methods.items()):
        print(f"  {method}: {count} experiments")

    # Configuration analysis
    print(f"\nConfiguration Analysis:")
    configs_with_data = [a for a in analyses if a.get("config_present")]
    if configs_with_data:
        # Detection methods
        detection_methods = {}
        for analysis in configs_with_data:
            det_method = analysis.get("detection_method", "unknown")
            if det_method not in detection_methods:
                detection_methods[det_method] = 0
            detection_methods[det_method] += 1

        print(f"  Detection methods: {detection_methods}")

        # Trials
        trials = [
            a.get("trials", 0) for a in configs_with_data if a.get("trials", 0) > 0
        ]
        if trials:
            print(
                f"  Trials per experiment: {min(trials)}-{max(trials)} (avg: {np.mean(trials):.1f})"
            )

        # Betting functions
        betting_funcs = {}
        for analysis in configs_with_data:
            betting_func = analysis.get("betting_function", "unknown")
            if betting_func not in betting_funcs:
                betting_funcs[betting_func] = 0
            betting_funcs[betting_func] += 1

        print(f"  Betting functions: {betting_funcs}")

        # Distance measures
        distance_measures = {}
        for analysis in configs_with_data:
            distance = analysis.get("distance_measure", "unknown")
            if distance not in distance_measures:
                distance_measures[distance] = 0
            distance_measures[distance] += 1

        print(f"  Distance measures: {distance_measures}")

        # Prediction enabled
        prediction_enabled = len(
            [a for a in configs_with_data if a.get("enable_prediction", False)]
        )
        prediction_disabled = len(
            [a for a in configs_with_data if not a.get("enable_prediction", False)]
        )
        print(
            f"  Prediction enabled: {prediction_enabled}, disabled: {prediction_disabled}"
        )

    # Results analysis
    print(f"\nResults Analysis:")
    results_with_data = [a for a in analyses if a.get("change_points_present")]
    if results_with_data:
        # Change points
        change_point_counts = [a.get("num_change_points", 0) for a in results_with_data]
        if change_point_counts:
            print(
                f"  Change points per experiment: {min(change_point_counts)}-{max(change_point_counts)} (avg: {np.mean(change_point_counts):.1f})"
            )

        # Detections
        detection_counts = [
            a.get("num_detections", 0)
            for a in analyses
            if a.get("detection_details_present")
        ]
        if detection_counts:
            print(
                f"  Detections per experiment: {min(detection_counts)}-{max(detection_counts)} (avg: {np.mean(detection_counts):.1f})"
            )

        # Raw data
        raw_data_rows = [
            a.get("raw_data_rows", 0) for a in analyses if a.get("raw_data_present")
        ]
        if raw_data_rows:
            print(
                f"  Raw data rows per experiment: {min(raw_data_rows)}-{max(raw_data_rows)} (avg: {np.mean(raw_data_rows):.1f})"
            )


def print_detailed_experiment_info(
    analyses: List[Dict[str, Any]], max_examples: int = 5
) -> None:
    """Print detailed information for a few example experiments."""
    print("\n" + "=" * 80)
    print("DETAILED EXPERIMENT EXAMPLES")
    print("=" * 80)

    # Show examples from different methods
    methods_shown = set()
    examples_shown = 0

    for analysis in analyses:
        if examples_shown >= max_examples:
            break

        method = analysis.get("method", "unknown")
        if method in methods_shown or analysis.get("status") == "parse_error":
            continue

        methods_shown.add(method)
        examples_shown += 1

        print(f"\nExample {examples_shown}: {analysis['folder_name']}")
        print(f"  Network: {analysis.get('network', 'unknown')}")
        print(f"  Scenario: {analysis.get('scenario', 'unknown')}")
        print(f"  Method: {analysis.get('method', 'unknown')}")
        print(f"  Status: {analysis.get('status', 'unknown')}")

        if analysis.get("config_present"):
            print(f"  Detection method: {analysis.get('detection_method', 'unknown')}")
            print(f"  Trials: {analysis.get('trials', 0)}")
            print(f"  Threshold: {analysis.get('threshold', 0)}")
            print(f"  Betting function: {analysis.get('betting_function', 'unknown')}")
            print(f"  Distance measure: {analysis.get('distance_measure', 'unknown')}")
            print(f"  Prediction enabled: {analysis.get('enable_prediction', False)}")

        if analysis.get("change_points_present"):
            print(f"  Change points: {analysis.get('num_change_points', 0)}")
            change_point_locations = analysis.get("change_point_locations", [])
            if change_point_locations:
                print(f"  Change point locations: {change_point_locations}")

        if analysis.get("detection_details_present"):
            print(f"  Total detections: {analysis.get('num_detections', 0)}")
            trials_with_detections = analysis.get("trials_with_detections", [])
            if trials_with_detections:
                print(f"  Trials with detections: {trials_with_detections}")

        if analysis.get("raw_data_present"):
            print(f"  Raw data rows: {analysis.get('raw_data_rows', 0)}")
            raw_columns = analysis.get("raw_data_columns", [])
            if raw_columns:
                print(
                    f"  Raw data columns: {raw_columns[:5]}{'...' if len(raw_columns) > 5 else ''}"
                )

        # Performance metrics
        if "TPR" in analysis:
            print(f"  Performance Metrics:")
            print(f"    TPR: {analysis['TPR']:.3f}")
            print(f"    FPR: {analysis['FPR']:.3f}")
            add_val = analysis["ADD"]
            if add_val == float("inf"):
                print(f"    ADD: ∞")
            else:
                print(f"    ADD: {add_val:.1f}")


def main():
    """Main function to analyze Table IV experiment structure."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Table IV experiment structure and generate performance table"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/plot_using_best_params",
        help="Path to Table IV results directory (default: results/plot_using_best_params)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information for example experiments",
    )
    parser.add_argument(
        "--generate-table",
        action="store_true",
        help="Generate Table IV performance comparison",
    )
    parser.add_argument(
        "--output",
        default="table_iv_results.xlsx",
        help="Output file for Table IV results (default: table_iv_results.xlsx)",
    )
    parser.add_argument(
        "--detection-window",
        type=int,
        default=40,
        help="Detection window size in steps after change point (default: 40)",
    )

    args = parser.parse_args()

    print("Analyzing Table IV experiment structure...")
    print(f"Results directory: {args.results_dir}")
    print(f"Detection window: {args.detection_window} steps after change point")

    # Analyze all experiments
    analyses = analyze_all_plot_using_best_params(args.results_dir, args.detection_window)

    if not analyses:
        print("No experiments found to analyze")
        return

    # Print structure summary
    print_structure_summary(analyses)

    # Print detailed examples if requested
    if args.detailed:
        print_detailed_experiment_info(analyses)

    # Generate Table IV if requested
    if args.generate_table:
        print("\n" + "=" * 80)
        print("GENERATING TABLE IV PERFORMANCE COMPARISON")
        print("=" * 80)

        table_df = create_table_iv_data(analyses)

        if not table_df.empty:
            # Display formatted table
            formatted_table = format_table_iv_for_display(table_df)
            print("\n" + formatted_table)

            # Save to Excel
            save_table_iv_results(table_df, args.output)

            print(f"\n✓ Table IV generation complete. Results saved to {args.output}")
            print(
                f"✓ Used detection window: {args.detection_window} steps after change point"
            )
        else:
            print("⚠️  No valid data found for Table IV generation")

    print(f"\n✓ Analysis complete. Processed {len(analyses)} experiments.")
    print(f"✓ Detection window used: {args.detection_window} steps after change point")


if __name__ == "__main__":
    main()
