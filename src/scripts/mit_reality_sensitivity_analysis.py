#!/usr/bin/env python3
"""
MIT Reality Sensitivity Analysis Data Reader

This script reads a single experiment folder from the MIT Reality parameter sweep
and extracts the configuration and detection results for analysis.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def read_experiment_folder(
    folder_path: str,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """
    Read a single MIT Reality experiment folder and extract config and results.

    Args:
        folder_path: Path to the experiment folder

    Returns:
        Tuple of (config_dict, change_point_metadata_df, detection_details_df, raw_detection_df)
        Returns (None, None, None, None) if files cannot be read
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder does not exist: {folder_path}")
        return None, None, None, None

    if not os.path.isdir(folder_path):
        print(f"Error: Path is not a directory: {folder_path}")
        return None, None, None, None

    # Read config.yaml
    config_path = os.path.join(folder_path, "config.yaml")
    config = None

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"✓ Successfully read config.yaml")
        except Exception as e:
            print(f"Error reading config.yaml: {e}")
            return None, None, None, None
    else:
        print(f"Error: config.yaml not found in {folder_path}")
        return None, None, None, None

    # Read detection results Excel file with multiple sheets
    detection_path = os.path.join(
        folder_path, "detection", "csv", "detection_results.xlsx"
    )
    change_point_metadata = None
    detection_details = None
    raw_detection_data = None

    if os.path.exists(detection_path):
        try:
            # Read all sheets to see what's available
            excel_file = pd.ExcelFile(detection_path)
            print(f"Available sheets: {excel_file.sheet_names}")

            # Read ChangePointMetadata sheet
            if "ChangePointMetadata" in excel_file.sheet_names:
                change_point_metadata = pd.read_excel(
                    detection_path, sheet_name="ChangePointMetadata"
                )
                print(
                    f"✓ Successfully read ChangePointMetadata sheet ({len(change_point_metadata)} rows)"
                )
            else:
                print("Warning: ChangePointMetadata sheet not found")

            # Read Detection Details sheet
            if "Detection Details" in excel_file.sheet_names:
                detection_details = pd.read_excel(
                    detection_path, sheet_name="Detection Details"
                )
                print(
                    f"✓ Successfully read Detection Details sheet ({len(detection_details)} rows)"
                )
            else:
                print("Warning: Detection Details sheet not found")

            # Read the first sheet (raw detection data) as fallback
            raw_detection_data = pd.read_excel(detection_path, sheet_name=0)
            print(
                f"✓ Successfully read raw detection data ({len(raw_detection_data)} rows)"
            )

        except Exception as e:
            print(f"Error reading detection_results.xlsx: {e}")
            return config, None, None, None
    else:
        print(f"Error: detection_results.xlsx not found at {detection_path}")
        return config, None, None, None

    return config, change_point_metadata, detection_details, raw_detection_data


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the configuration."""
    print("\n" + "=" * 50)
    print("CONFIGURATION SUMMARY")
    print("=" * 50)

    # Detection method info
    detection = config.get("detection", {})
    print(f"Detection method: {detection.get('method', 'unknown')}")
    print(f"Threshold: {detection.get('threshold', 'unknown')}")

    # Betting function info
    betting_config = detection.get("betting_func_config", {})
    betting_name = betting_config.get("name", "unknown")
    print(f"Betting function: {betting_name}")

    if betting_name == "power":
        epsilon = betting_config.get("power", {}).get("epsilon", "unknown")
        print(f"  - Epsilon: {epsilon}")
    elif betting_name == "beta":
        beta_config = betting_config.get("beta", {})
        print(f"  - Beta a: {beta_config.get('a', 'unknown')}")
        print(f"  - Beta b: {beta_config.get('b', 'unknown')}")
    elif betting_name == "mixture":
        epsilons = betting_config.get("mixture", {}).get("epsilons", [])
        print(f"  - Epsilons: {epsilons}")

    # Distance measure
    distance_config = detection.get("distance", {})
    print(f"Distance measure: {distance_config.get('measure', 'unknown')}")

    # Model info
    model = config.get("model", {})
    print(f"Model type: {model.get('type', 'unknown')}")
    print(f"Network: {model.get('network', 'unknown')}")

    # Trials info
    trials = config.get("trials", {})
    print(f"Number of trials: {trials.get('n_trials', 'unknown')}")


def print_change_point_summary(change_points: pd.DataFrame) -> None:
    """Print a summary of the change point metadata."""
    print("\n" + "=" * 50)
    print("CHANGE POINT METADATA SUMMARY")
    print("=" * 50)

    if change_points is not None and len(change_points) > 0:
        print(f"Total change points: {len(change_points)}")
        if "change_point" in change_points.columns:
            cp_values = sorted(change_points["change_point"].tolist())
            print(f"Change points: {cp_values}")
        print(f"\nChange point data:")
        print(change_points)
    else:
        print("No change point data available")


def print_detection_details_summary(detection_details: pd.DataFrame) -> None:
    """Print a summary of the detection details."""
    print("\n" + "=" * 50)
    print("DETECTION DETAILS SUMMARY")
    print("=" * 50)

    if detection_details is not None and len(detection_details) > 0:
        print(f"Total detections: {len(detection_details)}")
        print(f"Columns: {list(detection_details.columns)}")

        if "Trial" in detection_details.columns:
            trials = sorted(detection_details["Trial"].unique())
            print(f"Trials: {trials}")

        if "Type" in detection_details.columns:
            types = detection_details["Type"].unique()
            print(f"Detection types: {list(types)}")

        if "Is Within 30 Steps" in detection_details.columns:
            within_30 = detection_details["Is Within 30 Steps"].value_counts()
            print(f"Detections within 30 steps: {dict(within_30)}")

        # Show first few rows
        print(f"\nFirst 10 detections:")
        print(detection_details.head(10))
    else:
        print("No detection details available")


def print_raw_results_summary(results: pd.DataFrame) -> None:
    """Print a summary of the raw detection results."""
    print("\n" + "=" * 50)
    print("RAW DETECTION DATA SUMMARY")
    print("=" * 50)

    if results is not None and len(results) > 0:
        print(f"Total rows: {len(results)}")
        print(f"Columns: {list(results.columns)}")

        if "trial" in results.columns:
            trials = results["trial"].unique()
            print(f"Unique trials: {sorted(trials)}")

        if "detection_time" in results.columns:
            detections = results[results["detection_time"].notna()]
            print(f"Number of detections: {len(detections)}")

            if len(detections) > 0:
                print(
                    f"Detection times: {sorted(detections['detection_time'].tolist())}"
                )

        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(results.head())
    else:
        print("No raw detection data available")


def define_detection_ranges(change_points: List[int]) -> Dict[int, Tuple[int, int]]:
    """
    Define the valid detection ranges for each change point.

    Args:
        change_points: List of change point locations

    Returns:
        Dictionary mapping change_point -> (start, end) of valid detection range
    """
    # Define the specific ranges based on the rules provided
    ranges = {
        23: (23, 60),
        68: (68, 90),
        94: (94, 130),
        100: (100, 140),
        173: (173, 220),
        234: (234, 270),
    }

    # Only return ranges for change points that exist in the data
    return {cp: ranges[cp] for cp in change_points if cp in ranges}


def compute_performance_metrics(
    change_points_df: pd.DataFrame, detection_details_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute TPR, FPR, and ADD based on change points and detection details.

    Args:
        change_points_df: DataFrame with change point metadata
        detection_details_df: DataFrame with detection details

    Returns:
        Dictionary with performance metrics
    """
    if change_points_df is None or detection_details_df is None:
        return {"TPR": 0.0, "FPR": 0.0, "ADD": float("inf"), "error": "Missing data"}

    # Extract change points
    change_points = sorted(change_points_df["change_point"].tolist())
    detection_ranges = define_detection_ranges(change_points)

    print(f"Change points: {change_points}")
    print(f"Detection ranges: {detection_ranges}")

    # Get all detections
    detections = detection_details_df["Detection Index"].tolist()
    print(f"All detections: {sorted(detections)}")

    # Count true positives and compute delays
    true_positives = 0
    detection_delays = []
    detected_change_points = set()
    first_detection_delays = {}  # Track first detection delay for each CP

    for detection_idx in detections:
        is_true_positive = False

        for cp, (start, end) in detection_ranges.items():
            if start <= detection_idx <= end:
                # This is a true positive
                true_positives += 1
                detected_change_points.add(cp)
                delay = detection_idx - cp
                detection_delays.append(delay)

                # Track first detection for each change point for ADD calculation
                if cp not in first_detection_delays:
                    first_detection_delays[cp] = delay
                    print(
                        f"TP (first): Detection at {detection_idx} for CP {cp}, delay = {delay}"
                    )
                else:
                    print(
                        f"TP (additional): Detection at {detection_idx} for CP {cp}, delay = {delay}"
                    )

                is_true_positive = True
                break

        if not is_true_positive:
            print(f"FP: Detection at {detection_idx}")

    # Count false positives
    false_positives = len(detections) - true_positives

    # Count false negatives (undetected change points)
    false_negatives = len(change_points) - len(detected_change_points)

    # Compute metrics
    total_change_points = len(change_points)
    # TPR should be based on how many change points were detected, not total detections
    TPR = (
        len(detected_change_points) / total_change_points
        if total_change_points > 0
        else 0.0
    )

    # For FPR, we need to define the total possible false positive opportunities
    # Using the total time span minus the true positive windows
    total_time_span = 275  # Based on the data we saw
    true_positive_windows = sum(
        end - start + 1 for start, end in detection_ranges.values()
    )
    false_positive_opportunities = total_time_span - true_positive_windows

    FPR = (
        false_positives / false_positive_opportunities
        if false_positive_opportunities > 0
        else 0.0
    )

    # Compute ADD (Average Detection Delay) - use only first detection for each CP
    first_delays = list(first_detection_delays.values())
    ADD = np.mean(first_delays) if first_delays else float("inf")

    metrics = {
        "TPR": TPR,
        "FPR": FPR,
        "ADD": ADD,
        "true_positive_detections": true_positives,  # Total TP detections (can be > # CPs)
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_change_points": total_change_points,
        "detected_change_points": len(detected_change_points),
        "all_detection_delays": detection_delays,
        "first_detection_delays": first_delays,
        "total_detections": len(detections),
    }

    return metrics


def print_performance_summary(metrics: Dict[str, float]) -> None:
    """Print a summary of the performance metrics."""
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 50)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"True Positive Rate (TPR): {metrics['TPR']:.3f}")
    print(f"False Positive Rate (FPR): {metrics['FPR']:.6f}")
    print(f"Average Detection Delay (ADD): {metrics['ADD']:.2f}")
    print()
    print(f"True Positive Detections: {metrics['true_positive_detections']}")
    print(f"False Positive Detections: {metrics['false_positives']}")
    print(f"False Negatives (Missed CPs): {metrics['false_negatives']}")
    print(f"Total Change Points: {metrics['total_change_points']}")
    print(f"Detected Change Points: {metrics['detected_change_points']}")
    print(f"Total Detections: {metrics['total_detections']}")

    if metrics["first_detection_delays"]:
        print(
            f"First Detection Delays (used for ADD): {metrics['first_detection_delays']}"
        )
        print(f"Min First Delay: {min(metrics['first_detection_delays'])}")
        print(f"Max First Delay: {max(metrics['first_detection_delays'])}")

    if metrics["all_detection_delays"]:
        print(f"All Detection Delays: {metrics['all_detection_delays']}")
        print(f"Total TP Detections: {len(metrics['all_detection_delays'])}")


def process_all_experiments(results_dir: str) -> pd.DataFrame:
    """
    Process all experiment folders and compute metrics for each.

    Args:
        results_dir: Path to the directory containing all experiment folders

    Returns:
        DataFrame with metrics for all experiments
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return pd.DataFrame()

    # Get all experiment folders
    experiment_folders = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("mit_reality_")
    ]

    print(f"Found {len(experiment_folders)} experiment folders")

    all_results = []

    for i, folder_name in enumerate(experiment_folders):
        folder_path = os.path.join(results_dir, folder_name)
        print(f"\nProcessing [{i+1}/{len(experiment_folders)}]: {folder_name}")

        # Read experiment data
        config, change_points, detection_details, raw_results = read_experiment_folder(
            folder_path
        )

        if config is None or change_points is None or detection_details is None:
            print(f"  ⚠️  Skipping {folder_name} - missing data")
            continue

        # Extract experiment parameters from config
        detection_config = config.get("detection", {})
        betting_config = detection_config.get("betting_func_config", {})
        distance_config = detection_config.get("distance", {})

        # Parse experiment name to extract parameters
        exp_params = parse_experiment_name(folder_name)

        # Compute performance metrics
        metrics = compute_performance_metrics(change_points, detection_details)

        if "error" in metrics:
            print(f"  ⚠️  Error computing metrics for {folder_name}: {metrics['error']}")
            continue

        # Combine all information
        result = {
            "experiment_name": folder_name,
            "experiment_type": exp_params.get("type", "unknown"),
            "betting_function": betting_config.get("name", "unknown"),
            "distance_measure": distance_config.get("measure", "unknown"),
            "threshold": detection_config.get("threshold", None),
            **exp_params,  # Add parsed parameters
            **metrics,  # Add performance metrics
        }

        all_results.append(result)

        print(
            f"  ✓ TPR: {metrics['TPR']:.3f}, FPR: {metrics['FPR']:.6f}, ADD: {metrics['ADD']:.2f}"
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    print(f"\n✓ Successfully processed {len(all_results)} experiments")
    return results_df


def parse_experiment_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse experiment folder name to extract parameters.

    Args:
        folder_name: Name of the experiment folder

    Returns:
        Dictionary with parsed parameters
    """
    params = {}

    # Extract experiment type and parameters from folder name
    if "power_" in folder_name:
        params["type"] = "power"
        # Extract epsilon value
        parts = folder_name.split("_")
        for i, part in enumerate(parts):
            if part == "power" and i + 1 < len(parts):
                try:
                    params["epsilon"] = float(parts[i + 1])
                except ValueError:
                    pass
                break
    elif "mixture_" in folder_name:
        params["type"] = "mixture"
        params["mixture"] = True
    elif "beta_" in folder_name:
        params["type"] = "beta"
        # Extract beta_a and beta_b values
        parts = folder_name.split("_")
        for i, part in enumerate(parts):
            if part == "beta" and i + 2 < len(parts):
                try:
                    params["beta_a"] = float(parts[i + 1])
                    params["beta_b"] = float(parts[i + 2])
                except ValueError:
                    pass
                break
    elif "threshold_" in folder_name:
        params["type"] = "threshold"
        # Extract threshold value
        parts = folder_name.split("_")
        for i, part in enumerate(parts):
            if part == "threshold" and i + 1 < len(parts):
                try:
                    params["threshold_value"] = float(parts[i + 1])
                except ValueError:
                    pass
                break

    # Extract distance measure (usually the last meaningful part before timestamp)
    parts = folder_name.split("_")
    distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
    for part in parts:
        if part in distance_measures:
            params["distance"] = part
            break

    return params


def create_parameter_sensitivity_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a parameter sensitivity table in the format shown in TABLE III.

    Args:
        results_df: DataFrame with all experiment results

    Returns:
        Formatted table DataFrame
    """
    # Initialize the table structure
    table_data = []

    # MIT Reality is our single network type
    network = "MIT"

    # Process Power Betting results
    power_results = {}
    power_data = results_df[results_df["experiment_type"] == "power"]

    if not power_data.empty:
        # Group by epsilon values
        for epsilon in [0.2, 0.5, 0.7, 0.9]:
            epsilon_data = power_data[power_data["epsilon"] == epsilon]
            if not epsilon_data.empty:
                # Filter out infinite ADD values for proper averaging
                finite_add_data = epsilon_data[np.isfinite(epsilon_data["ADD"])]
                avg_add = (
                    finite_add_data["ADD"].mean()
                    if len(finite_add_data) > 0
                    else float("inf")
                )

                power_results[f"ε={epsilon}"] = {
                    "TPR": epsilon_data["TPR"].mean(),
                    "FPR": epsilon_data["FPR"].mean(),
                    "ADD": avg_add,
                }

    # Process Mixture Betting results
    mixture_results = {}
    mixture_data = results_df[results_df["experiment_type"] == "mixture"]

    if not mixture_data.empty:
        # Filter out infinite ADD values for proper averaging
        finite_add_data = mixture_data[np.isfinite(mixture_data["ADD"])]
        avg_add = (
            finite_add_data["ADD"].mean() if len(finite_add_data) > 0 else float("inf")
        )

        mixture_results["Mixture"] = {
            "TPR": mixture_data["TPR"].mean(),
            "FPR": mixture_data["FPR"].mean(),
            "ADD": avg_add,
        }

    # Process Beta Betting results
    beta_results = {}
    beta_data = results_df[results_df["experiment_type"] == "beta"]

    if not beta_data.empty:
        # Group by beta_a and beta_b combinations
        beta_combinations = [
            (0.2, 2.5),
            (0.4, 1.8),
            (0.6, 1.2),  # Representative combinations
        ]

        for beta_a, beta_b in beta_combinations:
            combo_data = beta_data[
                (beta_data["beta_a"] == beta_a) & (beta_data["beta_b"] == beta_b)
            ]
            if not combo_data.empty:
                # Filter out infinite ADD values for proper averaging
                finite_add_data = combo_data[np.isfinite(combo_data["ADD"])]
                avg_add = (
                    finite_add_data["ADD"].mean()
                    if len(finite_add_data) > 0
                    else float("inf")
                )

                beta_results[f"({beta_a},{beta_b})"] = {
                    "TPR": combo_data["TPR"].mean(),
                    "FPR": combo_data["FPR"].mean(),
                    "ADD": avg_add,
                }

    # Process Distance Metric results
    distance_results = {}
    distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]

    for distance in distance_measures:
        dist_data = results_df[results_df["distance_measure"] == distance]
        if not dist_data.empty:
            # Use proper abbreviations
            abbrev_map = {
                "euclidean": "Euc.",
                "mahalanobis": "Mah.",
                "cosine": "Cos.",
                "chebyshev": "Cheb.",
            }
            # Filter out infinite ADD values for proper averaging
            finite_add_data = dist_data[np.isfinite(dist_data["ADD"])]
            avg_add = (
                finite_add_data["ADD"].mean()
                if len(finite_add_data) > 0
                else float("inf")
            )

            distance_results[abbrev_map[distance]] = {
                "TPR": dist_data["TPR"].mean(),
                "FPR": dist_data["FPR"].mean(),
                "ADD": avg_add,
            }

    # Process Threshold results
    threshold_results = {}
    threshold_data = results_df[results_df["experiment_type"] == "threshold"]

    if not threshold_data.empty:
        for threshold_val in [20, 50, 100]:
            thresh_data = threshold_data[
                threshold_data["threshold_value"] == threshold_val
            ]
            if not thresh_data.empty:
                # Filter out infinite ADD values for proper averaging
                finite_add_data = thresh_data[np.isfinite(thresh_data["ADD"])]
                avg_add = (
                    finite_add_data["ADD"].mean()
                    if len(finite_add_data) > 0
                    else float("inf")
                )

                threshold_results[str(threshold_val)] = {
                    "TPR": thresh_data["TPR"].mean(),
                    "FPR": thresh_data["FPR"].mean(),
                    "ADD": avg_add,
                }

    # Create table rows for each metric
    metrics = ["TPR", "FPR", "ADD"]

    for metric in metrics:
        row = {"Network": network, "Metric": metric}

        # Power Betting columns
        for epsilon_key in ["ε=0.2", "ε=0.5", "ε=0.7", "ε=0.9"]:
            if epsilon_key in power_results:
                value = power_results[epsilon_key][metric]
                if metric == "ADD" and np.isinf(value):
                    row[epsilon_key] = "—"
                else:
                    row[epsilon_key] = (
                        f"{value:.3f}" if metric in ["TPR", "FPR"] else f"{value:.1f}"
                    )
            else:
                row[epsilon_key] = "—"

        # Mixture column
        if "Mixture" in mixture_results:
            value = mixture_results["Mixture"][metric]
            if metric == "ADD" and np.isinf(value):
                row["Mixture"] = "—"
            else:
                row["Mixture"] = (
                    f"{value:.3f}" if metric in ["TPR", "FPR"] else f"{value:.1f}"
                )
        else:
            row["Mixture"] = "—"

        # Beta Betting columns
        for beta_key in ["(0.2,2.5)", "(0.4,1.8)", "(0.6,1.2)"]:
            if beta_key in beta_results:
                value = beta_results[beta_key][metric]
                if metric == "ADD" and np.isinf(value):
                    row[beta_key] = "—"
                else:
                    row[beta_key] = (
                        f"{value:.3f}" if metric in ["TPR", "FPR"] else f"{value:.1f}"
                    )
            else:
                row[beta_key] = "—"

        # Distance Metric columns
        for dist_key in ["Euc.", "Mah.", "Cos.", "Cheb."]:
            if dist_key in distance_results:
                value = distance_results[dist_key][metric]
                if metric == "ADD" and np.isinf(value):
                    row[dist_key] = "—"
                else:
                    row[dist_key] = (
                        f"{value:.3f}" if metric in ["TPR", "FPR"] else f"{value:.1f}"
                    )
            else:
                row[dist_key] = "—"

        # Threshold columns
        for thresh_key in ["20", "50", "100"]:
            if thresh_key in threshold_results:
                value = threshold_results[thresh_key][metric]
                if metric == "ADD" and np.isinf(value):
                    row[thresh_key] = "—"
                else:
                    row[thresh_key] = (
                        f"{value:.3f}" if metric in ["TPR", "FPR"] else f"{value:.1f}"
                    )
            else:
                row[thresh_key] = "—"

        table_data.append(row)

    # Create DataFrame
    columns = [
        "Network",
        "Metric",
        "ε=0.2",
        "ε=0.5",
        "ε=0.7",
        "ε=0.9",
        "Mixture",
        "(0.2,2.5)",
        "(0.4,1.8)",
        "(0.6,1.2)",
        "Euc.",
        "Mah.",
        "Cos.",
        "Cheb.",
        "20",
        "50",
        "100",
    ]

    table_df = pd.DataFrame(table_data, columns=columns)

    return table_df


def save_results_summary(results_df: pd.DataFrame, output_path: str) -> None:
    """Save results summary to Excel file."""
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Save full results
            results_df.to_excel(writer, sheet_name="All Results", index=False)

            # Create and save parameter sensitivity table
            param_table = create_parameter_sensitivity_table(results_df)
            param_table.to_excel(
                writer, sheet_name="Parameter Sensitivity Table", index=False
            )

            # Create summary by experiment type
            if "experiment_type" in results_df.columns:
                summary_by_type = (
                    results_df.groupby("experiment_type")
                    .agg(
                        {
                            "TPR": ["mean", "std", "min", "max"],
                            "FPR": ["mean", "std", "min", "max"],
                            "ADD": ["mean", "std", "min", "max"],
                        }
                    )
                    .round(6)
                )
                summary_by_type.to_excel(writer, sheet_name="Summary by Type")

            # Create summary by distance measure
            if "distance_measure" in results_df.columns:
                summary_by_distance = (
                    results_df.groupby("distance_measure")
                    .agg(
                        {
                            "TPR": ["mean", "std", "min", "max"],
                            "FPR": ["mean", "std", "min", "max"],
                            "ADD": ["mean", "std", "min", "max"],
                        }
                    )
                    .round(6)
                )
                summary_by_distance.to_excel(writer, sheet_name="Summary by Distance")

        print(f"✓ Results saved to: {output_path}")
        print(
            f"✓ Parameter sensitivity table created in 'Parameter Sensitivity Table' sheet"
        )
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main function to process experiment folders."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process MIT Reality experiment folders"
    )
    parser.add_argument(
        "path", help="Path to experiment folder (single) or results directory (all)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all experiments in the directory"
    )
    parser.add_argument(
        "--output",
        default="mit_reality_sensitivity_results.xlsx",
        help="Output file for results summary (when processing all)",
    )
    parser.add_argument(
        "--show-config", action="store_true", help="Show detailed config (single mode)"
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Show detailed results (single mode)",
    )

    args = parser.parse_args()

    if args.all:
        # Process all experiments
        print("Processing all MIT Reality experiments...")
        results_df = process_all_experiments(args.path)

        if not results_df.empty:
            # Print summary statistics
            print("\n" + "=" * 80)
            print("OVERALL SUMMARY STATISTICS")
            print("=" * 80)

            print(f"Total experiments processed: {len(results_df)}")

            if "experiment_type" in results_df.columns:
                print(f"\nBy experiment type:")
                type_counts = results_df["experiment_type"].value_counts()
                for exp_type, count in type_counts.items():
                    print(f"  {exp_type}: {count} experiments")

            print(f"\nOverall performance metrics:")
            print(
                f"  TPR: {results_df['TPR'].mean():.3f} ± {results_df['TPR'].std():.3f}"
            )
            print(
                f"  FPR: {results_df['FPR'].mean():.6f} ± {results_df['FPR'].std():.6f}"
            )
            print(
                f"  ADD: {results_df['ADD'].mean():.2f} ± {results_df['ADD'].std():.2f}"
            )

            # Save results
            save_results_summary(results_df, args.output)

            # Print parameter sensitivity table
            print("\n" + "=" * 120)
            print("PARAMETER SENSITIVITY TABLE (TABLE III FORMAT)")
            print("=" * 120)
            param_table = create_parameter_sensitivity_table(results_df)
            print(param_table.to_string(index=False))
            print("=" * 120)
        else:
            print("No valid experiments found to process")

    else:
        # Process single experiment (original functionality)
        print(f"Reading experiment folder: {args.path}")

        config, change_points, detection_details, raw_results = read_experiment_folder(
            args.path
        )

        if (
            config is None
            and change_points is None
            and detection_details is None
            and raw_results is None
        ):
            print("Failed to read experiment data")
            return

        # Print summaries
        if config is not None:
            print_config_summary(config)

            if args.show_config:
                print("\n" + "=" * 50)
                print("FULL CONFIGURATION")
                print("=" * 50)
                print(yaml.dump(config, default_flow_style=False, sort_keys=True))

        # Print change point metadata
        print_change_point_summary(change_points)

        # Print detection details
        print_detection_details_summary(detection_details)

        # Compute and print performance metrics
        if change_points is not None and detection_details is not None:
            metrics = compute_performance_metrics(change_points, detection_details)
            print_performance_summary(metrics)
        else:
            print(
                "\n⚠️  Cannot compute performance metrics - missing change points or detection details"
            )

        # Print raw results summary
        if raw_results is not None:
            print_raw_results_summary(raw_results)

            if args.show_results:
                print("\n" + "=" * 50)
                print("FULL RAW RESULTS")
                print("=" * 50)
                print(raw_results.to_string())

        print(f"\n✓ Successfully processed experiment folder")


if __name__ == "__main__":
    main()
