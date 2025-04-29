#!/usr/bin/env python3
"""Find optimal parameters from parameter sweep results.

This script analyzes parameter sweep results to determine the best parameter values
for each network type based on different performance metrics.
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Parameter name mapping for pretty printing
PARAM_NAMES = {
    "threshold": "Detection Threshold (λ)",
    "window": "History Window (w)",
    "horizon": "Prediction Horizon (h)",
    "epsilon": "Epsilon",
    "betting_func": "Betting Function",
    "distance": "Distance Metric",
}

# Network name mapping for pretty printing
NETWORK_NAMES = {
    "sbm": "SBM",
    "ba": "BA",
    "er": "ER",
    "ws": "NWS",
}


def parse_directory_name(dir_name: str) -> Dict[str, Any]:
    """Parse parameter values from directory name.

    Args:
        dir_name: Name of the parameter directory

    Returns:
        Dictionary with extracted parameter values
    """
    # Extract parameters from directory name
    # Format: {network}_t{threshold}_w{window}_h{horizon}_e{epsilon}_{betting_func}_{distance}_{timestamp}
    pattern = r"^([a-z]+)_t(\d+)_w(\d+)_h(\d+)_e([\d\.]+)_([a-z]+)_([a-z]+)_\d+"
    match = re.match(pattern, os.path.basename(dir_name))

    if not match:
        logger.warning(f"Could not parse directory name: {dir_name}")
        return {}

    network, threshold, window, horizon, epsilon, betting_func, distance = (
        match.groups()
    )

    return {
        "network": network,
        "threshold": int(threshold),
        "window": int(window),
        "horizon": int(horizon),
        "epsilon": float(epsilon),
        "betting_func": betting_func,
        "distance": distance,
    }


def find_detection_file(dir_path: str) -> Optional[str]:
    """Find detection_results.xlsx file in the given directory.

    Args:
        dir_path: Path to parameter directory

    Returns:
        Path to detection_results.xlsx file if found, None otherwise
    """
    # Check if there's a detection_results.xlsx in the main directory
    detection_file = os.path.join(dir_path, "detection_results.xlsx")
    if os.path.exists(detection_file):
        return detection_file

    # If not, look for the proper subdirectory
    # The subdirectory pattern seems to be: {network}_graph_{distance}_{betting_func}_{timestamp}
    network_subdir = None
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path) and "_graph_" in item:
            network_subdir = item_path
            break

    if network_subdir:
        # Look for detection_results.xlsx in this subdirectory
        detection_file = os.path.join(network_subdir, "detection_results.xlsx")
        if os.path.exists(detection_file):
            return detection_file

    # If still not found, try a more general search for any Excel file
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".xlsx") and "detection" in file.lower():
                return os.path.join(root, file)

    return None


def extract_metrics_from_file(file_path: str) -> Dict[str, float]:
    """Extract performance metrics from detection_results.xlsx file.

    Args:
        file_path: Path to detection_results.xlsx file

    Returns:
        Dictionary with performance metrics
    """
    try:
        metrics = {}

        # Load Excel file
        xl = pd.ExcelFile(file_path)
        available_sheets = xl.sheet_names
        logger.debug(f"Available sheets in {file_path}: {available_sheets}")

        # Step 1: Extract true change points from ChangePointMetadata
        true_change_points = []
        if "ChangePointMetadata" in available_sheets:
            cp_metadata = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
            if not cp_metadata.empty and "change_point" in cp_metadata.columns:
                true_change_points = cp_metadata["change_point"].dropna().tolist()
                true_change_points = [int(cp) for cp in true_change_points]
                logger.debug(f"True change points from metadata: {true_change_points}")

        # If ChangePointMetadata didn't have change points, try to extract from other sheets
        if not true_change_points and "Detection Details" in available_sheets:
            det_details = pd.read_excel(file_path, sheet_name="Detection Details")
            if not det_details.empty and "Nearest True CP" in det_details.columns:
                true_change_points = (
                    det_details["Nearest True CP"].dropna().unique().tolist()
                )
                true_change_points = [int(cp) for cp in true_change_points]
                logger.debug(
                    f"True change points from detection details: {true_change_points}"
                )

        # Step 2: Extract all detections from Detection Details
        traditional_detections = []
        horizon_detections = []

        if "Detection Details" in available_sheets:
            det_details = pd.read_excel(file_path, sheet_name="Detection Details")
            if not det_details.empty:
                # Get traditional detections
                trad_rows = det_details[det_details["Type"] == "Traditional"]
                if not trad_rows.empty and "Detection Index" in trad_rows.columns:
                    traditional_detections = (
                        trad_rows["Detection Index"].dropna().tolist()
                    )
                    traditional_detections = [
                        int(idx) for idx in traditional_detections
                    ]

                # Get horizon detections
                horizon_rows = det_details[det_details["Type"] == "Horizon"]
                if not horizon_rows.empty and "Detection Index" in horizon_rows.columns:
                    horizon_detections = (
                        horizon_rows["Detection Index"].dropna().tolist()
                    )
                    horizon_detections = [int(idx) for idx in horizon_detections]

        # Step 3: Calculate TP, FP, FN for both detection methods
        # A detection is a true positive if it occurs within 20 steps after a true change point
        # and no other detection has already been matched to that change point

        # Function to classify detections
        def classify_detections(detections, true_cps, max_delay=20):
            # Sort detections and true change points
            detections = sorted(detections)
            true_cps = sorted(true_cps)

            true_positives = []
            false_positives = []
            detected_cps = set()

            for detection in detections:
                # Check if this detection is a true positive
                is_tp = False
                for cp in true_cps:
                    # Detection must be after change point and within max_delay steps
                    # And this change point hasn't been detected yet
                    if (cp <= detection <= cp + max_delay) and cp not in detected_cps:
                        true_positives.append((detection, cp))
                        detected_cps.add(cp)
                        is_tp = True
                        break

                # If not a true positive, it's a false positive
                if not is_tp:
                    false_positives.append(detection)

            # Calculate metrics
            num_tps = len(true_positives)
            num_fps = len(false_positives)
            num_fns = len(true_cps) - len(
                detected_cps
            )  # False negatives = missed change points

            # Calculate delays
            delays = [detection - cp for detection, cp in true_positives]
            avg_delay = np.mean(delays) if delays else np.nan

            # Calculate rates
            tpr = num_tps / len(true_cps) if true_cps else 0.0
            # FPR = FP / (total timesteps - # of true change points)
            # We don't know the total timesteps, so we use an estimate
            # from max detection index or change point
            max_time = max(
                max(detections) if detections else 0, max(true_cps) if true_cps else 0
            )

            # Calculate approximate false positive rate
            fpr = num_fps / max(1, max_time - len(true_cps))

            return {
                "tpr": tpr,
                "fpr": fpr,
                "avg_delay": avg_delay,
                "true_positives": num_tps,
                "false_positives": num_fps,
                "false_negatives": num_fns,
                "delay_details": delays,
            }

        # Only perform classification if we have true change points
        if true_change_points:
            # Classify traditional detections
            trad_metrics = classify_detections(
                traditional_detections, true_change_points
            )
            # Classify horizon detections
            horizon_metrics = classify_detections(
                horizon_detections, true_change_points
            )

            # Store metrics
            metrics.update(
                {
                    "tpr": trad_metrics["tpr"],
                    "fpr": trad_metrics["fpr"],
                    "avg_delay": trad_metrics["avg_delay"],
                    "true_positives": trad_metrics["true_positives"],
                    "false_positives": trad_metrics["false_positives"],
                    "false_negatives": trad_metrics["false_negatives"],
                    "horizon_tpr": horizon_metrics["tpr"],
                    "horizon_fpr": horizon_metrics["fpr"],
                    "horizon_avg_delay": horizon_metrics["avg_delay"],
                    "horizon_true_positives": horizon_metrics["true_positives"],
                    "horizon_false_positives": horizon_metrics["false_positives"],
                    "horizon_false_negatives": horizon_metrics["false_negatives"],
                }
            )

            # Calculate delay reduction if both methods have valid delays
            if (
                not np.isnan(trad_metrics["avg_delay"])
                and not np.isnan(horizon_metrics["avg_delay"])
                and trad_metrics["avg_delay"] > 0
            ):
                delay_reduction = (
                    trad_metrics["avg_delay"] - horizon_metrics["avg_delay"]
                ) / trad_metrics["avg_delay"]
                metrics["delay_reduction"] = delay_reduction

        # Step 4: If we couldn't calculate metrics from Detection Details,
        # fall back to using delay statistics from ChangePointMetadata
        if "avg_delay" not in metrics or np.isnan(metrics["avg_delay"]):
            if "ChangePointMetadata" in available_sheets:
                cp_metadata = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
                if not cp_metadata.empty:
                    # Extract delay metrics
                    if "traditional_avg_delay" in cp_metadata.columns:
                        metrics["avg_delay"] = float(
                            cp_metadata["traditional_avg_delay"].iloc[0]
                        )

                    if "horizon_avg_delay" in cp_metadata.columns:
                        metrics["horizon_avg_delay"] = float(
                            cp_metadata["horizon_avg_delay"].iloc[0]
                        )

                    if "delay_reduction" in cp_metadata.columns:
                        metrics["delay_reduction"] = float(
                            cp_metadata["delay_reduction"].iloc[0]
                        )

        # Step 5: If we still don't have detection rates, try to extract from Detection Summary
        if "tpr" not in metrics or "horizon_tpr" not in metrics:
            if "Detection Summary" in available_sheets:
                det_summary = pd.read_excel(file_path, sheet_name="Detection Summary")
                if not det_summary.empty:
                    # Get trial columns
                    trial_columns = [
                        col for col in det_summary.columns if "Trial" in col
                    ]
                    trad_count_cols = [
                        col
                        for col in trial_columns
                        if "Traditional Detection Count" in col
                    ]
                    horizon_count_cols = [
                        col for col in trial_columns if "Horizon Detection Count" in col
                    ]

                    # Calculate detection rates
                    if trad_count_cols:
                        # Get the detection counts for each true change point
                        trad_counts = det_summary[trad_count_cols].mean(axis=1).dropna()
                        if not trad_counts.empty:
                            trad_counts = (
                                trad_counts[:-2]
                                if len(trad_counts) > 2
                                else trad_counts
                            )
                            tpr = sum(count > 0 for count in trad_counts) / len(
                                trad_counts
                            )
                            metrics["tpr"] = tpr

                    if horizon_count_cols:
                        # Get the detection counts for each true change point
                        horizon_counts = (
                            det_summary[horizon_count_cols].mean(axis=1).dropna()
                        )
                        if not horizon_counts.empty:
                            horizon_counts = (
                                horizon_counts[:-2]
                                if len(horizon_counts) > 2
                                else horizon_counts
                            )
                            tpr = sum(count > 0 for count in horizon_counts) / len(
                                horizon_counts
                            )
                            metrics["horizon_tpr"] = tpr

        # Step 6: If needed, try Aggregate sheet as a last resort for detection rates
        if "tpr" not in metrics or "horizon_tpr" not in metrics:
            if "Aggregate" in available_sheets:
                agg_df = pd.read_excel(file_path, sheet_name="Aggregate")
                if not agg_df.empty:
                    # Look for detection rate columns
                    if "traditional_detection_rate" in agg_df.columns:
                        trad_rates = agg_df["traditional_detection_rate"].dropna()
                        if not trad_rates.empty:
                            metrics["tpr"] = float(trad_rates.max())

                    if "horizon_detection_rate" in agg_df.columns:
                        horizon_rates = agg_df["horizon_detection_rate"].dropna()
                        if not horizon_rates.empty:
                            metrics["horizon_tpr"] = float(horizon_rates.max())

        # Step 7: Handle null values and calculate any missing metrics
        for key in ["tpr", "horizon_tpr", "fpr", "horizon_fpr"]:
            if key not in metrics:
                metrics[key] = 0.0

        for key in ["avg_delay", "horizon_avg_delay"]:
            if key not in metrics or np.isnan(metrics[key]):
                metrics[key] = np.nan

        if (
            "delay_reduction" not in metrics
            and not np.isnan(metrics["avg_delay"])
            and not np.isnan(metrics["horizon_avg_delay"])
            and metrics["avg_delay"] > 0
        ):
            metrics["delay_reduction"] = (
                metrics["avg_delay"] - metrics["horizon_avg_delay"]
            ) / metrics["avg_delay"]

        # Step 8: Store the count of true change points for reference
        metrics["num_true_change_points"] = len(true_change_points)
        metrics["true_change_points"] = true_change_points

        # Log the extracted metrics
        if metrics:
            logger.info(f"Successfully extracted metrics from {file_path}: {metrics}")
        else:
            logger.warning(f"Could not extract any metrics from {file_path}")

        return metrics

    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return {}


def analyze_parameter_sweep(sweep_dir: str) -> pd.DataFrame:
    """Analyze parameter sweep results from all directories.

    Args:
        sweep_dir: Root directory containing parameter sweep results

    Returns:
        DataFrame with analysis results
    """
    logger.info(f"Analyzing parameter sweep results in {sweep_dir}")

    # Find all parameter directories
    param_dirs = glob.glob(os.path.join(sweep_dir, "*_t*_w*_h*_e*_*_*_*"))
    logger.info(f"Found {len(param_dirs)} parameter directories")

    # Process each directory
    results = []
    for dir_path in param_dirs:
        # Extract parameters from directory name
        params = parse_directory_name(dir_path)
        if not params:
            continue

        # Find detection.xlsx file
        detection_file = find_detection_file(dir_path)
        if not detection_file:
            logger.warning(f"No detection.xlsx found in {dir_path}")
            continue

        # Extract metrics from file
        metrics = extract_metrics_from_file(detection_file)
        if not metrics:
            logger.warning(f"No metrics extracted from {detection_file}")
            continue

        # Combine parameters and metrics
        result = {**params, **metrics}
        results.append(result)

    # Convert to DataFrame
    if not results:
        logger.error("No valid results found")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Check if we have the expected columns
    expected_columns = [
        "network",
        "threshold",
        "window",
        "horizon",
        "epsilon",
        "betting_func",
        "distance",
        "tpr",
        "avg_delay",
    ]
    missing_columns = [col for col in expected_columns if col not in result_df.columns]
    if missing_columns:
        logger.warning(f"Missing expected columns: {missing_columns}")

    return result_df


def find_best_parameters(result_df: pd.DataFrame, output_dir: str):
    """Find the best parameters for each network type and metric.

    Args:
        result_df: DataFrame with analysis results
        output_dir: Directory to save output files
    """
    if result_df.empty:
        logger.error("Empty result DataFrame, cannot find best parameters")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group by network type and find best parameters for each metric
    network_results = {}

    for network in result_df["network"].unique():
        network_df = result_df[result_df["network"] == network]

        # Best parameters for highest TPR (traditional)
        try:
            best_tpr_idx = network_df["tpr"].idxmax()
            best_tpr_params = network_df.loc[best_tpr_idx]
        except:
            best_tpr_params = None

        # Best parameters for lowest average delay (traditional)
        try:
            best_delay_idx = network_df["avg_delay"].idxmin()
            best_delay_params = network_df.loc[best_delay_idx]
        except:
            best_delay_params = None

        # Best parameters for horizon TPR if available
        best_horizon_tpr_params = None
        if "horizon_tpr" in network_df.columns:
            try:
                best_horizon_tpr_idx = network_df["horizon_tpr"].idxmax()
                best_horizon_tpr_params = network_df.loc[best_horizon_tpr_idx]
            except:
                pass

        # Best parameters for horizon delay if available
        best_horizon_delay_params = None
        if "horizon_avg_delay" in network_df.columns:
            try:
                best_horizon_delay_idx = network_df["horizon_avg_delay"].idxmin()
                best_horizon_delay_params = network_df.loc[best_horizon_delay_idx]
            except:
                pass

        # Store results
        network_results[network] = {
            "best_tpr": best_tpr_params,
            "best_delay": best_delay_params,
            "best_horizon_tpr": best_horizon_tpr_params,
            "best_horizon_delay": best_horizon_delay_params,
        }

    # Generate LaTeX table
    generate_latex_table(
        network_results, os.path.join(output_dir, "best_parameters.tex")
    )

    # Generate summary report
    generate_summary_report(
        network_results, result_df, os.path.join(output_dir, "parameter_summary.txt")
    )

    # Save detailed results to CSV
    result_df.to_csv(os.path.join(output_dir, "all_parameters.csv"), index=False)

    # Generate parameter effect plots
    generate_parameter_effect_analysis(result_df, output_dir)


def generate_latex_table(
    network_results: Dict[str, Dict[str, pd.Series]], output_file: str
):
    """Generate LaTeX table with best parameter values for each network.

    Args:
        network_results: Dictionary with best parameters for each network
        output_file: Path to output file
    """
    # Get list of parameters to include in the table
    parameters = [
        "threshold",
        "window",
        "horizon",
        "epsilon",
        "betting_func",
        "distance",
    ]

    # Start building the LaTeX table
    latex_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.3}",
        "\\footnotesize",
        "\\begin{tabular}{|l|" + "c|" * len(parameters) + "}",
        "\\hline",
        "\\textbf{Network Type} & "
        + " & ".join([PARAM_NAMES[param] for param in parameters])
        + " \\\\",
        "\\hline",
    ]

    # Add rows for each network type
    for network, results in network_results.items():
        # Use the parameters that give best TPR
        best_params = results["best_tpr"]
        if best_params is not None:
            network_name = NETWORK_NAMES.get(network, network.upper())
            param_values = []
            for param in parameters:
                if param == "betting_func":
                    # Capitalize betting function
                    value = best_params[param].capitalize()
                elif param == "distance":
                    # Capitalize distance metric
                    value = best_params[param].capitalize()
                else:
                    value = best_params[param]
                param_values.append(str(value))

            row = f"{network_name} & " + " & ".join(param_values) + " \\\\"
            latex_lines.append(row)
            latex_lines.append("\\hline")

    # Finish the table
    latex_lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Optimal hyperparameter configurations for different network types, determined through comprehensive parameter sweep experiments.}",
            "\\label{tab:best_hyperparams}",
            "\\end{table}",
        ]
    )

    # Write to file with UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Generated LaTeX table at {output_file}")


def generate_summary_report(
    network_results: Dict[str, Dict[str, pd.Series]],
    result_df: pd.DataFrame,
    output_file: str,
):
    """Generate summary report with parameter analysis.

    Args:
        network_results: Dictionary with best parameters for each network
        result_df: DataFrame with all parameter combinations and results
        output_file: Path to output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PARAMETER ANALYSIS SUMMARY\n")
        f.write("=========================\n\n")

        # Summary of all results
        f.write(f"Total parameter combinations analyzed: {len(result_df)}\n")
        f.write(f"Networks: {', '.join(result_df['network'].unique())}\n\n")

        # Best parameters for each network
        f.write("BEST PARAMETERS BY NETWORK TYPE\n")
        f.write("------------------------------\n\n")

        for network, results in network_results.items():
            network_name = NETWORK_NAMES.get(network, network.upper())
            f.write(f"{network_name} Network:\n")

            # Best for traditional TPR
            if results["best_tpr"] is not None:
                f.write("  Best for detection rate (TPR):\n")
                best_tpr = results["best_tpr"]
                tpr_value = best_tpr.get("tpr", "N/A")
                f.write(f"    TPR: {tpr_value:.2%}\n")
                for param in [
                    "threshold",
                    "window",
                    "horizon",
                    "epsilon",
                    "betting_func",
                    "distance",
                ]:
                    param_name = PARAM_NAMES[param]
                    param_value = best_tpr.get(param, "N/A")
                    f.write(f"    {param_name}: {param_value}\n")

            # Best for traditional delay
            if results["best_delay"] is not None:
                f.write("\n  Best for detection delay:\n")
                best_delay = results["best_delay"]
                delay_value = best_delay.get("avg_delay", "N/A")
                f.write(f"    Avg Delay: {delay_value:.2f} steps\n")
                for param in [
                    "threshold",
                    "window",
                    "horizon",
                    "epsilon",
                    "betting_func",
                    "distance",
                ]:
                    param_name = PARAM_NAMES[param]
                    param_value = best_delay.get(param, "N/A")
                    f.write(f"    {param_name}: {param_value}\n")

            # Best for horizon TPR if available
            if results["best_horizon_tpr"] is not None:
                f.write("\n  Best for horizon detection rate (TPR):\n")
                best_h_tpr = results["best_horizon_tpr"]
                h_tpr_value = best_h_tpr.get("horizon_tpr", "N/A")
                f.write(f"    Horizon TPR: {h_tpr_value:.2%}\n")
                for param in [
                    "threshold",
                    "window",
                    "horizon",
                    "epsilon",
                    "betting_func",
                    "distance",
                ]:
                    param_name = PARAM_NAMES[param]
                    param_value = best_h_tpr.get(param, "N/A")
                    f.write(f"    {param_name}: {param_value}\n")

            f.write("\n" + "-" * 40 + "\n\n")

        # Parameter effect analysis
        f.write("PARAMETER EFFECT ANALYSIS\n")
        f.write("------------------------\n\n")

        for param in [
            "threshold",
            "window",
            "horizon",
            "epsilon",
            "betting_func",
            "distance",
        ]:
            f.write(f"Effect of {PARAM_NAMES[param]}:\n")

            # Group by parameter and calculate mean metrics
            try:
                param_effect = result_df.groupby(param)[["tpr", "avg_delay"]].mean()
                f.write(param_effect.to_string() + "\n\n")

                # Network-specific effect
                for network in result_df["network"].unique():
                    network_name = NETWORK_NAMES.get(network, network.upper())
                    network_df = result_df[result_df["network"] == network]
                    network_effect = network_df.groupby(param)[
                        ["tpr", "avg_delay"]
                    ].mean()
                    f.write(f"{network_name} Network:\n")
                    f.write(network_effect.to_string() + "\n\n")
            except Exception as e:
                f.write(f"Error analyzing {param}: {e}\n\n")

        f.write("\nAnalysis completed.\n")

    logger.info(f"Generated summary report at {output_file}")


def generate_parameter_effect_analysis(result_df: pd.DataFrame, output_dir: str):
    """Generate CSV files for parameter effects by network.

    Args:
        result_df: DataFrame with all results
        output_dir: Directory to save output files
    """
    # Parameters to analyze
    parameters = [
        "threshold",
        "window",
        "horizon",
        "epsilon",
        "betting_func",
        "distance",
    ]

    # Create output directory for parameter effects
    param_dir = os.path.join(output_dir, "parameter_effects")
    os.makedirs(param_dir, exist_ok=True)

    # Generate CSV for each parameter
    for param in parameters:
        # Overall effect
        try:
            param_effect = (
                result_df.groupby(param)[["tpr", "avg_delay"]].mean().reset_index()
            )
            param_effect.to_csv(
                os.path.join(param_dir, f"{param}_effect.csv"),
                index=False,
                encoding="utf-8",
            )

            # Effect by network
            param_network_effect = (
                result_df.groupby([param, "network"])[["tpr", "avg_delay"]]
                .mean()
                .reset_index()
            )
            param_network_effect.to_csv(
                os.path.join(param_dir, f"{param}_effect_by_network.csv"),
                index=False,
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Error generating parameter effect for {param}: {e}")

    logger.info(f"Generated parameter effect analysis in {param_dir}")


def main():
    """Main entry point for finding best parameters."""
    parser = argparse.ArgumentParser(
        description="Find best parameters from parameter sweep results."
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="results/parameter_sweep",
        help="Directory containing parameter sweep results",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/best_parameters",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    # Analyze parameter sweep results
    result_df = analyze_parameter_sweep(args.dir)

    # Find best parameters
    find_best_parameters(result_df, args.output)

    # Save the complete results dataframe to CSV
    os.makedirs(args.output, exist_ok=True)
    result_df.to_csv(
        os.path.join(args.output, "all_results.csv"), index=False, encoding="utf-8"
    )

    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
