#!/usr/bin/env python3
"""
LaTeX Table Generator for Sensitivity Analysis

This script reads the sensitivity analysis CSV files and generates LaTeX table data
formatted for the paper's parameter sensitivity table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sensitivity_data(csv_path: str) -> pd.DataFrame:
    """Load and clean sensitivity analysis data"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()


def group_by_parameters(df: pd.DataFrame) -> dict:
    """Group data by parameter configurations"""
    grouped_data = defaultdict(lambda: defaultdict(dict))

    # Network name mapping from lowercase to uppercase
    network_mapping = {"sbm": "SBM", "er": "ER", "ba": "BA", "ws": "NWS"}

    for _, row in df.iterrows():
        network = network_mapping.get(row["network"].lower(), row["network"].upper())

        # Group by different parameter types
        if "betting_type" in row and pd.notna(row["betting_type"]):
            if (
                row["betting_type"] == "power"
                and "epsilon" in row
                and pd.notna(row["epsilon"])
            ):
                param_key = f"power_{row['epsilon']}"
            elif row["betting_type"] == "mixture":
                param_key = "mixture"
            elif (
                row["betting_type"] == "beta"
                and "beta_a" in row
                and pd.notna(row["beta_a"])
            ):
                param_key = f"beta_{row['beta_a']}"
            else:
                param_key = "other"
        else:
            param_key = "default"

        # Add distance measure info
        if "distance" in row and pd.notna(row["distance"]):
            distance_key = f"distance_{row['distance']}"
        else:
            distance_key = "distance_euclidean"

        # Add threshold info
        if "threshold" in row and pd.notna(row["threshold"]):
            threshold_key = f"threshold_{int(row['threshold'])}"
        else:
            threshold_key = "threshold_50"

        # Store metrics (average Traditional and Horizon for each parameter combination)
        for metric in ["TPR", "FPR", "ADD"]:
            if metric in row and pd.notna(row[metric]):
                # Store values by parameter type
                if param_key not in grouped_data[network]:
                    grouped_data[network][param_key] = {}
                if metric not in grouped_data[network][param_key]:
                    grouped_data[network][param_key][metric] = []
                grouped_data[network][param_key][metric].append(row[metric])

                # Store values by distance measure
                if distance_key not in grouped_data[network]:
                    grouped_data[network][distance_key] = {}
                if metric not in grouped_data[network][distance_key]:
                    grouped_data[network][distance_key][metric] = []
                grouped_data[network][distance_key][metric].append(row[metric])

                # Store values by threshold
                if threshold_key not in grouped_data[network]:
                    grouped_data[network][threshold_key] = {}
                if metric not in grouped_data[network][threshold_key]:
                    grouped_data[network][threshold_key][metric] = []
                grouped_data[network][threshold_key][metric].append(row[metric])

    # Average the collected values
    for network in grouped_data:
        for param_key in grouped_data[network]:
            for metric in grouped_data[network][param_key]:
                if isinstance(grouped_data[network][param_key][metric], list):
                    values = grouped_data[network][param_key][metric]
                    avg_val = np.mean(values)
                    grouped_data[network][param_key][metric] = avg_val

                    # Debug: Print FPR values to check if they're being read correctly
                    if metric == "FPR" and avg_val > 0:
                        print(
                            f"DEBUG: {network} {param_key} FPR = {avg_val:.6f} (from {len(values)} values: {values})"
                        )

    return grouped_data


def format_latex_table(grouped_data: dict) -> str:
    """Generate LaTeX table with corrected bolding logic"""
    latex_rows = []

    # Networks in order
    networks = ["SBM", "ER", "BA", "NWS"]
    metrics = ["TPR", "FPR", "ADD"]

    def format_value(val, metric):
        """Format value with appropriate precision"""
        if metric == "FPR":
            return f"{val:.3f}"
        else:
            return f"{val:.2f}"

    def format_bold_value(val, metric):
        """Format bold value with appropriate precision"""
        if metric == "FPR":
            return f"\\textbf{{{val:.3f}}}"
        else:
            return f"\\textbf{{{val:.2f}}}"

    for network in networks:
        if network not in grouped_data:
            logger.warning(f"Network {network} not found in data")
            continue

        network_data = grouped_data[network]

        for i, metric in enumerate(metrics):
            row_parts = []

            # Add network name and metric (only for first metric)
            if i == 0:
                row_parts.append(f"\\multirow{{3}}{{*}}{{{network}}}")
            else:
                row_parts.append("")

            row_parts.append(metric)

            # Power betting values
            power_values = []

            for epsilon in [0.2, 0.5, 0.7, 0.9]:
                param_key = f"power_{epsilon}"
                if param_key in network_data and metric in network_data[param_key]:
                    val = network_data[param_key][metric]
                    power_values.append(val)
                else:
                    power_values.append(0.0)

            # Add mixture
            if "mixture" in network_data and metric in network_data["mixture"]:
                mixture_val = network_data["mixture"][metric]
                power_values.append(mixture_val)
            else:
                power_values.append(0.0)

            # Find best value for this metric type
            non_zero_values = [v for v in power_values if v > 0]
            if non_zero_values:
                if metric == "TPR":
                    # TPR: higher is better
                    best_power = max(non_zero_values)
                else:
                    # FPR and ADD: lower is better
                    best_power = min(non_zero_values)
            else:
                best_power = 0.0

            # Format power betting values with bold for best
            for val in power_values:
                if val > 0 and abs(val - best_power) < 0.001:
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Beta betting values
            beta_values = []

            for beta_a in [0.3, 0.5, 0.7]:
                param_key = f"beta_{beta_a}"
                if param_key in network_data and metric in network_data[param_key]:
                    val = network_data[param_key][metric]
                    beta_values.append(val)
                else:
                    beta_values.append(0.0)

            # Find best value for this metric type
            non_zero_beta_values = [v for v in beta_values if v > 0]
            if non_zero_beta_values:
                if metric == "TPR":
                    # TPR: higher is better
                    best_beta = max(non_zero_beta_values)
                else:
                    # FPR and ADD: lower is better
                    best_beta = min(non_zero_beta_values)
            else:
                best_beta = 0.0

            for val in beta_values:
                if val > 0 and abs(val - best_beta) < 0.001:
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Distance measures (averaged across all parameter combinations)
            distance_values = []
            all_distance_values = []

            for distance in ["euclidean", "mahalanobis", "cosine", "chebyshev"]:
                param_key = f"distance_{distance}"
                if param_key in network_data and metric in network_data[param_key]:
                    val = network_data[param_key][metric]
                    distance_values.append(val)
                    all_distance_values.append(val)
                else:
                    distance_values.append(0.0)

            # Find best distance value
            if all_distance_values:
                if metric == "TPR":
                    best_distance = max(all_distance_values)
                else:
                    best_distance = min(all_distance_values)
            else:
                best_distance = 0.0

            for val in distance_values:
                if val > 0 and abs(val - best_distance) < 0.001:
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Threshold values
            threshold_values = []

            for threshold in [20, 50, 100]:
                param_key = f"threshold_{threshold}"
                if param_key in network_data and metric in network_data[param_key]:
                    val = network_data[param_key][metric]
                    threshold_values.append(val)
                else:
                    threshold_values.append(0.0)

            # Find best value for this metric type
            non_zero_threshold_values = [v for v in threshold_values if v > 0]
            if non_zero_threshold_values:
                if metric == "TPR":
                    # TPR: higher is better
                    best_threshold = max(non_zero_threshold_values)
                else:
                    # FPR and ADD: lower is better
                    best_threshold = min(non_zero_threshold_values)
            else:
                best_threshold = 0.0

            for val in threshold_values:
                if val > 0 and abs(val - best_threshold) < 0.001:
                    row_parts.append(format_bold_value(val, metric))
                else:
                    row_parts.append(format_value(val, metric))

            # Join row parts
            latex_row = " & ".join(row_parts) + " \\\\"
            latex_rows.append(latex_row)

        # Add horizontal line after each network
        latex_rows.append("\\hline")

    return "\n".join(latex_rows)


def main():
    """Main execution"""
    # Load the main results
    results_dir = Path("results/sensitivity_analysis_5_summary")
    sensitivity_file = results_dir / "detailed_results.csv"

    if not sensitivity_file.exists():
        logger.error(f"Sensitivity results file not found: {sensitivity_file}")
        return

    # Load and process data
    df = load_sensitivity_data(str(sensitivity_file))
    if df.empty:
        logger.error("No data loaded")
        return

    logger.info(f"Processing {len(df)} experiments")
    logger.info(f"Networks: {df['network'].unique()}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Group by parameters
    grouped_data = group_by_parameters(df)

    # Generate LaTeX table
    latex_table = format_latex_table(grouped_data)

    # Save LaTeX table
    output_file = results_dir / "parameter_sensitivity_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)

    logger.info(f"LaTeX table saved to: {output_file}")

    # Print the table for immediate use
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY TABLE (LaTeX FORMAT)")
    print("=" * 80)
    print(latex_table)
    print("=" * 80 + "\n")

    # Print summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 40)
    for network in grouped_data:
        print(f"\n{network}:")
        print(f"  Available parameters: {list(grouped_data[network].keys())}")

        # Show best performing configs for each metric
        for metric in ["TPR", "FPR", "ADD"]:
            values = []
            configs = []
            for config, data in grouped_data[network].items():
                if metric in data:
                    values.append(data[metric])
                    configs.append(config)

            if values:
                if metric == "TPR":
                    best_idx = np.argmax(values)
                    print(
                        f"  Best {metric}: {values[best_idx]:.3f} ({configs[best_idx]})"
                    )
                else:
                    best_idx = np.argmin(values)
                    print(
                        f"  Best {metric}: {values[best_idx]:.3f} ({configs[best_idx]})"
                    )


if __name__ == "__main__":
    main()
