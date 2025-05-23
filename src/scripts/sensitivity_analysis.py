#!/usr/bin/env python3
"""
Sensitivity Analysis Script for Network Change Detection Experiments

This script analyzes the results from the sensitivity analysis experiments,
extracting data from individual experiment directories and creating the
normalized relative performance table for the paper.
"""

import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """Analyzer for sensitivity analysis experimental results."""

    def __init__(self, results_dir: str = "results/sensitivity_analysis"):
        """Initialize the analyzer with the results directory."""
        self.results_dir = Path(results_dir)
        self.experiment_data = []
        self.summary_stats = {}

        # Define expected parameter values
        self.network_types = ["sbm", "er", "ba", "ws"]
        self.distance_measures = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
        self.epsilon_values = [0.2, 0.5, 0.7, 0.9]
        self.threshold_values = [20, 50, 100]
        self.beta_a_values = [0.3, 0.5, 0.7]

        # Experiment groups
        self.experiment_groups = [
            "power_betting",
            "mixture_betting",
            "beta_betting",
            "thresholds",
        ]

    def analyze_directory_structure(self):
        """Analyze the directory structure and categorize experiments."""
        logger.info(f"Analyzing directory structure in: {self.results_dir}")

        if not self.results_dir.exists():
            logger.error(f"Results directory not found: {self.results_dir}")
            return

        # Get all subdirectories
        subdirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(subdirs)} subdirectories")

        # Categorize by experiment type
        experiment_counts = defaultdict(int)
        successful_experiments = 0
        failed_experiments = 0

        for subdir in subdirs:
            # Skip consolidated directory
            if subdir.name == "consolidated":
                continue

            # Parse experiment info from directory name
            exp_info = self.parse_experiment_name(subdir.name)
            if exp_info:
                experiment_counts[exp_info.get("group", "unknown")] += 1

                # Check if experiment has results
                results_files = list(subdir.glob("*_results.xlsx")) + list(
                    subdir.glob("*_results.csv")
                )
                config_files = list(subdir.glob("config.yaml"))

                if results_files:
                    successful_experiments += 1
                    logger.debug(f"✓ {subdir.name} - Has results")
                elif config_files:
                    failed_experiments += 1
                    logger.debug(f"✗ {subdir.name} - Config only, no results")
                else:
                    logger.debug(f"? {subdir.name} - No config or results")

        # Print summary
        logger.info("=" * 60)
        logger.info("DIRECTORY STRUCTURE ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total subdirectories: {len(subdirs)}")
        logger.info(f"Successful experiments: {successful_experiments}")
        logger.info(f"Failed experiments: {failed_experiments}")
        logger.info("")
        logger.info("Experiments by type:")
        for exp_type, count in experiment_counts.items():
            logger.info(f"  {exp_type}: {count}")

        self.summary_stats = {
            "total_dirs": len(subdirs),
            "successful": successful_experiments,
            "failed": failed_experiments,
            "by_type": dict(experiment_counts),
        }

        return self.summary_stats

    def parse_experiment_name(self, dir_name: str) -> Optional[Dict]:
        """Parse experiment information from directory name."""
        # Remove timestamp suffix
        name_parts = dir_name.split("_")
        if (
            len(name_parts) >= 2
            and name_parts[-1].isdigit()
            and len(name_parts[-1]) >= 8
        ):
            name_parts = name_parts[:-1]  # Remove timestamp

        name_without_timestamp = "_".join(name_parts)

        # Handle martingale_graph experiments (these seem to be the actual sensitivity experiments)
        if "martingale_graph" in name_without_timestamp:
            # Pattern: network_martingale_graph_distance_betting_params
            # e.g., sbm_martingale_graph_euclidean_power_20250522_154949

            # Extract network (first part)
            network = name_parts[0] if name_parts else "unknown"

            # Find distance (after martingale_graph)
            try:
                mg_idx = name_parts.index("martingale")  # Find 'martingale'
                if mg_idx + 2 < len(
                    name_parts
                ):  # Should be followed by 'graph' and distance
                    distance = name_parts[mg_idx + 2]

                    # Find betting function type
                    if mg_idx + 3 < len(name_parts):
                        betting_type = name_parts[mg_idx + 3]

                        if betting_type == "power":
                            # Extract epsilon if available
                            if mg_idx + 4 < len(name_parts):
                                try:
                                    epsilon = float(name_parts[mg_idx + 4])
                                    return {
                                        "group": "power_betting",
                                        "network": network,
                                        "epsilon": epsilon,
                                        "distance": distance,
                                    }
                                except ValueError:
                                    # Default epsilon if not parseable
                                    return {
                                        "group": "power_betting",
                                        "network": network,
                                        "epsilon": 0.5,  # Default
                                        "distance": distance,
                                    }
                        elif betting_type == "mixture":
                            return {
                                "group": "mixture_betting",
                                "network": network,
                                "distance": distance,
                            }
                        elif betting_type == "beta":
                            # Extract a parameter if available
                            if mg_idx + 4 < len(name_parts):
                                try:
                                    a_param = float(name_parts[mg_idx + 4])
                                    return {
                                        "group": "beta_betting",
                                        "network": network,
                                        "a": a_param,
                                        "distance": distance,
                                    }
                                except ValueError:
                                    return {
                                        "group": "beta_betting",
                                        "network": network,
                                        "a": 0.5,  # Default
                                        "distance": distance,
                                    }
            except (ValueError, IndexError):
                pass

        # Pattern matching for standard experiment types
        patterns = {
            "power_betting": r"^(\w+)_power_([0-9.]+)_(\w+)$",
            "mixture_betting": r"^(\w+)_mixture_(\w+)$",
            "beta_betting": r"^(\w+)_beta_([0-9.]+)_(\w+)$",
            "thresholds": r"^(\w+)_threshold_(\d+)_(\w+)$",
        }

        for group, pattern in patterns.items():
            match = re.match(pattern, name_without_timestamp)
            if match:
                if group == "power_betting":
                    return {
                        "group": group,
                        "network": match.group(1),
                        "epsilon": float(match.group(2)),
                        "distance": match.group(3),
                    }
                elif group == "mixture_betting":
                    return {
                        "group": group,
                        "network": match.group(1),
                        "distance": match.group(2),
                    }
                elif group == "beta_betting":
                    return {
                        "group": group,
                        "network": match.group(1),
                        "a": float(match.group(2)),
                        "distance": match.group(3),
                    }
                elif group == "thresholds":
                    return {
                        "group": group,
                        "network": match.group(1),
                        "threshold": int(match.group(2)),
                        "distance": match.group(3),
                    }

        return None

    def extract_experiment_data(self):
        """Extract experimental data from all successful experiments."""
        logger.info("Extracting experimental data from result files...")

        subdirs = [
            d
            for d in self.results_dir.iterdir()
            if d.is_dir() and d.name != "consolidated"
        ]

        for subdir in subdirs:
            # Parse experiment info
            exp_info = self.parse_experiment_name(subdir.name)
            if not exp_info:
                continue  # Skip unknown experiments

            # Look for detection_results.xlsx files specifically
            results_files = list(subdir.glob("detection_results.xlsx"))
            if not results_files:
                results_files = list(subdir.glob("detection_results.csv"))

            if not results_files:
                logger.debug(f"No detection_results file found in {subdir.name}")
                continue

            # Load the results
            try:
                results_file = results_files[0]
                if results_file.suffix == ".xlsx":
                    df = pd.read_excel(results_file)
                else:
                    df = pd.read_csv(results_file)

                # Calculate summary metrics from time-series data
                if len(df) > 1:  # Time-series data
                    summary_metrics = self.calculate_summary_metrics(df)

                    if summary_metrics:
                        # Combine experiment info with calculated metrics
                        experiment_record = {
                            "experiment_name": subdir.name,
                            "group": exp_info["group"],
                            "network": exp_info["network"],
                            "TPR": summary_metrics["TPR"],
                            "FPR": summary_metrics["FPR"],
                            "ADD": summary_metrics["ADD"],
                            "TPR_horizon": summary_metrics.get("TPR_horizon", 0),
                            "ADD_horizon": summary_metrics.get("ADD_horizon", 0),
                            "delay_reduction": summary_metrics.get(
                                "delay_reduction", 0
                            ),
                        }

                        # Add group-specific parameters
                        if exp_info["group"] == "power_betting":
                            experiment_record["epsilon"] = exp_info["epsilon"]
                            experiment_record["distance"] = exp_info["distance"]
                        elif exp_info["group"] == "mixture_betting":
                            experiment_record["distance"] = exp_info["distance"]
                        elif exp_info["group"] == "beta_betting":
                            experiment_record["a"] = exp_info["a"]
                            experiment_record["distance"] = exp_info["distance"]
                        elif exp_info["group"] == "thresholds":
                            experiment_record["threshold"] = exp_info["threshold"]
                            experiment_record["distance"] = exp_info["distance"]

                        self.experiment_data.append(experiment_record)
                        logger.debug(f"✓ Extracted data from {subdir.name}")
                    else:
                        logger.warning(
                            f"Could not calculate metrics from {results_file}"
                        )

                elif len(df) == 1:  # Summary data (original format)
                    row = df.iloc[0]

                    # Combine experiment info with results
                    experiment_record = {
                        "experiment_name": subdir.name,
                        "group": exp_info["group"],
                        "network": exp_info["network"],
                        "TPR": float(row.get("TPR", 0)),
                        "FPR": float(row.get("FPR", 0)),
                        "ADD": float(row.get("ADD", 0)),
                        "TPR_horizon": float(row.get("TPR_horizon", 0)),
                        "ADD_horizon": float(row.get("ADD_horizon", 0)),
                        "delay_reduction": float(row.get("delay_reduction", 0)),
                    }

                    # Add group-specific parameters
                    if exp_info["group"] == "power_betting":
                        experiment_record["epsilon"] = exp_info["epsilon"]
                        experiment_record["distance"] = exp_info["distance"]
                    elif exp_info["group"] == "mixture_betting":
                        experiment_record["distance"] = exp_info["distance"]
                    elif exp_info["group"] == "beta_betting":
                        experiment_record["a"] = exp_info["a"]
                        experiment_record["distance"] = exp_info["distance"]
                    elif exp_info["group"] == "thresholds":
                        experiment_record["threshold"] = exp_info["threshold"]
                        experiment_record["distance"] = exp_info["distance"]

                    self.experiment_data.append(experiment_record)
                    logger.debug(f"✓ Extracted data from {subdir.name}")
                else:
                    logger.warning(
                        f"Unexpected number of rows ({len(df)}) in {results_file}"
                    )

            except Exception as e:
                logger.error(f"Error reading {results_file}: {str(e)}")

        logger.info(
            f"Successfully extracted data from {len(self.experiment_data)} experiments"
        )
        return self.experiment_data

    def calculate_summary_metrics(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate summary metrics from time-series detection data."""
        try:
            # Extract true change points
            true_changes = df[df["true_change_point"] == 1]["timestep"].tolist()

            if not true_changes:
                logger.warning("No true change points found in data")
                return None

            # Calculate metrics for traditional method
            traditional_metrics = self.calculate_detection_metrics(
                df, true_changes, "traditional_detected"
            )

            # Calculate metrics for horizon method if available
            horizon_metrics = {}
            if "horizon_detected" in df.columns:
                horizon_metrics = self.calculate_detection_metrics(
                    df, true_changes, "horizon_detected"
                )

                # Calculate delay reduction
                if traditional_metrics["ADD"] > 0:
                    delay_reduction = (
                        (traditional_metrics["ADD"] - horizon_metrics["ADD"])
                        / traditional_metrics["ADD"]
                    ) * 100
                else:
                    delay_reduction = 0
                horizon_metrics["delay_reduction"] = delay_reduction

            # Combine metrics
            result = {
                "TPR": traditional_metrics["TPR"],
                "FPR": traditional_metrics["FPR"],
                "ADD": traditional_metrics["ADD"],
                "TPR_horizon": horizon_metrics.get("TPR", 0),
                "ADD_horizon": horizon_metrics.get("ADD", 0),
                "delay_reduction": horizon_metrics.get("delay_reduction", 0),
            }

            return result

        except Exception as e:
            logger.error(f"Error calculating summary metrics: {str(e)}")
            return None

    def calculate_detection_metrics(
        self, df: pd.DataFrame, true_changes: List[int], detection_col: str
    ) -> Dict:
        """Calculate TPR, FPR, and ADD for a detection method."""
        detected_changes = df[df[detection_col] == 1]["timestep"].tolist()

        # Calculate TPR (True Positive Rate)
        true_positives = 0
        detection_delays = []

        for true_change in true_changes:
            # Look for detection within reasonable window (e.g., 20 timesteps)
            detection_window = [
                d for d in detected_changes if true_change <= d <= true_change + 20
            ]
            if detection_window:
                true_positives += 1
                # Calculate delay as time from change to first detection
                delay = min(detection_window) - true_change
                detection_delays.append(delay)

        tpr = true_positives / len(true_changes) if true_changes else 0

        # Calculate FPR (False Positive Rate)
        false_positives = 0
        for detection in detected_changes:
            # Check if this detection corresponds to any true change within window
            if not any(
                true_change <= detection <= true_change + 20
                for true_change in true_changes
            ):
                false_positives += 1

        # FPR = false positives / total possible non-change points
        total_timesteps = len(df)
        total_non_change_points = total_timesteps - len(true_changes)
        fpr = (
            false_positives / total_non_change_points
            if total_non_change_points > 0
            else 0
        )

        # Calculate ADD (Average Detection Delay)
        add = np.mean(detection_delays) if detection_delays else 0

        return {"TPR": tpr, "FPR": fpr, "ADD": add}

    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame from extracted experiment data."""
        if not self.experiment_data:
            logger.warning(
                "No experiment data available. Run extract_experiment_data() first."
            )
            return pd.DataFrame()

        df = pd.DataFrame(self.experiment_data)
        logger.info(
            f"Created summary DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )

        # Print basic statistics
        logger.info("\nExperiment counts by group:")
        group_counts = df["group"].value_counts()
        for group, count in group_counts.items():
            logger.info(f"  {group}: {count}")

        logger.info("\nExperiment counts by network:")
        network_counts = df["network"].value_counts()
        for network, count in network_counts.items():
            logger.info(f"  {network}: {count}")

        return df

    def analyze_parameter_performance(self, df: pd.DataFrame):
        """Analyze parameter performance across different configurations."""
        logger.info("\n" + "=" * 60)
        logger.info("PARAMETER PERFORMANCE ANALYSIS")
        logger.info("=" * 60)

        metrics = ["TPR", "FPR", "ADD"]

        for metric in metrics:
            logger.info(f"\n{metric} Analysis:")
            logger.info("-" * 40)

            # Power betting analysis
            power_df = df[df["group"] == "power_betting"]
            if not power_df.empty:
                logger.info("\nPower Betting (by epsilon):")
                epsilon_performance = power_df.groupby("epsilon")[metric].agg(
                    ["mean", "std", "count"]
                )
                for epsilon, stats in epsilon_performance.iterrows():
                    logger.info(
                        f"  ε={epsilon}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                    )

            # Distance measure analysis
            distance_df = df[df["distance"].notna()]
            if not distance_df.empty:
                logger.info(f"\nDistance Measures:")
                distance_performance = distance_df.groupby("distance")[metric].agg(
                    ["mean", "std", "count"]
                )
                for distance, stats in distance_performance.iterrows():
                    logger.info(
                        f"  {distance}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                    )

            # Threshold analysis
            threshold_df = df[df["group"] == "thresholds"]
            if not threshold_df.empty:
                logger.info(f"\nThreshold Analysis:")
                threshold_performance = threshold_df.groupby("threshold")[metric].agg(
                    ["mean", "std", "count"]
                )
                for threshold, stats in threshold_performance.iterrows():
                    logger.info(
                        f"  λ={threshold}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                    )

    def create_relative_performance_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the relative performance table as shown in the paper."""
        logger.info("\n" + "=" * 60)
        logger.info("CREATING RELATIVE PERFORMANCE TABLE")
        logger.info("=" * 60)

        # This will create the normalized table structure from the paper
        # For now, let's create a basic version and we can refine it

        networks = ["sbm", "er", "ba", "ws"]
        metrics = ["TPR", "FPR", "ADD"]

        # Initialize results dictionary
        results = {}

        for network in networks:
            network_data = df[df["network"] == network]
            if network_data.empty:
                logger.warning(f"No data found for network: {network}")
                continue

            results[network] = {}

            for metric in metrics:
                results[network][metric] = {}

                # Power betting analysis
                power_data = network_data[network_data["group"] == "power_betting"]
                if not power_data.empty:
                    epsilon_results = {}
                    for epsilon in self.epsilon_values:
                        eps_data = power_data[power_data["epsilon"] == epsilon]
                        if not eps_data.empty:
                            epsilon_results[f"ε={epsilon}"] = eps_data[metric].mean()

                    # Normalize to best performance = 1.0
                    if epsilon_results:
                        if metric == "FPR":  # Lower is better for FPR
                            best_val = min(epsilon_results.values())
                            epsilon_results = {
                                k: best_val / v if v > 0 else 1.0
                                for k, v in epsilon_results.items()
                            }
                        else:  # Higher is better for TPR, lower is better for ADD
                            if metric == "TPR":
                                best_val = max(epsilon_results.values())
                                epsilon_results = {
                                    k: v / best_val if best_val > 0 else 1.0
                                    for k, v in epsilon_results.items()
                                }
                            else:  # ADD - lower is better
                                best_val = min(epsilon_results.values())
                                epsilon_results = {
                                    k: best_val / v if v > 0 else 1.0
                                    for k, v in epsilon_results.items()
                                }

                    results[network][metric].update(epsilon_results)

                # Distance measure analysis
                distance_results = {}
                for distance in self.distance_measures:
                    dist_data = network_data[network_data["distance"] == distance]
                    if not dist_data.empty:
                        distance_results[distance] = dist_data[metric].mean()

                # Normalize distance results
                if distance_results:
                    if metric == "FPR":  # Lower is better
                        best_val = min(distance_results.values())
                        distance_results = {
                            k: best_val / v if v > 0 else 1.0
                            for k, v in distance_results.items()
                        }
                    else:
                        if metric == "TPR":  # Higher is better
                            best_val = max(distance_results.values())
                            distance_results = {
                                k: v / best_val if best_val > 0 else 1.0
                                for k, v in distance_results.items()
                            }
                        else:  # ADD - lower is better
                            best_val = min(distance_results.values())
                            distance_results = {
                                k: best_val / v if v > 0 else 1.0
                                for k, v in distance_results.items()
                            }

                results[network][metric].update(distance_results)

                # Threshold analysis
                threshold_data = network_data[network_data["group"] == "thresholds"]
                if not threshold_data.empty:
                    threshold_results = {}
                    for threshold in self.threshold_values:
                        thresh_data = threshold_data[
                            threshold_data["threshold"] == threshold
                        ]
                        if not thresh_data.empty:
                            threshold_results[f"λ={threshold}"] = thresh_data[
                                metric
                            ].mean()

                    # Normalize threshold results
                    if threshold_results:
                        if metric == "FPR":  # Lower is better
                            best_val = min(threshold_results.values())
                            threshold_results = {
                                k: best_val / v if v > 0 else 1.0
                                for k, v in threshold_results.items()
                            }
                        else:
                            if metric == "TPR":  # Higher is better
                                best_val = max(threshold_results.values())
                                threshold_results = {
                                    k: v / best_val if best_val > 0 else 1.0
                                    for k, v in threshold_results.items()
                                }
                            else:  # ADD - lower is better
                                best_val = min(threshold_results.values())
                                threshold_results = {
                                    k: best_val / v if v > 0 else 1.0
                                    for k, v in threshold_results.items()
                                }

                    results[network][metric].update(threshold_results)

        # Convert to DataFrame format suitable for display
        rows = []
        for network in networks:
            if network in results:
                for metric in metrics:
                    row = {"Network": network.upper(), "Metric": metric}
                    row.update(results[network][metric])
                    rows.append(row)

        relative_df = pd.DataFrame(rows)
        logger.info(f"Created relative performance table with {len(relative_df)} rows")

        return relative_df

    def save_results(
        self,
        df: pd.DataFrame,
        relative_df: pd.DataFrame,
        output_dir: str = "results/analysis",
    ):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save raw summary data
        raw_file = output_path / "sensitivity_raw_data.xlsx"
        df.to_excel(raw_file, index=False)
        logger.info(f"Saved raw data to: {raw_file}")

        # Save relative performance table
        relative_file = output_path / "sensitivity_relative_performance.xlsx"
        relative_df.to_excel(relative_file, index=False)
        logger.info(f"Saved relative performance table to: {relative_file}")

        # Save summary statistics
        summary_file = output_path / "sensitivity_summary.txt"
        with open(summary_file, "w") as f:
            f.write("SENSITIVITY ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total experiments analyzed: {len(df)}\n")
            f.write(
                f"Successful experiments: {self.summary_stats.get('successful', 0)}\n"
            )
            f.write(f"Failed experiments: {self.summary_stats.get('failed', 0)}\n\n")

            f.write("Experiments by type:\n")
            for exp_type, count in self.summary_stats.get("by_type", {}).items():
                f.write(f"  {exp_type}: {count}\n")

        logger.info(f"Saved summary to: {summary_file}")

    def run_full_analysis(self):
        """Run the complete sensitivity analysis pipeline."""
        logger.info("Starting full sensitivity analysis...")

        # Step 1: Analyze directory structure
        self.analyze_directory_structure()

        # Step 2: Extract experimental data
        self.extract_experiment_data()

        # Step 3: Create summary DataFrame
        df = self.create_summary_dataframe()

        if df.empty:
            logger.error("No experimental data found. Analysis cannot continue.")
            return None, None

        # Step 4: Analyze parameter performance
        self.analyze_parameter_performance(df)

        # Step 5: Create relative performance table
        relative_df = self.create_relative_performance_table(df)

        # Step 6: Save results
        self.save_results(df, relative_df)

        logger.info("Full sensitivity analysis completed!")
        return df, relative_df


def main():
    """Main function to run sensitivity analysis."""
    # Initialize analyzer
    analyzer = SensitivityAnalyzer()

    # Run full analysis
    raw_df, relative_df = analyzer.run_full_analysis()

    if raw_df is not None:
        print("\nAnalysis completed successfully!")
        print(f"Raw data shape: {raw_df.shape}")
        print(f"Relative performance table shape: {relative_df.shape}")

        # Display sample of results
        print("\nSample of raw data:")
        print(raw_df.head())

        print("\nSample of relative performance data:")
        print(relative_df.head())
    else:
        print("Analysis failed - no data found.")


if __name__ == "__main__":
    main()
