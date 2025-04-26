#!/usr/bin/env python3
# SHAP Analysis for Martingale-based Graph Change Detection

import pandas as pd
import argparse
import os
import sys
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from changepoint.threshold import CustomThresholdModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_martingale_data(file_path):
    """
    Load martingale values from an Excel file with expected sheet structure.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing martingale data

    Returns
    -------
    tuple
        A tuple containing:
        - df: DataFrame with the loaded martingale data
        - change_points: List of true change points
        - timesteps: Array of timestep values
        - feature_cols: List of column names for individual feature martingales
        - original_feature_names: List of human-readable feature names
    """
    print(f"Loading martingale data from {file_path}")

    # Read the Aggregate sheet
    try:
        df = pd.read_excel(file_path, sheet_name="Aggregate")
        print(f"Data loaded with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading Aggregate sheet: {str(e)}")
        return None, [], [], [], []

    # Read the ChangePointMetadata sheet
    try:
        metadata_df = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
        change_points = metadata_df["change_point"].values.tolist()
        print(f"Found {len(change_points)} change points: {change_points}")

        # Print delay information
        if (
            "traditional_avg_delay" in metadata_df.columns
            and "horizon_avg_delay" in metadata_df.columns
        ):
            for i, cp in enumerate(change_points):
                trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]
                reduction = metadata_df.iloc[i]["delay_reduction"]
                print(
                    f"CP {cp}: Trad delay={trad_delay}, Horizon delay={hor_delay}, Reduction={reduction:.2%}"
                )
    except Exception as e:
        print(f"Error reading ChangePointMetadata sheet: {str(e)}")
        change_points = []

    # Get timesteps
    timesteps = df["timestep"].values

    # Find all individual traditional martingale mean columns
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("individual_traditional_martingales_feature")
        and col.endswith("_mean")
    ]

    # Sort feature columns by feature number
    feature_cols.sort(key=lambda x: int(x.split("feature")[1].split("_")[0]))

    print(f"Found {len(feature_cols)} feature martingale columns: {feature_cols}")

    # Define feature name mapping based on threshold.py
    feature_id_to_name = {
        "0": "Degree",
        "1": "Density",
        "2": "Clustering",
        "3": "Betweenness",
        "4": "Eigenvector",
        "5": "Closeness",
        "6": "Spectral",
        "7": "Laplacian",
    }

    # Create human-readable feature names
    original_feature_names = []
    for col in feature_cols:
        feature_id = col.split("feature")[1].split("_")[0]
        original_feature_names.append(
            feature_id_to_name.get(feature_id, f"Feature {feature_id}")
        )

    return df, change_points, timesteps, feature_cols, original_feature_names


def run_safe_shap_analysis(
    model,
    df,
    feature_cols,
    sum_col,
    change_points,
    threshold,
    timesteps,
    output_dir,
    feature_names,
):
    """Run SHAP analysis with error handling for edge cases.

    This wrapper safely handles the SHAP analysis to avoid index errors.
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # First create the time-based visualization which is more reliable
        print("Generating time-based SHAP visualization...")
        contributions_df = model.visualize_shap_over_time(
            df=df,
            feature_cols=feature_cols,
            sum_col=sum_col,
            change_points=change_points,
            timesteps=timesteps,
            output_path=os.path.join(output_dir, "shap_over_time.png"),
            threshold=threshold,
            feature_names=feature_names,
        )

        # Find detection points
        detection_indices = []
        for i in range(1, len(df)):
            if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
                detection_indices.append(i)

        # If no detection points with threshold, find peaks near change points
        if not detection_indices and change_points:
            for cp in change_points:
                cp_idx = np.argmin(np.abs(timesteps - cp))
                window_start = cp_idx
                window_end = min(len(df), cp_idx + 10)
                max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                detection_indices.append(max_idx)

        # Save contributions to CSV if we have them
        if not contributions_df.empty:
            contributions_df.to_csv(
                os.path.join(output_dir, "feature_contributions.csv"), index=False
            )

        # Print analysis summary
        for idx, detection_index in enumerate(detection_indices):
            detection_time = timesteps[detection_index]
            print(
                f"\nFeature contributions at detection point {idx+1} (timestep {detection_time}):"
            )

            # Calculate contributions
            feature_values = df[feature_cols].iloc[detection_index].values
            total = sum(feature_values)
            if total > 0:
                contributions = [
                    (name, val, (val / total) * 100)
                    for name, val in zip(feature_names, feature_values)
                ]
                contributions.sort(key=lambda x: x[2], reverse=True)

                # Print table
                print(
                    f"{'Feature':<15} {'Martingale Value':<15} {'Contribution %':<15}"
                )
                print("-" * 50)
                for name, val, pct in contributions:
                    print(f"{name:<15} {val:<15.6f} {pct:<15.2f}")

        return True

    except Exception as e:
        print(f"Error during SHAP analysis: {str(e)}")
        print("Falling back to simplified analysis...")

        # Fallback to a simplified analysis
        sum_martingale = df[sum_col]

        # Find peaks around change points
        if change_points:
            print("\nFeature contributions at change points:")
            for cp in change_points:
                cp_idx = np.argmin(np.abs(timesteps - cp))
                window = 10
                window_start = max(0, cp_idx)
                window_end = min(len(df), cp_idx + window)

                # Find peak in window
                peak_idx = sum_martingale.iloc[window_start:window_end].idxmax()
                peak_time = timesteps[peak_idx]
                print(f"\nPeak at timestep {peak_time} (near change point {cp}):")

                # Calculate contributions
                feature_values = df[feature_cols].iloc[peak_idx].values
                total = sum(feature_values)
                if total > 0:
                    contributions = [
                        (name, val, (val / total) * 100)
                        for name, val in zip(feature_names, feature_values)
                    ]
                    contributions.sort(key=lambda x: x[2], reverse=True)

                    # Print table
                    print(
                        f"{'Feature':<15} {'Martingale Value':<15} {'Contribution %':<15}"
                    )
                    print("-" * 50)
                    for name, val, pct in contributions:
                        print(f"{name:<15} {val:<15.6f} {pct:<15.2f}")

        return False


def main():
    """Main function to parse arguments and execute the SHAP analysis workflow."""
    parser = argparse.ArgumentParser(description="Analyze martingale data with SHAP.")
    parser.add_argument(
        "file_path", help="Path to Excel file containing martingale data"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Detection threshold (default: 60.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/shap_analysis",
        help="Directory to save output files (default: results/shap_analysis)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce verbosity of output"
    )
    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)

    # Load data
    result = load_martingale_data(args.file_path)
    df, change_points, timesteps, feature_cols, original_feature_names = result

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Check for NaN values and replace with zeros
    print("Checking for NaN values...")
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"Replacing {nan_count} NaN values in {col} with zeros")
            df[col] = df[col].fillna(0)

    sum_col = "traditional_sum_martingales_mean"
    if sum_col in df.columns:
        nan_count = df[sum_col].isna().sum()
        if nan_count > 0:
            print(f"Replacing {nan_count} NaN values in {sum_col} with zeros")
            df[sum_col] = df[sum_col].fillna(0)

    # Create threshold model
    model = CustomThresholdModel(threshold=args.threshold)

    # Print mapping between feature IDs and names
    print("\nFeature mapping:")
    for i, (col, name) in enumerate(zip(feature_cols, original_feature_names)):
        print(f"  Feature {i}: {name}")

    # Run SHAP analysis with error handling
    print(f"\nRunning SHAP analysis with threshold={args.threshold}")
    success = run_safe_shap_analysis(
        model=model,
        df=df,
        feature_cols=feature_cols,
        sum_col=sum_col,
        change_points=change_points,
        threshold=args.threshold,
        timesteps=timesteps,
        output_dir=args.output_dir,
        feature_names=original_feature_names,
    )

    if success:
        print(f"Analysis complete. Results saved to {args.output_dir}")
    else:
        print(f"Simplified analysis complete. Some visualizations may be missing.")


if __name__ == "__main__":
    main()
