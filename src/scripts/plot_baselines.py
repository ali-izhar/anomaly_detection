#!/usr/bin/env python

"""
Script to plot CUSUM and EWMA change detection results.
Similar to plot_martingale.py but specialized for these baseline methods.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.patches import Patch
import argparse


def setup_plot_style():
    """Set up consistent plotting style for research visualizations."""
    plt.style.use("seaborn-v0_8-paper")
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "figure.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": (12, 7),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.axisbelow": True,
            "lines.linewidth": 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# Feature display names mapping
FEATURE_INFO = {
    "0": "Degree",
    "1": "Density",
    "2": "Clustering",
    "3": "Betweenness",
    "4": "Eigenvector",
    "5": "Closeness",
    "6": "Spectral",
    "7": "Laplacian",
}


def load_data(file_path, sheet_name="Aggregate"):
    """Load detection data from an Excel file. Return
    a dataframe with the data and a list of change points."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully read sheet: {sheet_name} from {file_path}")
        try:
            metadata_df = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
            change_points = metadata_df["change_point"].values.tolist()
            print(f"Found {len(change_points)} change points in metadata")
        except Exception:
            change_points = []
            if "true_change_point" in df.columns:
                change_points = df.loc[
                    ~df["true_change_point"].isna(), "timestep"
                ].values.tolist()
                print(f"Found {len(change_points)} change points in main sheet")

        return df, change_points

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, []


def load_trial_data(file_path):
    """Load data from all trial sheets and metadata.

    Returns:
        trial_dfs: List of dataframes for each trial
        change_points: List of change points
        metadata_df: DataFrame with change point metadata
    """
    try:
        excel = pd.ExcelFile(file_path)
        trial_sheets = [
            sheet for sheet in excel.sheet_names if sheet.startswith("Trial")
        ]
        print(f"Found {len(trial_sheets)} trial sheets: {trial_sheets}")

        # Load change point metadata if available
        try:
            metadata_df = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
            change_points = metadata_df["change_point"].values.tolist()
            print(f"Found {len(change_points)} change points in metadata")

            if all(col in metadata_df.columns for col in ["traditional_avg_delay"]):
                for i, cp in enumerate(change_points):
                    trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                    print(f"CP {cp}: Detection delay={trad_delay:.2f}")
        except Exception as e:
            print(f"Warning: Could not read metadata: {str(e)}")
            metadata_df = None
            change_points = []

        # Load each trial sheet
        trial_dfs = []
        for sheet in trial_sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            trial_dfs.append(df)
            # If we don't have change points yet, try to get them from this sheet
            if not change_points and "true_change_point" in df.columns:
                change_points = df.loc[
                    ~df["true_change_point"].isna(), "timestep"
                ].values.tolist()

        return trial_dfs, change_points, metadata_df

    except Exception as e:
        print(f"Error loading trial data: {str(e)}")
        return [], [], None


def plot_cusum(df, output_path, change_points=None, trial_dfs=None, metadata_df=None):
    """Create a CUSUM plot with both positive and negative statistics.

    Args:
        df: DataFrame containing CUSUM data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        trial_dfs: List of DataFrames for individual trials
        metadata_df: DataFrame with change point metadata
    """
    # Check for required columns
    required_cols = ["timestep"]

    # Check different naming variations that could exist in the data
    cusum_pos_options = ["cusum_positives", "cusum_positives_mean", "cusum_pos"]
    cusum_neg_options = ["cusum_negatives", "cusum_negatives_mean", "cusum_neg"]
    combined_options = ["combined_cusum", "combined_cusum_mean"]

    # Also check for standard martingale column names as fallbacks
    martingale_options = ["traditional_martingales", "traditional_martingales_mean"]
    sum_options = ["traditional_sum_martingales", "traditional_sum_martingales_mean"]
    avg_options = ["traditional_avg_martingales", "traditional_avg_martingales_mean"]

    # Try to find appropriate columns
    cusum_pos_col = next((col for col in df.columns if col in cusum_pos_options), None)
    cusum_neg_col = next((col for col in df.columns if col in cusum_neg_options), None)
    combined_col = next((col for col in df.columns if col in combined_options), None)

    # If CUSUM-specific columns aren't found, use martingale columns as fallbacks
    martingale_col = next(
        (col for col in df.columns if col in martingale_options), None
    )
    sum_col = next((col for col in df.columns if col in sum_options), None)
    avg_col = next((col for col in df.columns if col in avg_options), None)

    # Use the best available column for combined CUSUM
    if combined_col is None:
        if sum_col:
            combined_col = sum_col
            print(f"Using {sum_col} as combined CUSUM")
        elif avg_col:
            combined_col = avg_col
            print(f"Using {avg_col} as combined CUSUM")
        elif martingale_col:
            combined_col = martingale_col
            print(f"Using {martingale_col} as combined CUSUM")

    if not (cusum_pos_col or cusum_neg_col or combined_col):
        print(
            "No CUSUM data columns found in the dataset, and no suitable fallback columns found"
        )
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = df["timestep"].values
    x_min, x_max = min(x), max(x)
    x_ticks = np.arange(0, x_max + 40, 40)
    x_limits = (x_min, x_max + 5)

    # Plot data
    if cusum_pos_col:
        ax.plot(x, df[cusum_pos_col], "b-", label="CUSUM+", linewidth=1.5, alpha=0.8)

    if cusum_neg_col:
        ax.plot(x, df[cusum_neg_col], "r-", label="CUSUM-", linewidth=1.5, alpha=0.8)

    if combined_col:
        ax.plot(x, df[combined_col], "k-", label="CUSUM Statistic", linewidth=2)

    # Identify threshold from data
    # First check for columns with threshold in their name
    threshold_col = next(
        (col for col in df.columns if "threshold" in col.lower()), None
    )
    if threshold_col:
        threshold = df[threshold_col].iloc[0]
    else:
        # Look for threshold in configuration or metadata
        if metadata_df is not None and "threshold" in metadata_df.columns:
            threshold = metadata_df["threshold"].iloc[0]
        else:
            # Estimate threshold based on detection points if change points exist
            if change_points and combined_col:
                # Find the CUSUM value when it crosses at change points
                threshold_values = []
                for cp in change_points:
                    # Find detection point (first point after change where CUSUM exceeds some value)
                    detection_window = df[
                        (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                    ]
                    if not detection_window.empty:
                        # Find the maximum value in this window as a proxy for threshold
                        max_val = detection_window[combined_col].max()
                        if max_val > 0:
                            threshold_values.append(max_val)

                threshold = min(threshold_values) if threshold_values else 8.0
            else:
                # Default threshold if we can't determine it
                threshold = 8.0

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="g",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.1f})",
    )

    # Mark change points
    detection_points = []
    if change_points:
        for cp in change_points:
            # Add vertical line for change point
            ax.axvline(
                x=cp,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="True Change Point" if len(detection_points) == 0 else "",
            )

            # Find detection point (first point after change where CUSUM exceeds threshold)
            if combined_col:
                detection_window = df[
                    (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                ]
                detection_idx = detection_window[
                    detection_window[combined_col] >= threshold
                ].index.min()
                if pd.notna(detection_idx):
                    detection_point = detection_window.loc[detection_idx, "timestep"]
                    detection_points.append(detection_point)

                    # Mark detection point
                    ax.scatter(
                        [detection_point],
                        [threshold],
                        color="red",
                        s=100,
                        zorder=10,
                        marker="o",
                        label="Detection Point" if len(detection_points) == 1 else "",
                    )

                    # Add annotation for detection delay
                    delay = detection_point - cp
                    ax.annotate(
                        f"Delay: {delay:.1f}",
                        xy=(detection_point, threshold),
                        xytext=(detection_point + 2, threshold + 1),
                        fontweight="bold",
                        fontsize=12,
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="red"
                        ),
                    )

    # Improve plot appearance
    ax.set_xlim(x_limits)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_ylabel("CUSUM Statistic", fontweight="bold")
    ax.set_title("CUSUM Detection Results", fontsize=16, fontweight="bold")

    # Add legend with improved appearance
    ax.legend(loc="upper right", framealpha=0.95, frameon=True, fancybox=True)

    # Add grid for readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Highlight change regions
    if change_points:
        for cp in change_points:
            # Add subtle background shading to highlight change points
            ax.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CUSUM plot to {output_path}")


def plot_ewma(df, output_path, change_points=None, trial_dfs=None, metadata_df=None):
    """Create an EWMA plot with control limits.

    Args:
        df: DataFrame containing EWMA data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        trial_dfs: List of DataFrames for individual trials
        metadata_df: DataFrame with change point metadata
    """
    # Check for required columns
    required_cols = ["timestep"]

    # Check different naming variations that could exist in the data
    ewma_options = ["ewma_values", "ewma_values_mean", "ewma"]
    upper_options = ["upper_limits", "upper_limits_mean", "ucl"]
    lower_options = ["lower_limits", "lower_limits_mean", "lcl"]
    stat_options = ["ewma_statistics", "ewma_statistics_mean", "ewma_stats"]

    # Also check for standard martingale column names as fallbacks
    martingale_options = ["traditional_martingales", "traditional_martingales_mean"]
    sum_options = ["traditional_sum_martingales", "traditional_sum_martingales_mean"]
    avg_options = ["traditional_avg_martingales", "traditional_avg_martingales_mean"]

    # Try to find EWMA-specific columns
    ewma_col = next((col for col in df.columns if col in ewma_options), None)
    upper_col = next((col for col in df.columns if col in upper_options), None)
    lower_col = next((col for col in df.columns if col in lower_options), None)
    stat_col = next((col for col in df.columns if col in stat_options), None)

    # If EWMA-specific columns aren't found, use martingale columns as fallbacks
    martingale_col = next(
        (col for col in df.columns if col in martingale_options), None
    )
    sum_col = next((col for col in df.columns if col in sum_options), None)
    avg_col = next((col for col in df.columns if col in avg_options), None)

    # Use the best available column for EWMA values
    if ewma_col is None:
        if martingale_col:
            ewma_col = martingale_col
            print(f"Using {martingale_col} as EWMA values")
        elif avg_col:
            ewma_col = avg_col
            print(f"Using {avg_col} as EWMA values")

    # Use the best available column for EWMA statistics
    if stat_col is None:
        if sum_col:
            stat_col = sum_col
            print(f"Using {sum_col} as EWMA statistics")
        elif avg_col and avg_col != ewma_col:
            stat_col = avg_col
            print(f"Using {avg_col} as EWMA statistics")

    if not (ewma_col or stat_col):
        print(
            "No EWMA data columns found in the dataset, and no suitable fallback columns found"
        )
        return

    # For EWMA without control limits, we'll use a single panel plot instead of two panels
    if not (upper_col and lower_col):
        fig, ax = plt.subplots(figsize=(12, 7))

        x = df["timestep"].values
        x_min, x_max = min(x), max(x)
        x_ticks = np.arange(0, x_max + 40, 40)
        x_limits = (x_min, x_max + 5)

        # Plot EWMA values
        if ewma_col:
            # Check if we have individual features or just an aggregate
            ewma_feature_cols = [
                col
                for col in df.columns
                if col.startswith("individual_") and "martingales_feature" in col
            ]
            if ewma_feature_cols:
                # We have individual features - plot them with transparency
                for col in ewma_feature_cols:
                    feature_id = col.split("_")[-1].replace("feature", "")
                    if feature_id.isdigit() and int(feature_id) in FEATURE_INFO:
                        feature_name = FEATURE_INFO[feature_id]
                        ax.plot(
                            x, df[col], alpha=0.3, label=f"{feature_name}", linewidth=1
                        )

                # Plot average as a more prominent line
                ax.plot(x, df[ewma_col], "k-", label="EWMA Statistic", linewidth=2)
            else:
                # Just plot the main EWMA value
                ax.plot(x, df[ewma_col], "b-", label="EWMA Statistic", linewidth=2)

        # If we have a separate statistical column, plot it too
        if stat_col and stat_col != ewma_col:
            ax.plot(
                x,
                df[stat_col],
                "r-",
                label="EWMA Control Statistic",
                linewidth=1.5,
                alpha=0.7,
            )

        # Determine threshold from data or use a default
        threshold = None

        # First check metadata
        if metadata_df is not None and "threshold" in metadata_df.columns:
            threshold = metadata_df["threshold"].iloc[0]

        # If not found in metadata, try to detect from detection points
        if threshold is None and ewma_col and change_points:
            threshold_values = []
            for cp in change_points:
                detection_window = df[
                    (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                ]
                if not detection_window.empty:
                    max_val = detection_window[ewma_col].max()
                    if max_val > 0:
                        threshold_values.append(max_val)
            threshold = min(threshold_values) if threshold_values else None

        # Default threshold
        if threshold is None:
            threshold = (
                df[ewma_col].mean() + 2 * df[ewma_col].std() if ewma_col else 3.0
            )

        # Add threshold line
        ax.axhline(
            y=threshold,
            color="g",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.1f})",
        )
    else:
        # Create figure - two panels, one for EWMA values with control limits, one for statistics
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        x = df["timestep"].values
        x_min, x_max = min(x), max(x)
        x_ticks = np.arange(0, x_max + 40, 40)
        x_limits = (x_min, x_max + 5)

        # Panel 1: EWMA values with control limits
        if ewma_col:
            # Check if we have individual features or just an aggregate
            ewma_feature_cols = [
                col
                for col in df.columns
                if col.startswith("ewma_values_") and col.endswith("_mean")
            ]
            if ewma_feature_cols:
                # We have individual features - plot them with transparency
                for col in ewma_feature_cols:
                    feature_id = col.split("_")[2].replace("feature", "")
                    if feature_id.isdigit() and int(feature_id) in FEATURE_INFO:
                        feature_name = FEATURE_INFO[feature_id]
                        ax1.plot(
                            x, df[col], alpha=0.3, label=f"{feature_name}", linewidth=1
                        )

                # Plot average as a more prominent line
                if "ewma_values_mean" in df.columns:
                    ax1.plot(
                        x,
                        df["ewma_values_mean"],
                        "k-",
                        label="Average EWMA",
                        linewidth=2,
                    )
            else:
                # Just plot the main EWMA value
                ax1.plot(x, df[ewma_col], "b-", label="EWMA", linewidth=2)

        # Add control limits if available
        if upper_col and lower_col:
            ax1.plot(
                x, df[upper_col], "r--", label="Upper Control Limit", linewidth=1.5
            )
            ax1.plot(
                x, df[lower_col], "r--", label="Lower Control Limit", linewidth=1.5
            )

            # Shade the area between control limits
            ax1.fill_between(x, df[lower_col], df[upper_col], color="red", alpha=0.1)

        # Panel 2: EWMA statistics (normalized distance from center)
        if stat_col:
            ax2.plot(x, df[stat_col], "b-", label="EWMA Statistic", linewidth=2)

            # Identify threshold from data or hardcode a reasonable value
            threshold = 2.0  # Default threshold (typically 2-3 standard deviations)
            ax2.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold ({threshold:.1f})",
            )

        # Mark change points and detections on both panels
        detection_points = []
        if change_points:
            for cp in change_points:
                # Add vertical lines for change points on both panels
                ax1.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    label="True Change Point" if len(detection_points) == 0 else "",
                )
                ax2.axvline(x=cp, color="gray", linestyle="--", linewidth=1.5)

                # Find detection point using traditional detection method (threshold crossing)
                detection_point = None

                if ewma_col:
                    detection_window = df[
                        (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                    ]

                    if upper_col and lower_col:
                        # Find where EWMA exceeds either control limit
                        outside_limits = detection_window[
                            (detection_window[ewma_col] > detection_window[upper_col])
                            | (detection_window[ewma_col] < detection_window[lower_col])
                        ]

                        if not outside_limits.empty:
                            detection_idx = outside_limits.index.min()
                            detection_point = outside_limits.loc[
                                detection_idx, "timestep"
                            ]
                            detection_points.append(detection_point)

                            # Determine which limit was exceeded
                            ewma_val = outside_limits.loc[detection_idx, ewma_col]
                            upper_val = outside_limits.loc[detection_idx, upper_col]

                            limit_exceeded = (
                                upper_val
                                if ewma_val > upper_val
                                else outside_limits.loc[detection_idx, lower_col]
                            )

                            # Mark detection point on both panels
                            ax1.scatter(
                                [detection_point],
                                [ewma_val],
                                color="red",
                                s=100,
                                zorder=10,
                                marker="o",
                                label=(
                                    "Detection Point"
                                    if len(detection_points) == 1
                                    else ""
                                ),
                            )

                            if stat_col:
                                stat_val = outside_limits.loc[detection_idx, stat_col]
                                ax2.scatter(
                                    [detection_point],
                                    [stat_val],
                                    color="red",
                                    s=100,
                                    zorder=10,
                                    marker="o",
                                )

                            # Add annotation for detection delay on top panel
                            delay = detection_point - cp
                            ax1.annotate(
                                f"Delay: {delay:.1f}",
                                xy=(detection_point, ewma_val),
                                xytext=(detection_point + 2, ewma_val),
                                fontweight="bold",
                                fontsize=12,
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    fc="white",
                                    alpha=0.8,
                                    ec="red",
                                ),
                            )
                    else:
                        # Without control limits, just use a fixed threshold
                        threshold_val = (
                            threshold
                            if threshold is not None
                            else df[ewma_col].mean() + 2 * df[ewma_col].std()
                        )
                        over_threshold = detection_window[
                            detection_window[ewma_col] >= threshold_val
                        ]

                        if not over_threshold.empty:
                            detection_idx = over_threshold.index.min()
                            detection_point = over_threshold.loc[
                                detection_idx, "timestep"
                            ]
                            detection_points.append(detection_point)

                            ewma_val = over_threshold.loc[detection_idx, ewma_col]

                            # Mark detection point
                            ax1.scatter(
                                [detection_point],
                                [ewma_val],
                                color="red",
                                s=100,
                                zorder=10,
                                marker="o",
                                label=(
                                    "Detection Point"
                                    if len(detection_points) == 1
                                    else ""
                                ),
                            )

                            if stat_col:
                                try:
                                    stat_val = over_threshold.loc[
                                        detection_idx, stat_col
                                    ]
                                    ax2.scatter(
                                        [detection_point],
                                        [stat_val],
                                        color="red",
                                        s=100,
                                        zorder=10,
                                        marker="o",
                                    )
                                except:
                                    pass

                            # Add annotation for detection delay
                            delay = detection_point - cp
                            ax1.annotate(
                                f"Delay: {delay:.1f}",
                                xy=(detection_point, ewma_val),
                                xytext=(detection_point + 2, ewma_val),
                                fontweight="bold",
                                fontsize=12,
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    fc="white",
                                    alpha=0.8,
                                    ec="red",
                                ),
                            )

        # Improve plot appearance
        ax1.set_xlim(x_limits)
        ax1.set_ylabel("EWMA Value", fontweight="bold")
        ax1.set_title("EWMA Change Detection Results", fontsize=16, fontweight="bold")

        ax2.set_xlim(x_limits)
        ax2.set_xticks(x_ticks)
        ax2.set_xlabel("Time", fontweight="bold")
        ax2.set_ylabel("EWMA Statistic", fontweight="bold")

        # Add legends with improved appearance
        ax1.legend(loc="upper right", framealpha=0.95, frameon=True, fancybox=True)
        ax2.legend(loc="upper right", framealpha=0.95, frameon=True, fancybox=True)

        # Add grid for readability
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Highlight change regions
        if change_points:
            for cp in change_points:
                # Add subtle background shading to highlight change points
                ax1.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)
                ax2.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)

    # Single panel setup if we're not using the dual-panel approach
    if "ax" in locals():
        # Mark change points
        if change_points:
            for cp in change_points:
                ax.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    label="True Change Point" if cp == change_points[0] else "",
                )

                # Add subtle background shading
                ax.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)

        # Find detection points
        if ewma_col and change_points:
            for cp in change_points:
                detection_window = df[
                    (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                ]
                if not detection_window.empty:
                    # Find first point that exceeds threshold
                    over_threshold = detection_window[
                        detection_window[ewma_col] >= threshold
                    ]
                    if not over_threshold.empty:
                        detection_idx = over_threshold.index.min()
                        detection_point = over_threshold.loc[detection_idx, "timestep"]

                        # Mark detection point
                        ewma_val = over_threshold.loc[detection_idx, ewma_col]
                        ax.scatter(
                            [detection_point],
                            [ewma_val],
                            color="red",
                            s=100,
                            zorder=10,
                            marker="o",
                            label="Detection Point",
                        )

                        # Add annotation for detection delay
                        delay = detection_point - cp
                        ax.annotate(
                            f"Delay: {delay:.1f}",
                            xy=(detection_point, ewma_val),
                            xytext=(detection_point + 2, ewma_val),
                            fontweight="bold",
                            fontsize=12,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="white",
                                alpha=0.8,
                                ec="red",
                            ),
                        )

        # Finalize single panel plot
        ax.set_xlim(x_limits)
        ax.set_xticks(x_ticks)
        ax.set_xlabel("Time", fontweight="bold")
        ax.set_ylabel("EWMA Value", fontweight="bold")
        ax.set_title("EWMA Change Detection Results", fontsize=16, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.95, frameon=True, fancybox=True)
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create figure - two panels, one for EWMA values with control limits, one for statistics
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    x = df["timestep"].values
    x_min, x_max = min(x), max(x)
    x_ticks = np.arange(0, x_max + 40, 40)
    x_limits = (x_min, x_max + 5)

    # Panel 1: EWMA values with control limits
    if ewma_col:
        # Check if we have individual features or just an aggregate
        ewma_feature_cols = [
            col
            for col in df.columns
            if col.startswith("ewma_values_") and col.endswith("_mean")
        ]
        if ewma_feature_cols:
            # We have individual features - plot them with transparency
            for col in ewma_feature_cols:
                feature_id = col.split("_")[2].replace("feature", "")
                if feature_id.isdigit() and int(feature_id) in FEATURE_INFO:
                    feature_name = FEATURE_INFO[feature_id]
                    ax1.plot(
                        x, df[col], alpha=0.3, label=f"{feature_name}", linewidth=1
                    )

            # Plot average as a more prominent line
            if "ewma_values_mean" in df.columns:
                ax1.plot(
                    x, df["ewma_values_mean"], "k-", label="Average EWMA", linewidth=2
                )
        else:
            # Just plot the main EWMA value
            ax1.plot(x, df[ewma_col], "b-", label="EWMA", linewidth=2)

    # Add control limits if available
    if upper_col and lower_col:
        ax1.plot(x, df[upper_col], "r--", label="Upper Control Limit", linewidth=1.5)
        ax1.plot(x, df[lower_col], "r--", label="Lower Control Limit", linewidth=1.5)

        # Shade the area between control limits
        ax1.fill_between(x, df[lower_col], df[upper_col], color="red", alpha=0.1)

    # Panel 2: EWMA statistics (normalized distance from center)
    if stat_col:
        ax2.plot(x, df[stat_col], "b-", label="EWMA Statistic", linewidth=2)

        # Identify threshold from data or hardcode a reasonable value
        threshold = 2.0  # Default threshold (typically 2-3 standard deviations)
        ax2.axhline(
            y=threshold,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.1f})",
        )

    # Mark change points and detections on both panels
    detection_points = []
    if change_points:
        for cp in change_points:
            # Add vertical lines for change points on both panels
            ax1.axvline(
                x=cp,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="True Change Point" if len(detection_points) == 0 else "",
            )
            ax2.axvline(x=cp, color="gray", linestyle="--", linewidth=1.5)

            # Find detection point
            # This is complex for EWMA since detection can happen when value exceeds either control limit
            detection_point = None

            if ewma_col and upper_col and lower_col:
                # Check for points where EWMA exceeds either control limit
                detection_window = df[
                    (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                ]

                if not detection_window.empty:
                    # Find where EWMA exceeds either limit
                    outside_limits = detection_window[
                        (detection_window[ewma_col] > detection_window[upper_col])
                        | (detection_window[ewma_col] < detection_window[lower_col])
                    ]

                    if not outside_limits.empty:
                        detection_idx = outside_limits.index.min()
                        detection_point = outside_limits.loc[detection_idx, "timestep"]
                        detection_points.append(detection_point)

                        # Determine which limit was exceeded
                        ewma_val = outside_limits.loc[detection_idx, ewma_col]
                        upper_val = outside_limits.loc[detection_idx, upper_col]

                        limit_exceeded = (
                            upper_val
                            if ewma_val > upper_val
                            else outside_limits.loc[detection_idx, lower_col]
                        )

                        # Mark detection point on both panels
                        ax1.scatter(
                            [detection_point],
                            [ewma_val],
                            color="red",
                            s=100,
                            zorder=10,
                            marker="o",
                            label=(
                                "Detection Point" if len(detection_points) == 1 else ""
                            ),
                        )

                        if stat_col:
                            stat_val = outside_limits.loc[detection_idx, stat_col]
                            ax2.scatter(
                                [detection_point],
                                [stat_val],
                                color="red",
                                s=100,
                                zorder=10,
                                marker="o",
                            )

                        # Add annotation for detection delay on top panel
                        delay = detection_point - cp
                        ax1.annotate(
                            f"Delay: {delay:.1f}",
                            xy=(detection_point, ewma_val),
                            xytext=(detection_point + 2, ewma_val),
                            fontweight="bold",
                            fontsize=12,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="white",
                                alpha=0.8,
                                ec="red",
                            ),
                        )

    # Improve plot appearance
    ax1.set_xlim(x_limits)
    ax1.set_ylabel("EWMA Value", fontweight="bold")
    ax1.set_title("EWMA Change Detection Results", fontsize=16, fontweight="bold")

    ax2.set_xlim(x_limits)
    ax2.set_xticks(x_ticks)
    ax2.set_xlabel("Time", fontweight="bold")
    ax2.set_ylabel("EWMA Statistic", fontweight="bold")

    # Add legends with improved appearance
    ax1.legend(loc="upper right", framealpha=0.95, frameon=True, fancybox=True)
    ax2.legend(loc="upper right", framealpha=0.95, frameon=True, fancybox=True)

    # Add grid for readability
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Highlight change regions
    if change_points:
        for cp in change_points:
            # Add subtle background shading to highlight change points
            ax1.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)
            ax2.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved EWMA plot to {output_path}")


def plot_comparison(file_paths, output_dir, method_names=None):
    """Create comparison plots for different detection methods.

    Args:
        file_paths: List of paths to Excel files with detection results
        output_dir: Directory to save output plots
        method_names: Optional list of method names corresponding to file_paths
    """
    if method_names is None:
        # Extract method names from file paths if not provided
        method_names = []
        for path in file_paths:
            # Try to extract method name from the directory name
            parts = os.path.basename(os.path.dirname(path)).split("_")
            if len(parts) > 1:
                method_names.append(
                    parts[1].upper()
                )  # e.g., extract "cusum" from "ws_cusum_graph..."
            else:
                method_names.append(f"Method {len(method_names)+1}")

    # Load data for all methods
    all_data = []
    all_change_points = []
    all_detection_points = []

    for i, path in enumerate(file_paths):
        df, change_points = load_data(path)
        if df is None:
            continue

        # Determine detection method type and find detection points
        method = method_names[i].lower()
        detection_points = []

        if "cusum" in method:
            # Find combined CUSUM column
            combined_col = next(
                (col for col in df.columns if "combined_cusum" in col), None
            )
            threshold_val = 8.0  # Default if not found

            if combined_col and change_points:
                for cp in change_points:
                    # Find first point after change where CUSUM exceeds threshold
                    detection_window = df[
                        (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                    ]
                    if not detection_window.empty:
                        detection_idx = detection_window[
                            detection_window[combined_col] >= threshold_val
                        ].index.min()
                        if pd.notna(detection_idx):
                            detection_points.append(
                                detection_window.loc[detection_idx, "timestep"]
                            )

        elif "ewma" in method:
            # Find EWMA column and control limits
            ewma_col = next((col for col in df.columns if "ewma_values" in col), None)
            upper_col = next((col for col in df.columns if "upper_limits" in col), None)
            lower_col = next((col for col in df.columns if "lower_limits" in col), None)

            if ewma_col and upper_col and lower_col and change_points:
                for cp in change_points:
                    # Find first point after change where EWMA exceeds either control limit
                    detection_window = df[
                        (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                    ]
                    if not detection_window.empty:
                        outside_limits = detection_window[
                            (detection_window[ewma_col] > detection_window[upper_col])
                            | (detection_window[ewma_col] < detection_window[lower_col])
                        ]
                        if not outside_limits.empty:
                            detection_points.append(outside_limits.iloc[0]["timestep"])

        else:  # Martingale or other
            # Find traditional martingale column
            trad_col = next(
                (
                    col
                    for col in df.columns
                    if "traditional_martingales" in col
                    or "traditional_sum_martingales" in col
                ),
                None,
            )
            threshold_val = 60.0  # Default martingale threshold

            if trad_col and change_points:
                for cp in change_points:
                    # Find first point after change where martingale exceeds threshold
                    detection_window = df[
                        (df["timestep"] >= cp) & (df["timestep"] <= cp + 15)
                    ]
                    if not detection_window.empty:
                        detection_idx = detection_window[
                            detection_window[trad_col] >= threshold_val
                        ].index.min()
                        if pd.notna(detection_idx):
                            detection_points.append(
                                detection_window.loc[detection_idx, "timestep"]
                            )

        all_data.append(df)
        all_change_points.append(change_points)
        all_detection_points.append(detection_points)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for different methods
    colors = ["blue", "red", "green", "purple", "orange"]

    # Get common x-axis limits
    x_min = min([df["timestep"].min() for df in all_data])
    x_max = max([df["timestep"].max() for df in all_data])
    x_ticks = np.arange(0, x_max + 40, 40)
    x_limits = (x_min, x_max + 5)

    # Determine overall change points (use first file's change points if available)
    overall_change_points = all_change_points[0] if all_change_points else []

    # Mark change points
    if overall_change_points:
        for cp in overall_change_points:
            ax.axvline(
                x=cp,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="True Change Point" if cp == overall_change_points[0] else "",
            )

            # Add subtle background shading to highlight change points
            ax.axvspan(cp - 1, cp + 1, color="lightgray", alpha=0.3, zorder=0)

    # Plot normalized detection statistics for each method
    for i, (df, detection_points, method) in enumerate(
        zip(all_data, all_detection_points, method_names)
    ):
        color = colors[i % len(colors)]

        # Determine which column to use for detection statistic
        if "cusum" in method.lower():
            stat_col = next(
                (col for col in df.columns if "combined_cusum" in col), None
            )
        elif "ewma" in method.lower():
            stat_col = next(
                (col for col in df.columns if "ewma_statistics" in col), None
            )
        else:  # Martingale or other
            stat_col = next(
                (
                    col
                    for col in df.columns
                    if "traditional_martingales" in col
                    or "traditional_sum_martingales" in col
                ),
                None,
            )

        if stat_col:
            # Normalize the statistic to 0-1 range for fair comparison
            max_val = df[stat_col].max()
            if max_val > 0:
                normalized_stat = df[stat_col] / max_val
                ax.plot(
                    df["timestep"],
                    normalized_stat,
                    color=color,
                    linewidth=2,
                    label=f"{method} Statistic",
                    alpha=0.8,
                )

            # Mark detection points
            for dp in detection_points:
                try:
                    dp_idx = df[df["timestep"] == dp].index[0]
                    dp_val = normalized_stat.iloc[dp_idx]
                    ax.scatter(
                        [dp], [dp_val], color=color, s=100, zorder=10, marker="o"
                    )

                    # Calculate delay if we can match to a change point
                    if overall_change_points:
                        # Find closest change point
                        closest_cp = min(
                            overall_change_points, key=lambda cp: abs(cp - dp)
                        )
                        if abs(closest_cp - dp) <= 15:  # Only show if reasonably close
                            delay = dp - closest_cp
                            ax.annotate(
                                f"{method}: {delay:.1f}",
                                xy=(dp, dp_val),
                                xytext=(dp + 2, dp_val),
                                color=color,
                                fontweight="bold",
                                fontsize=11,
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    fc="white",
                                    alpha=0.8,
                                    ec=color,
                                ),
                            )
                except:
                    continue

    # Improve plot appearance
    ax.set_xlim(x_limits)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Time", fontweight="bold", fontsize=14)
    ax.set_ylabel("Normalized Detection Statistic", fontweight="bold", fontsize=14)
    ax.set_title("Comparison of Detection Methods", fontsize=16, fontweight="bold")

    # Add legend with improved appearance
    ax.legend(
        loc="upper right", framealpha=0.95, frameon=True, fancybox=True, fontsize=12
    )

    # Add grid for readability
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "method_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"Saved method comparison plot to {os.path.join(output_dir, 'method_comparison.png')}"
    )


def plot_detection_results(file_path, output_dir=None, method=None):
    """Main function to plot detection results from an Excel file.

    Args:
        file_path: Path to the Excel file
        output_dir: Directory to save output plots (defaults to same directory as file)
        method: Detection method ('cusum', 'ewma', or None to auto-detect)
    """
    setup_plot_style()

    # Default output directory to the same directory as the file if not specified
    if output_dir is None:
        output_dir = os.path.dirname(file_path)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df, change_points = load_data(file_path)
    if df is None:
        return

    # Auto-detect method if not specified
    if method is None:
        # Check for specific columns to determine the method
        if any("cusum" in col.lower() for col in df.columns):
            method = "cusum"
        elif any("ewma" in col.lower() for col in df.columns):
            method = "ewma"
        else:
            print("Could not auto-detect method, defaulting to CUSUM")
            method = "cusum"

    # Load trial data
    trial_dfs, trial_change_points, metadata_df = load_trial_data(file_path)

    # If we found change points in trials but not in the aggregate sheet
    if not change_points and trial_change_points:
        change_points = trial_change_points

    # Create appropriate plots based on method
    if method.lower() == "cusum":
        plot_cusum(
            df=df,
            output_path=os.path.join(output_dir, "cusum_results.png"),
            change_points=change_points,
            trial_dfs=trial_dfs,
            metadata_df=metadata_df,
        )
    elif method.lower() == "ewma":
        plot_ewma(
            df=df,
            output_path=os.path.join(output_dir, "ewma_results.png"),
            change_points=change_points,
            trial_dfs=trial_dfs,
            metadata_df=metadata_df,
        )
    else:
        print(f"Unsupported method: {method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot CUSUM and EWMA detection results"
    )
    parser.add_argument(
        "--file_path", "-f", type=str, required=True, help="Path to Excel file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None, help="Output directory"
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["cusum", "ewma"],
        default=None,
        help="Detection method (auto-detected if not specified)",
    )
    parser.add_argument(
        "--compare",
        "-c",
        type=str,
        nargs="+",
        default=None,
        help="List of file paths to compare multiple methods",
    )
    parser.add_argument(
        "--method_names",
        "-n",
        type=str,
        nargs="+",
        default=None,
        help="Names for each method in comparison (must match number of files)",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare multiple methods
        file_paths = [args.file_path] + args.compare
        method_names = args.method_names
        if method_names and len(method_names) != len(file_paths):
            print("Warning: Number of method names does not match number of files")
            method_names = None

        plot_comparison(
            file_paths=file_paths,
            output_dir=args.output_dir or os.path.dirname(args.file_path),
            method_names=method_names,
        )
    else:
        # Plot a single method
        plot_detection_results(
            file_path=args.file_path, output_dir=args.output_dir, method=args.method
        )
