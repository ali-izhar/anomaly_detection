# src/utils/plot_martingale.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def setup_plot_style():
    """Set up consistent plotting style for research visualizations optimized for 2-column layout."""
    plt.style.use("seaborn-v0_8-paper")
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "font.size": 16,  # Increased from 13
            "axes.labelsize": 18,  # Increased from 14
            "axes.titlesize": 20,  # Increased from 16
            "figure.titlesize": 22,  # Increased from 18
            "xtick.labelsize": 14,  # Increased from 12
            "ytick.labelsize": 14,  # Increased from 12
            "legend.fontsize": 14,  # Increased from 12
            "figure.figsize": (
                3.5,
                2.5,
            ),  # Much smaller from (12, 7) - single column width
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,  # Slightly more visible
            "axes.axisbelow": True,
            "lines.linewidth": 2.5,  # Slightly thicker from 2
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
    """Load martingale data from an Excel file. Return
    a dataframe with the martingale data and a list of change points."""
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


def calculate_correct_detection_delays(
    trial_dfs, change_points, threshold=50.0, max_delay=50
):
    """Calculate correct detection delays from trial data, filtering out false positives.

    Args:
        trial_dfs: List of trial DataFrames
        change_points: List of change points
        threshold: Detection threshold
        max_delay: Maximum reasonable delay to consider (filters out false positives)

    Returns:
        dict: Dictionary with corrected delays for each change point
    """
    if not trial_dfs or not change_points:
        return {}

    corrected_delays = {}

    for cp in change_points:
        trad_delays = []
        hor_delays = []

        for trial_df in trial_dfs:
            if "traditional_sum_martingales" not in trial_df.columns:
                continue

            # Find first detection after change point (within reasonable range)
            post_cp_data = trial_df[trial_df["timestep"] >= cp].copy()

            # Traditional detection: first time threshold is crossed
            trad_detections = post_cp_data[
                post_cp_data["traditional_sum_martingales"] >= threshold
            ]
            if not trad_detections.empty:
                first_trad_detection = trad_detections.iloc[0]["timestep"]
                trad_delay = first_trad_detection - cp
                # Only include if delay is reasonable (filter out false positives)
                if 0 <= trad_delay <= max_delay:
                    trad_delays.append(trad_delay)

            # Horizon detection: first time threshold is crossed
            if "horizon_sum_martingales" in trial_df.columns:
                hor_detections = post_cp_data[
                    post_cp_data["horizon_sum_martingales"] >= threshold
                ]
                if not hor_detections.empty:
                    first_hor_detection = hor_detections.iloc[0]["timestep"]
                    hor_delay = first_hor_detection - cp
                    # Only include if delay is reasonable (filter out false positives)
                    if 0 <= hor_delay <= max_delay:
                        hor_delays.append(hor_delay)

        # Calculate average delays for this change point
        if trad_delays and hor_delays:
            avg_trad_delay = sum(trad_delays) / len(trad_delays)
            avg_hor_delay = sum(hor_delays) / len(hor_delays)
            reduction = (
                (avg_trad_delay - avg_hor_delay) / avg_trad_delay
                if avg_trad_delay > 0
                else 0
            )

            corrected_delays[cp] = {
                "traditional_avg_delay": avg_trad_delay,
                "horizon_avg_delay": avg_hor_delay,
                "delay_reduction": reduction,
                "trad_count": len(trad_delays),
                "hor_count": len(hor_delays),
            }

            print(
                f"Corrected CP {cp}: Trad delay={avg_trad_delay:.2f} (n={len(trad_delays)}), "
                f"Horizon delay={avg_hor_delay:.2f} (n={len(hor_delays)}), "
                f"Reduction={reduction:.2%}"
            )

    return corrected_delays


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

            if all(
                col in metadata_df.columns
                for col in ["horizon_avg_delay", "traditional_avg_delay"]
            ):
                for i, cp in enumerate(change_points):
                    trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                    horizon_delay = metadata_df.iloc[i]["horizon_avg_delay"]
                    reduction = metadata_df.iloc[i].get(
                        "delay_reduction", 1 - horizon_delay / trad_delay
                    )
                    print(
                        f"CP {cp}: Trad delay={trad_delay:.2f}, Horizon delay={horizon_delay:.2f}, Reduction={reduction:.2%}"
                    )
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


def plot_individual_martingales(
    df,
    output_path,
    change_points=None,
    trial_dfs=None,
    metadata_df=None,
    threshold=50.0,
):
    """Create a grid of plots for individual feature martingales.

    Args:
        df: DataFrame containing martingale data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        trial_dfs: List of DataFrames for individual trials (for box plots)
        metadata_df: DataFrame with change point metadata
    """
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("individual_traditional_martingales_feature")
        and col.endswith("_mean")
    ]

    if not feature_cols:
        print("No individual feature martingale columns found")
        return

    # Calculate grid dimensions
    n_features = len(feature_cols)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create figure - optimized for 2-column research layout
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7, 1.8 * n_rows), sharex=True
    )  # Much smaller from (12, 2.8 * n_rows)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    x = df["timestep"].values

    # Determine full data range for consistent x-axis limits
    x_min, x_max = min(x), max(x)

    # Use cleaner tick marks - only to 200 to reduce clutter
    x_ticks = np.array([0, 40, 80, 120, 160, 200])

    # Set proper x-axis limits to include the last tick mark at 200 with small margin
    x_limits = (x_min, 210)  # Reduced from 285

    # Find global maximum for consistent y-axis limits
    y_max = 0
    for col in feature_cols:
        col_max = df[col].max()
        if col_max > y_max:
            y_max = col_max

    # Round up to nearest threshold
    y_max = ((y_max // threshold) + 1) * threshold

    # Find max feature martingale values at change points to determine importance
    feature_importance = {}
    if change_points and trial_dfs:
        for feature_id in range(len(feature_cols)):
            trad_col = f"individual_traditional_martingales_feature{feature_id}"
            hor_col = f"individual_horizon_martingales_feature{feature_id}"
            max_val = 0

            for cp in change_points:
                # Look at window after change point
                for trial_df in trial_dfs:
                    cp_idx = np.argmin(np.abs(trial_df["timestep"].values - cp))
                    window_end = min(len(trial_df), cp_idx + 10)
                    if cp_idx < len(trial_df):
                        window = trial_df.iloc[cp_idx:window_end]
                        if trad_col in window.columns:
                            max_val = max(max_val, window[trad_col].max())
                        if hor_col in window.columns:
                            max_val = max(max_val, window[hor_col].max())

            feature_importance[feature_id] = max_val

    # If we have trial data, prepare for box plots
    has_trial_data = trial_dfs is not None and len(trial_dfs) > 0

    # Sample points for box plots (plotting at every timestep would be too crowded)
    if has_trial_data:
        # Find important points around change points and sample regularly elsewhere
        sample_points = []
        if change_points:
            window = 5  # Points before and after change points
            for cp in change_points:
                for i in range(max(0, cp - window), min(len(x), cp + window + 1)):
                    if i in x:
                        sample_points.append(i)

        # Add regular samples (about 10-15 points total)
        if len(x) > 0:
            step = max(1, len(x) // 10)
            for i in range(0, len(x), step):
                sample_points.append(x[i])

        # Remove duplicates and sort
        sample_points = sorted(set(sample_points))

    # Plot each feature
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            ax = axes[i]

            # Extract feature ID and get name
            feature_id = col.split("feature")[1].split("_")[0]
            feature_name = FEATURE_INFO.get(feature_id, f"Feature {feature_id}")

            # Determine if this is an important feature
            is_important = False
            if feature_importance and int(feature_id) in feature_importance:
                importance_val = feature_importance[int(feature_id)]
                # Consider important if max value is at least 30% of the threshold
                is_important = importance_val > 15

            # Use background shading for important features
            if is_important:
                ax.set_facecolor(
                    "#f8f8ff"
                )  # Very light blue background for important features
                title_fontweight = "bold"
            else:
                title_fontweight = "normal"

            if has_trial_data:
                # Traditional martingales
                trad_col_base = (
                    f"individual_traditional_martingales_feature{feature_id}"
                )
                hor_col_base = f"individual_horizon_martingales_feature{feature_id}"

                # Create box plot data for each sample point
                trad_data = []
                trad_positions = []
                hor_data = []
                hor_positions = []

                for time_point in sample_points:
                    # Collect values across trials for this time point
                    trad_values = []
                    hor_values = []

                    for trial_df in trial_dfs:
                        if "timestep" in trial_df.columns:
                            # Find exact or closest timestep
                            if time_point in trial_df["timestep"].values:
                                idx = trial_df.index[
                                    trial_df["timestep"] == time_point
                                ][0]
                                if trad_col_base in trial_df.columns:
                                    trad_values.append(trial_df.loc[idx, trad_col_base])
                                if hor_col_base in trial_df.columns:
                                    hor_values.append(trial_df.loc[idx, hor_col_base])

                    if trad_values:
                        trad_data.append(trad_values)
                        trad_positions.append(time_point)
                    if hor_values:
                        hor_data.append(hor_values)
                        hor_positions.append(time_point)

                # Create box plots with narrower width for better visibility
                box_width = min(3.0, threshold / len(sample_points))

                if trad_data:
                    trad_boxes = ax.boxplot(
                        trad_data,
                        positions=trad_positions,
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor="#ADD8E6", color="#0000CD", alpha=0.7
                        ),  # Improved colors
                        whiskerprops=dict(
                            color="#0000CD", linewidth=1.5
                        ),  # Increased from 1.0
                        medianprops=dict(
                            color="#00008B", linewidth=2.0
                        ),  # Increased from 1.5
                        showfliers=False,
                        zorder=3,
                    )

                if hor_data:
                    hor_boxes = ax.boxplot(
                        hor_data,
                        positions=hor_positions,
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor="#FFD580", color="#FF8C00", alpha=0.7
                        ),  # Improved colors
                        whiskerprops=dict(
                            color="#FF8C00", linewidth=1.5
                        ),  # Increased from 1.0
                        medianprops=dict(
                            color="#FF4500", linewidth=2.0
                        ),  # Increased from 1.5
                        showfliers=False,
                        zorder=2,
                    )

                # Also plot the means from the aggregate data as a line
                ax.plot(
                    x, df[col].values, "#0000CD", linewidth=1.5, alpha=0.5, zorder=1
                )

                horizon_col = col.replace("traditional", "horizon")
                if horizon_col in df.columns:
                    ax.plot(
                        x,
                        df[horizon_col].values,
                        "#FF8C00",
                        linewidth=1.5,
                        alpha=0.5,
                        zorder=1,
                    )
            else:
                # Regular line plot (original behavior)
                ax.plot(
                    x, df[col].values, "#0000CD", linewidth=2.0, label="Traditional"
                )

                horizon_col = col.replace("traditional", "horizon")
                if horizon_col in df.columns:
                    ax.plot(
                        x,
                        df[horizon_col].values,
                        "#FF8C00",
                        linewidth=2.0,
                        label="Horizon",
                    )

            # Mark change points if provided
            if change_points:
                for cp in change_points:
                    # Add subtle background shading to highlight change points
                    ax.axvspan(cp - 1, cp + 1, color="gray", alpha=0.15, zorder=0)
                    ax.axvline(
                        x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=2.0
                    )

            # Set title and labels
            ax.set_title(
                feature_name,
                fontsize=16,  # Increased from 13
                fontweight=title_fontweight,
                color="#444444" if not is_important else "#000066",
            )
            if i % n_cols == 0:
                ax.set_ylabel("Martingale Value", fontsize=15)  # Increased from 12
            if i >= n_features - n_cols:
                ax.set_xlabel("Time", fontsize=15)  # Increased from 12

            ax.set_ylim(0, y_max)
            ax.set_yticks(range(0, int(y_max) + 1, int(threshold)))
            ax.set_xticks(x_ticks)
            ax.set_xlim(x_limits)
            ax.set_xticklabels([str(int(tick)) for tick in ax.get_xticks()])

            # Add legend on first plot only
            if i == 0:
                if has_trial_data:
                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D

                    legend_elements = [
                        Patch(
                            facecolor="#ADD8E6",
                            edgecolor="#0000CD",
                            alpha=0.7,
                            label="Traditional",
                        ),
                        Patch(
                            facecolor="#FFD580",
                            edgecolor="#FF8C00",
                            alpha=0.7,
                            label="Horizon",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="red",
                            linestyle="--",
                            alpha=0.8,
                            label="Threshold",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="green",
                            linestyle="--",
                            alpha=0.7,
                            label="Prediction",
                        ),
                    ]
                    ax.legend(
                        handles=legend_elements,
                        loc="upper left",
                        fontsize=8,
                        framealpha=0.9,
                        bbox_to_anchor=(0.02, 0.98),
                        ncol=2,
                    )  # 2 columns with smaller text
                else:
                    # For non-trial data, create a simplified legend
                    from matplotlib.lines import Line2D

                    legend_elements = [
                        Line2D(
                            [0],
                            [0],
                            color="#0000CD",
                            linewidth=2.0,
                            label="Traditional",
                        ),
                        Line2D(
                            [0], [0], color="#FF8C00", linewidth=2.0, label="Horizon"
                        ),
                    ]
                    ax.legend(handles=legend_elements, loc="upper right", fontsize=13)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    for ax in axes:
        if ax.get_visible():
            ax.set_xlim(x_limits)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.95)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved individual martingales plot to {output_path}")


def plot_sum_martingales(
    df,
    output_path,
    change_points=None,
    threshold=50.0,
    trial_dfs=None,
    metadata_df=None,
    enable_annot=True,
):
    """Create a comparison plot of sum martingales.

    Args:
        df: DataFrame containing martingale data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        threshold: Detection threshold value
        trial_dfs: List of DataFrames for individual trials (for box plots)
        metadata_df: DataFrame with change point metadata for delay annotations
        enable_annot: Whether to show delay reduction annotations
    """
    # Find column names for sum martingales
    trad_sum_col = next(
        (
            col
            for col in df.columns
            if col
            in ["traditional_sum_martingales_mean", "traditional_sum_martingales"]
        ),
        None,
    )

    hor_sum_col = next(
        (
            col
            for col in df.columns
            if col in ["horizon_sum_martingales_mean", "horizon_sum_martingales"]
        ),
        None,
    )

    if not trad_sum_col:
        print("Traditional sum martingale column not found")
        return

    # Create figure - optimized for 2-column research layout
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Much smaller from (10, 6)

    x = df["timestep"].values

    # Clean up x-axis ticks - use fewer ticks to reduce clutter
    x_min, x_max = min(x), max(x)

    # Use cleaner tick marks - only to 200 to reduce clutter
    x_ticks = np.array([0, 40, 80, 120, 160, 200])

    # Set proper x-axis limits to include the last tick mark at 200 with small margin
    x_limits = (x_min, 210)  # Reduced from 285

    # Check if we have trial data for box plots
    has_trial_data = trial_dfs is not None and len(trial_dfs) > 0

    # If we have change points, add subtle background highlighting
    if change_points:
        for cp in change_points:
            # Add light shading after change point to highlight detection region
            ax.axvspan(cp, min(cp + 10, x_max), color="#f5f5f5", zorder=0)

    # Add prediction start line back for legend
    ax.axvline(
        x=10,
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        zorder=0,
    )

    if has_trial_data:
        # Reduce sample points for box plots to reduce clutter
        sample_points = []
        if change_points:
            window = 3  # Reduced from 5 - fewer points before and after change points
            for cp in change_points:
                for i in range(max(0, cp - window), min(max(x) + 1, cp + window + 1)):
                    if i in x:
                        sample_points.append(i)

        # Add fewer regular samples to reduce clutter
        if len(x) > 0:
            step = max(1, len(x) // 8)  # Reduced from 15 to 8 - fewer sample points
            regular_points = list(range(int(min(x)), int(max(x)) + 1, step))
            sample_points.extend(regular_points)

        # Remove duplicates and sort
        sample_points = sorted(set(sample_points))

        # Create box plot data for each sample point
        trad_data = []
        trad_positions = []
        hor_data = []
        hor_positions = []

        for time_point in sample_points:
            # Collect values across trials for this time point
            trad_values = []
            hor_values = []

            for trial_df in trial_dfs:
                if "timestep" in trial_df.columns:
                    # Find rows with this timestep
                    matches = trial_df[trial_df["timestep"] == time_point]
                    if not matches.empty:
                        if "traditional_sum_martingales" in trial_df.columns:
                            trad_values.append(
                                matches["traditional_sum_martingales"].values[0]
                            )
                        if "horizon_sum_martingales" in trial_df.columns:
                            hor_values.append(
                                matches["horizon_sum_martingales"].values[0]
                            )

            if trad_values:
                trad_data.append(trad_values)
                trad_positions.append(time_point)
            if hor_values:
                hor_data.append(hor_values)
                hor_positions.append(time_point)

        # Create box plots with improved styling
        box_width = min(3.0, 60 / len(sample_points))

        if trad_data:
            trad_boxes = ax.boxplot(
                trad_data,
                positions=trad_positions,
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor="#ADD8E6", color="#0000CD", alpha=0.7),
                whiskerprops=dict(color="#0000CD", linewidth=1.5),  # Increased from 1.0
                medianprops=dict(color="#00008B", linewidth=2.0),  # Increased from 1.5
                showfliers=False,
                zorder=3,
            )

        if hor_data:
            hor_boxes = ax.boxplot(
                hor_data,
                positions=hor_positions,
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor="#FFD580", color="#FF8C00", alpha=0.7),
                whiskerprops=dict(color="#FF8C00", linewidth=1.5),  # Increased from 1.0
                medianprops=dict(color="#FF4500", linewidth=2.0),  # Increased from 1.5
                showfliers=False,
                zorder=2,
            )

        # Add thin lines for mean values from aggregate data
        ax.plot(
            x,
            df[trad_sum_col].values,
            "#0000CD",
            linewidth=2.5,  # Increased from 2.0
            alpha=0.5,
            zorder=1,
        )

        if hor_sum_col:
            ax.plot(
                x,
                df[hor_sum_col].values,
                "#FF8C00",
                linewidth=2.5,  # Increased from 2.0
                alpha=0.5,
                zorder=1,
            )

        # Add comprehensive legend with 2 columns and smaller text
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(
                facecolor="#ADD8E6", edgecolor="#0000CD", alpha=0.7, label="Traditional"
            ),
            Patch(facecolor="#FFD580", edgecolor="#FF8C00", alpha=0.7, label="Horizon"),
            Line2D([0], [0], color="red", linestyle="--", alpha=0.8, label="Threshold"),
            Line2D(
                [0], [0], color="green", linestyle="--", alpha=0.7, label="Prediction"
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.02, 0.98),
            ncol=2,
        )  # 2 columns with smaller text

    else:
        # Standard line plots (original behavior)
        ax.plot(
            x,
            df[trad_sum_col].values,
            "#0000CD",
            linewidth=2.5,  # Increased from 2.0
            label="Traditional Sum",
            zorder=5,
        )

        if hor_sum_col:
            ax.plot(
                x,
                df[hor_sum_col].values,
                "#FF8C00",
                linewidth=2.5,  # Increased from 2.0
                label="Horizon Sum",
                zorder=3,
            )

        # Add comprehensive legend with 2 columns and smaller text
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(
                facecolor="#ADD8E6", edgecolor="#0000CD", alpha=0.7, label="Traditional"
            ),
            Patch(facecolor="#FFD580", edgecolor="#FF8C00", alpha=0.7, label="Horizon"),
            Line2D([0], [0], color="red", linestyle="--", alpha=0.8, label="Threshold"),
            Line2D(
                [0], [0], color="green", linestyle="--", alpha=0.7, label="Prediction"
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.02, 0.98),
            ncol=2,
        )  # 2 columns with smaller text

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label="Threshold",
        alpha=0.8,
        linewidth=2.0,  # Increased from 1.5
    )

    # Mark change points and add detection delay annotations if available
    if change_points:
        # Calculate corrected delays to filter out false positives
        corrected_delays = {}
        if trial_dfs:
            corrected_delays = calculate_correct_detection_delays(
                trial_dfs, change_points, threshold
            )

        for i, cp in enumerate(change_points):
            # Add vertical line for change point
            ax.axvline(
                x=cp,
                color="gray",
                linestyle="--",
                alpha=0.8,
                linewidth=2.0,  # Increased from 1.2
                label="Change Point" if i == 0 else "",
            )

            # Use corrected delays if available, otherwise fall back to metadata
            if cp in corrected_delays:
                delay_data = corrected_delays[cp]
                trad_delay = delay_data["traditional_avg_delay"]
                hor_delay = delay_data["horizon_avg_delay"]
                reduction = delay_data["delay_reduction"]
            elif metadata_df is not None and i < len(metadata_df):
                trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]
                reduction = metadata_df.iloc[i].get(
                    "delay_reduction", 1 - hor_delay / trad_delay
                )
            else:
                continue  # Skip if no delay data available

            # Find detection points (where lines cross threshold)
            trad_detection = cp + trad_delay
            hor_detection = cp + hor_delay

            # Add detection points and annotations only if enabled
            if enable_annot:
                # Add detection points only - remove detailed text annotations to reduce clutter
                ax.scatter(
                    [trad_detection],
                    [threshold],
                    color="#0000CD",
                    s=30,  # Much smaller from 100
                    zorder=10,
                    marker="o",
                )
                ax.scatter(
                    [hor_detection],
                    [threshold],
                    color="#FF8C00",
                    s=30,  # Much smaller from 100
                    zorder=10,
                    marker="o",
                )

                # Only add the delay reduction annotation (most important info)
                mid_point = (trad_detection + hor_detection) / 2
                ax.annotate(
                    f"{reduction:.0%} faster",  # Simplified format
                    xy=(mid_point, threshold),
                    xytext=(mid_point, threshold * 1.2),  # Reduced from 1.3
                    color="#006400",
                    fontweight="bold",
                    fontsize=9,  # Further reduced from 11
                    ha="center",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        fc="#E8F8E8",
                        alpha=0.8,  # Smaller padding
                    ),
                )

    ax.set_xticks(x_ticks)
    tick_labels = [str(int(tick)) for tick in x_ticks]
    ax.set_xticklabels(tick_labels)

    ax.set_xlim(x_limits)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel("Time", fontsize=16, fontweight="bold")  # Increased from 14
    ax.set_ylabel(
        "Martingale Value", fontsize=16, fontweight="bold"
    )  # Increased from 14

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sum martingales plot to {output_path}")


def plot_martingales(
    file_path,
    sheet_name="Aggregate",
    output_dir="results",
    threshold=50.0,
    use_boxplots=True,
    enable_annot=True,
):
    """Main function to plot martingale data from an Excel file.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        output_dir: Directory to save output plots
        threshold: Detection threshold value
        use_boxplots: Whether to use box plots for showing distributions
        enable_annot: Whether to show delay reduction annotations
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    df, change_points = load_data(file_path, sheet_name)
    if df is None:
        return

    # Load trial data if box plots are requested
    trial_dfs = None
    metadata_df = None
    if use_boxplots:
        trial_dfs, trial_change_points, metadata_df = load_trial_data(file_path)
        # If we found change points in trials but not in the aggregate sheet
        if not change_points and trial_change_points:
            change_points = trial_change_points

    # Plot individual martingales
    plot_individual_martingales(
        df=df,
        output_path=os.path.join(output_dir, "individual_martingales.png"),
        change_points=change_points,
        trial_dfs=trial_dfs if use_boxplots else None,
        metadata_df=metadata_df,
        threshold=threshold,
    )

    # Plot sum martingales
    plot_sum_martingales(
        df=df,
        output_path=os.path.join(output_dir, "sum_martingales.png"),
        change_points=change_points,
        threshold=threshold,
        trial_dfs=trial_dfs if use_boxplots else None,
        metadata_df=metadata_df,
        enable_annot=enable_annot,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot martingale data from Excel file")
    parser.add_argument(
        "--file_path", "-f", type=str, required=True, help="Path to Excel file"
    )
    parser.add_argument(
        "--sheet_name", "-s", type=str, default="Aggregate", help="Sheet name"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=50.0, help="Detection threshold"
    )
    parser.add_argument(
        "--no_boxplots", action="store_true", help="Disable box plots for distributions"
    )
    parser.add_argument(
        "--no_annotations",
        action="store_true",
        help="Disable delay reduction annotations",
    )

    args = parser.parse_args()

    plot_martingales(
        file_path=args.file_path,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        threshold=args.threshold,
        use_boxplots=not args.no_boxplots,
        enable_annot=not args.no_annotations,
    )
