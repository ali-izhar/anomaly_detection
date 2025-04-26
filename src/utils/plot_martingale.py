# src/utils/plot_martingale.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def setup_plot_style():
    """Set up consistent plotting style for research visualizations."""
    plt.style.use("seaborn-v0_8-paper")
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "figure.titlesize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),  # Slightly wider for better visualization
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.2,  # More subtle grid
            "axes.axisbelow": True,
            "lines.linewidth": 2,
            "axes.spines.top": False,  # Remove top spine for cleaner look
            "axes.spines.right": False,  # Remove right spine for cleaner look
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


def load_trial_data(file_path):
    """Load data from all trial sheets and metadata.

    Returns:
        trial_dfs: List of dataframes for each trial
        change_points: List of change points
        metadata_df: DataFrame with change point metadata
    """
    try:
        # Read the Excel file
        excel = pd.ExcelFile(file_path)

        # Identify trial sheets
        trial_sheets = [
            sheet for sheet in excel.sheet_names if sheet.startswith("Trial")
        ]
        print(f"Found {len(trial_sheets)} trial sheets: {trial_sheets}")

        # Load change point metadata if available
        try:
            metadata_df = pd.read_excel(file_path, sheet_name="ChangePointMetadata")
            change_points = metadata_df["change_point"].values.tolist()
            print(f"Found {len(change_points)} change points in metadata")

            # Print delay information if available
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
    df, output_path, change_points=None, trial_dfs=None, metadata_df=None
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

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharex=True)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    x = df["timestep"].values

    # Determine full data range for consistent x-axis limits
    x_min, x_max = min(x), max(x)

    # Use 40 instead of 20 as the tick spacing
    tick_spacing = max(40, (x_max - x_min) // 5)

    # Generate ticks from 0 to 200 by 40s
    x_ticks = np.array([0, 40, 80, 120, 160, 200])

    # Explicitly set the x-axis limits to show just a little beyond 200
    x_limits = (x_min, 205)

    # Find global maximum for consistent y-axis limits
    y_max = 0
    for col in feature_cols:
        col_max = df[col].max()
        if col_max > y_max:
            y_max = col_max

    # Round up to nearest 40
    y_max = ((y_max // 40) + 1) * 40

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
                box_width = min(3.0, 50 / len(sample_points))

                if trad_data:
                    trad_boxes = ax.boxplot(
                        trad_data,
                        positions=trad_positions,
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor="#ADD8E6", color="#0000CD", alpha=0.7
                        ),  # Improved colors
                        whiskerprops=dict(color="#0000CD", linewidth=1.0),
                        medianprops=dict(color="#00008B", linewidth=1.5),
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
                        whiskerprops=dict(color="#FF8C00", linewidth=1.0),
                        medianprops=dict(color="#FF4500", linewidth=1.5),
                        showfliers=False,
                        zorder=2,
                    )

                # Also plot the means from the aggregate data as a line
                ax.plot(
                    x, df[col].values, "#0000CD", linewidth=1.0, alpha=0.5, zorder=1
                )

                horizon_col = col.replace("traditional", "horizon")
                if horizon_col in df.columns:
                    ax.plot(
                        x,
                        df[horizon_col].values,
                        "#FF8C00",
                        linewidth=1.0,
                        alpha=0.5,
                        zorder=1,
                    )
            else:
                # Regular line plot (original behavior)
                ax.plot(
                    x, df[col].values, "#0000CD", linewidth=1.5, label="Traditional"
                )

                horizon_col = col.replace("traditional", "horizon")
                if horizon_col in df.columns:
                    ax.plot(
                        x,
                        df[horizon_col].values,
                        "#FF8C00",
                        linewidth=1.5,
                        label="Horizon",
                    )

            # Mark change points if provided
            if change_points:
                for cp in change_points:
                    # Add subtle background shading to highlight change points
                    ax.axvspan(cp - 1, cp + 1, color="gray", alpha=0.15, zorder=0)
                    ax.axvline(
                        x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.2
                    )

            # Set title and labels
            ax.set_title(
                feature_name,
                fontsize=11,
                fontweight=title_fontweight,
                color="#444444" if not is_important else "#000066",
            )
            if i % n_cols == 0:
                ax.set_ylabel("Martingale Value", fontsize=10)
            if i >= n_features - n_cols:
                ax.set_xlabel("Time", fontsize=10)

            # Set consistent y-axis limits
            ax.set_ylim(0, y_max)
            ax.set_yticks(range(0, int(y_max) + 1, 40))

            # Set consistent x-axis ticks
            ax.set_xticks(x_ticks)
            # Set x-axis limits to be exactly the same as in sum_martingales plot
            ax.set_xlim(x_limits)
            # Make sure we use the same tick labels
            ax.set_xticklabels([str(int(tick)) for tick in ax.get_xticks()])

            # Add legend on first plot only
            if i == 0:
                if has_trial_data:
                    from matplotlib.patches import Patch

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
                    ]
                    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
                else:
                    ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    # Before saving, enforce consistent x-axis limits across all subplots once more
    for ax in axes:
        if ax.get_visible():
            ax.set_xlim(x_limits)

    # Adjust spacing for better tick label visibility
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
):
    """Create a comparison plot of sum martingales.

    Args:
        df: DataFrame containing martingale data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        threshold: Detection threshold value
        trial_dfs: List of DataFrames for individual trials (for box plots)
        metadata_df: DataFrame with change point metadata for delay annotations
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

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["timestep"].values

    # Clean up x-axis ticks - use fewer ticks to reduce clutter
    x_min, x_max = min(x), max(x)

    # Use same fixed tick marks as in individual plot
    x_ticks = np.array([0, 40, 80, 120, 160, 200])

    # Set proper x-axis limits to include the last tick mark at 200 with small margin
    x_limits = (x_min, 205)

    # Check if we have trial data for box plots
    has_trial_data = trial_dfs is not None and len(trial_dfs) > 0

    # If we have change points, add subtle background highlighting
    if change_points:
        for cp in change_points:
            # Add light shading after change point to highlight detection region
            ax.axvspan(cp, min(cp + 10, x_max), color="#f5f5f5", zorder=0)

    if has_trial_data:
        # Sample points for box plots (plotting at every timestep would be too crowded)
        sample_points = []
        if change_points:
            window = 5  # Points before and after change points
            for cp in change_points:
                for i in range(max(0, cp - window), min(max(x) + 1, cp + window + 1)):
                    if i in x:
                        sample_points.append(i)

        # Add regular samples (about 10-15 points total)
        if len(x) > 0:
            step = max(1, len(x) // 15)
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
                whiskerprops=dict(color="#0000CD", linewidth=1.0),
                medianprops=dict(color="#00008B", linewidth=1.5),
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
                whiskerprops=dict(color="#FF8C00", linewidth=1.0),
                medianprops=dict(color="#FF4500", linewidth=1.5),
                showfliers=False,
                zorder=2,
            )

        # Add thin lines for mean values from aggregate data
        ax.plot(
            x,
            df[trad_sum_col].values,
            "#0000CD",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )

        if hor_sum_col:
            ax.plot(
                x,
                df[hor_sum_col].values,
                "#FF8C00",
                linewidth=1.0,
                alpha=0.5,
                zorder=1,
            )

        # Add legend with improved appearance
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="#ADD8E6", edgecolor="#0000CD", alpha=0.7, label="Traditional"
            ),
            Patch(facecolor="#FFD580", edgecolor="#FF8C00", alpha=0.7, label="Horizon"),
            Patch(facecolor="red", edgecolor="red", alpha=0.5, label="Threshold"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    else:
        # Standard line plots (original behavior)
        ax.plot(
            x,
            df[trad_sum_col].values,
            "#0000CD",
            linewidth=2.0,
            label="Traditional Sum",
            zorder=5,
        )

        if hor_sum_col:
            ax.plot(
                x,
                df[hor_sum_col].values,
                "#FF8C00",
                linewidth=2.0,
                label="Horizon Sum",
                zorder=3,
            )

        # Add legend
        ax.legend(loc="upper right", fontsize=10)

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label="Threshold",
        alpha=0.8,
        linewidth=1.5,
    )

    # Mark change points and add detection delay annotations if available
    if change_points:
        for i, cp in enumerate(change_points):
            # Add vertical line for change point
            ax.axvline(
                x=cp,
                color="gray",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
                label="Change Point" if i == 0 else "",
            )

            # Add detection delay annotations if metadata is available
            if metadata_df is not None and i < len(metadata_df):
                try:
                    trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                    hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]

                    # Find detection points (where lines cross threshold)
                    trad_detection = cp + trad_delay
                    hor_detection = cp + hor_delay

                    # Mark detection points
                    ax.scatter(
                        [trad_detection],
                        [threshold],
                        color="#0000CD",
                        s=80,
                        zorder=10,
                        marker="o",
                    )
                    ax.scatter(
                        [hor_detection],
                        [threshold],
                        color="#FF8C00",
                        s=80,
                        zorder=10,
                        marker="o",
                    )

                    # Add delay annotations
                    ax.annotate(
                        f"Trad: {trad_delay:.1f}",
                        xy=(trad_detection, threshold),
                        xytext=(trad_detection + 2, threshold * 1.1),
                        color="#00008B",
                        fontweight="bold",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    )

                    ax.annotate(
                        f"Horizon: {hor_delay:.1f}",
                        xy=(hor_detection, threshold),
                        xytext=(hor_detection - 2, threshold * 0.9),
                        color="#FF4500",
                        fontweight="bold",
                        fontsize=9,
                        ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    )

                    # Add delay reduction info between arrows
                    reduction = metadata_df.iloc[i].get(
                        "delay_reduction", 1 - hor_delay / trad_delay
                    )
                    mid_point = (trad_detection + hor_detection) / 2
                    ax.annotate(
                        f"{reduction:.1%} faster",
                        xy=(mid_point, threshold),
                        xytext=(mid_point, threshold * 1.3),
                        color="#006400",
                        fontweight="bold",
                        fontsize=9,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8E8", alpha=0.9),
                    )
                except Exception as e:
                    print(f"Error adding delay annotations: {e}")

    # Set clean x-ticks
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
    ax.set_xlim(x_limits)

    # If tick labels still overlap, rotate them
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Set axis labels
    ax.set_xlabel("Timestep", fontsize=12, fontweight="bold")
    ax.set_ylabel("Martingale Value", fontsize=12, fontweight="bold")

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
):
    """Main function to plot martingale data from an Excel file.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        output_dir: Directory to save output plots
        threshold: Detection threshold value
        use_boxplots: Whether to use box plots for showing distributions
    """
    # Setup plot style
    setup_plot_style()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
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
    )

    # Plot sum martingales
    plot_sum_martingales(
        df=df,
        output_path=os.path.join(output_dir, "sum_martingales.png"),
        change_points=change_points,
        threshold=threshold,
        trial_dfs=trial_dfs if use_boxplots else None,
        metadata_df=metadata_df,
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

    args = parser.parse_args()

    plot_martingales(
        file_path=args.file_path,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        threshold=args.threshold,
        use_boxplots=not args.no_boxplots,
    )
