# src/utils/plot_martingale.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import numpy as np

# Add the path to access changepoint module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from changepoint.threshold import CustomThresholdModel


def setup_research_plot_style():
    plt.style.use("seaborn-v0_8-paper")
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "Palatino"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "figure.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (8, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "#CCCCCC",
        }
    )


# Features from algorithm.yaml (with display-friendly names)
FEATURE_INFO = {
    "mean_degree": {"name": "Degree", "feature_id": "0"},
    "density": {"name": "Density", "feature_id": "1"},
    "mean_clustering": {"name": "Clustering", "feature_id": "2"},
    "mean_betweenness": {"name": "Betweenness", "feature_id": "3"},
    "mean_eigenvector": {"name": "Eigenvector", "feature_id": "4"},
    "mean_closeness": {"name": "Closeness", "feature_id": "5"},
    "max_singular_value": {"name": "Spectral", "feature_id": "6"},
    "min_nonzero_laplacian": {"name": "Laplacian", "feature_id": "7"},
}

FEATURE_ID_TO_NAME = {v["feature_id"]: v["name"] for k, v in FEATURE_INFO.items()}


def plot_martingales_from_csv(
    csv_path,
    sheet_name="Aggregate",
    output_dir="results",
    threshold=60.0,
    plot_shap=False,
):
    """
    Plot martingale data from a CSV file.

    Args:
        csv_path: Path to the CSV file containing martingale data
        sheet_name: Name of the sheet in the CSV file
        output_dir: Directory to save output plots
        threshold: Detection threshold value
        plot_shap: Whether to create SHAP analysis plots
    """
    setup_research_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df, true_change_points, metadata_df = load_data(csv_path, sheet_name)
    if df is None:
        return

    print(f"Using change points: {true_change_points}")

    # Create feature plots in a 2x2 grid for selected features
    _plot_feature_grid(
        df=df,
        output_path=os.path.join(output_dir, "feature_comparison.png"),
        true_change_points=true_change_points,
        threshold=threshold,
    )

    # Create comparison plot between traditional and horizon martingales
    _plot_comparison(
        df=df,
        output_path=os.path.join(output_dir, "martingale_comparison.png"),
        true_change_points=true_change_points,
        threshold=threshold,
        metadata_df=metadata_df if "metadata_df" in locals() else None,
    )

    # Add SHAP analysis plots if requested
    if plot_shap:
        print("Generating SHAP analysis plots...")
        run_shap_analysis(
            df=df,
            output_dir=output_dir,
            true_change_points=true_change_points,
            threshold=threshold,
        )


def load_data(csv_path, sheet_name="Aggregate"):
    """
    Load martingale data from a CSV or Excel file.

    Args:
        csv_path: Path to the CSV or Excel file
        sheet_name: Sheet name for Excel files

    Returns:
        Tuple of (DataFrame, change_points, metadata_df)
        If loading fails, returns (None, [], None)
    """
    try:
        # Check file extension
        file_ext = os.path.splitext(csv_path)[1].lower()
        if file_ext == ".xlsx" or file_ext == ".xls":
            # Read Excel file
            df = pd.read_excel(csv_path, sheet_name=sheet_name)
            print(f"Successfully read Excel file: {csv_path}, sheet: {sheet_name}")

            # Try to read ChangePointMetadata sheet if it exists
            try:
                metadata_df = pd.read_excel(csv_path, sheet_name="ChangePointMetadata")
                true_change_points = metadata_df["change_point"].values.tolist()
                print(f"Successfully read change points from ChangePointMetadata sheet")

                # If the metadata has delay information, print it
                if (
                    "traditional_avg_delay" in metadata_df.columns
                    and "horizon_avg_delay" in metadata_df.columns
                ):
                    for i, cp in enumerate(true_change_points):
                        trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                        hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]
                        reduction = metadata_df.iloc[i].get("delay_reduction", 0)
                        print(
                            f"Change point {cp}: Trad delay={trad_delay}, Horizon delay={hor_delay}, Reduction={reduction:.2%}"
                        )
            except Exception as e:
                print(f"Warning: Could not read ChangePointMetadata: {str(e)}")
                # Fall back to extracting change points from main sheet
                true_change_points = []
                metadata_df = None
                if "true_change_point" in df.columns:
                    true_change_points = df.loc[
                        ~df["true_change_point"].isna(), "timestep"
                    ].values.astype(int)
        else:
            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"Successfully read CSV file: {csv_path}")
            # For CSV files, extract from main sheet
            true_change_points = []
            metadata_df = None
            if "true_change_point" in df.columns:
                true_change_points = df.loc[
                    ~df["true_change_point"].isna(), "timestep"
                ].values.astype(int)

        return df, true_change_points, metadata_df

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, [], None


def run_shap_analysis(
    df,
    output_dir="results/shap_analysis",
    true_change_points=None,
    threshold=60.0,
):
    """
    Run SHAP analysis on martingale data using the CustomThresholdModel.

    This function delegates the actual SHAP analysis to the threshold.py functionality.
    It's a simple wrapper that prepares the data and calls the appropriate methods.

    Args:
        df: DataFrame containing martingale data
        output_dir: Directory to save output plots
        true_change_points: List of true change points to mark on plots
        threshold: Detection threshold value
    """
    from changepoint.threshold import CustomThresholdModel

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract feature columns and prepare data
    feature_cols, feature_names, sum_col, df_clean, timesteps = prepare_shap_data(df)

    if feature_cols is None:
        return  # Data preparation failed

    # Convert true_change_points to list if it's a numpy array
    if isinstance(true_change_points, np.ndarray):
        true_change_points = true_change_points.tolist()

    # Create threshold model
    model = CustomThresholdModel(threshold=threshold)

    # Run SHAP analysis using the threshold model
    try:
        print("Running SHAP analysis...")

        # Use visualize_shap_over_time for the SHAP plot
        contributions_df = model.visualize_shap_over_time(
            df=df_clean,
            feature_cols=feature_cols,
            sum_col=sum_col,
            change_points=true_change_points,
            timesteps=timesteps,
            output_path=os.path.join(output_dir, "shap_over_time.png"),
            threshold=threshold,
            feature_names=feature_names,
        )

        # Save contributions to CSV
        if not contributions_df.empty:
            contributions_df.to_csv(
                os.path.join(output_dir, "feature_contributions.csv"), index=False
            )
            print(
                f"Saved feature contributions to {os.path.join(output_dir, 'feature_contributions.csv')}"
            )

        print(f"SHAP analysis complete. Results saved to {output_dir}")

        # Find and print feature contributions (this should ideally be in threshold.py)
        print_feature_contributions(
            model=model,
            df=df_clean,
            feature_cols=feature_cols,
            feature_names=feature_names,
            sum_col=sum_col,
            threshold=threshold,
            timesteps=timesteps,
            true_change_points=true_change_points,
        )

    except Exception as e:
        print(f"Error during SHAP analysis: {str(e)}")
        print("Simplified SHAP analysis may be incomplete.")


def prepare_shap_data(df):
    """
    Prepare data for SHAP analysis by extracting columns and handling NaN values.

    Args:
        df: DataFrame containing martingale data

    Returns:
        Tuple of (feature_cols, feature_names, sum_col, cleaned_df, timesteps)
        If preparation fails, returns (None, None, None, None, None)
    """
    # Find feature columns for individual martingales
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("individual_traditional_martingales_feature")
        and col.endswith("_mean")
    ]

    # Sort feature columns by feature number
    feature_cols.sort(key=lambda x: int(x.split("feature")[1].split("_")[0]))

    if not feature_cols:
        print("No individual feature martingale columns found in data")
        return None, None, None, None, None

    # Get feature names based on feature IDs
    feature_names = []
    for col in feature_cols:
        feature_id = col.split("feature")[1].split("_")[0]
        feature_names.append(
            FEATURE_ID_TO_NAME.get(feature_id, f"Feature {feature_id}")
        )

    # Sum column for traditional martingales
    sum_col = "traditional_sum_martingales_mean"
    if sum_col not in df.columns:
        print(f"Sum martingale column '{sum_col}' not found in data")
        return None, None, None, None, None

    # Create timesteps array
    timesteps = (
        df["timestep"].values if "timestep" in df.columns else np.arange(len(df))
    )

    # Check for NaN values and handle them
    print("Checking for NaN values in data...")
    has_nans = False
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            has_nans = True
            print(f"Found {nan_count} NaN values in {col}")

    if has_nans:
        print("Filling NaN values with zeros to avoid LinearRegression errors")
        # Create a copy of the DataFrame with NaN values filled
        df_clean = df.copy()
        for col in feature_cols:
            df_clean[col] = df_clean[col].fillna(0)

        # Also check and fill NaNs in the sum column
        if df_clean[sum_col].isna().sum() > 0:
            print(f"Filling NaN values in {sum_col}")
            df_clean[sum_col] = df_clean[sum_col].fillna(0)
    else:
        df_clean = df

    return feature_cols, feature_names, sum_col, df_clean, timesteps


def print_feature_contributions(
    model,
    df,
    feature_cols,
    feature_names,
    sum_col,
    threshold,
    timesteps,
    true_change_points,
):
    """
    Print feature contributions for detection points.

    Note: This function should ideally be part of CustomThresholdModel in threshold.py.

    Args:
        model: CustomThresholdModel instance
        df: DataFrame containing martingale data
        feature_cols: List of feature column names
        feature_names: List of feature display names
        sum_col: Column name for sum martingale
        threshold: Detection threshold value
        timesteps: Array of timestep values
        true_change_points: List of true change points
    """
    # Find detection points for threshold crossing
    detection_indices = []
    for i in range(1, len(df)):
        if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
            detection_indices.append(i)

    # If no threshold crossings, find peaks near change points
    if len(detection_indices) == 0 and true_change_points:
        for cp in true_change_points:
            # Find closest index to change point
            cp_idx = np.argmin(np.abs(timesteps - cp))
            window = 10
            window_start = max(0, cp_idx)
            window_end = min(len(df), cp_idx + window)

            # Find peak in window
            window_values = df[sum_col].iloc[window_start:window_end]
            if not window_values.empty:
                max_idx = window_values.idxmax()
                detection_indices.append(max_idx)

    # Print feature contributions table for each detection point
    if detection_indices:
        for idx, detection_idx in enumerate(detection_indices):
            if 0 <= detection_idx < len(df):
                detection_time = timesteps[detection_idx]
                print(
                    f"\nFeature contributions at detection point {idx+1} (timestep {detection_time}):"
                )

                # Calculate contributions
                feature_values = df[feature_cols].iloc[detection_idx].values
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


def _plot_feature_grid(df, output_path, true_change_points=None, threshold=20.0):
    """
    Create a grid of all feature martingales comparing traditional vs horizon.

    Args:
        df: DataFrame containing the data
        output_path: Path to save the plot
        true_change_points: List of true change points to mark on the plot
        threshold: Detection threshold
    """
    # Use all features
    feature_ids = list(FEATURE_ID_TO_NAME.keys())

    # Calculate grid dimensions
    n_features = len(feature_ids)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with appropriate grid - reduced height per grid cell
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
    axes = axes.flatten()

    # Get x values (timestep)
    x = df["timestep"].values

    # Find global maximum y-value across all features to set consistent y-axis limits
    global_max = 0
    for feature_id in feature_ids:
        trad_mean_col = f"individual_traditional_martingales_feature{feature_id}_mean"
        hor_mean_col = f"individual_horizon_martingales_feature{feature_id}_mean"

        if trad_mean_col in df.columns:
            feature_max = df[trad_mean_col].max()
            if feature_max > global_max:
                global_max = feature_max

        if hor_mean_col in df.columns:
            feature_max = df[hor_mean_col].max()
            if feature_max > global_max:
                global_max = feature_max

    # Round up to nearest 40
    y_max = ((global_max // 40) + 1) * 40

    # Plot each feature
    for i, feature_id in enumerate(feature_ids):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            ax = axes[i]
            feature_name = FEATURE_ID_TO_NAME.get(feature_id, f"Feature {feature_id}")

            # Set up subtle grid lines
            ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)

            # Column names
            trad_mean_col = (
                f"individual_traditional_martingales_feature{feature_id}_mean"
            )
            trad_std_col = f"individual_traditional_martingales_feature{feature_id}_std"
            trad_upper_col = (
                f"individual_traditional_martingales_feature{feature_id}_upper"
            )
            trad_lower_col = (
                f"individual_traditional_martingales_feature{feature_id}_lower"
            )

            hor_mean_col = f"individual_horizon_martingales_feature{feature_id}_mean"
            hor_std_col = f"individual_horizon_martingales_feature{feature_id}_std"
            hor_upper_col = f"individual_horizon_martingales_feature{feature_id}_upper"
            hor_lower_col = f"individual_horizon_martingales_feature{feature_id}_lower"

            # Plot traditional martingale
            if trad_mean_col in df.columns:
                ax.plot(x, df[trad_mean_col].values, "b-", linewidth=1.5, label="Act.")

                # Plot confidence bands
                if trad_upper_col in df.columns and trad_lower_col in df.columns:
                    ax.fill_between(
                        x,
                        df[trad_lower_col].values,
                        df[trad_upper_col].values,
                        color="b",
                        alpha=0.2,
                    )
                elif trad_std_col in df.columns:
                    ax.fill_between(
                        x,
                        df[trad_mean_col].values - df[trad_std_col].values,
                        df[trad_mean_col].values + df[trad_std_col].values,
                        color="b",
                        alpha=0.2,
                    )

            # Plot horizon martingale
            if hor_mean_col in df.columns:
                ax.plot(
                    x, df[hor_mean_col].values, "orange", linewidth=1.5, label="Pred."
                )

                # Plot confidence bands
                if hor_upper_col in df.columns and hor_lower_col in df.columns:
                    ax.fill_between(
                        x,
                        df[hor_lower_col].values,
                        df[hor_upper_col].values,
                        color="orange",
                        alpha=0.2,
                    )
                elif hor_std_col in df.columns:
                    ax.fill_between(
                        x,
                        df[hor_mean_col].values - df[hor_std_col].values,
                        df[hor_mean_col].values + df[hor_std_col].values,
                        color="orange",
                        alpha=0.2,
                    )

            # Mark true change points - use light gray color
            cp_added_to_legend = False
            if true_change_points is not None and len(true_change_points) > 0:
                for cp in true_change_points:
                    if not cp_added_to_legend:
                        ax.axvline(
                            x=cp,
                            color="gray",
                            linestyle="--",
                            alpha=0.8,
                            linewidth=1.2,
                            label="CP",
                        )
                        cp_added_to_legend = True
                    else:
                        ax.axvline(
                            x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.2
                        )

            # Add title and grid
            ax.set_title(feature_name, fontsize=10)

            # Set y-axis label on leftmost plots only
            if i % n_cols == 0:
                ax.set_ylabel("Mart. Value", fontsize=9)

            # Set x-axis label on bottom row plots only
            if i >= n_features - n_cols or i >= len(axes) - n_cols:
                ax.set_xlabel("Time", fontsize=9)

            # Add legend on first plot only
            if i == 0:
                ax.legend(loc="upper right", fontsize=7)

            # Set consistent y-axis limits with ticks every 40 units
            ax.set_ylim(-5, y_max)
            ax.set_yticks(range(0, int(y_max) + 1, 40))
            ax.tick_params(
                axis="both", which="major", labelsize=8
            )  # Smaller tick labels

            # Use thin solid lines for y-axis grid
            ax.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    # Make the plot more compact with tighter spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.3)  # Further reduced vertical spacing

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature grid plot to {output_path}")


def _plot_comparison(
    df,
    output_path,
    true_change_points=None,
    threshold=20.0,
    metadata_df=None,
    use_boxplot=False,
    trial_dfs=None,
):
    """
    Create a comparison plot between traditional and horizon martingales.

    Args:
        df: DataFrame containing the aggregated data (means across trials)
        output_path: Path to save the plot
        true_change_points: List of true change points to mark on the plot
        threshold: Detection threshold
        metadata_df: DataFrame containing change point metadata
        use_boxplot: Whether to use box plots instead of line plots (requires trial_dfs)
        trial_dfs: List of DataFrames from individual trials (required for box plots)
    """
    # Column names can be with or without _mean suffix, handle both
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
    trad_avg_col = next(
        (
            col
            for col in df.columns
            if col
            in ["traditional_avg_martingales_mean", "traditional_avg_martingales"]
        ),
        None,
    )
    hor_avg_col = next(
        (
            col
            for col in df.columns
            if col in ["horizon_avg_martingales_mean", "horizon_avg_martingales"]
        ),
        None,
    )

    # Check if necessary columns exist
    if not (trad_sum_col and hor_sum_col):
        print("Missing required columns for comparison plot")
        print(
            f"Columns needed: traditional_sum_martingales or traditional_sum_martingales_mean, "
            f"horizon_sum_martingales or horizon_sum_martingales_mean"
        )
        print(f"Columns available: {', '.join(df.columns)}")
        return

    # Get x values (timestep)
    x = df["timestep"].values

    # Determine data range for x-axis limits
    x_min = min(x)
    x_max = max(x)

    # Set up styling for the plot
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # If using box plots and trial data is provided
    if use_boxplot and trial_dfs is not None and len(trial_dfs) > 0:
        # Create figure with more appropriate dimensions for boxplots
        fig, ax = plt.subplots(figsize=(12, 7))

        # Set up improved grid
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.7)
        ax.set_axisbelow(True)  # Place grid below data elements

        # Prepare data for box plots - we need to be smarter about sampling
        traditional_data = []
        horizon_data = []
        timesteps = []

        # Identify important regions around change points and sparse samples elsewhere
        if true_change_points:
            # Collect all timesteps in windows around change points
            important_timesteps = []
            window_size = 7  # Focus on fewer points around change points for clarity

            for cp in true_change_points:
                start = max(x_min, cp - window_size)
                end = min(x_max, cp + window_size + 1)
                important_timesteps.extend(range(int(start), int(end) + 1))

            # Determine spacing for remaining samples
            # Get approximately 10 samples outside important regions
            remaining_count = min(10, len(x) - len(important_timesteps))
            if remaining_count > 0:
                step = max(1, int((x_max - x_min) / remaining_count))

                # Start from the first timestep and sample at regular intervals
                regular_samples = list(range(int(x_min), int(x_max) + 1, step))

                # Filter out points that are too close to important regions
                filtered_samples = [
                    t
                    for t in regular_samples
                    if not any(
                        abs(t - imp) < window_size / 2 for imp in important_timesteps
                    )
                ]

                # Combine important and regular samples
                sample_timesteps = sorted(set(important_timesteps + filtered_samples))
            else:
                sample_timesteps = important_timesteps
        else:
            # If no change points, sample regularly but sparingly
            step = max(1, int(len(x) / 15))  # About 15 samples total
            sample_timesteps = list(range(0, len(x), step))

        # Now collect data for these carefully selected timesteps
        for t_idx in sample_timesteps:
            if t_idx < len(x) and t_idx >= 0:
                t = x[t_idx]

                # Skip if t is out of bounds
                if t < x_min or t > x_max:
                    continue

                timesteps.append(t)

                # Collect values across trials for this timestep
                trad_values = []
                hor_values = []

                for trial_df in trial_dfs:
                    # Find the closest timestep in this trial's data
                    if "timestep" in trial_df.columns:
                        # Use a more efficient approach for finding closest index
                        closest_idx = np.argmin(np.abs(trial_df["timestep"].values - t))

                        # Check for both possible column names
                        for col_name in [
                            "traditional_sum_martingales",
                            "traditional_sum_martingales_mean",
                        ]:
                            if col_name in trial_df.columns:
                                trad_values.append(trial_df[col_name].iloc[closest_idx])
                                break

                        for col_name in [
                            "horizon_sum_martingales",
                            "horizon_sum_martingales_mean",
                        ]:
                            if col_name in trial_df.columns:
                                hor_values.append(trial_df[col_name].iloc[closest_idx])
                                break

                if trad_values and hor_values:  # Only add if we have data
                    traditional_data.append(trad_values)
                    horizon_data.append(hor_values)
                else:
                    # Remove this timestep if no data was found
                    timesteps.pop()

        # Create boxplots with improved appearance
        # Wider boxes for better visibility
        box_width = min(2.0, 30 / len(timesteps))  # Scale width based on density

        # Set positions for better spacing
        trad_positions = np.array(timesteps)
        hor_positions = np.array(timesteps)

        # Traditional boxplot
        trad_boxes = ax.boxplot(
            traditional_data,
            positions=trad_positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue", alpha=0.7),
            whiskerprops=dict(color="blue", linewidth=1.5),
            medianprops=dict(color="darkblue", linewidth=2.0),
            capprops=dict(linewidth=1.5),
            showfliers=False,
            zorder=5,
        )

        # Horizon boxplot - using different color and pattern
        hor_boxes = ax.boxplot(
            horizon_data,
            positions=hor_positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor="bisque", color="orange", alpha=0.7),
            whiskerprops=dict(color="orange", linewidth=1.5),
            medianprops=dict(color="darkorange", linewidth=2.0),
            capprops=dict(linewidth=1.5),
            showfliers=False,
            zorder=3,
        )

        # Add lines connecting median points for better trend visibility
        trad_medians = [box.get_ydata()[0] for box in trad_boxes["medians"]]
        hor_medians = [box.get_ydata()[0] for box in hor_boxes["medians"]]

        # Plot lines with improved appearance
        ax.plot(
            trad_positions,
            trad_medians,
            "b-",
            alpha=0.7,
            linewidth=1.5,
            zorder=6,
            label="Trad. Median",
        )
        ax.plot(
            hor_positions,
            hor_medians,
            "orange",
            alpha=0.7,
            linewidth=1.5,
            zorder=4,
            label="Horizon Median",
        )

        # Improved legend with custom patches
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="lightblue", edgecolor="blue", alpha=0.7, label="Trad. Sum"
            ),
            Patch(
                facecolor="bisque", edgecolor="orange", alpha=0.7, label="Horizon Sum"
            ),
        ]

    else:
        # Create figure for line plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up grid lines
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)

        # Traditional line plot approach
        # Plot traditional sum martingales
        ax.plot(
            x,
            df[trad_sum_col].values,
            "b-",
            linewidth=2.0,
            label="Trad. Sum",
            zorder=5,
        )

        # Add confidence bands for traditional sum if available
        trad_sum_upper = next(
            (
                col
                for col in df.columns
                if col
                in [
                    "traditional_sum_martingales_upper",
                    "traditional_sum_martingales_std",
                ]
            ),
            None,
        )
        trad_sum_lower = next(
            (col for col in df.columns if col in ["traditional_sum_martingales_lower"]),
            None,
        )

        if trad_sum_upper and trad_sum_lower:
            ax.fill_between(
                x,
                df[trad_sum_lower].values,
                df[trad_sum_upper].values,
                color="b",
                alpha=0.15,
                zorder=4,
            )
        elif trad_sum_upper:  # Using std as upper
            ax.fill_between(
                x,
                df[trad_sum_col].values - df[trad_sum_upper].values,
                df[trad_sum_col].values + df[trad_sum_upper].values,
                color="b",
                alpha=0.15,
                zorder=4,
            )

        # Plot horizon sum martingales
        ax.plot(
            x,
            df[hor_sum_col].values,
            "orange",
            linewidth=2.0,
            label="Horizon Sum",
            zorder=3,
        )

        # Add confidence bands for horizon sum if available
        hor_sum_upper = next(
            (
                col
                for col in df.columns
                if col
                in ["horizon_sum_martingales_upper", "horizon_sum_martingales_std"]
            ),
            None,
        )
        hor_sum_lower = next(
            (col for col in df.columns if col in ["horizon_sum_martingales_lower"]),
            None,
        )

        if hor_sum_upper and hor_sum_lower:
            ax.fill_between(
                x,
                df[hor_sum_lower].values,
                df[hor_sum_upper].values,
                color="orange",
                alpha=0.15,
                zorder=2,
            )
        elif hor_sum_upper:  # Using std as upper
            ax.fill_between(
                x,
                df[hor_sum_col].values - df[hor_sum_upper].values,
                df[hor_sum_col].values + df[hor_sum_upper].values,
                color="orange",
                alpha=0.15,
                zorder=2,
            )

        # Plot traditional avg martingales if available
        if trad_avg_col:
            ax.plot(
                x,
                df[trad_avg_col].values,
                "g--",
                linewidth=1.5,
                label="Trad. Avg",
                alpha=0.7,
                zorder=5,
            )

        # Plot horizon avg martingales if available
        if hor_avg_col:
            ax.plot(
                x,
                df[hor_avg_col].values,
                "purple",
                linestyle="--",
                linewidth=1.5,
                label="Horizon Avg",
                alpha=0.7,
                zorder=3,
            )

        # Set up legend elements for line plots
        legend_elements = None  # Use default legend

    # Add threshold line with improved appearance
    ax.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label="Threshold",
        alpha=0.8,
        linewidth=1.5,
        zorder=1,
    )

    # Create better x-ticks - use round numbers at regular intervals
    # Start with major increments (50, 100, etc.) and add change points
    major_step = 50
    major_ticks = list(range(0, int(x_max) + major_step, major_step))

    # Add change points to ticks if they're not already close to a major tick
    if true_change_points:
        for cp in true_change_points:
            if not any(abs(cp - tick) < major_step / 5 for tick in major_ticks):
                major_ticks.append(cp)

    # Sort ticks and round to nearest integer
    major_ticks = sorted([int(round(tick)) for tick in major_ticks])

    # Set x-axis ticks
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(tick) for tick in major_ticks])

    # Mark true change points with improved styling
    if true_change_points and len(true_change_points) > 0:
        for i, cp in enumerate(true_change_points):
            # Add vertical line for change point with better visibility
            if i == 0:  # Only add to legend once
                ax.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                    label="Change Point",
                )
            else:
                ax.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                )

            # Add delay annotations if metadata is available
            if (
                metadata_df is not None
                and "traditional_avg_delay" in metadata_df.columns
                and i < len(metadata_df)
            ):
                try:
                    trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                    hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]

                    # Find nearby peak values for better annotation placement
                    cp_idx = np.argmin(np.abs(x - cp))
                    window = min(10, len(df) // 10)  # Adaptive window size

                    # Traditional martingale annotation
                    cp_window_start = max(0, cp_idx)
                    cp_window_end = min(len(df) - 1, cp_idx + int(trad_delay * 2))
                    if cp_window_start < cp_window_end:
                        window_df = df.iloc[cp_window_start : cp_window_end + 1]
                        if not window_df.empty:
                            peak_idx = window_df[trad_sum_col].idxmax()
                            peak_x = df.loc[peak_idx, "timestep"]
                            peak_y = df.loc[peak_idx, trad_sum_col]

                            # Position annotation better
                            text_x = peak_x + (x_max - x_min) * 0.02
                            text_y = peak_y * 0.9

                            # Add better annotation
                            ax.annotate(
                                f"Trad. delay: {trad_delay:.2f}",
                                xy=(peak_x, peak_y),
                                xytext=(text_x, text_y),
                                color="blue",
                                fontsize=12,
                                fontweight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.3", fc="white", alpha=0.7
                                ),
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color="blue",
                                    connectionstyle="arc3,rad=0.2",
                                    shrinkA=5,
                                    shrinkB=5,
                                ),
                            )

                    # Horizon martingale annotation
                    cp_window_start = max(0, cp_idx)
                    cp_window_end = min(len(df) - 1, cp_idx + int(hor_delay * 2))
                    if cp_window_start < cp_window_end:
                        window_df = df.iloc[cp_window_start : cp_window_end + 1]
                        if not window_df.empty:
                            peak_idx = window_df[hor_sum_col].idxmax()
                            peak_x = df.loc[peak_idx, "timestep"]
                            peak_y = df.loc[peak_idx, hor_sum_col]

                            # Position annotation better
                            text_x = peak_x - (x_max - x_min) * 0.02
                            text_y = peak_y * 1.1

                            # Add better annotation
                            ax.annotate(
                                f"Horizon delay: {hor_delay:.2f}",
                                xy=(peak_x, peak_y),
                                xytext=(text_x, text_y),
                                color="orange",
                                fontsize=12,
                                fontweight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.3", fc="white", alpha=0.7
                                ),
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color="orange",
                                    connectionstyle="arc3,rad=-0.2",
                                    shrinkA=5,
                                    shrinkB=5,
                                ),
                                ha="right",
                            )

                except Exception as e:
                    print(f"Error adding delay annotations: {str(e)}")

    # Set better axis limits with appropriate padding
    padding = (x_max - x_min) * 0.05  # 5% padding
    ax.set_xlim(x_min - padding, x_max + padding)

    # Determine appropriate y-axis limits
    y_values = []
    if use_boxplot:
        # For boxplots, collect all data points
        for data in traditional_data + horizon_data:
            y_values.extend(data)
    else:
        # For line plots, use the column values
        y_values = np.concatenate([df[trad_sum_col].values, df[hor_sum_col].values])

    # Set y-axis limits with headroom
    y_max = max(max(y_values) * 1.1, threshold * 1.2)  # Ensure threshold is visible
    ax.set_ylim(0, y_max)

    # Set improved axis labels and title
    ax.set_xlabel("Timestep", fontsize=14, fontweight="bold")
    ax.set_ylabel("Martingale Value", fontsize=14, fontweight="bold")

    # Add title based on plot type
    if use_boxplot:
        ax.set_title(
            "Martingale Distribution Across Trials", fontsize=16, fontweight="bold"
        )
    else:
        ax.set_title("Martingale Values Over Time", fontsize=16, fontweight="bold")

    # Add improved legend
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            ncol=2,
            fontsize=12,
            framealpha=0.8,
            title="Martingale Types",
        )
    else:
        ax.legend(
            loc="upper right",
            ncol=2,
            fontsize=12,
            framealpha=0.8,
        )

    # Make the plot more compact and professional
    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def plot_martingale_box_comparison(
    excel_path,
    output_dir="results",
    trial_sheets=None,
    threshold=50.0,
):
    """
    Create box plot comparison from multiple trial sheets in an Excel file.

    Args:
        excel_path: Path to Excel file containing multiple trial sheets
        output_dir: Directory to save output plots
        trial_sheets: List of sheet names for trials (if None, will look for 'Trial1', 'Trial2', etc.)
        threshold: Detection threshold value
    """
    setup_research_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # Verify it's an Excel file
    file_ext = os.path.splitext(excel_path)[1].lower()
    if file_ext not in [".xlsx", ".xls"]:
        print(f"Error: {excel_path} is not an Excel file")
        return

    # Get all sheets in the Excel file
    try:
        excel = pd.ExcelFile(excel_path)
        all_sheets = excel.sheet_names
        print(f"Found sheets in Excel file: {all_sheets}")

        # If trial_sheets not specified, look for 'Trial1', 'Trial2', etc.
        if trial_sheets is None:
            trial_sheets = [sheet for sheet in all_sheets if sheet.startswith("Trial")]

        if not trial_sheets:
            print("No trial sheets found in Excel file")
            return

        print(f"Using trial sheets: {trial_sheets}")
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return

    # Load data from all trial sheets
    trial_dfs = []
    true_change_points = None
    metadata_df = None

    # Try to load metadata first if it exists
    try:
        if "ChangePointMetadata" in all_sheets:
            metadata_df = pd.read_excel(excel_path, sheet_name="ChangePointMetadata")
            true_change_points = metadata_df["change_point"].values.tolist()
            print(f"Successfully read change points from ChangePointMetadata sheet")

            # Print delay information if available
            if (
                "traditional_avg_delay" in metadata_df.columns
                and "horizon_avg_delay" in metadata_df.columns
            ):
                for i, cp in enumerate(true_change_points):
                    trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                    hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]
                    reduction = metadata_df.iloc[i].get("delay_reduction", 0)
                    print(
                        f"Change point {cp}: Trad delay={trad_delay}, Horizon delay={hor_delay}, Reduction={reduction:.2%}"
                    )
    except Exception as e:
        print(f"Warning: Could not read ChangePointMetadata: {str(e)}")

    # Load trial data from sheets
    for sheet_name in trial_sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"Successfully read trial data from sheet: {sheet_name}")

            # Add sheet name as a column for reference
            df["trial_name"] = sheet_name

            # If we don't have change points yet, try to get them from this sheet
            if true_change_points is None and "true_change_point" in df.columns:
                true_change_points = df.loc[
                    ~df["true_change_point"].isna(), "timestep"
                ].values.tolist()
                print(
                    f"Extracted change points from {sheet_name}: {true_change_points}"
                )

            trial_dfs.append(df)
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {str(e)}")

    if not trial_dfs:
        print("No valid trial data loaded")
        return

    # Create an aggregate DataFrame (average across trials)
    # This assumes all trials have the same timesteps
    agg_df = trial_dfs[0][["timestep"]].copy()

    # Map column names for consistency (with and without _mean suffix)
    column_mapping = {
        "traditional_sum_martingales": "traditional_sum_martingales_mean",
        "horizon_sum_martingales": "horizon_sum_martingales_mean",
        "traditional_avg_martingales": "traditional_avg_martingales_mean",
        "horizon_avg_martingales": "horizon_avg_martingales_mean",
    }

    # Function to get source column name
    def get_source_col(df, target_cols):
        for col in target_cols:
            if col in df.columns:
                return col
        return None

    # Compute means across trials for each column type
    for target_col, source_cols in {
        "traditional_sum_martingales_mean": [
            "traditional_sum_martingales",
            "traditional_sum_martingales_mean",
        ],
        "horizon_sum_martingales_mean": [
            "horizon_sum_martingales",
            "horizon_sum_martingales_mean",
        ],
        "traditional_avg_martingales_mean": [
            "traditional_avg_martingales",
            "traditional_avg_martingales_mean",
        ],
        "horizon_avg_martingales_mean": [
            "horizon_avg_martingales",
            "horizon_avg_martingales_mean",
        ],
    }.items():
        # Initialize with zeros
        agg_df[target_col] = 0

        # Sum values from all trials
        valid_trials = 0
        for df in trial_dfs:
            source_col = get_source_col(df, source_cols)
            if source_col:
                agg_df[target_col] += df[source_col]
                valid_trials += 1

        # Compute average
        if valid_trials > 0:
            agg_df[target_col] = agg_df[target_col] / valid_trials

    # Create box plot
    _plot_comparison(
        df=agg_df,
        output_path=os.path.join(output_dir, "martingale_comparison_boxplot.png"),
        true_change_points=true_change_points,
        threshold=threshold,
        metadata_df=metadata_df,
        use_boxplot=True,
        trial_dfs=trial_dfs,
    )

    # Also create standard line plot for comparison
    _plot_comparison(
        df=agg_df,
        output_path=os.path.join(output_dir, "martingale_comparison.png"),
        true_change_points=true_change_points,
        threshold=threshold,
        metadata_df=metadata_df,
        use_boxplot=False,
    )


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Plot martingale data from CSV file")
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to CSV or Excel file"
    )
    parser.add_argument(
        "--sheet_name", type=str, default="Aggregate", help="Sheet name for Excel files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=50.0, help="Detection threshold"
    )
    parser.add_argument(
        "--plot_shap",
        action="store_true",
        help="Generate SHAP analysis plots",
    )
    parser.add_argument(
        "--box_plot",
        action="store_true",
        help="Generate box plots from multiple trial sheets in the Excel file",
    )
    parser.add_argument(
        "--trial_sheets",
        nargs="*",
        help="Sheet names for trials (defaults to Trial1, Trial2, etc.)",
    )

    args = parser.parse_args()

    if args.box_plot:
        print(
            f"Creating box plot comparison from Excel file with multiple trial sheets"
        )
        plot_martingale_box_comparison(
            excel_path=args.csv_path,
            output_dir=args.output_dir,
            trial_sheets=args.trial_sheets,
            threshold=args.threshold,
        )
    else:
        plot_martingales_from_csv(
            csv_path=args.csv_path,
            sheet_name=args.sheet_name,
            output_dir=args.output_dir,
            threshold=args.threshold,
            plot_shap=args.plot_shap,
        )
