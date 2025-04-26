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
    # Check if necessary columns exist
    if not (
        "traditional_sum_martingales_mean" in df.columns
        and "horizon_sum_martingales_mean" in df.columns
    ):
        print("Missing required columns for comparison plot")
        return

    # Create figure for combined plot - more compact
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up subtle grid lines
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)

    # Get x values (timestep)
    x = df["timestep"].values

    # Determine data range for x-axis limits
    x_min = min(x)
    x_max = max(x)

    # If using box plots and trial data is provided
    if use_boxplot and trial_dfs is not None and len(trial_dfs) > 0:
        # Prepare data for box plots
        # We need to collect martingale values at each timestep across all trials
        traditional_data = []
        horizon_data = []
        timesteps = []

        # Select a subset of timesteps to make the plot readable
        # Either select every nth timestep or focus on areas around change points
        if true_change_points:
            # Focus on regions around change points
            sample_timesteps = []
            for cp in true_change_points:
                # Add timesteps in a window around each change point
                window = 10  # Adjust as needed
                sample_timesteps.extend(
                    range(max(0, cp - window), min(int(x_max), cp + window + 1))
                )
            # Add some regular samples in between for context
            step = max(1, int(len(x) / 20))  # At most 20 points outside CP regions
            for t in range(0, len(x), step):
                if t not in sample_timesteps:
                    sample_timesteps.append(t)
            sample_timesteps.sort()
        else:
            # Regular sampling throughout the time series
            step = max(1, int(len(x) / 40))  # At most 40 box plots total
            sample_timesteps = list(range(0, len(x), step))

        # Collect data for each selected timestep
        for t_idx in sample_timesteps:
            if t_idx < len(x):
                t = x[t_idx]
                timesteps.append(t)

                # Collect values across trials for this timestep
                trad_values = []
                hor_values = []

                for trial_df in trial_dfs:
                    # Find the closest timestep in this trial's data
                    if "timestep" in trial_df.columns:
                        closest_idx = np.argmin(np.abs(trial_df["timestep"].values - t))
                        if "traditional_sum_martingales_mean" in trial_df.columns:
                            trad_values.append(
                                trial_df["traditional_sum_martingales_mean"].iloc[
                                    closest_idx
                                ]
                            )
                        if "horizon_sum_martingales_mean" in trial_df.columns:
                            hor_values.append(
                                trial_df["horizon_sum_martingales_mean"].iloc[
                                    closest_idx
                                ]
                            )

                traditional_data.append(trad_values)
                horizon_data.append(hor_values)

        # Create box plots
        # Offset the boxes slightly to avoid overlap
        trad_positions = np.array(timesteps) - 0.3
        hor_positions = np.array(timesteps) + 0.3

        # Traditional sum martingales box plot
        trad_boxes = ax.boxplot(
            traditional_data,
            positions=trad_positions,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            whiskerprops=dict(color="blue"),
            medianprops=dict(color="blue", linewidth=1.5),
            showfliers=False,
            zorder=5,
        )

        # Horizon sum martingales box plot
        hor_boxes = ax.boxplot(
            horizon_data,
            positions=hor_positions,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="bisque", color="orange"),
            whiskerprops=dict(color="orange"),
            medianprops=dict(color="orange", linewidth=1.5),
            showfliers=False,
            zorder=3,
        )

        # Add custom legend for box plots
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightblue", edgecolor="blue", label="Trad. Sum"),
            Patch(facecolor="bisque", edgecolor="orange", label="Horizon Sum"),
        ]

        # Connect median points with lines for better visualization
        trad_medians = [box.get_ydata()[0] for box in trad_boxes["medians"]]
        hor_medians = [box.get_ydata()[0] for box in hor_boxes["medians"]]

        ax.plot(trad_positions, trad_medians, "b-", alpha=0.7, linewidth=1.0, zorder=6)
        ax.plot(
            hor_positions, hor_medians, "orange", alpha=0.7, linewidth=1.0, zorder=4
        )

    else:
        # Traditional line plot approach
        # Plot traditional sum martingales
        ax.plot(
            x,
            df["traditional_sum_martingales_mean"].values,
            "b-",
            linewidth=2.0,
            label="Trad. Sum",
            zorder=5,
        )

        # Add confidence bands for traditional sum
        if all(
            col in df.columns
            for col in [
                "traditional_sum_martingales_upper",
                "traditional_sum_martingales_lower",
            ]
        ):
            ax.fill_between(
                x,
                df["traditional_sum_martingales_lower"].values,
                df["traditional_sum_martingales_upper"].values,
                color="b",
                alpha=0.15,
                zorder=4,
            )
        elif "traditional_sum_martingales_std" in df.columns:
            ax.fill_between(
                x,
                df["traditional_sum_martingales_mean"].values
                - df["traditional_sum_martingales_std"].values,
                df["traditional_sum_martingales_mean"].values
                + df["traditional_sum_martingales_std"].values,
                color="b",
                alpha=0.15,
                zorder=4,
            )

        # Plot horizon sum martingales
        ax.plot(
            x,
            df["horizon_sum_martingales_mean"].values,
            "orange",
            linewidth=2.0,
            label="Horizon Sum",
            zorder=3,
        )

        # Add confidence bands for horizon sum
        if all(
            col in df.columns
            for col in [
                "horizon_sum_martingales_upper",
                "horizon_sum_martingales_lower",
            ]
        ):
            ax.fill_between(
                x,
                df["horizon_sum_martingales_lower"].values,
                df["horizon_sum_martingales_upper"].values,
                color="orange",
                alpha=0.15,
                zorder=2,
            )
        elif "horizon_sum_martingales_std" in df.columns:
            ax.fill_between(
                x,
                df["horizon_sum_martingales_mean"].values
                - df["horizon_sum_martingales_std"].values,
                df["horizon_sum_martingales_mean"].values
                + df["horizon_sum_martingales_std"].values,
                color="orange",
                alpha=0.15,
                zorder=2,
            )

        # Plot traditional avg martingales
        if "traditional_avg_martingales_mean" in df.columns:
            ax.plot(
                x,
                df["traditional_avg_martingales_mean"].values,
                "g--",
                linewidth=1.5,
                label="Trad. Avg",
                alpha=0.7,
                zorder=5,
            )

        # Plot horizon avg martingales
        if "horizon_avg_martingales_mean" in df.columns:
            ax.plot(
                x,
                df["horizon_avg_martingales_mean"].values,
                "purple",
                linestyle="--",
                linewidth=1.5,
                label="Horizon Avg",
                alpha=0.7,
                zorder=3,
            )

        # Set up legend elements for line plots
        legend_elements = None  # Use default legend

    # Add threshold line - keep as red
    ax.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label="Threshold",
        alpha=0.7,
        linewidth=1.5,
        zorder=1,
    )

    # Setup x-ticks with intervals of 50
    x_ticks = list(range(0, int(x_max) + 51, 50))

    # Mark true change points and add annotations for delays - use light gray for consistency
    cp_added_to_legend = False

    if true_change_points is not None and len(true_change_points) > 0:
        for i, cp in enumerate(true_change_points):
            # Add vertical line for change point
            if not cp_added_to_legend:
                ax.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                    label="CP",
                )
                cp_added_to_legend = True
            else:
                ax.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                )

            # Add CP value to x-ticks if not already close to an existing tick
            # Only add if it's not within 10 units of an existing tick
            if not any(abs(cp - tick) < 10 for tick in x_ticks):
                x_ticks.append(cp)

            # Get peak values near change point
            if (
                metadata_df is not None
                and "traditional_avg_delay" in metadata_df.columns
            ):
                try:
                    trad_delay = metadata_df.iloc[i]["traditional_avg_delay"]
                    hor_delay = metadata_df.iloc[i]["horizon_avg_delay"]

                    # Find peak values for annotation
                    detection_point_trad = cp + int(trad_delay)
                    detection_point_hor = cp + int(hor_delay)

                    # Safely find detection points within dataframe bounds
                    if 0 <= detection_point_trad < len(df):
                        # Find peak in the vicinity of detection point for better annotation
                        window = 5  # Look 5 steps before and after detection point
                        start_idx = max(0, detection_point_trad - window)
                        end_idx = min(len(df) - 1, detection_point_trad + window)

                        # Get the slice of data around detection point
                        slice_df = df.iloc[start_idx : end_idx + 1]
                        trad_values = slice_df["traditional_sum_martingales_mean"]

                        # Find the peak value and its index
                        peak_value = trad_values.max()
                        peak_idx = slice_df["traditional_sum_martingales_mean"].idxmax()

                        # Format delay to 2 decimal places
                        trad_delay_str = f"{trad_delay:.2f}"

                        # Add detection delay annotation with horizontal arrow
                        # Position text to the right of the peak
                        ax.annotate(
                            f"delay:{trad_delay_str}",
                            xy=(df.loc[peak_idx, "timestep"], peak_value),
                            xytext=(
                                df.loc[peak_idx, "timestep"] + 15,
                                peak_value - 10,
                            ),  # Offset horizontally
                            textcoords="data",
                            arrowprops=dict(
                                arrowstyle="->",
                                color="blue",
                                connectionstyle="arc3,rad=0.2",  # Curved arrow
                                shrinkA=5,
                                shrinkB=5,
                            ),
                            color="blue",
                            fontsize=12,
                            fontweight="bold",
                            ha="left",
                            va="center",
                            zorder=10,
                        )

                    if 0 <= detection_point_hor < len(df):
                        # Find peak for horizon in the vicinity of detection point
                        window = 5
                        start_idx = max(0, detection_point_hor - window)
                        end_idx = min(len(df) - 1, detection_point_hor + window)

                        # Get the slice of data around detection point
                        slice_df = df.iloc[start_idx : end_idx + 1]
                        hor_values = slice_df["horizon_sum_martingales_mean"]

                        # Find the peak value and its index
                        peak_value = hor_values.max()
                        peak_idx = slice_df["horizon_sum_martingales_mean"].idxmax()

                        # Format delay to 2 decimal places
                        hor_delay_str = f"{hor_delay:.2f}"

                        # Add detection delay annotation with horizontal arrow
                        # Position text to the left of the peak to avoid overlap
                        ax.annotate(
                            f"delay:{hor_delay_str}",
                            xy=(df.loc[peak_idx, "timestep"], peak_value),
                            xytext=(
                                df.loc[peak_idx, "timestep"] - 15,
                                peak_value + 10,
                            ),  # Offset horizontally
                            textcoords="data",
                            arrowprops=dict(
                                arrowstyle="->",
                                color="orange",
                                connectionstyle="arc3,rad=-0.2",  # Curved arrow in opposite direction
                                shrinkA=5,
                                shrinkB=5,
                            ),
                            color="orange",
                            fontsize=12,
                            fontweight="bold",
                            ha="right",
                            va="center",
                            zorder=10,
                        )
                except Exception as e:
                    print(f"Error adding delay annotations: {str(e)}")

    # Sort the x_ticks to ensure they're in ascending order
    x_ticks.sort()

    # Set x-axis ticks
    ax.set_xticks(x_ticks)

    # Set x-axis limits to actual data range with small padding
    padding = (x_max - x_min) * 0.05  # 5% padding on each side
    ax.set_xlim(x_min - padding, x_max + padding)

    # Set axis labels and title
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Martingale Value", fontsize=14)

    # Add legend with custom elements if specified
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            ncol=2,
            fontsize=12,
            framealpha=0.8,
        )
    else:
        ax.legend(
            loc="upper right",
            ncol=2,
            fontsize=12,
            framealpha=0.8,
        )

    # Make the plot more compact
    plt.tight_layout()

    # Save the figure
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
    agg_df = trial_dfs[0].copy()

    # Compute means across trials for each column
    for col in agg_df.columns:
        if col != "timestep" and col != "trial_name":
            # Initialize with zeros
            agg_df[col] = 0

            # Sum values from all trials
            valid_trials = 0
            for df in trial_dfs:
                if col in df.columns:
                    agg_df[col] += df[col]
                    valid_trials += 1

            # Compute average
            if valid_trials > 0:
                agg_df[col] = agg_df[col] / valid_trials

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
