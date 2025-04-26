# src/utils/plot_shap.py

"""SHAP (SHapley Additive exPlanations) analysis for anomaly detection."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
from sklearn.linear_model import LinearRegression

# Feature display names for better visualization
FEATURE_NAMES = {
    "0": "Degree",
    "1": "Density",
    "2": "Clustering",
    "3": "Betweenness",
    "4": "Eigenvector",
    "5": "Closeness",
    "6": "Spectral",
    "7": "Laplacian",
}


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
            "figure.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
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


def load_data(file_path, sheet_name="Aggregate"):
    """Load data from an Excel file. Returns a tuple of (DataFrame with data, list of change points, metadata DataFrame)."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully read sheet: {sheet_name} from {file_path}")

        # Try to read change points from metadata sheet
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

            # Try to extract change points from main sheet
            if "true_change_point" in df.columns:
                change_points = df.loc[
                    ~df["true_change_point"].isna(), "timestep"
                ].values.tolist()
                print(f"Found {len(change_points)} change points in main sheet")

        return df, change_points, metadata_df

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, [], None


def load_trial_data(file_path):
    """Load data from all trial sheets in an Excel file.

    Args:
        file_path: Path to the Excel file

    Returns:
        tuple: (list of DataFrames for trials, list of change points, metadata DataFrame)
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


def get_feature_cols(
    df, prefix="individual_traditional_martingales_feature", suffix="_mean"
):
    """Extract feature column names from DataFrame based on prefix and suffix.

    Args:
        df: DataFrame containing the data
        prefix: Prefix of feature columns
        suffix: Suffix of feature columns

    Returns:
        list: Feature column names
    """
    feature_cols = [
        col for col in df.columns if col.startswith(prefix) and col.endswith(suffix)
    ]

    # Sort by feature number
    feature_cols.sort(key=lambda x: int(x.split(prefix)[1].split("_")[0]))

    return feature_cols


def get_display_names(feature_cols):
    """Convert feature column names to human-readable display names.

    Args:
        feature_cols: List of feature column names

    Returns:
        list: Display names for features
    """
    display_names = []
    for col in feature_cols:
        # Extract feature ID
        feature_id = col.split("feature")[1].split("_")[0]
        # Get display name from mapping or use default
        display_name = FEATURE_NAMES.get(feature_id, f"Feature {feature_id}")
        display_names.append(display_name)

    return display_names


def compute_shap_values(df, feature_cols, sum_col, threshold=50.0, timesteps=None):
    """Compute SHAP values for feature importance analysis.

    Args:
        df: DataFrame containing martingale data
        feature_cols: List of column names for individual feature martingales
        sum_col: Column name for sum martingale
        timesteps: Optional array of timesteps

    Returns:
        tuple: (SHAP values, feature contributions at detection points)
    """
    if timesteps is None:
        timesteps = np.arange(len(df))

    # Create feature matrix
    X = df[feature_cols].copy()

    # Handle NaN values by filling with zeros
    X = X.fillna(0)
    y = df[sum_col].fillna(0)

    # Check for any remaining NaN values
    if X.isna().any().any() or y.isna().any():
        print("Warning: After filling, some NaN values remain. These will be dropped.")
        # Drop rows with any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        print(f"Dropped {len(df) - len(X)} rows with NaN values")

    print(f"Shape of feature matrix after handling NaNs: {X.shape}")

    # Fit linear model to approximate the sum martingale
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Verify model accuracy
    predictions = model.predict(X)
    r2 = np.corrcoef(predictions, y)[0, 1] ** 2
    print(f"Linear model R² score: {r2:.6f}")

    # Compute SHAP values
    try:
        # Sample background data for the explainer
        background_indices = np.random.choice(
            len(X), size=min(100, len(X)), replace=False
        )
        background = X.iloc[background_indices]

        # Create explainer and compute SHAP values
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

        print("Successfully computed SHAP values using KernelExplainer")
    except Exception as e:
        print(f"Error computing SHAP values with KernelExplainer: {e}")
        print("Using feature values * coefficients as SHAP approximation")

        # Approximate SHAP values for linear model
        shap_values = np.zeros(X.shape)
        for i, col in enumerate(X.columns):
            shap_values[:, i] = X[col].values * model.coef_[i]

    # Find detection points (threshold crossings)
    detection_indices = []

    for i in range(1, len(df)):
        if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
            detection_indices.append(i)

    # If no threshold crossings, find peaks after change points
    if not detection_indices:
        change_points = []
        if "true_change_point" in df.columns:
            change_points = df.loc[
                ~df["true_change_point"].isna(), "timestep"
            ].values.tolist()

        if change_points:
            for cp in change_points:
                # Find closest index to change point
                cp_idx = np.argmin(np.abs(timesteps - cp))

                # Define window after change point
                window_start = cp_idx
                window_end = min(len(df), cp_idx + 10)

                # Find index of maximum sum martingale
                if window_start < window_end:
                    max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                    detection_indices.append(max_idx)

    # Compute contributions at detection points
    contributions = []

    for idx in detection_indices:
        if 0 <= idx < len(df):
            detection_time = timesteps[idx]

            # Calculate feature values and contributions
            feature_values = df[feature_cols].iloc[idx].fillna(0).values
            total = sum(feature_values)

            if total > 0:
                contrib_data = {
                    "Feature": get_display_names(feature_cols),
                    "Martingale Value": feature_values,
                    "Contribution %": (feature_values / total) * 100,
                    "Detection Point": detection_time,
                }

                # Create DataFrame and sort by contribution
                contrib_df = pd.DataFrame(contrib_data)
                contrib_df = contrib_df.sort_values("Contribution %", ascending=False)

                contributions.append(contrib_df)

                print(f"\nFeature contributions at detection point {detection_time}:")
                print(
                    f"{'Feature':<15} {'Martingale Value':<15} {'Contribution %':<15}"
                )
                print("-" * 50)

                for _, row in contrib_df.iterrows():
                    print(
                        f"{row['Feature']:<15} {row['Martingale Value']:<15.6f} {row['Contribution %']:<15.2f}"
                    )

    if contributions:
        all_contributions = pd.concat(contributions)
        return shap_values, all_contributions
    else:
        return shap_values, pd.DataFrame()


def plot_shap_summary(shap_values, X, feature_names, output_path):
    """Create SHAP summary plot to visualize feature importance.

    Args:
        shap_values: SHAP values from compute_shap_values
        X: Feature matrix used for SHAP analysis
        feature_names: List of feature display names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {output_path}")


def plot_shap_waterfall(shap_values, X, feature_names, output_path, sample_idx=None):
    """Create SHAP waterfall plot for a specific sample.

    Args:
        shap_values: SHAP values from compute_shap_values
        X: Feature matrix used for SHAP analysis
        feature_names: List of feature display names
        output_path: Path to save the plot
        sample_idx: Index of sample to explain (defaults to highest sum of SHAP values)
    """
    if sample_idx is None:
        # Pick sample with highest sum of absolute SHAP values
        sample_idx = np.argmax(np.sum(np.abs(shap_values), axis=1))

    plt.figure(figsize=(10, 6))

    try:
        # Get expected value (base value)
        expected_value = np.mean(X.sum(axis=1).values)

        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=expected_value,
                data=X.iloc[sample_idx].values,
                feature_names=feature_names,
            ),
            show=False,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP waterfall plot for sample {sample_idx} to {output_path}")
    except Exception as e:
        print(f"Error creating waterfall plot: {e}")
        plt.close()


def plot_shap_over_time(
    shap_values, df, feature_cols, sum_col, change_points, threshold, output_path
):
    """Create visualization of SHAP values over time with annotated change points.

    Args:
        shap_values: SHAP values from compute_shap_values
        df: DataFrame containing martingale data
        feature_cols: List of column names for individual feature martingales
        sum_col: Column name for sum martingale
        change_points: List of change points
        threshold: Detection threshold
        output_path: Path to save the plot
    """
    # Get display names for features
    feature_names = get_display_names(feature_cols)

    # Get timesteps
    timesteps = (
        df["timestep"].values if "timestep" in df.columns else np.arange(len(df))
    )

    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    # Create figure with two panels
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1},
    )

    # Panel 1: SHAP Values
    for i, col in enumerate(feature_cols):
        axs[0].plot(
            timesteps,
            shap_values[:, i],
            label=feature_names[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Mark change points
    if change_points:
        for cp in change_points:
            axs[0].axvline(x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.5)

    # Note: Removed annotations for significant SHAP values as requested

    axs[0].set_title("Feature Contributions Over Time (SHAP Values)", fontsize=14)
    axs[0].set_ylabel("SHAP Value", fontsize=12)
    axs[0].legend(loc="upper right", fontsize=9)
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Classifier Feature Contributions
    # Calculate classifier feature contributions
    classifier_contributions = np.zeros((len(df), len(feature_cols)))
    detection_indices = []

    # Find detection points (threshold crossings)
    for i in range(1, len(df)):
        if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
            detection_indices.append(i)

    # If no threshold crossings, find peaks after change points
    if not detection_indices and change_points:
        for cp in change_points:
            # Find closest index to change point
            cp_idx = np.argmin(np.abs(timesteps - cp))

            # Define window after change point
            window_start = cp_idx
            window_end = min(len(df), cp_idx + 10)

            # Find index of maximum sum martingale
            if window_start < window_end:
                max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                detection_indices.append(max_idx)

    # Process all detection points if available
    if detection_indices:
        for detection_idx in detection_indices:
            # Calculate contribution at detection point
            feature_values = df[feature_cols].iloc[detection_idx].values
            total = sum(feature_values)

            if total > 0:
                # Contributions as percentages
                for j, val in enumerate(feature_values):
                    classifier_contributions[detection_idx, j] = val / total

                # Add decaying contributions around detection point
                window = 2
                for i in range(
                    max(0, detection_idx - window),
                    min(len(df), detection_idx + window + 1),
                ):
                    if i != detection_idx:
                        decay = 0.2 ** abs(i - detection_idx)
                        vals = df[feature_cols].iloc[i].values
                        val_sum = sum(vals)
                        if val_sum > 0:
                            for j, val in enumerate(vals):
                                classifier_contributions[i, j] = (val / val_sum) * decay

    # Get contribution percentages for each feature at detection points
    feature_percentages = {}
    detection_timestamps = []

    # Extract contribution percentages at each detection point
    if detection_indices and len(detection_indices) > 0:
        for i, idx in enumerate(detection_indices):
            if 0 <= idx < len(timesteps):
                detection_timestamps.append(timesteps[idx])
                for j, col in enumerate(feature_cols):
                    if j not in feature_percentages:
                        feature_percentages[j] = []
                    feature_percentages[j].append(
                        classifier_contributions[idx, j] * 100
                    )

    # Create legend labels with percentages
    legend_labels = []
    for i, name in enumerate(feature_names):
        if i in feature_percentages and len(feature_percentages[i]) > 0:
            # Format percentages string based on number of detection points
            if len(feature_percentages[i]) == 1:
                pct_str = f"{feature_percentages[i][0]:.1f}%"
            elif len(feature_percentages[i]) == 2:
                pct_str = f"{feature_percentages[i][0]:.1f}%, {feature_percentages[i][1]:.1f}%"
            else:
                pct_str = ", ".join([f"{pct:.1f}%" for pct in feature_percentages[i]])

            legend_labels.append(f"{name} ({pct_str})")
        else:
            legend_labels.append(name)

    # Plot classifier contributions
    for i, col in enumerate(feature_cols):
        axs[1].plot(
            timesteps,
            classifier_contributions[:, i],
            label=legend_labels[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Mark change points and detection points
    if change_points:
        for cp in change_points:
            axs[1].axvline(x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.5)

    if detection_indices:
        for idx in detection_indices:
            if 0 <= idx < len(timesteps):
                dp = timesteps[idx]
                axs[1].axvline(
                    x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
                )

    axs[1].set_title("Feature Contributions at Detection Points", fontsize=14)
    axs[1].set_xlabel("Timestep", fontsize=12)
    axs[1].set_ylabel("Feature Contribution (0-1)", fontsize=12)

    # Create custom legend with more space
    axs[1].legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        handlelength=1,
        handleheight=1.5,
        labelspacing=0.4,
    )

    axs[1].grid(True, alpha=0.3)

    # Set common x-axis limits
    x_min, x_max = min(timesteps), max(timesteps)
    x_limits = (x_min, 205)  # Same as in original code

    # Set common x-ticks
    x_ticks = np.array([0, 40, 80, 120, 160, 200])

    for ax in axs:
        ax.set_xlim(x_limits)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in ax.get_xticks()])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP over time plot to {output_path}")

    # Generate feature contribution report for detection points
    all_contributions = []
    if detection_indices:
        for idx in detection_indices:
            if 0 <= idx < len(df):
                detection_time = timesteps[idx]

                # Get contributions for this detection point
                contrib_data = {
                    "Feature": feature_names,
                    "Martingale Value": df[feature_cols].iloc[idx].values,
                    "Contribution %": classifier_contributions[idx] * 100,
                    "Detection Point": detection_time,
                }

                # Create DataFrame and sort by contribution
                contrib_df = pd.DataFrame(contrib_data)
                contrib_df = contrib_df.sort_values("Contribution %", ascending=False)
                all_contributions.append(contrib_df)

                print(f"\nFeature contributions at detection point {detection_time}:")
                print(
                    f"{'Feature':<15} {'Martingale Value':<15} {'Contribution %':<15}"
                )
                print("-" * 50)

                for _, row in contrib_df.iterrows():
                    print(
                        f"{row['Feature']:<15} {row['Martingale Value']:<15.6f} {row['Contribution %']:<15.2f}"
                    )

    return all_contributions


def plot_feature_contributions(contributions_df, output_path):
    """Create horizontal bar chart of feature contributions.

    Args:
        contributions_df: DataFrame with feature contributions
        output_path: Path to save the plot
    """
    if contributions_df.empty:
        print("No feature contributions to plot")
        return

    # Group by feature and compute average contribution
    avg_contrib = (
        contributions_df.groupby("Feature")["Contribution %"].mean().reset_index()
    )
    avg_contrib = avg_contrib.sort_values(
        "Contribution %", ascending=True
    )  # Ascending for horizontal bars

    # Create colors based on contribution percentage
    colors = plt.cm.YlOrRd(avg_contrib["Contribution %"] / 100)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(avg_contrib["Feature"], avg_contrib["Contribution %"], color=colors)

    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center"
        )

    plt.xlabel("Average Contribution %")
    plt.title("Feature Contributions to Anomaly Detection")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature contributions plot to {output_path}")


def analyze_shap(
    file_path, sheet_name="Aggregate", output_dir="results/shap", threshold=50.0
):
    """Main function to analyze SHAP values and create visualizations.

    Args:
        file_path: Path to Excel file
        sheet_name: Name of sheet to analyze
        output_dir: Directory to save output plots
        threshold: Detection threshold
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    df, change_points, metadata_df = load_data(file_path, sheet_name)

    if df is None:
        return

    # Get feature columns
    feature_cols = get_feature_cols(df)
    if not feature_cols:
        print("No feature columns found in the data")
        return

    # Get display names
    feature_names = get_display_names(feature_cols)

    # Determine sum column name
    sum_col = next(
        (
            col
            for col in df.columns
            if col
            in ["traditional_sum_martingales_mean", "traditional_sum_martingales"]
        ),
        None,
    )

    if sum_col is None:
        print("Sum martingale column not found in the data")
        return

    print(f"Analyzing {len(feature_cols)} features with {sum_col}")

    # Use provided threshold
    print(f"Using threshold: {threshold}")

    # Compute SHAP values
    shap_values, contributions_df = compute_shap_values(
        df=df, feature_cols=feature_cols, sum_col=sum_col, threshold=threshold
    )

    # Create SHAP summary plot
    plot_shap_summary(
        shap_values=shap_values,
        X=df[feature_cols],
        feature_names=feature_names,
        output_path=os.path.join(output_dir, "shap_summary.png"),
    )

    # Create SHAP waterfall plot
    # Find detection point if any
    detection_idx = None
    for i in range(1, len(df)):
        if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
            detection_idx = i
            break

    if detection_idx is not None:
        plot_shap_waterfall(
            shap_values=shap_values,
            X=df[feature_cols],
            feature_names=feature_names,
            output_path=os.path.join(output_dir, "shap_waterfall.png"),
            sample_idx=detection_idx,
        )

    # Create SHAP over time plot
    plot_shap_over_time(
        shap_values=shap_values,
        df=df,
        feature_cols=feature_cols,
        sum_col=sum_col,
        change_points=change_points,
        threshold=threshold,
        output_path=os.path.join(output_dir, "shap_over_time.png"),
    )

    # Create feature contributions plot
    if not contributions_df.empty:
        plot_feature_contributions(
            contributions_df=contributions_df,
            output_path=os.path.join(output_dir, "feature_contributions.png"),
        )

        # Save contributions to CSV
        contributions_df.to_csv(
            os.path.join(output_dir, "feature_contributions.csv"), index=False
        )

        print(
            f"Saved feature contributions to {os.path.join(output_dir, 'feature_contributions.csv')}"
        )

    print(f"SHAP analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze SHAP values for anomaly detection"
    )
    parser.add_argument(
        "--file_path", "-f", type=str, required=True, help="Path to Excel file"
    )
    parser.add_argument(
        "--sheet_name", "-s", type=str, default="Aggregate", help="Sheet name"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="results/shap", help="Output directory"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=50.0, help="Detection threshold"
    )

    args = parser.parse_args()

    analyze_shap(
        file_path=args.file_path,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
