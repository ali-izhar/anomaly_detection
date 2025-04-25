#!/usr/bin/env python3
# SHAP Analysis for Martingale-based Graph Change Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sklearn
from sklearn.linear_model import LinearRegression
import argparse
import os


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


def plot_combined_analysis(
    df,
    change_points,
    timesteps,
    feature_cols,
    original_feature_names,
    threshold=60.0,
):
    """
    Create a combined plot with martingale values, SHAP values, and classifier contributions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing martingale values
    change_points : list
        List of true change points
    timesteps : numpy.ndarray
        Array of timestep values
    feature_cols : list
        List of column names for individual feature martingales
    original_feature_names : list
        List of human-readable feature names
    threshold : float
        Detection threshold value (default: 60.0)
    """
    print("Creating combined analysis plot...")

    # Create a DataFrame for feature martingales
    X = pd.DataFrame()
    for i, col in enumerate(feature_cols):
        # Extract feature ID to create simpler column names
        feature_id = col.split("feature")[1].split("_")[0]
        X[f"M^{feature_id}"] = df[col]

    # Get the sum martingale
    sum_martingale_col = "traditional_sum_martingales_mean"
    if sum_martingale_col not in df.columns:
        print(
            f"Warning: {sum_martingale_col} not found. Computing from individual martingales."
        )
        df[sum_martingale_col] = df[feature_cols].sum(axis=1)

    # Handle NaN values by replacing them with zeros
    print("Checking for NaN values...")
    nan_count_X = X.isna().sum().sum()
    nan_count_y = df[sum_martingale_col].isna().sum()

    if nan_count_X > 0:
        print(f"Found {nan_count_X} NaN values in feature data. Replacing with zeros.")
        X = X.fillna(0)

    if nan_count_y > 0:
        print(
            f"Found {nan_count_y} NaN values in sum martingale. Replacing with zeros."
        )
        df[sum_martingale_col] = df[sum_martingale_col].fillna(0)

    # Get the average martingale
    avg_martingale_col = "traditional_avg_martingales_mean"

    # Find detection points (where sum martingale crosses threshold)
    detection_indices = []
    for i in range(1, len(df)):
        if (
            df[sum_martingale_col].iloc[i - 1] <= threshold
            and df[sum_martingale_col].iloc[i] > threshold
        ):
            detection_indices.append(i)

    # If no threshold crossing is found, use maximum martingale values near change points
    if not detection_indices:
        print(
            "No threshold crossing found. Using maximum martingale values near change points."
        )

        for cp in change_points:
            # Find the closest index to the change point
            cp_idx = np.argmin(np.abs(timesteps - cp))

            # Define a window around the change point (e.g., 10 steps after)
            window_start = cp_idx
            window_end = min(len(df), cp_idx + 10)

            # Find the index of maximum sum martingale in this window
            max_idx = df[sum_martingale_col].iloc[window_start:window_end].idxmax()
            detection_indices.append(max_idx)

        if detection_indices:
            print(
                f"Found {len(detection_indices)} peaks near change points at timesteps: {[timesteps[i] for i in detection_indices]}"
            )

    if detection_indices:
        print(
            f"Using detection points at timesteps: {[timesteps[i] for i in detection_indices]}"
        )
        # Use all detection indices instead of just the first one
        detection_indices = sorted(detection_indices)
    else:
        print("No detection point found. Analysis will be limited.")
        detection_indices = []

    # Compute SHAP values for feature contributions
    print("Computing SHAP values...")

    # Create a simple linear model (additive martingale is sum of features)
    model = LinearRegression(fit_intercept=False)  # No intercept since M^A is exact sum
    model.fit(X, df[sum_martingale_col])

    # Verify the model accuracy
    predictions = model.predict(X)
    r2 = sklearn.metrics.r2_score(df[sum_martingale_col], predictions)
    print(f"R² score of linear model: {r2:.6f} (should be close to 1.0)")

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
    except Exception as e:
        print(f"Error computing SHAP values with KernelExplainer: {e}")
        print("Using feature values * coefficients as SHAP approximation")
        # Approximate SHAP values for linear model
        shap_values = np.zeros(X.shape)
        for i, col in enumerate(X.columns):
            shap_values[:, i] = X[col].values * model.coef_[i]

    # Compute threshold-based classifier SHAP values for each detection point
    print("Computing threshold-based classifier contributions...")
    classifier_shap_values = np.zeros(X.shape)

    # Process all detection points
    for detection_index in detection_indices:
        # Calculate contribution at detection point
        feature_values = X.iloc[detection_index].values
        total = sum(feature_values)

        if total > 0:
            # Contributions as percentages
            for j, val in enumerate(feature_values):
                classifier_shap_values[detection_index, j] = val / total

            # Add decaying contributions around detection point
            window = 2
            for i in range(
                max(0, detection_index - window),
                min(len(df), detection_index + window + 1),
            ):
                if i != detection_index:
                    decay = 0.2 ** abs(i - detection_index)
                    vals = X.iloc[i].values
                    val_sum = sum(vals)
                    if val_sum > 0:
                        for j, val in enumerate(vals):
                            classifier_shap_values[i, j] = (val / val_sum) * decay

    # If no detection indices found, use feature peaks for visualization
    if not detection_indices:
        print(
            "No detection point found. Creating feature contribution visualization anyway."
        )

        # Find overall peak for each feature for visualization
        for j in range(X.shape[1]):
            peak_idx = X.iloc[:, j].idxmax()
            if peak_idx is not None:
                feature_val = X.iloc[peak_idx, j]
                # Only assign non-zero value if feature actually has some contribution
                if feature_val > 0:
                    classifier_shap_values[peak_idx, j] = (
                        0.5  # Use a modest value for visualization
                    )

    # Set up publication-quality plot style
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.figsize": (7, 8),
            "figure.dpi": 300,
        }
    )

    # Create a consistent color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    # Create the combined 3-panel figure
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(7, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.15},
    )

    # Panel 1: Feature Martingales
    for i, col in enumerate(feature_cols):
        axs[0].plot(
            timesteps,
            df[col],
            label=original_feature_names[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Add sum martingale
    axs[0].plot(
        timesteps,
        df[sum_martingale_col],
        label="Sum Martingale",
        color="black",
        linewidth=2,
    )

    # Add avg martingale if available
    if avg_martingale_col in df.columns:
        axs[0].plot(
            timesteps,
            df[avg_martingale_col],
            label="Avg Martingale",
            color="purple",
            linewidth=1.5,
            linestyle="--",
        )

    # Add threshold
    axs[0].axhline(
        y=threshold,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold})",
    )

    # Mark change points
    for cp in change_points:
        cp_idx = np.where(timesteps == cp)[0]
        if len(cp_idx) > 0:
            axs[0].axvline(x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5)

    # Mark all detection points
    for detection_index in detection_indices:
        dp = timesteps[detection_index]
        axs[0].axvline(x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5)

    axs[0].set_title("Martingale Values Over Time", fontsize=11)
    axs[0].set_ylabel("Martingale Value", fontsize=10)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(True, alpha=0.3)

    # Panel 2: SHAP Values
    feature_names = [
        f"M^{col.split('feature')[1].split('_')[0]}" for col in feature_cols
    ]
    for i, col in enumerate(feature_names):
        axs[1].plot(
            timesteps,
            shap_values[:, i],
            label=original_feature_names[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Mark change points
    for cp in change_points:
        cp_idx = np.where(timesteps == cp)[0]
        if len(cp_idx) > 0:
            axs[1].axvline(x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5)

    # Mark all detection points
    for detection_index in detection_indices:
        dp = timesteps[detection_index]
        axs[1].axvline(x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5)

    axs[1].set_title("SHAP Values Over Time", fontsize=11)
    axs[1].set_ylabel("SHAP Value", fontsize=10)
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Classifier SHAP Values
    for i, col in enumerate(feature_names):
        axs[2].plot(
            timesteps,
            classifier_shap_values[:, i],
            label=original_feature_names[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Mark change points
    for cp in change_points:
        cp_idx = np.where(timesteps == cp)[0]
        if len(cp_idx) > 0:
            axs[2].axvline(x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5)

    # If we have detection points, annotate them with percentages
    panel_title = "Feature Contributions at Peak Martingales"
    if detection_indices:
        # Mark all detection points
        for detection_index in detection_indices:
            dp = timesteps[detection_index]
            axs[2].axvline(
                x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
            )

            # Add text annotation near each detection point
            text_y = 0.8  # Starting y position for text
            for i, col in enumerate(feature_names):
                if i % 2 == 0:  # Split annotations to either side
                    text_x = dp + 5  # Right side
                    ha = "left"
                else:
                    text_x = dp - 5  # Left side
                    ha = "right"

                percentage = classifier_shap_values[detection_index, i] * 100
                if percentage > 3.0:  # Only show significant contributors
                    axs[2].annotate(
                        f"{original_feature_names[i]}: {percentage:.1f}%",
                        xy=(text_x, text_y - i * 0.05),
                        color=colors[i],
                        fontsize=8,
                        fontweight="bold",
                        ha=ha,
                        va="center",
                    )

        # Set title based on whether this was a threshold crossing or peaks
        if any(
            df[sum_martingale_col].iloc[idx] > threshold for idx in detection_indices
        ):
            panel_title = "Threshold-Based Classifier SHAP Values"
        else:
            panel_title = "Feature Contributions at Peak Martingales"
    else:
        panel_title = "Feature Contributions at Maximum Points"

    axs[2].set_title(panel_title, fontsize=11)
    axs[2].set_xlabel("Timestep", fontsize=10)
    axs[2].set_ylabel("Feature Contribution (0 to 1)", fontsize=10)
    axs[2].legend(loc="upper right", fontsize=8)
    axs[2].grid(True, alpha=0.3)

    # Save the figure
    plt.tight_layout()
    plt.savefig("combined_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig("combined_analysis.pdf", format="pdf", bbox_inches="tight")
    print("Saved plots to combined_analysis.png and combined_analysis.pdf")

    # Generate feature contribution report for all detection points
    if detection_indices:
        # Create a DataFrame to hold all contributions
        all_contributions = []

        for idx, detection_index in enumerate(detection_indices):
            detection_time = timesteps[detection_index]
            print(
                f"\nFeature contributions at detection point {idx+1} (timestep {detection_time}):"
            )

            # Get contributions for this detection point
            contrib_df = pd.DataFrame(
                {
                    "Feature": original_feature_names,
                    "Martingale Value": X.iloc[detection_index].values,
                    "Contribution %": classifier_shap_values[detection_index] * 100,
                    "Detection Point": detection_time,  # Add timestep information
                }
            )

            # Sort by contribution percentage
            contrib_df = contrib_df.sort_values("Contribution %", ascending=False)
            print(
                contrib_df[["Feature", "Martingale Value", "Contribution %"]].to_string(
                    index=False
                )
            )

            # Add to the collection of all contributions
            all_contributions.append(contrib_df)

        # Combine all detection point analyses
        if all_contributions:
            combined_df = pd.concat(all_contributions)
            combined_df.to_csv("detection_contributions.csv", index=False)
            print("Saved feature contributions to detection_contributions.csv")

            # Create a separate plot comparing feature contributions across detection points
            if len(detection_indices) > 1:
                _plot_comparison_across_detections(
                    all_contributions, original_feature_names, colors
                )


def _plot_comparison_across_detections(all_contributions, feature_names, colors):
    """Create a bar chart comparing feature contributions across multiple detection points."""
    # Create a pivot table with features as rows and detection points as columns
    pivot_df = pd.concat(all_contributions)
    pivot_df = pivot_df.pivot(
        index="Feature", columns="Detection Point", values="Contribution %"
    )

    # Sort features by average contribution
    pivot_df["Average"] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values("Average", ascending=False)
    pivot_df = pivot_df.drop("Average", axis=1)

    # Create the bar chart
    plt.figure(figsize=(10, 6))

    # Set bar width based on number of detection points
    bar_width = 0.8 / len(pivot_df.columns)

    # Plot bars for each detection point
    for i, (detection_point, contributions) in enumerate(pivot_df.items()):
        # Calculate position for this group of bars
        positions = np.arange(len(pivot_df.index)) + i * bar_width

        # Plot with feature-specific colors
        for j, (feature, value) in enumerate(contributions.items()):
            color_idx = feature_names.index(feature) if feature in feature_names else j
            plt.bar(
                positions[j],
                value,
                bar_width,
                label=f"CP {int(detection_point)}" if j == 0 else None,
                color=colors[color_idx % len(colors)],
                alpha=0.7,
            )

    # Add labels and title
    plt.xlabel("Feature")
    plt.ylabel("Contribution %")
    plt.title("Feature Contributions Across Detection Points")
    plt.xticks(
        np.arange(len(pivot_df.index)) + bar_width * (len(pivot_df.columns) - 1) / 2,
        pivot_df.index,
        rotation=45,
        ha="right",
    )
    plt.legend(title="Detection Point")
    plt.tight_layout()

    # Save the comparison plot
    plt.savefig("contribution_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved feature contribution comparison to contribution_comparison.png")


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
    args = parser.parse_args()

    # Load data
    df, change_points, timesteps, feature_cols, original_feature_names = (
        load_martingale_data(args.file_path)
    )

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Create combined analysis plot
    plot_combined_analysis(
        df,
        change_points,
        timesteps,
        feature_cols,
        original_feature_names,
        args.threshold,
    )


if __name__ == "__main__":
    main()
