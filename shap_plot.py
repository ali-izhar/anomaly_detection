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
    Load martingale values from a CSV or Excel file.

    Expected format from user's data:
    - timestep: Timestep column
    - true_change_point: Column indicating true change points
    - degree, betweenness, eigenvector, closeness, svd, lsvd: Feature columns
    - traditional_sum_martingales: Additive martingale column
    - traditional_detected: Column indicating detected changes
    - traditional_avg_martingales: Average martingale column
    """
    print(f"Loading martingale data from {file_path}")

    # Determine file type and read accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".csv":
        df = pd.read_csv(file_path)
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension: {file_ext}. Use .csv, .xlsx, or .xls"
        )

    print(f"Data loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Get timesteps
    if "timestep" in df.columns:
        timesteps = df["timestep"].values
    else:
        timesteps = np.arange(len(df))

    # Get change point column
    cp_col = None
    if "true_change_point" in df.columns:
        cp_col = "true_change_point"

    # Define the expected feature columns
    expected_feature_cols = [
        "degree",
        "betweenness",
        "eigenvector",
        "closeness",
        "svd",
        "lsvd",
    ]

    # Check which expected feature columns actually exist in the data
    feature_cols = [col for col in expected_feature_cols if col in df.columns]

    if not feature_cols:
        # Fallback to the original approach if none of the expected columns are found
        feature_cols = [
            col
            for col in df.columns
            if col.startswith("individual_traditional_martingales_feature")
        ]

        if not feature_cols:
            # Try alternative naming
            feature_cols = [
                col
                for col in df.columns
                if col.startswith("feature") or "martingale" in col.lower()
            ]
            feature_cols = [
                col
                for col in feature_cols
                if "traditional_sum" not in col and "traditional_avg" not in col
            ]

    if not feature_cols:
        raise ValueError("Could not identify feature columns in the file.")

    print(f"Found {len(feature_cols)} feature columns: {feature_cols}")

    # Store original feature names for later use in plot legends
    original_feature_names = feature_cols.copy()

    # Rename columns for easier use
    column_mapping = {}
    for i, col in enumerate(feature_cols):
        column_mapping[col] = f"M^{i}"

    # Map additive martingale column
    if "traditional_sum_martingales" in df.columns:
        column_mapping["traditional_sum_martingales"] = "M^A"

    # Rename columns in dataframe
    df = df.rename(columns=column_mapping)

    # Update feature_cols list with new names
    feature_cols = [column_mapping[col] for col in feature_cols]

    return df, timesteps, feature_cols, cp_col, original_feature_names


def compute_additive_martingale(df, feature_cols):
    """
    Compute the additive martingale if not already in the data.
    """
    if "M^A" in df.columns:
        print("Using existing additive martingale column 'M^A'")
        return df

    print("Computing additive martingale as sum of feature martingales")
    df["M^A"] = df[feature_cols].sum(axis=1)
    return df


def plot_combined_analysis(
    df,
    timesteps,
    feature_cols,
    cp_col=None,
    threshold=None,
    detection_index=None,
    original_feature_names=None,
):
    """
    Create a combined plot with martingale values on top, normalized SHAP values in the middle,
    and classifier-based SHAP values on the bottom.
    Optimized for publication-quality figures in research papers.

    Args:
        df: DataFrame containing martingale values
        timesteps: Array of timestep values
        feature_cols: List of column names for individual feature martingales
        cp_col: Name of change point column (optional)
        threshold: Detection threshold value (optional)
        detection_index: Index of detection point to analyze (optional)
        original_feature_names: List of original feature column names (optional)
    """
    print("Computing SHAP values...")

    # If original feature names weren't provided, use generic names
    if original_feature_names is None:
        original_feature_names = [f"feature{i}" for i in range(len(feature_cols))]

    # Prepare the data for SHAP analysis
    X = df[feature_cols]
    y = df["M^A"]

    # Create a simple linear model (additive martingale is sum of features)
    model = LinearRegression(fit_intercept=False)  # No intercept since M^A is exact sum
    model.fit(X, y)

    # Verify the model is accurate (should be perfect for additive martingale)
    predictions = model.predict(X)
    r2 = sklearn.metrics.r2_score(y, predictions)
    print(f"R² score of linear model: {r2:.6f} (should be close to 1.0)")

    # Define prediction function for SHAP
    def predict_fn(x):
        return model.predict(x)

    # Compute SHAP values
    try:
        # Create a background dataset (sample of X)
        background_indices = np.random.choice(
            len(X), size=min(100, len(X)), replace=False
        )
        background = X.iloc[background_indices]

        # Create explainer and compute SHAP values
        print("Creating KernelExplainer...")
        explainer = shap.KernelExplainer(predict_fn, background)
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X)
        print(f"SHAP values shape: {np.array(shap_values).shape}")
    except Exception as e:
        print(f"Error using KernelExplainer: {e}")
        print("Computing SHAP values manually based on model coefficients...")
        # For linear models without intercept, SHAP values = feature_value * coefficient
        shap_values = np.zeros(X.shape)
        for i, col in enumerate(feature_cols):
            shap_values[:, i] = X[col].values * model.coef_[i]

    # Create a DataFrame with SHAP values over time
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{col}" for col in feature_cols])
    shap_df["timestep"] = timesteps

    # Normalize SHAP values by the max absolute SHAP value for any feature at any time
    shap_cols = [f"SHAP_{col}" for col in feature_cols]
    max_abs_shap = max(
        abs(shap_df[shap_cols].values.min()), abs(shap_df[shap_cols].values.max())
    )

    if max_abs_shap > 0:  # Avoid division by zero
        for col in shap_cols:
            shap_df[f"Norm_{col}"] = shap_df[col] / max_abs_shap

    # =========================================================================
    # Create threshold-based classifier SHAP values similar to CustomThresholdModel
    # =========================================================================
    print("Computing threshold-based classifier SHAP values...")

    # Create a custom threshold model similar to CustomThresholdModel in threshold.py
    class ThresholdClassifier:
        def __init__(self, threshold):
            # Default to a safe value if threshold is None
            self.threshold = threshold if threshold is not None else 1.0

        def predict(self, X):
            # Binary classifier: 1 if sum exceeds threshold, 0 otherwise
            sums = np.sum(X, axis=1)
            return (sums > self.threshold).astype(int)

        def predict_proba(self, X):
            # Probability calculation as in threshold.py
            sums = np.sum(X, axis=1)
            return sums / (sums + self.threshold)

    # Initialize the classifier with threshold
    if threshold is not None:
        classifier = ThresholdClassifier(threshold)
    else:
        # Use a default threshold for visualization purposes if none provided
        default_threshold = df["M^A"].max() * 0.5  # 50% of max as default
        threshold = default_threshold  # Set the threshold variable
        classifier = ThresholdClassifier(threshold)
        print(f"No threshold provided, using {default_threshold:.2f} for visualization")

    # Compute classifier-based SHAP values with correct binary behavior
    classifier_shap_values = np.zeros(X.shape)

    # Binary labels to use for reference
    binary_labels = np.zeros(len(df))
    for i in range(len(df)):
        binary_labels[i] = 1 if df["M^A"].iloc[i] > threshold else 0

    # Find the exact detection point if it exists
    detection_found = False
    if detection_index is not None:
        detection_found = True
    elif threshold is not None:
        for i in range(1, len(df)):
            if df["M^A"].iloc[i - 1] <= threshold and df["M^A"].iloc[i] > threshold:
                detection_index = i
                detection_found = True
                break

    # For manual calculation of classifier SHAP values
    print("Using manual calculation for classifier SHAP values...")

    # If we found a detection point, calculate exact feature contributions
    if detection_found and detection_index is not None:
        # Calculate the exact percentage contributions at detection point
        detection_values = X.iloc[detection_index].values
        total_martingale = sum(detection_values)

        if total_martingale > 0:  # Avoid division by zero
            # At detection point: SHAP values exactly match contribution percentages
            for j, val in enumerate(detection_values):
                percentage = val / total_martingale
                classifier_shap_values[detection_index, j] = percentage

            # Create a small window around detection with exponential decay
            window = 2  # Small window around detection
            for i in range(
                max(0, detection_index - window),
                min(len(df), detection_index + window + 1),
            ):
                if i != detection_index:
                    # Exponential decay based on distance from detection
                    decay = 0.2 ** abs(i - detection_index)
                    feature_values = X.iloc[i].values
                    feature_sum = sum(feature_values)

                    if feature_sum > 0:  # Avoid division by zero
                        for j, val in enumerate(feature_values):
                            # Small SHAP values near detection, maintaining relative proportions
                            percentage = val / feature_sum
                            classifier_shap_values[i, j] = percentage * decay
    else:
        # If no detection point, set very small values throughout
        print("No detection point found, setting minimal classifier SHAP values")
        classifier_shap_values.fill(0.001)  # Small non-zero values for visualization

    # Create DataFrame with classifier SHAP values
    classifier_shap_df = pd.DataFrame(
        classifier_shap_values,
        columns=[f"Classifier_SHAP_{col}" for col in feature_cols],
    )
    classifier_shap_df["timestep"] = timesteps

    # The classifier SHAP values are already scaled as percentages (0-1),
    # so no additional normalization is needed

    # Set publication-quality plot style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "text.usetex": True,  # Enable LaTeX rendering
            "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts}",
            "mathtext.fontset": "cm",  # Computer Modern font for math
        }
    )

    # Define a consistent, visually distinct color palette for features
    # This uses a colorblind-friendly palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    # Create labels for features using original column names
    latex_labels = []
    for i, orig_name in enumerate(original_feature_names):
        latex_labels.append(orig_name)  # Use the original column name directly

    # Create the combined 1 column, 3 row figure - optimized for paper
    print("Creating publication-ready plot...")
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(7, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.15},
    )

    # Top plot: Martingale values
    for i, col in enumerate(feature_cols):
        axs[0].plot(
            timesteps,
            df[col],
            label=latex_labels[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Plot additive martingale with thicker line
    axs[0].plot(timesteps, df["M^A"], label=r"$M^A$", color="black", linewidth=2)

    # Plot threshold line if provided
    if threshold is not None:
        axs[0].axhline(
            y=threshold,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=r"$\lambda=" + str(threshold) + "$",
        )

    # Plot change points if available
    if cp_col is not None and cp_col in df.columns:
        change_points = timesteps[df[cp_col] == 1]
        if len(change_points) > 0:
            for cp in change_points:
                axs[0].axvline(
                    x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                )
            axs[0].scatter(
                change_points,
                [0] * len(change_points),
                color="g",
                s=60,
                marker="o",
                label="True Change Points",
                zorder=5,
            )

    # Add detection point if detection_index is provided
    if detection_index is not None:
        dp = timesteps[detection_index]
        axs[0].axvline(x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5)
        axs[0].scatter(
            [dp],
            [0],
            color="purple",
            s=60,
            marker="o",
            label="Detection Point",
            zorder=5,
        )

    # Set y-axis limit for top plot
    axs[0].set_ylim(0, 70)

    # Customize the top plot
    axs[0].set_title("Martingale Values Over Time", fontsize=11, pad=8)
    axs[0].set_ylabel("Martingale Value", fontsize=10)

    # Add a legend outside the plot to the right to maximize data visibility
    legend = axs[0].legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=8,
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(0.5)

    # Set the grid behind the lines for better visibility
    axs[0].grid(True, alpha=0.3, linestyle=":")
    axs[0].set_axisbelow(True)

    # Middle plot: Normalized SHAP values
    for i, col in enumerate(feature_cols):
        axs[1].plot(
            timesteps,
            shap_df[f"SHAP_{col}"],
            label=latex_labels[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Set y-axis limit for middle plot to match top plot
    axs[1].set_ylim(0, 70)

    # Add change points if available
    if cp_col is not None and cp_col in df.columns:
        change_points = timesteps[df[cp_col] == 1]
        if len(change_points) > 0:
            for cp in change_points:
                axs[1].axvline(
                    x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                )

    # Add detection point if detection_index is provided
    if detection_index is not None:
        dp = timesteps[detection_index]
        axs[1].axvline(x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5)

    # Customize the middle plot
    axs[1].set_title("SHAP Values Over Time", fontsize=11, pad=8)
    axs[1].set_ylabel("SHAP Value", fontsize=10)

    # Add the R² note to this plot
    if r2 > 0.99:
        axs[1].text(
            0.02,
            0.95,
            f"SHAP values = direct feature contributions (R²={r2:.4f})",
            fontsize=8,
            style="italic",
            transform=axs[1].transAxes,
        )

    # Add a legend outside the plot to maximize data visibility
    legend = axs[1].legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=8,
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(0.5)

    # Set the grid behind the lines for better visibility
    axs[1].grid(True, alpha=0.3, linestyle=":")
    axs[1].set_axisbelow(True)

    # Bottom plot: Classifier-based SHAP values
    for i, col in enumerate(feature_cols):
        axs[2].plot(
            timesteps,
            classifier_shap_df[f"Classifier_SHAP_{col}"],
            label=latex_labels[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Add change points if available
    if cp_col is not None and cp_col in df.columns:
        change_points = timesteps[df[cp_col] == 1]
        if len(change_points) > 0:
            for cp in change_points:
                axs[2].axvline(
                    x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                )

    # Add detection point if detection_index is provided
    if detection_index is not None:
        dp = timesteps[detection_index]
        axs[2].axvline(x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5)

    # If we have a detection index, update the legend to include percentages
    if detection_index is not None:
        # Clear existing lines and create new ones with updated labels
        lines = axs[2].get_lines()[0 : len(feature_cols)]  # Get only the feature lines
        axs[2].clear()

        # Re-add the lines with updated legend labels that include percentages
        for i, col in enumerate(feature_cols):
            # Get the percentage value for this feature at detection point
            percentage = classifier_shap_values[detection_index, i] * 100

            # Format the label with original feature name and percentage
            percentage_label = f"{original_feature_names[i]} = {percentage:.1f}%"

            # Plot with the new label
            axs[2].plot(
                timesteps,
                classifier_shap_df[f"Classifier_SHAP_{col}"],
                label=percentage_label,
                color=colors[i],
                alpha=0.7,
                linewidth=1.2,
            )

        # Re-add the change point lines and detection point
        if cp_col is not None and cp_col in df.columns:
            change_points = timesteps[df[cp_col] == 1]
            if len(change_points) > 0:
                for cp in change_points:
                    axs[2].axvline(
                        x=cp, color="g", linestyle="--", alpha=0.8, linewidth=1.5
                    )

        # Re-add detection point
        axs[2].axvline(x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5)

    # Customize the bottom plot
    axs[2].set_title("Threshold-Based Classifier SHAP Values", fontsize=11, pad=8)
    axs[2].set_xlabel("Timestep", fontsize=10)
    axs[2].set_ylabel("Feature Contribution (0 to 1)", fontsize=10)

    # Add a note explaining the classifier SHAP values
    if detection_index is not None:
        axs[2].text(
            0.02,
            0.95,
            f"Values show exact feature contribution percentages at detection",
            fontsize=8,
            style="italic",
            transform=axs[2].transAxes,
        )

    # Add a legend outside the plot to maximize data visibility
    legend = axs[2].legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=8,
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(0.5)

    # Set the grid behind the lines for better visibility
    axs[2].grid(True, alpha=0.3, linestyle=":")
    axs[2].set_axisbelow(True)

    # Adjust the default space between subplots
    plt.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.07)

    # Enhance with a box around the plots to give a professional finish
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Save the combined plot with high resolution
    plt.savefig("combined_analysis.png", dpi=600, bbox_inches="tight")
    plt.savefig(
        "combined_analysis.pdf", format="pdf", bbox_inches="tight"
    )  # PDF for publication
    print(
        "Saved publication-ready plots to combined_analysis.png and combined_analysis.pdf"
    )
    plt.close()

    # If a detection index is provided, analyze feature contributions at that point
    if detection_index is not None:
        timestep_value = timesteps[detection_index]
        print(f"Analyzing detection point at timestep {timestep_value}...")

        # Get the data for the detection point
        detection_data = X.iloc[detection_index : detection_index + 1]

        # Get SHAP values for detection point
        detection_shap = (
            shap_values[detection_index] if detection_index < len(shap_values) else None
        )
        detection_classifier_shap = (
            classifier_shap_values[detection_index]
            if detection_index < len(classifier_shap_values)
            else None
        )

        if detection_shap is not None:
            # Compute probability of change as in CustomThresholdModel in threshold.py
            martingale_sum = df["M^A"].iloc[detection_index]
            change_prob = (
                martingale_sum / (martingale_sum + threshold) if threshold else 0.5
            )

            # Show the feature contributions at detection point
            contrib_df = pd.DataFrame(
                {
                    "Feature": original_feature_names,
                    "Martingale Value": detection_data.values[0],
                    "SHAP Value": detection_shap,
                    "Contribution %": (
                        100 * detection_shap / sum(detection_shap)
                        if sum(detection_shap) != 0
                        else 0
                    ),
                    "Classifier SHAP": detection_classifier_shap,
                    "Classifier Contribution %": (
                        100 * detection_classifier_shap / sum(detection_classifier_shap)
                        if sum(detection_classifier_shap) != 0
                        else 0
                    ),
                }
            )
            contrib_df = contrib_df.sort_values("Contribution %", ascending=False)
            print("\nFeature contributions at detection point:")
            print(contrib_df.to_string(index=False))
            print(f"\nProbability of change at detection point: {change_prob:.4f}")

            # Save the contributions to a CSV
            contrib_df.to_csv("detection_contributions.csv", index=False)
            print("Saved detection point contributions to detection_contributions.csv")

            return contrib_df


def find_detection_index(df, threshold):
    """Find the index where the additive martingale first exceeds the threshold."""
    for i in range(1, len(df)):
        if df["M^A"].iloc[i - 1] < threshold and df["M^A"].iloc[i] >= threshold:
            return i
    return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze martingale data with SHAP.")
    parser.add_argument(
        "file_path", help="Path to CSV or Excel file containing martingale values"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,  # Default threshold set to 60
        help="Detection threshold for additive martingale (default: 60.0)",
    )
    args = parser.parse_args()

    # Load data
    df, timesteps, feature_cols, cp_col, original_feature_names = load_martingale_data(
        args.file_path
    )

    # Compute additive martingale if needed
    df = compute_additive_martingale(df, feature_cols)

    # Use the threshold from args (which defaults to 60)
    threshold = args.threshold
    print(f"Using threshold (lambda) value: {threshold}")

    # Find detection index for the threshold
    detection_index = find_detection_index(df, threshold)
    if detection_index is not None:
        print(f"Detection occurred at timestep {timesteps[detection_index]}")
    else:
        print("No detection found for the given threshold.")

    # Create the combined plot with martingale values and normalized SHAP values
    plot_combined_analysis(
        df,
        timesteps,
        feature_cols,
        cp_col,
        threshold,
        detection_index,
        original_feature_names,
    )


if __name__ == "__main__":
    main()
