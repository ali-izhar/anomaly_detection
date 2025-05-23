#!/usr/bin/env python3
"""
Martingale SHAP Classifier Analysis Script

This script uses the CustomThresholdModel from threshold.py to analyze traditional
martingale data from an Excel file and create a single comprehensive visualization with:
1. Martingale streams over time
2. SHAP analysis showing feature contributions
3. Classifier output and detection points

The script focuses on traditional martingales and uses proper feature names.

Usage:
    python plot_mart_shap_classifier.py --input data.xlsx --threshold 60.0 --output results/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from changepoint.threshold import (
    CustomThresholdModel,
    ShapConfig,
    FEATURE_NAMES,
    FEATURE_ORDER,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_validate_data(
    excel_path: str, sheet_name: str = "Trial1"
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and validate the Excel data.

    Args:
        excel_path: Path to the Excel file
        sheet_name: Name of the sheet to read

    Returns:
        Tuple of (DataFrame with validated data, list of feature columns)

    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    logger.info(f"Loading data from {excel_path}, sheet: {sheet_name}")

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Validate required columns
    required_cols = ["timestep", "true_change_point", "traditional_sum_martingales"]

    # Check for individual traditional martingale features (0-7)
    feature_cols = []
    for i in range(8):
        col_name = f"individual_traditional_martingales_feature{i}"
        if col_name in df.columns:
            feature_cols.append(col_name)
        else:
            logger.warning(f"Missing feature column: {col_name}")

    if len(feature_cols) < 8:
        logger.warning(
            f"Only found {len(feature_cols)} out of 8 expected feature columns"
        )

    # Validate that we have the minimum required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not feature_cols:
        raise ValueError("No individual traditional martingale feature columns found")

    logger.info(f"Found {len(feature_cols)} feature columns: {feature_cols}")

    return df, feature_cols


def clean_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    sum_col: str = "traditional_sum_martingales",
) -> pd.DataFrame:
    """Clean the data by handling NaN values and other data quality issues.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        sum_col: Name of sum martingale column

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning...")

    # Create a copy to avoid modifying original data
    df_clean = df.copy()

    # Check for NaN values in key columns
    key_cols = feature_cols + [sum_col, "timestep", "true_change_point"]

    # Report NaN statistics
    nan_counts = {}
    for col in key_cols:
        if col in df_clean.columns:
            nan_count = df_clean[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = nan_count
                logger.warning(
                    f"Column '{col}' has {nan_count} NaN values out of {len(df_clean)} rows"
                )

    if nan_counts:
        logger.info(f"Total columns with NaN values: {len(nan_counts)}")

        # Strategy 1: Forward fill for time series data (martingales should be continuous)
        logger.info("Applying forward fill for feature columns...")
        for col in feature_cols:
            if col in df_clean.columns:
                # Use new pandas methods instead of deprecated fillna(method=)
                df_clean[col] = df_clean[col].ffill().bfill()

        # Handle sum martingale
        if sum_col in df_clean.columns:
            df_clean[sum_col] = df_clean[sum_col].ffill().bfill()

        # For categorical columns like true_change_point, fill with 0 (no change)
        if "true_change_point" in df_clean.columns:
            df_clean["true_change_point"] = df_clean["true_change_point"].fillna(0)

        # Check if any NaN values remain
        remaining_nan = {}
        for col in key_cols:
            if col in df_clean.columns:
                nan_count = df_clean[col].isna().sum()
                if nan_count > 0:
                    remaining_nan[col] = nan_count

        if remaining_nan:
            logger.warning(f"Remaining NaN values after cleaning: {remaining_nan}")
            # If still have NaN values, drop those rows
            logger.info("Dropping rows with remaining NaN values...")
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=key_cols)
            final_rows = len(df_clean)
            if final_rows < initial_rows:
                logger.warning(
                    f"Dropped {initial_rows - final_rows} rows due to NaN values"
                )
        else:
            logger.info("All NaN values successfully handled")
    else:
        logger.info("No NaN values found in key columns")

    # Validate that we still have data
    if len(df_clean) == 0:
        raise ValueError("No data remaining after cleaning")

    # Ensure numeric columns are properly typed
    for col in feature_cols + [sum_col]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Final check for any remaining NaN values after type conversion
    final_nan_check = df_clean[key_cols].isna().sum().sum()
    if final_nan_check > 0:
        logger.warning(
            f"Found {final_nan_check} NaN values after type conversion, dropping affected rows"
        )
        df_clean = df_clean.dropna(subset=key_cols)

    logger.info(f"Data cleaning complete. Final dataset: {len(df_clean)} rows")

    return df_clean


def extract_change_points(
    df: pd.DataFrame, timestep_col: str = "timestep"
) -> List[int]:
    """Extract true change points from the data.

    Args:
        df: DataFrame containing the data
        timestep_col: Name of the timestep column

    Returns:
        List of timesteps where changes occurred
    """
    if "true_change_point" not in df.columns:
        logger.warning("No true_change_point column found")
        return []

    # Find rows where true_change_point is 1
    change_point_rows = df[df["true_change_point"] == 1]
    change_points = change_point_rows[timestep_col].tolist()

    logger.info(f"Found {len(change_points)} true change points: {change_points}")
    return change_points


def get_feature_display_names(feature_cols: List[str]) -> List[str]:
    """Get display names for features using the FEATURE_NAMES mapping.

    Args:
        feature_cols: List of feature column names

    Returns:
        List of display names
    """
    display_names = []
    for i, col in enumerate(feature_cols):
        if i < len(FEATURE_ORDER):
            feature_key = FEATURE_ORDER[i]
            display_name = FEATURE_NAMES.get(feature_key, f"Feature {i}")
        else:
            display_name = f"Feature {i}"
        display_names.append(display_name)

    return display_names


def create_comprehensive_plot(
    df: pd.DataFrame,
    feature_cols: List[str],
    change_points: List[int],
    threshold: float,
    output_path: str,
    sum_col: str = "traditional_sum_martingales",
) -> None:
    """Create the main 3-panel comprehensive plot.

    Args:
        df: DataFrame with martingale data
        feature_cols: List of feature column names
        change_points: List of true change points
        threshold: Detection threshold
        output_path: Path to save the plot
        sum_col: Name of the sum martingale column
    """
    logger.info("Creating comprehensive 3-panel plot...")

    # Initialize the threshold model
    model = CustomThresholdModel(threshold=threshold)

    # Get timesteps and feature display names
    timesteps = df["timestep"].values
    feature_names = get_feature_display_names(feature_cols)

    # Setup plot style for publication (single column) - VERY COMPACT
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,  # Slightly smaller base font
            "axes.labelsize": 11,  # Axis labels
            "axes.titlesize": 12,  # Panel titles
            "legend.fontsize": 8,  # Smaller legend text
            "xtick.labelsize": 9,  # X-axis tick labels
            "ytick.labelsize": 9,  # Y-axis tick labels
            "figure.figsize": (6, 6),  # Much more compact height
            "figure.dpi": 300,  # Publication quality
            "lines.linewidth": 1.5,  # Slightly thinner lines
            "axes.linewidth": 1.0,  # Standard axes
            "grid.linewidth": 0.6,  # Thinner grid
        }
    )

    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    # Create figure with three panels - VERY COMPACT layout
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(6, 6),  # Very tight height
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.15},  # Very tight spacing
    )

    # Find detection points (threshold crossings) first - we'll need these for all panels
    detection_indices = []
    if sum_col in df.columns:
        for i in range(1, len(df)):
            if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
                detection_indices.append(i)

    # Calculate y-axis range for Panel 1 (martingales) to use for both Panel 1 and Panel 2
    martingale_data = []
    for col in feature_cols:
        if col in df.columns:
            martingale_data.extend(df[col].values)
    if sum_col in df.columns:
        martingale_data.extend(df[sum_col].values)

    if martingale_data:
        y_min = min(martingale_data)
        y_max = max(martingale_data)
        # Add some padding (5% on each side)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        shared_y_min = y_min - y_padding
        shared_y_max = y_max + y_padding
    else:
        shared_y_min, shared_y_max = 0, 100  # fallback

    # Panel 1: Martingale Values
    logger.info("Creating Panel 1: Martingale Values")

    # Create labels with martingale values at detection time
    martingale_labels = []
    if detection_indices:
        detection_idx = detection_indices[0]  # Use first detection point for values
        for i, col in enumerate(feature_cols):
            if col in df.columns:
                detection_value = df[col].iloc[detection_idx]
                martingale_labels.append(f"{feature_names[i]} ({detection_value:.2f})")
            else:
                martingale_labels.append(feature_names[i])

        # Sum martingale label with detection value
        if sum_col in df.columns:
            sum_detection_value = df[sum_col].iloc[detection_idx]
            sum_label = f"Sum Martingale ({sum_detection_value:.2f})"
        else:
            sum_label = "Sum Martingale"
    else:
        # No detection points, use regular labels
        martingale_labels = feature_names
        sum_label = "Sum Martingale"

    # Plot individual features with enhanced labels
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            axs[0].plot(
                timesteps,
                df[col],
                label=martingale_labels[i],
                color=colors[i],
                alpha=0.7,
                linewidth=1.2,
            )

    # Add sum martingale
    if sum_col in df.columns:
        axs[0].plot(timesteps, df[sum_col], label=sum_label, color="black", linewidth=2)

    # Add threshold
    axs[0].axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold})",
    )

    # Mark change points
    for cp in change_points:
        axs[0].axvline(x=cp, color="green", linestyle="--", alpha=0.8, linewidth=1.5)

    # Mark detection points
    for idx in detection_indices:
        if 0 <= idx < len(timesteps):
            dp = timesteps[idx]
            axs[0].axvline(
                x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
            )

    axs[0].set_ylabel("Martingale Value", fontsize=11)
    axs[0].legend(
        loc="upper right",
        fontsize=7,
        ncol=2,
        frameon=True,
        fancybox=False,
        shadow=False,
    )
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylim(shared_y_min, shared_y_max)  # Set shared y-axis range

    # Panel 2: SHAP Values
    logger.info("Creating Panel 2: SHAP Values")
    try:
        # Use the proper SHAP methods from threshold.py instead of manual linear regression
        logger.info(
            "Using CustomThresholdModel.compute_shap_from_dataframe() method..."
        )

        # Compute SHAP values using the threshold.py methods
        shap_values, shap_feature_names = model.compute_shap_from_dataframe(
            df=df,
            feature_cols=feature_cols,
            change_points=change_points,
            timesteps=timesteps,
            detection_index=detection_indices[0] if detection_indices else None,
            normalize=False,  # Use raw SHAP values, not normalized
        )

        # DIAGNOSTIC ANALYSIS: Compare with manual linear regression
        logger.info("=== DIAGNOSTIC ANALYSIS ===")

        # Manual linear regression for comparison
        X = df[feature_cols].values
        y = df[sum_col].values if sum_col in df.columns else np.sum(X, axis=1)

        from sklearn.linear_model import LinearRegression

        model_lr = LinearRegression(fit_intercept=False)
        model_lr.fit(X, y)

        logger.info(f"Manual linear regression coefficients: {model_lr.coef_}")

        # Check if sum martingale is exactly the sum of individual martingales
        manual_sum = np.sum(X, axis=1)
        sum_difference = np.abs(y - manual_sum)
        max_sum_diff = np.max(sum_difference)

        logger.info(f"Sum martingale vs manual sum of features:")
        logger.info(f"  - Max difference: {max_sum_diff:.10f}")

        if max_sum_diff < 1e-10:
            logger.info(
                "  ✓ Sum martingale IS EXACTLY the sum of individual martingales"
            )
            logger.info("  → Perfect linear additive relationship confirmed!")
        else:
            logger.info(
                f"  → Sum martingale is NOT exactly the sum (difference: {max_sum_diff:.6f})"
            )

        # Compare threshold.py SHAP vs manual linear regression SHAP
        manual_shap = np.zeros(X.shape)
        for i in range(len(feature_cols)):
            manual_shap[:, i] = X[:, i] * model_lr.coef_[i]

        # Since coefficients should be [1,1,1,1,1,1,1,1], manual SHAP should equal feature values
        max_manual_diff = np.max(np.abs(manual_shap - X))
        logger.info(
            f"Manual SHAP vs Feature values max difference: {max_manual_diff:.10f}"
        )

        # Compare threshold.py SHAP vs manual SHAP
        shap_comparison_diff = np.max(np.abs(shap_values - manual_shap))
        logger.info(
            f"threshold.py SHAP vs Manual SHAP max difference: {shap_comparison_diff:.10f}"
        )

        logger.info("=== END DIAGNOSTIC ===")

        # Calculate R² for display
        y_pred = model_lr.predict(X)
        from sklearn.metrics import r2_score

        r2 = r2_score(y, y_pred)
        logger.info(f"Linear model R² = {r2:.6f}")

        if r2 > 0.999:
            logger.info(
                "Near-perfect linear relationship: SHAP ≈ Martingale values (theoretical equivalence)"
            )
            if max_sum_diff < 1e-10:
                logger.info(
                    "EXPLANATION: Sum martingale = Σ(individual martingales) → Perfect R² expected!"
                )
        else:
            logger.info(
                f"Approximate linear relationship: SHAP approximates Martingale values (R² = {r2:.3f})"
            )

        # Verify that SHAP approximation sums correctly
        shap_sum = np.sum(shap_values, axis=1)
        y_actual = df[sum_col].values if sum_col in df.columns else np.sum(X, axis=1)
        max_shap_sum_diff = np.max(np.abs(shap_sum - y_actual))
        logger.info(
            f"SHAP sum validation: max difference = {max_shap_sum_diff:.6f} (should be ~0)"
        )

        # Create labels with SHAP values at detection time
        shap_labels = []
        if detection_indices:
            detection_idx = detection_indices[0]  # Use first detection point for values
            for i, col in enumerate(feature_cols):
                shap_detection_value = shap_values[detection_idx, i]
                shap_labels.append(f"{feature_names[i]} ({shap_detection_value:.2f})")
        else:
            # No detection points, use regular labels
            shap_labels = feature_names

        # Plot SHAP values with enhanced labels
        for i, col in enumerate(feature_cols):
            axs[1].plot(
                timesteps,
                shap_values[:, i],
                label=shap_labels[i],
                color=colors[i],
                alpha=0.7,
                linewidth=1.2,
            )

        # Mark change points and detection points
        for cp in change_points:
            axs[1].axvline(
                x=cp, color="green", linestyle="--", alpha=0.8, linewidth=1.5
            )
        for idx in detection_indices:
            if 0 <= idx < len(timesteps):
                dp = timesteps[idx]
                axs[1].axvline(
                    x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
                )

        # Enhanced title with R² information
        if r2 > 0.999:
            title_suffix = f" (R² = {r2:.4f} - Perfect Linear Model)"
        else:
            title_suffix = f" (R² = {r2:.3f} - Approximate)"

        axs[1].set_ylabel("SHAP Value", fontsize=11)
        axs[1].legend(
            loc="upper right",
            fontsize=7,
            ncol=2,
            frameon=True,
            fancybox=False,
            shadow=False,
        )
        axs[1].grid(True, alpha=0.3)
        axs[1].set_ylim(shared_y_min, shared_y_max)  # Use same y-axis range as Panel 1

        # Add text annotation about theoretical equivalence - COMPACT
        if r2 > 0.999 and max_sum_diff < 1e-10:
            annotation_text = "Perfect Equivalence:\nSHAP = Martingale"
            text_color = "green"
        elif r2 > 0.999:
            annotation_text = f"Near-Perfect:\nR²={r2:.4f}"
            text_color = "darkgreen"
        else:
            annotation_text = f"Approximation:\nR²={r2:.3f}"
            text_color = "orange"

        axs[1].text(
            0.02,
            0.98,
            annotation_text,
            transform=axs[1].transAxes,
            fontsize=9,  # Smaller annotation
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor=text_color,
                linewidth=1.0,
            ),
            color=text_color,
            fontweight="bold",
        )

    except Exception as e:
        logger.warning(f"Error computing SHAP values: {e}")
        axs[1].text(
            0.5,
            0.5,
            "SHAP analysis unavailable",
            transform=axs[1].transAxes,
            ha="center",
            va="center",
        )
        axs[1].set_title(
            "Panel 2: SHAP Analysis (Unavailable)", fontsize=12, fontweight="bold"
        )
        axs[1].set_ylim(shared_y_min, shared_y_max)  # Still set the same range

    # Panel 3: Classifier Output (Feature Contributions at Detection Points)
    logger.info("Creating Panel 3: Classifier Output")

    # Calculate feature contributions at detection points
    classifier_contributions = np.zeros(X.shape)

    if detection_indices:
        for detection_index in detection_indices:
            feature_values = X[detection_index]
            total = np.sum(feature_values)

            if total > 0:
                # Contributions as percentages
                for j, val in enumerate(feature_values):
                    classifier_contributions[detection_index, j] = val / total

                # Add decaying contributions around detection point
                window = 3
                for i in range(
                    max(0, detection_index - window),
                    min(len(df), detection_index + window + 1),
                ):
                    if i != detection_index:
                        decay = 0.3 ** abs(i - detection_index)
                        vals = X[i]
                        val_sum = np.sum(vals)
                        if val_sum > 0:
                            for j, val in enumerate(vals):
                                classifier_contributions[i, j] = (val / val_sum) * decay

    # Plot classifier contributions
    if detection_indices:
        # Create labels with percentages from detection points
        labels_with_percentages = []
        for i, col in enumerate(feature_cols):
            if detection_indices:
                idx = detection_indices[0]
                percentage = classifier_contributions[idx, i] * 100
                labels_with_percentages.append(
                    f"{feature_names[i]} ({percentage:.1f}%)"
                )
            else:
                labels_with_percentages.append(feature_names[i])

        for i, col in enumerate(feature_cols):
            axs[2].plot(
                timesteps,
                classifier_contributions[:, i],
                label=labels_with_percentages[i],
                color=colors[i],
                alpha=0.7,
                linewidth=1.2,
            )
    else:
        # If no detection points, show raw feature proportions
        for i, col in enumerate(feature_cols):
            if col in df.columns:
                proportions = df[col] / df[feature_cols].sum(axis=1)
                axs[2].plot(
                    timesteps,
                    proportions,
                    label=feature_names[i],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.2,
                )

    # Mark change points and detection points
    for cp in change_points:
        axs[2].axvline(x=cp, color="green", linestyle="--", alpha=0.8, linewidth=1.5)
    for idx in detection_indices:
        if 0 <= idx < len(timesteps):
            dp = timesteps[idx]
            axs[2].axvline(
                x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
            )

    axs[2].set_xlabel("Timestep", fontsize=11)
    axs[2].set_ylabel("Classifier Contribution", fontsize=11)
    axs[2].legend(
        loc="upper right",
        fontsize=7,
        ncol=2,
        frameon=True,
        fancybox=False,
        shadow=False,
    )
    axs[2].grid(True, alpha=0.3)

    # Set x-axis ticks to increments of 40
    max_timestep = timesteps[-1] if len(timesteps) > 0 else 200
    x_ticks = np.arange(0, max_timestep + 1, 40)
    axs[2].set_xticks(x_ticks)

    # Add legend for vertical lines - with proper padding to avoid overlap
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="True Change Point",
        ),
        Line2D(
            [0],
            [0],
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label="Detection Point",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),  # Add more padding below
        ncol=2,
        fontsize=9,
        frameon=False,  # Remove frame for compactness
    )

    # Save the plot with proper padding for bottom legend
    plt.subplots_adjust(
        bottom=0.10, top=0.97, hspace=0.15
    )  # More bottom margin for legend
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    logger.info(f"Comprehensive plot saved to: {output_path}")

    # Log summary statistics with clean integer formatting
    logger.info(f"Analysis Summary:")
    logger.info(f"  - Data points: {len(df)}")
    logger.info(
        f"  - True change points: {len(change_points)} at timesteps {change_points}"
    )
    # Convert numpy integers to regular integers for cleaner logging
    clean_detection_timesteps = [int(timesteps[i]) for i in detection_indices]
    logger.info(
        f"  - Detection points: {len(detection_indices)} at timesteps {clean_detection_timesteps}"
    )
    logger.info(f"  - Threshold: {threshold}")
    logger.info(
        f"  - Y-axis range (Panels 1&2): {shared_y_min:.2f} to {shared_y_max:.2f}"
    )

    # Log detection point values for reference
    if detection_indices:
        detection_idx = detection_indices[0]
        clean_detection_timestep = int(timesteps[detection_idx])
        logger.info(
            f"Values at first detection point (timestep {clean_detection_timestep}):"
        )

        # Calculate differences and show theoretical relationship
        total_martingale_diff = 0
        total_abs_diff = 0

        for i, col in enumerate(feature_cols):
            if col in df.columns:
                mart_val = df[col].iloc[detection_idx]
                if "shap_values" in locals():
                    shap_val = shap_values[detection_idx, i]
                    diff = abs(mart_val - shap_val)
                    total_martingale_diff += mart_val
                    total_abs_diff += diff

                    if diff < 0.001:
                        status = "✓ Perfect"
                    elif diff < 0.01:
                        status = "≈ Very Close"
                    else:
                        status = f"△ Diff={diff:.3f}"

                    logger.info(
                        f"  - {feature_names[i]}: Martingale={mart_val:.2f}, SHAP={shap_val:.2f} {status}"
                    )
                else:
                    logger.info(f"  - {feature_names[i]}: Martingale={mart_val:.2f}")

        # Show overall equivalence metrics
        if "shap_values" in locals():
            logger.info(f"Theoretical Relationship Validation:")
            if sum_col in df.columns:
                sum_martingale = df[sum_col].iloc[detection_idx]
                sum_shap = np.sum(shap_values[detection_idx, :])
                sum_diff = abs(sum_martingale - sum_shap)
                logger.info(f"  - Sum Martingale: {sum_martingale:.2f}")
                logger.info(f"  - Sum SHAP: {sum_shap:.2f}")
                logger.info(f"  - Sum Difference: {sum_diff:.6f}")

                if sum_diff < 0.001:
                    logger.info(
                        "  ✓ Perfect theoretical equivalence: Sum(SHAP) = Sum(Martingales)"
                    )
                else:
                    logger.info(
                        f"  ≈ Approximate equivalence: Sum difference = {sum_diff:.6f}"
                    )

            avg_relative_error = (
                (total_abs_diff / total_martingale_diff) * 100
                if total_martingale_diff > 0
                else 0
            )
            logger.info(f"  - Average relative error: {avg_relative_error:.4f}%")

            if avg_relative_error < 0.1:
                logger.info("  ✓ Excellent SHAP approximation quality")
            elif avg_relative_error < 1.0:
                logger.info("  ≈ Good SHAP approximation quality")
            else:
                logger.info("  △ Moderate SHAP approximation quality")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze martingale data using threshold-based classifier with SHAP"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input Excel file")
    parser.add_argument(
        "--sheet", "-s", default="Trial10", help="Sheet name to read (default: Trial1)"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=60.0,
        help="Detection threshold (default: 60.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/martingale_shap_equivalence",
        help="Output directory (default: results/martingale_shap_equivalence)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load and validate data
        df, feature_cols = load_and_validate_data(args.input, args.sheet)

        # Clean the data
        df_clean = clean_data(df, feature_cols)

        # Extract change points
        change_points = extract_change_points(df_clean)

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Create the main comprehensive plot
        output_path = os.path.join(
            args.output, "martingale_shap_classifier_analysis.png"
        )
        create_comprehensive_plot(
            df=df_clean,
            feature_cols=feature_cols,
            change_points=change_points,
            threshold=args.threshold,
            output_path=output_path,
        )

        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        print(f"📊 Main output: martingale_shap_classifier_analysis.png")
        print(f"🎯 Threshold used: {args.threshold}")
        print(
            f"📈 Features analyzed: {', '.join(get_feature_display_names(feature_cols))}"
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
