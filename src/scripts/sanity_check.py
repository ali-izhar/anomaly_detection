#!/usr/bin/env python3
"""Network Hyperparameter Analysis Sanity Check.
This script validates the results of hyperparameter analysis and checks theoretical bounds.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List


def load_analysis_results(network_name: str) -> Optional[pd.DataFrame]:
    """Load hyperparameter analysis results from CSV file."""
    csv_path = (
        f"results/hyperparameter_analysis/{network_name}_hyperparameter_analysis.csv"
    )
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def validate_probability_metrics(df: pd.DataFrame) -> Dict[str, bool]:
    """Check if probability metrics are in valid range [0, 1]."""
    probability_cols = [
        "trad_tpr",
        "horizon_tpr",
        "trad_missed_rate",
        "horizon_missed_rate",
        "trad_fpr",
        "horizon_fpr",
    ]

    results = {}
    for col in probability_cols:
        if col in df.columns:
            values = df[col].dropna()
            is_valid = ((values >= 0) & (values <= 1)).all()
            results[col] = is_valid
            num_invalid = ((values < 0) | (values > 1)).sum()
            if num_invalid > 0:
                print(f"Warning: {num_invalid} invalid values found in {col}")

    return results


def check_complementary_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Check if TPR + missed_rate = 1."""

    results = {}

    # Traditional detection
    if "trad_tpr" in df.columns and "trad_missed_rate" in df.columns:
        trad_sum = df["trad_tpr"] + df["trad_missed_rate"]
        mean_diff = abs(trad_sum.mean() - 1.0)
        max_diff = abs(trad_sum - 1.0).max()
        results["trad_mean_diff"] = mean_diff
        results["trad_max_diff"] = max_diff

        if mean_diff > 1e-10 or max_diff > 1e-10:
            print(
                f"Warning: trad_tpr + trad_missed_rate ≠ 1 (mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f})"
            )

    # Horizon detection
    if "horizon_tpr" in df.columns and "horizon_missed_rate" in df.columns:
        horizon_sum = df["horizon_tpr"] + df["horizon_missed_rate"]
        mean_diff = abs(horizon_sum.mean() - 1.0)
        max_diff = abs(horizon_sum - 1.0).max()
        results["horizon_mean_diff"] = mean_diff
        results["horizon_max_diff"] = max_diff

        if mean_diff > 1e-10 or max_diff > 1e-10:
            print(
                f"Warning: horizon_tpr + horizon_missed_rate ≠ 1 (mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f})"
            )

    return results


def check_doobs_ville_inequality(df: pd.DataFrame) -> Tuple[bool, float, List[int]]:
    """Check if FPR <= 1/threshold (Doob's/Ville's inequality)."""

    if "threshold" not in df.columns or "trad_fpr" not in df.columns:
        print(
            "Cannot check Doob's/Ville's inequality: missing threshold or trad_fpr columns"
        )
        return False, 0.0, []

    # Check if theoretical_bound is already calculated in the data
    if "theoretical_bound" in df.columns:
        print("Using pre-calculated theoretical bound from data")
        # Use the stored theoretical bound
        violations = df[df["trad_fpr"] > df["theoretical_bound"] + 1e-10]
    else:
        # Calculate theoretical upper bound (original method)
        df = df.copy()
        df["theoretical_bound"] = 1.0 / df["threshold"]

        # Check if FPR <= 1/threshold
        violations = df[df["trad_fpr"] > df["theoretical_bound"] + 1e-10]

    violation_count = len(violations)
    violation_percent = 100 * violation_count / len(df)

    if violation_count > 0:
        print(
            f"Warning: Doob's/Ville's inequality violated in {violation_count} cases ({violation_percent:.2f}%)"
        )
        violated_thresholds = violations["threshold"].unique().tolist()
        print(f"Violated thresholds: {sorted(violated_thresholds)}")

        # Show some example violations
        print("\nExample violations:")
        if "theoretical_bound" in df.columns:
            print(violations[["threshold", "trad_fpr", "theoretical_bound"]].head())
        else:
            bound_column = df["theoretical_bound"]
            print(
                violations[["threshold", "trad_fpr"]]
                .head()
                .assign(theoretical_bound=bound_column)
            )
    else:
        print(f"✓ Doob's/Ville's inequality satisfied for all {len(df)} configurations")

    return (
        violation_count == 0,
        violation_percent,
        violations["threshold"].unique().tolist() if violation_count > 0 else [],
    )


def check_horizon_delay_reduction(df: pd.DataFrame) -> Dict[str, float]:
    """Check if horizon detection reduces delay compared to traditional detection."""

    if (
        "traditional_avg_delay" not in df.columns
        or "horizon_avg_delay" not in df.columns
    ):
        print("Cannot check horizon delay reduction: missing delay columns")
        return {}

    # Calculate delay difference and percent reduction
    df_with_delays = df.dropna(subset=["traditional_avg_delay", "horizon_avg_delay"])
    df_with_delays = df_with_delays.copy()
    df_with_delays["delay_diff"] = (
        df_with_delays["traditional_avg_delay"] - df_with_delays["horizon_avg_delay"]
    )

    # Check percentage of cases where horizon provides improvement
    improved_count = (df_with_delays["delay_diff"] > 0).sum()
    improved_percent = 100 * improved_count / len(df_with_delays)

    # Mean delay reduction
    mean_reduction = df_with_delays["delay_diff"].mean()

    print(
        f"Horizon detection reduces delay in {improved_count}/{len(df_with_delays)} cases ({improved_percent:.2f}%)"
    )
    print(f"Mean delay reduction: {mean_reduction:.2f} timesteps")

    return {"improved_percent": improved_percent, "mean_reduction": mean_reduction}


def check_threshold_fpr_relationship(df: pd.DataFrame) -> None:
    """Check if higher thresholds correspond to lower FPRs."""

    if "threshold" not in df.columns or "trad_fpr" not in df.columns:
        print("Cannot check threshold-FPR relationship: missing columns")
        return

    # Group by threshold and calculate mean FPR
    threshold_groups = df.groupby("threshold")["trad_fpr"].mean().reset_index()
    threshold_groups = threshold_groups.sort_values("threshold")

    # Is FPR generally decreasing as threshold increases?
    is_monotonic = threshold_groups["trad_fpr"].is_monotonic_decreasing

    if is_monotonic:
        print("✓ FPR monotonically decreases as threshold increases")
    else:
        print("Warning: FPR does not monotonically decrease with increasing threshold")

    # Plot threshold vs. FPR
    plt.figure(figsize=(10, 6))
    plt.plot(
        threshold_groups["threshold"],
        threshold_groups["trad_fpr"],
        "o-",
        label="Mean FPR",
    )

    # Add theoretical bound
    thresholds = threshold_groups["threshold"].values
    theoretical_bound = 1.0 / thresholds
    plt.plot(thresholds, theoretical_bound, "r--", label="Theoretical Bound (1/λ)")

    plt.xlabel("Detection Threshold (λ)")
    plt.ylabel("False Positive Rate")
    plt.title("Threshold vs. FPR (with Theoretical Bound)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    os.makedirs("results/sanity_checks", exist_ok=True)
    plt.savefig("results/sanity_checks/threshold_fpr_relationship.png", dpi=300)
    plt.close()


def check_parameter_impact(df: pd.DataFrame) -> None:
    """Check impact of different parameters on performance metrics."""

    metrics = ["trad_tpr", "trad_fpr", "traditional_avg_delay"]
    parameters = [
        "threshold",
        "window",
        "horizon",
        "epsilon",
        "betting_func",
        "distance",
    ]

    parameter_impact = {}

    for param in parameters:
        if param not in df.columns:
            continue

        # Skip if parameter has only one unique value
        if len(df[param].unique()) <= 1:
            continue

        impact = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            # Group by parameter and calculate mean metric value
            if df[param].dtype == "object" or df[param].dtype == "category":
                # Categorical parameter
                grouped = df.groupby(param)[metric].mean()
                min_value = grouped.min()
                max_value = grouped.max()
                range_value = max_value - min_value
            else:
                # Numerical parameter
                # Use correlation instead
                correlation = df[[param, metric]].corr().iloc[0, 1]
                range_value = abs(correlation)

            impact[metric] = range_value

        parameter_impact[param] = impact

    # Print parameter impact summary
    print("\nParameter Impact Summary:")
    for param, metrics_impact in parameter_impact.items():
        print(f"\n{param}:")
        for metric, impact in metrics_impact.items():
            print(f"  - Impact on {metric}: {impact:.4f}")


def main():
    """Main entry point for sanity check script."""
    parser = argparse.ArgumentParser(
        description="Run sanity checks on hyperparameter analysis results."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Network type to check (e.g., sbm, ba, er, ws)",
    )

    args = parser.parse_args()

    # Load data
    df = load_analysis_results(args.network)
    if df is None:
        return

    print(
        f"Running sanity checks for {args.network} network data with {len(df)} configurations...\n"
    )

    # Run checks
    print("1. Validating probability metrics...")
    validate_probability_metrics(df)

    print("\n2. Checking complementary rates...")
    check_complementary_rates(df)

    print("\n3. Checking Doob's/Ville's inequality (FPR <= 1/threshold)...")
    check_doobs_ville_inequality(df)

    print("\n4. Checking threshold-FPR relationship...")
    check_threshold_fpr_relationship(df)

    print("\n5. Checking horizon delay reduction...")
    check_horizon_delay_reduction(df)

    print("\n6. Checking parameter impact...")
    check_parameter_impact(df)

    print("\nSanity checks completed.")


if __name__ == "__main__":
    main()
