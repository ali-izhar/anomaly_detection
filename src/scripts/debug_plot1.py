#!/usr/bin/env python3
"""Debug script to validate the betting function TPR comparison in the first plot."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# Betting function colors (same as original script)
BETTING_COLORS = {
    "beta": "#fd8d3c",  # Light orange
    "mixture": "#637939",  # Olive green
    "power": "#7b4173",  # Purple
    "normal": "#3182bd",  # Blue
}


def load_analysis_results(network_name="sbm"):
    """Load hyperparameter analysis results from CSV file."""
    csv_path = (
        f"results/hyperparameter_analysis/{network_name}_hyperparameter_analysis.csv"
    )
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def analyze_tpr_by_betting_function(df):
    """Analyze TPR values by betting function to debug first plot."""
    print("\n=== TPR ANALYSIS BY BETTING FUNCTION ===")

    # Overall statistics by betting function
    print("\nOverall Statistics:")
    tpr_by_betting = df.groupby("betting_func")[["trad_tpr", "horizon_tpr"]].agg(
        ["count", "mean", "median", "min", "max"]
    )
    print(tpr_by_betting)

    # Count perfect detection rates (TPR = 1.0)
    print("\nPerfect Detection Count (TPR = 1.0):")
    perfect_detection = df.groupby("betting_func").apply(
        lambda x: pd.Series(
            {
                "perfect_trad_count": (x["trad_tpr"] == 1.0).sum(),
                "perfect_trad_pct": (x["trad_tpr"] == 1.0).mean() * 100,
                "perfect_horizon_count": (x["horizon_tpr"] == 1.0).sum(),
                "perfect_horizon_pct": (x["horizon_tpr"] == 1.0).mean() * 100,
            }
        )
    )
    print(perfect_detection)

    # Analyze by distance metric
    print("\nTPR by Betting Function and Distance Metric:")
    tpr_by_dist = (
        df.groupby(["betting_func", "distance"])["trad_tpr"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    # Find best distance metric for each betting function
    best_dist = tpr_by_dist.loc[tpr_by_dist.groupby("betting_func")["mean"].idxmax()]
    print("\nBest Distance Metric for Each Betting Function:")
    print(best_dist[["betting_func", "distance", "mean"]])

    return tpr_by_betting, perfect_detection, tpr_by_dist


def visualize_tpr_comparison(df):
    """Create validation visualizations for TPR by betting function."""
    # Create figure with multiple plots for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 1. Boxplot (similar to original)
    sns.boxplot(
        x="betting_func",
        y="trad_tpr",
        data=df,
        ax=axes[0],
        palette=BETTING_COLORS,
        showfliers=False,
    )
    sns.stripplot(
        x="betting_func",
        y="trad_tpr",
        data=df,
        ax=axes[0],
        color="black",
        size=3,
        alpha=0.3,
        jitter=True,
    )
    axes[0].set_title("Traditional TPR Distribution\n(Boxplot with Data Points)")
    axes[0].set_xlabel("Betting Function")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].set_ylim(0, 1.05)

    # 2. Histogram/Distribution
    for i, bf in enumerate(sorted(df["betting_func"].unique())):
        subset = df[df["betting_func"] == bf]
        sns.histplot(
            subset["trad_tpr"],
            ax=axes[1],
            color=BETTING_COLORS.get(bf),
            alpha=0.6,
            label=bf.capitalize(),
            kde=True,
            stat="probability",
        )
    axes[1].set_title("TPR Distribution by Betting Function\n(Showing Bimodal Nature)")
    axes[1].set_xlabel("True Positive Rate (TPR)")
    axes[1].set_ylabel("Probability Density")
    axes[1].legend()

    # 3. Bar plot with confidence intervals by distance
    best_configs = (
        df.groupby(["betting_func", "distance"])["trad_tpr"].mean().reset_index()
    )
    best_configs = best_configs.pivot(
        index="distance", columns="betting_func", values="trad_tpr"
    )
    best_configs.plot(
        kind="bar",
        ax=axes[2],
        color=[BETTING_COLORS.get(bf) for bf in best_configs.columns],
        rot=45,
    )
    axes[2].set_title("TPR by Distance Metric\n(Shows Power's Strong Performance)")
    axes[2].set_xlabel("Distance Metric")
    axes[2].set_ylabel("Mean TPR")
    axes[2].set_ylim(0, 1.05)

    # Save figure
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig("debug_output/tpr_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved validation plot to debug_output/tpr_validation.png")


def main():
    """Main function to debug the first plot's TPR representation."""
    # Load and analyze data
    df = load_analysis_results()
    if df is None:
        return

    print(f"Loaded {len(df)} rows of hyperparameter analysis data")

    # Display basic info
    print("\nBetting functions in dataset:", df["betting_func"].unique())
    print("Distance metrics in dataset:", df["distance"].unique())

    # Analyze TPR by betting function
    tpr_stats, perfect_detection, tpr_by_dist = analyze_tpr_by_betting_function(df)

    # Create validation visualizations
    visualize_tpr_comparison(df)

    # Final summary
    print("\n=== SUMMARY ===")
    power_perf = tpr_stats.loc["power", "trad_tpr"]
    print(f"Power betting mean TPR: {power_perf['mean']:.4f}")
    print(
        f"Power betting perfect detection rate: {perfect_detection.loc['power', 'perfect_trad_pct']:.1f}%"
    )

    print("\nPossible issues with original plot:")
    print(
        "1. Data aggregation may not properly represent bimodal distribution (many 1.0 TPRs, some 0)"
    )
    print(
        "2. The boxplot visualization might be suppressing outliers, hiding perfect detection cases"
    )
    print(
        "3. The original plot may be filtering data in a way that excludes power's best configurations"
    )

    # Compare best configurations
    print("\nBest configurations for Power betting:")
    power_best = df[(df["betting_func"] == "power") & (df["trad_tpr"] == 1.0)]
    if not power_best.empty:
        top_configs = power_best.groupby(["distance"]).size().reset_index(name="count")
        print(top_configs)


if __name__ == "__main__":
    main()
