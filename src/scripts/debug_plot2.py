#!/usr/bin/env python3
"""Debug script to analyze and validate the detection accuracy grid plot."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

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

# Stream colors for Traditional vs Horizon
STREAM_COLORS = {
    "trad_color": "#2C5F7F",  # Deeper blue for traditional
    "horizon_color": "#FF8C38",  # Warmer orange for horizon
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


def analyze_fpr_scaling_issues(df):
    """Analyze FPR values to identify potential scaling issues in the plot."""
    print("\n=== FPR SCALING ANALYSIS ===")

    # Overall FPR statistics
    print("\nOverall FPR Statistics:")
    fpr_stats = df[["trad_fpr", "horizon_fpr"]].describe()
    print(fpr_stats)

    # FPR by betting function
    print("\nFPR Statistics by Betting Function:")
    betting_fpr = df.groupby("betting_func")[["trad_fpr", "horizon_fpr"]].agg(
        ["count", "mean", "median", "min", "max"]
    )
    print(betting_fpr)

    # Check if FPR values are very small
    small_fpr_count = ((df["trad_fpr"] < 0.001) | (df["horizon_fpr"] < 0.001)).sum()
    very_small_fpr_count = (
        (df["trad_fpr"] < 0.0001) | (df["horizon_fpr"] < 0.0001)
    ).sum()
    print(
        f"\nCount of very small FPR values (<0.001): {small_fpr_count}/{len(df)} rows ({small_fpr_count/len(df)*100:.1f}%)"
    )
    print(
        f"Count of extremely small FPR values (<0.0001): {very_small_fpr_count}/{len(df)} rows ({very_small_fpr_count/len(df)*100:.1f}%)"
    )

    return fpr_stats


def analyze_threshold_fpr_relationship(df):
    """Analyze the relationship between threshold and FPR to validate Ville's inequality."""
    print("\n=== THRESHOLD-FPR RELATIONSHIP ANALYSIS ===")

    thresholds = sorted(df["threshold"].unique())
    print(f"Thresholds in dataset: {thresholds}")

    # Calculate theoretical bounds (1/λ)
    theoretical_bounds = {float(t): 1 / float(t) for t in thresholds}
    print("\nTheoretical FPR Bounds:")
    for t, bound in theoretical_bounds.items():
        print(f"Threshold {int(t)}: bound = {bound:.6f}")

    # Calculate actual FPR by threshold
    threshold_df = df.groupby("threshold").agg(
        {"trad_fpr": ["mean", "max"], "horizon_fpr": ["mean", "max"]}
    )

    print("\nActual FPR vs Theoretical Bound:")
    for t in thresholds:
        t_float = float(t)
        bound = theoretical_bounds[t_float]
        trad_mean = threshold_df.loc[t, ("trad_fpr", "mean")]
        trad_max = threshold_df.loc[t, ("trad_fpr", "max")]
        hor_mean = threshold_df.loc[t, ("horizon_fpr", "mean")]
        hor_max = threshold_df.loc[t, ("horizon_fpr", "max")]

        trad_margin = bound - trad_mean
        hor_margin = bound - hor_mean

        print(f"Threshold {int(t)}:")
        print(f"  Bound: {bound:.6f}")
        print(f"  Traditional FPR (mean): {trad_mean:.6f} (margin: {trad_margin:.6f})")
        print(f"  Horizon FPR (mean): {hor_mean:.6f} (margin: {hor_margin:.6f})")
        print(f"  Maximum FPRs - Traditional: {trad_max:.6f}, Horizon: {hor_max:.6f}")

    return threshold_df


def analyze_distance_metrics(df):
    """Analyze impact of distance metrics on TPR and FPR."""
    print("\n=== DISTANCE METRIC ANALYSIS ===")

    distance_metrics = sorted(df["distance"].unique())
    print(f"Distance metrics in dataset: {distance_metrics}")

    # TPR and FPR by distance metric for each betting function
    dist_stats = (
        df.groupby(["betting_func", "distance"])
        .agg(
            {
                "trad_tpr": "mean",
                "horizon_tpr": "mean",
                "trad_fpr": "mean",
                "horizon_fpr": "mean",
            }
        )
        .reset_index()
    )

    # Reshape for easier comparison
    dist_wide = dist_stats.pivot(
        index=["betting_func", "distance"],
        columns=[],
        values=["trad_tpr", "horizon_tpr", "trad_fpr", "horizon_fpr"],
    )

    print("\nPerformance by Distance Metric:")
    print(dist_wide)

    # Find best distance metric for each betting function (based on TPR)
    print("\nBest Distance Metric by Betting Function (highest TPR):")
    best_dist = dist_stats.loc[dist_stats.groupby("betting_func")["trad_tpr"].idxmax()]
    print(best_dist[["betting_func", "distance", "trad_tpr", "trad_fpr"]])

    return dist_stats


def analyze_epsilon_impact(df):
    """Analyze how epsilon affects TPR and FPR for different betting functions."""
    print("\n=== EPSILON IMPACT ANALYSIS ===")

    # Filter out beta betting as it doesn't use epsilon
    epsilon_df = df[df["betting_func"] != "beta"].copy()

    if "epsilon" not in epsilon_df.columns:
        print("Epsilon parameter not found in dataset")
        return None

    epsilon_values = sorted(epsilon_df["epsilon"].unique())
    print(f"Epsilon values in dataset: {epsilon_values}")

    # Calculate metrics by epsilon and betting function
    epsilon_stats = (
        epsilon_df.groupby(["betting_func", "epsilon"])
        .agg(
            {
                "trad_tpr": "mean",
                "horizon_tpr": "mean",
                "trad_fpr": "mean",
                "horizon_fpr": "mean",
            }
        )
        .reset_index()
    )

    print("\nPerformance by Epsilon Value:")
    for bf in sorted(epsilon_df["betting_func"].unique()):
        bf_stats = epsilon_stats[epsilon_stats["betting_func"] == bf]
        print(f"\n{bf.capitalize()} Betting Function:")
        for _, row in bf_stats.iterrows():
            eps = row["epsilon"]
            print(f"  Epsilon {eps}:")
            print(
                f"    Traditional TPR: {row['trad_tpr']:.4f}, FPR: {row['trad_fpr']:.6f}"
            )
            print(
                f"    Horizon TPR: {row['horizon_tpr']:.4f}, FPR: {row['horizon_fpr']:.6f}"
            )

    # Calculate sensitivity (slope) of TPR and FPR to epsilon changes
    print("\nSensitivity to Epsilon Changes:")
    for bf in sorted(epsilon_df["betting_func"].unique()):
        bf_stats = epsilon_stats[epsilon_stats["betting_func"] == bf]
        if len(bf_stats) >= 2:
            # Simple linear regression slope calculation
            x = bf_stats["epsilon"].values
            y_tpr_trad = bf_stats["trad_tpr"].values
            y_tpr_hor = bf_stats["horizon_tpr"].values
            y_fpr_trad = bf_stats["trad_fpr"].values
            y_fpr_hor = bf_stats["horizon_fpr"].values

            # Calculate slope if we have multiple epsilon values
            if len(set(x)) > 1:
                tpr_trad_slope = np.polyfit(x, y_tpr_trad, 1)[0]
                tpr_hor_slope = np.polyfit(x, y_tpr_hor, 1)[0]
                fpr_trad_slope = np.polyfit(x, y_fpr_trad, 1)[0]
                fpr_hor_slope = np.polyfit(x, y_fpr_hor, 1)[0]

                print(f"\n{bf.capitalize()} Betting Function:")
                print(
                    f"  Traditional TPR sensitivity: {tpr_trad_slope:.4f} per epsilon unit"
                )
                print(
                    f"  Horizon TPR sensitivity: {tpr_hor_slope:.4f} per epsilon unit"
                )
                print(
                    f"  Traditional FPR sensitivity: {fpr_trad_slope:.6f} per epsilon unit"
                )
                print(
                    f"  Horizon FPR sensitivity: {fpr_hor_slope:.6f} per epsilon unit"
                )

    return epsilon_stats


def create_debug_plots(df, fpr_stats, threshold_df, dist_stats, epsilon_stats):
    """Create debug plots to visualize the issues in the detection accuracy grid."""
    os.makedirs("debug_output", exist_ok=True)

    # 1. Threshold vs FPR plot with appropriate scaling
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot theoretical bound
    thresholds = sorted(df["threshold"].unique())
    x = np.array(thresholds)
    theoretical_values = 1 / x
    ax.plot(
        x,
        theoretical_values,
        "k--",
        label="Theoretical bound (1/λ)",
        linewidth=2.5,
        alpha=0.8,
    )

    # Group by threshold and betting function
    for bf in sorted(df["betting_func"].unique()):
        subset = df[df["betting_func"] == bf]
        if not subset.empty:
            # Traditional FPR
            trad_threshold_groups = (
                subset.groupby("threshold")["trad_fpr"].mean().reset_index()
            )
            if not trad_threshold_groups.empty:
                ax.plot(
                    trad_threshold_groups["threshold"],
                    trad_threshold_groups["trad_fpr"],
                    "o-",
                    label=f"{bf.capitalize()} (Trad)",
                    color=BETTING_COLORS[bf],
                    linewidth=2.0,
                    markersize=6,
                    alpha=0.8,
                )

            # Horizon FPR
            hor_threshold_groups = (
                subset.groupby("threshold")["horizon_fpr"].mean().reset_index()
            )
            if not hor_threshold_groups.empty:
                ax.plot(
                    hor_threshold_groups["threshold"],
                    hor_threshold_groups["horizon_fpr"],
                    "s--",
                    label=f"{bf.capitalize()} (Horizon)",
                    color=BETTING_COLORS[bf],
                    linewidth=1.5,
                    markersize=5,
                    alpha=0.6,
                )

    # Configure y-axis with appropriate scaling
    min_fpr = df[["trad_fpr", "horizon_fpr"]].min().min()
    max_fpr = df[["trad_fpr", "horizon_fpr"]].max().max()

    # Use scientific notation for very small numbers
    if max_fpr < 0.01:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-4, 4))
        ax.yaxis.set_major_formatter(formatter)

    # Set appropriate limits
    ax.set_ylim(0, min(max(max_fpr * 1.2, 0.01), 0.05))

    ax.set_xlabel("Detection Threshold (λ)", fontsize=14)
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right", ncol=2)
    ax.set_title("Debug: Threshold vs FPR Relationship")

    plt.savefig("debug_output/threshold_fpr_debug.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. TPR by distance metric with both Traditional and Horizon
    if dist_stats is not None:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Reshape data for plotting
        plot_data = []
        for _, row in dist_stats.iterrows():
            # Add Traditional row
            plot_data.append(
                {
                    "betting_func": row["betting_func"],
                    "distance": row["distance"],
                    "Method": "Traditional",
                    "TPR": row["trad_tpr"],
                }
            )
            # Add Horizon row
            plot_data.append(
                {
                    "betting_func": row["betting_func"],
                    "distance": row["distance"],
                    "Method": "Horizon",
                    "TPR": row["horizon_tpr"],
                }
            )

        plot_df = pd.DataFrame(plot_data)

        # Create the grouped bar plot
        sns.barplot(
            x="distance",
            y="TPR",
            hue="Method",
            data=plot_df,
            palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
            saturation=0.8,
            ax=ax,
        )

        ax.set_xlabel("Distance Metric", fontsize=14)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_title("Debug: TPR by Distance Metric")
        ax.legend(title="Method")

        plt.savefig("debug_output/distance_tpr_debug.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3. Epsilon vs TPR and FPR plots
    if epsilon_stats is not None:
        # TPR plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for bf in sorted(epsilon_stats["betting_func"].unique()):
            bf_data = epsilon_stats[epsilon_stats["betting_func"] == bf]

            # Traditional TPR
            ax.plot(
                bf_data["epsilon"],
                bf_data["trad_tpr"],
                "o-",
                label=f"{bf.capitalize()} (Trad)",
                color=BETTING_COLORS[bf],
                linewidth=2.0,
                markersize=6,
                alpha=0.8,
            )

            # Horizon TPR
            ax.plot(
                bf_data["epsilon"],
                bf_data["horizon_tpr"],
                "s--",
                label=f"{bf.capitalize()} (Horizon)",
                color=BETTING_COLORS[bf],
                linewidth=1.5,
                markersize=5,
                alpha=0.6,
            )

        ax.set_xlabel("Epsilon (ε)", fontsize=14)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=10, loc="best", ncol=2)
        ax.set_title("Debug: Epsilon vs TPR")

        plt.savefig("debug_output/epsilon_tpr_debug.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # FPR plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for bf in sorted(epsilon_stats["betting_func"].unique()):
            bf_data = epsilon_stats[epsilon_stats["betting_func"] == bf]

            # Traditional FPR
            ax.plot(
                bf_data["epsilon"],
                bf_data["trad_fpr"],
                "o-",
                label=f"{bf.capitalize()} (Trad)",
                color=BETTING_COLORS[bf],
                linewidth=2.0,
                markersize=6,
                alpha=0.8,
            )

            # Horizon FPR
            ax.plot(
                bf_data["epsilon"],
                bf_data["horizon_fpr"],
                "s--",
                label=f"{bf.capitalize()} (Horizon)",
                color=BETTING_COLORS[bf],
                linewidth=1.5,
                markersize=5,
                alpha=0.6,
            )

        ax.set_xlabel("Epsilon (ε)", fontsize=14)
        ax.set_ylabel("False Positive Rate (FPR)", fontsize=14)

        # Use scientific notation for very small numbers
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-4, 4))
        ax.yaxis.set_major_formatter(formatter)

        # Set appropriate limits
        min_fpr = epsilon_stats[["trad_fpr", "horizon_fpr"]].min().min()
        max_fpr = epsilon_stats[["trad_fpr", "horizon_fpr"]].max().max()
        ax.set_ylim(0, min(max(max_fpr * 1.2, 0.01), 0.05))

        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=10, loc="best", ncol=2)
        ax.set_title("Debug: Epsilon vs FPR")

        plt.savefig("debug_output/epsilon_fpr_debug.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Debug plots saved to debug_output/ directory")


def main():
    """Main function to debug the detection accuracy grid plot."""
    # Load the data
    df = load_analysis_results()
    if df is None:
        return

    print(f"Loaded {len(df)} rows of hyperparameter analysis data")

    # Run analyses
    fpr_stats = analyze_fpr_scaling_issues(df)
    threshold_df = analyze_threshold_fpr_relationship(df)
    dist_stats = analyze_distance_metrics(df)
    epsilon_stats = analyze_epsilon_impact(df)

    # Create debug plots
    create_debug_plots(df, fpr_stats, threshold_df, dist_stats, epsilon_stats)

    # Final summary
    print("\n=== SUMMARY OF POTENTIAL ISSUES ===")
    print("1. Y-axis scaling issues for FPR plots:")
    if df["trad_fpr"].min() < 0.001 or df["horizon_fpr"].min() < 0.001:
        print("   - Very small FPR values might be causing y-axis labeling issues")
        print(
            f"   - Range of FPR values: {df['trad_fpr'].min():.8f} to {df['trad_fpr'].max():.8f}"
        )

    print("\n2. Epsilon impact observations:")
    if "epsilon" in df.columns:
        for bf in sorted(df[df["betting_func"] != "beta"]["betting_func"].unique()):
            bf_data = df[df["betting_func"] == bf]
            tpr_corr = bf_data["epsilon"].corr(bf_data["trad_tpr"])
            fpr_corr = bf_data["epsilon"].corr(bf_data["trad_fpr"])
            print(
                f"   - {bf.capitalize()}: TPR correlation with epsilon: {tpr_corr:.4f}"
            )
            print(f"     FPR correlation with epsilon: {fpr_corr:.4f}")

    print("\n3. Distance metric observations:")
    if dist_stats is not None and not dist_stats.empty:
        best_dist = dist_stats.loc[dist_stats["trad_tpr"].idxmax()]["distance"]
        worst_dist = dist_stats.loc[dist_stats["trad_tpr"].idxmin()]["distance"]
        print(f"   - Best distance metric for mean TPR: {best_dist}")
        print(f"   - Worst distance metric for mean TPR: {worst_dist}")


if __name__ == "__main__":
    main()
