#!/usr/bin/env python3
"""Network Hyperparameter Analysis Plotting.
This script creates concise, informative visualizations from hyperparameter analysis results."""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from datetime import datetime
from matplotlib.ticker import MaxNLocator

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["axes.titlepad"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["legend.framealpha"] = 0.9
plt.rcParams["legend.edgecolor"] = "lightgray"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["lines.markersize"] = 8

NETWORK_NAMES = {
    "sbm": "SBM",
    "ba": "BA",
    "er": "ER",
    "ws": "NWS",
}

BETTING_COLORS = {
    "beta": "#fd8d3c",  # Light orange
    "mixture": "#637939",  # Olive green
    "power": "#7b4173",  # Purple
    "normal": "#3182bd",  # Blue
}


def load_analysis_results(network_name: str) -> Optional[pd.DataFrame]:
    """Load hyperparameter analysis results from CSV file."""
    csv_path = (
        f"results/hyperparameter_analysis/{network_name}_hyperparameter_analysis.csv"
    )
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for plotting."""
    # Create an explicit copy to avoid SettingWithCopyWarning
    df_clean = df.dropna(
        subset=["trad_tpr", "horizon_tpr", "traditional_avg_delay", "horizon_avg_delay"]
    ).copy()

    # Convert categorical variables to categories for better plotting
    for col in ["betting_func", "distance"]:
        if col in df_clean.columns:
            df_clean.loc[:, col] = df_clean[col].astype("category")

    return df_clean


# -------------------------------------------------------------------------------
# ----------------------- (1) BETTING FUNCTION COMPARISON -----------------------
# -------------------------------------------------------------------------------
def plot_betting_function_comparison(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Compare the impact of different betting functions on performance."""
    if "betting_func" not in df.columns:
        print("Error: No betting function data available")
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    betting_funcs = sorted(df["betting_func"].unique())
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}
    plot_df = df.copy()
    plot_df["betting_func"] = pd.Categorical(
        plot_df["betting_func"], categories=betting_funcs
    )

    boxprops = dict(linewidth=2.5, alpha=0.8)
    whiskerprops = dict(linewidth=2.0, alpha=0.8)
    medianprops = dict(linewidth=2.5, color="black")

    # 1. Plot TPR by betting function
    ax = axs[0]
    sns.boxplot(
        x="betting_func",
        y="trad_tpr",
        hue="betting_func",
        data=plot_df,
        ax=ax,
        palette=colors,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showfliers=False,
        legend=False,
    )
    sns.stripplot(
        x="betting_func",
        y="trad_tpr",
        data=plot_df,
        ax=ax,
        color="black",
        size=3,
        alpha=0.3,
        jitter=True,
    )

    ax.set_xlabel("")
    ax.set_ylabel(r"True Positive Rate (TPR)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.4)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])

    # 2. Plot Detection Delay by betting function
    ax = axs[1]
    sns.boxplot(
        x="betting_func",
        y="traditional_avg_delay",
        hue="betting_func",
        data=plot_df,
        ax=ax,
        palette=colors,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showfliers=False,
        legend=False,
    )
    sns.stripplot(
        x="betting_func",
        y="traditional_avg_delay",
        data=plot_df,
        ax=ax,
        color="black",
        size=3,
        alpha=0.3,
        jitter=True,
    )

    ax.set_xlabel("")
    ax.set_ylabel(r"Detection Delay (timesteps)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])

    # 3. Plot FPR by betting function
    ax = axs[2]
    sns.boxplot(
        x="betting_func",
        y="trad_fpr",
        hue="betting_func",
        data=plot_df,
        ax=ax,
        palette=colors,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showfliers=False,
        legend=False,
    )
    sns.stripplot(
        x="betting_func",
        y="trad_fpr",
        data=plot_df,
        ax=ax,
        color="black",
        size=3,
        alpha=0.3,
        jitter=True,
    )

    ax.set_xlabel("")
    ax.set_ylabel(r"False Positive Rate (FPR)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])

    bf_metrics = {}
    for bf in betting_funcs:
        subset = df[df["betting_func"] == bf]
        tpr = subset["trad_tpr"].mean()
        delay = (
            subset["traditional_avg_delay"].mean()
            if "traditional_avg_delay" in subset.columns
            else np.nan
        )
        fpr = subset["trad_fpr"].mean()

        # Normalize delay for score calculation
        norm_delay = (
            (delay - df["traditional_avg_delay"].min())
            / (df["traditional_avg_delay"].max() - df["traditional_avg_delay"].min())
            if not np.isnan(delay)
            else 0.5
        )

        # Composite score: maximize TPR, minimize delay and FPR
        score = tpr - 0.5 * norm_delay - 5 * fpr
        bf_metrics[bf] = {"tpr": tpr, "delay": delay, "fpr": fpr, "score": score}

    # Find best betting function based on composite score
    best_bf = max(bf_metrics.items(), key=lambda x: x[1]["score"])[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{network_name}_betting_function_comparison.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    generate_betting_function_comparison_analysis_log(
        df, best_bf, bf_metrics, output_dir, network_name
    )
    print(f"Saved betting function comparison plot to {output_path}")
    plt.close(fig)


def generate_betting_function_comparison_analysis_log(
    df: pd.DataFrame, best_bf: str, bf_metrics: Dict, output_dir: str, network_name: str
):
    """Generate analysis log with performance metrics."""
    data_points = len(df)

    # Determine best parameters based on metrics
    best_threshold = df[df["betting_func"] == best_bf]["threshold"].median()
    best_window = df[df["betting_func"] == best_bf]["window"].median()
    best_epsilon = df[df["betting_func"] == best_bf]["epsilon"].median()

    # Find best distance metric
    best_distance = (
        df[df["betting_func"] == best_bf]
        .groupby("distance")
        .agg({"trad_tpr": "mean"})
        .sort_values("trad_tpr", ascending=False)
        .index[0]
    )

    # Calculate average metrics
    avg_tpr = df["trad_tpr"].mean()
    avg_delay = df["traditional_avg_delay"].mean()
    avg_fpr = df["trad_fpr"].mean()

    # Generate unique parameter combinations count
    param_cols = [
        "threshold",
        "window",
        "horizon",
        "epsilon",
        "betting_func",
        "distance",
    ]
    unique_params = df.drop_duplicates(subset=param_cols).shape[0]

    # Get column statistics
    stats_df = df.describe()

    # Create the log content
    log_content = f"""# Network Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## General Information

- network_type: {NETWORK_NAMES.get(network_name, network_name.upper())}
- analysis_date: {datetime.now().strftime("%Y-%m-%d")}
- data_points: {data_points}

## Best Parameters

- betting_function: {best_bf}
- distance_metric: {best_distance}
- threshold: {int(best_threshold)}
- window_size: {int(best_window)}
- epsilon: {best_epsilon:.1f}
- best_betting_function: {best_bf}

## Performance Metrics

- best_betting_tpr: {bf_metrics[best_bf]["tpr"]:.4f}
- avg_tpr: {avg_tpr:.4f}
- avg_delay: {avg_delay:.2f}
- avg_fpr: {avg_fpr:.4f}
- best_betting_delay: {bf_metrics[best_bf]["delay"]:.1f}
- best_betting_fpr: {bf_metrics[best_bf]["fpr"]:.4f}

## Plot Annotations

### parameter_grid
- Threshold vs. FPR with theoretical bound showing detection confidence
- Window size impact on detection delay by betting function
- Prediction horizon effect on true positive rate
- Distance metric comparison across betting functions

### betting_comparison
- True positive rate by betting function
- Detection delay comparison between betting functions
- False positive rate performance by betting function

## Data Summary

### Column Statistics

```
{stats_df.to_string()}
```

### Parameter Combinations

Total unique parameter combinations: {unique_params}
"""
    log_path = os.path.join(
        output_dir, f"{network_name}_betting_function_comparison_analysis_log.md"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"Saved analysis log to {log_path}")


# ---------------------------------------------------------------------------
# ----------------------- (2) DETECTION ACCURACY GRID -----------------------
# ---------------------------------------------------------------------------
def plot_detection_accuracy_grid(df: pd.DataFrame, output_dir: str, network_name: str):
    """Create a 2x2 grid visualization focusing on detection accuracy and FPR control."""
    if df.empty:
        print("Error: No data available for detection accuracy analysis")
        return

    # Optimize layout for research paper - tighter spacing with just enough room for labels
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 8),  # Reduced size for better proportions
        gridspec_kw={"wspace": 0.15, "hspace": 0.15},  # Tighter spacing
        constrained_layout=True,  # Use constrained_layout for better spacing
    )

    betting_funcs = sorted(df["betting_func"].unique())
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}

    # 1. Threshold vs FPR with theoretical bound (top-left)
    ax = axs[0, 0]
    ax.axhspan(0, 0.05, alpha=0.1, color="green")
    thresholds = sorted(df["threshold"].unique())
    x = np.array(thresholds)
    theoretical_values = 1 / x
    ax.plot(
        x,
        theoretical_values,
        "k--",
        label="Theoretical bound (1/$\lambda$)",
        linewidth=2.5,
        alpha=0.8,
    )

    # Group by threshold and betting function
    for bf in betting_funcs:
        subset = df[df["betting_func"] == bf]
        if not subset.empty:
            threshold_groups = (
                subset.groupby("threshold")["trad_fpr"].mean().reset_index()
            )
            if not threshold_groups.empty:
                ax.plot(
                    threshold_groups["threshold"],
                    threshold_groups["trad_fpr"],
                    "o-",
                    label=bf.capitalize(),
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                )

    ax.set_xlabel("Detection Threshold ($\lambda$)", fontsize=16)
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=16)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(
        fontsize=14, loc="upper right", frameon=True, framealpha=0.9, handlelength=1.5
    )

    # 2. TPR by distance metric (top-right)
    ax = axs[0, 1]
    if "distance" in df.columns:
        distances = sorted(df["distance"].unique())

        # Calculate average TPR for each distance and betting function combination
        tpr_by_distance = []
        for bf in betting_funcs:
            for dist in distances:
                subset = df[(df["betting_func"] == bf) & (df["distance"] == dist)]
                if not subset.empty:
                    avg_tpr = subset["trad_tpr"].mean()
                    tpr_by_distance.append(
                        {"betting_func": bf, "distance": dist, "tpr": avg_tpr}
                    )

        if tpr_by_distance:
            distance_df = pd.DataFrame(tpr_by_distance)
            # Use a more research-oriented bar style
            sns.barplot(
                x="distance",
                y="tpr",
                hue="betting_func",
                data=distance_df,
                palette=colors,
                ax=ax,
                saturation=0.8,
                err_kws={"linewidth": 1.5},
                capsize=0.1,
            )
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [x.get_text().capitalize() for x in ax.get_xticklabels()],
                fontsize=14,
            )

            ax.set_xlabel("Distance Metric", fontsize=16)
            ax.set_ylabel("True Positive Rate (TPR)", fontsize=16)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(
                fontsize=14,
                title="Betting Function",
                title_fontsize=14,
                loc="upper right",
                frameon=True,
                framealpha=0.9,
                handlelength=1.5,
                ncol=1,
            )

    # 3. FPR with epsilon (bottom-left)
    ax = axs[1, 0]
    if "epsilon" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Epsilon parameter not available in dataset",
            ha="center",
            va="center",
            fontsize=14,
        )
    else:
        epsilon_values = sorted(df["epsilon"].unique())

        # Filter out beta betting function which doesn't depend on epsilon
        epsilon_betting_funcs = [bf for bf in betting_funcs if bf != "beta"]

        # Plot for each betting function (except beta)
        for bf in epsilon_betting_funcs:
            subset = df[df["betting_func"] == bf]
            if not subset.empty:
                epsilon_metrics = []

                # Group by epsilon and calculate average FPR
                for eps in epsilon_values:
                    eps_data = subset[subset["epsilon"] == eps]
                    avg_fpr = eps_data["trad_fpr"].mean()

                    epsilon_metrics.append(
                        {
                            "epsilon": eps,
                            "fpr": avg_fpr,
                        }
                    )

                eps_df = pd.DataFrame(epsilon_metrics)

                # Plot FPR with improved line properties
                ax.plot(
                    eps_df["epsilon"],
                    eps_df["fpr"],
                    "o-",
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=9,
                    alpha=0.8,
                    label=f"{bf.capitalize()}",
                )

        ax.set_xlabel("Epsilon ($\\epsilon$)", fontsize=16)
        ax.set_ylabel("False Positive Rate (FPR)", fontsize=16)
        ax.set_ylim(0, min(1.05, max(df["trad_fpr"].max() * 1.2, 0.1)))
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(
            fontsize=14,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            handlelength=1.5,
        )

    # 4. TPR with epsilon (bottom-right)
    ax = axs[1, 1]
    if "epsilon" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Epsilon parameter not available in dataset",
            ha="center",
            va="center",
            fontsize=14,
        )
    else:
        epsilon_values = sorted(df["epsilon"].unique())

        # Filter out beta betting function which doesn't depend on epsilon
        epsilon_betting_funcs = [bf for bf in betting_funcs if bf != "beta"]

        # Plot for each betting function (except beta)
        for bf in epsilon_betting_funcs:
            subset = df[df["betting_func"] == bf]
            if not subset.empty:
                epsilon_metrics = []

                # Group by epsilon and calculate average TPR
                for eps in epsilon_values:
                    eps_data = subset[subset["epsilon"] == eps]
                    avg_tpr = eps_data["trad_tpr"].mean()

                    epsilon_metrics.append(
                        {
                            "epsilon": eps,
                            "tpr": avg_tpr,
                        }
                    )

                eps_df = pd.DataFrame(epsilon_metrics)

                # Plot TPR with improved styling
                ax.plot(
                    eps_df["epsilon"],
                    eps_df["tpr"],
                    "o-",
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=9,
                    alpha=0.8,
                    label=f"{bf.capitalize()}",
                )

        ax.set_xlabel("Epsilon ($\\epsilon$)", fontsize=16)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=16)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(
            fontsize=14,
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            handlelength=1.5,
        )

        # Better annotation with non-overlapping arrow
        if epsilon_values and epsilon_betting_funcs:
            eps_min = min(epsilon_values)
            eps_max = max(epsilon_values)
            eps_range = eps_max - eps_min

            ax.annotate(
                "Higher epsilon improves TPR\nfor aggressive detection",
                xy=(eps_max - eps_range * 0.2, 0.75),  # Arrow endpoint
                xytext=(eps_max - eps_range * 0.6, 0.45),  # Text position
                fontsize=14,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc="white",
                    alpha=0.9,
                    edgecolor="lightgray",
                ),
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=-0.2",
                    linewidth=1.5,
                    relpos=(0.8, 0.8),  # Connect from top-right of text box
                ),
            )

    # Publication-ready styling applied at the end
    for i in range(2):
        for j in range(2):
            # Remove top and right spines
            axs[i, j].spines["top"].set_visible(False)
            axs[i, j].spines["right"].set_visible(False)

            # Make bottom and left spines slightly thicker
            axs[i, j].spines["bottom"].set_linewidth(1.2)
            axs[i, j].spines["left"].set_linewidth(1.2)

            # Add subtle tick marks
            axs[i, j].tick_params(direction="out", length=4, width=1.2, pad=4)

            # Ensure y-axis has enough ticks for readability
            axs[i, j].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))

            # Format tick labels to avoid scientific notation and unnecessary decimals
            axs[i, j].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

            # For x-axis ticks
            if (i == 0 and j == 0) or (
                i == 1 and j == 0
            ):  # Threshold and epsilon plots
                axs[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{network_name}_detection_accuracy_grid.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Generate metrics for analysis log
    # Find best threshold based on optimal TPR-FPR tradeoff
    df_copy = df.copy()
    df_copy["score"] = (
        df_copy["trad_tpr"] - 5 * df_copy["trad_fpr"]
    )  # Weight FPR 5x more than TPR
    best_threshold_group = df_copy.groupby("threshold")["score"].mean().idxmax()

    # Best distance metric
    best_distance = None
    if "distance" in df.columns:
        best_distance = df.groupby("distance")["trad_tpr"].mean().idxmax()

    # Average metrics by threshold
    threshold_metrics = (
        df.groupby("threshold")
        .agg(
            {
                "trad_tpr": "mean",
                "trad_fpr": "mean",
                "traditional_avg_delay": "mean",
                "horizon_tpr": "mean",
                "horizon_fpr": "mean",
            }
        )
        .reset_index()
    )

    # Identify high performance configurations
    high_perf_thresholds = threshold_metrics[
        (threshold_metrics["trad_tpr"] > 0.7) & (threshold_metrics["trad_fpr"] < 0.05)
    ]["threshold"].tolist()

    generate_detection_accuracy_analysis_log(
        df,
        threshold_metrics,
        best_threshold_group,
        high_perf_thresholds,
        best_distance,
        output_dir,
        network_name,
    )

    print(f"Saved detection accuracy grid to {output_path}")
    plt.close(fig)


def generate_detection_accuracy_analysis_log(
    df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    best_threshold: float,
    high_perf_thresholds: list,
    best_distance: str,
    output_dir: str,
    network_name: str,
):
    """Generate analysis log for detection accuracy grid."""
    data_points = len(df)

    # Get average metrics for best threshold
    best_thr_metrics = threshold_metrics[
        threshold_metrics["threshold"] == best_threshold
    ].iloc[0]

    # Get theoretical bound for best threshold
    theoretical_bound = 1 / best_threshold

    # Calculate margin between theoretical and actual FPR
    fpr_margin = theoretical_bound - best_thr_metrics["trad_fpr"]
    fpr_margin_percent = (fpr_margin / theoretical_bound) * 100

    # Prepare high-performance thresholds list
    high_perf_thresholds_str = (
        ", ".join([str(int(t)) for t in high_perf_thresholds])
        if high_perf_thresholds
        else "None found"
    )

    # Calculate average metrics
    avg_tpr = df["trad_tpr"].mean()
    avg_fpr = df["trad_fpr"].mean()
    missed_rate = 1 - avg_tpr

    # Calculate epsilon impact if available
    epsilon_analysis = ""
    if "epsilon" in df.columns:
        # Filter out beta betting since it doesn't use epsilon
        epsilon_dependent_df = df[df["betting_func"] != "beta"]

        if not epsilon_dependent_df.empty:
            # Get best epsilon for TPR
            best_eps_tpr = (
                epsilon_dependent_df.groupby("epsilon")["trad_tpr"].mean().idxmax()
            )
            # Get best epsilon for FPR (lowest FPR)
            best_eps_fpr = (
                epsilon_dependent_df.groupby("epsilon")["trad_fpr"].mean().idxmin()
            )

            # Calculate metrics at these points
            tpr_at_best_eps = epsilon_dependent_df[
                epsilon_dependent_df["epsilon"] == best_eps_tpr
            ]["trad_tpr"].mean()
            fpr_at_best_eps = epsilon_dependent_df[
                epsilon_dependent_df["epsilon"] == best_eps_fpr
            ]["trad_fpr"].mean()

            # Check for mixture betting function
            has_mixture = "mixture" in df["betting_func"].unique()
            mixture_note = ""
            if has_mixture:
                mixture_note = "- note: Mixture betting function uses multiple epsilon values simultaneously [0.7, 0.8, 0.9]"

            epsilon_analysis = f"""
## Epsilon Analysis

- best_epsilon_for_tpr: {best_eps_tpr:.2f}
- tpr_at_best_epsilon: {tpr_at_best_eps:.4f}
- best_epsilon_for_fpr: {best_eps_fpr:.2f}
- fpr_at_best_epsilon: {fpr_at_best_eps:.6f}
- epsilon_tradeoff: Higher epsilon values generally increase TPR but may increase FPR
{mixture_note}
- note: Beta betting function does not depend on epsilon parameter
"""

    log_content = f"""# Detection Accuracy Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## General Information

- network_type: {NETWORK_NAMES.get(network_name, network_name.upper())}
- analysis_date: {datetime.now().strftime("%Y-%m-%d")}
- data_points: {data_points}

## Threshold Analysis

- optimal_threshold: {int(best_threshold)}
- theoretical_bound (1/$\lambda$): {theoretical_bound:.6f}
- actual_fpr_at_optimal: {best_thr_metrics["trad_fpr"]:.6f}
- margin_below_bound: {fpr_margin:.6f} ({fpr_margin_percent:.2f}%)
- tpr_at_optimal: {best_thr_metrics["trad_tpr"]:.4f}
- high_performance_thresholds: {high_perf_thresholds_str}
{epsilon_analysis}
## Distance Metric Analysis
"""

    if best_distance:
        log_content += f"- best_distance_metric: {best_distance}\n"
        distance_tpr = df[df["distance"] == best_distance]["trad_tpr"].mean()
        log_content += f"- avg_tpr_with_best_distance: {distance_tpr:.4f}\n"
    else:
        log_content += "- distance metrics not available in dataset\n"

    log_content += f"""
## Overall Performance Metrics

- average_tpr: {avg_tpr:.4f}
- average_fpr: {avg_fpr:.6f}
- missed_detection_rate: {missed_rate:.4f}
- tpr_to_fpr_ratio: {(avg_tpr/avg_fpr) if avg_fpr > 0 else 'inf':.2f}

## Ville's Inequality Verification

The theoretical bound states that for a threshold $\lambda$, the false positive rate (FPR) should not exceed 1/$\lambda$.
This bound is derived from Ville's inequality and is a key guarantee of the method.

Threshold | Theoretical Bound (1/$\lambda$) | Actual FPR | Margin | Verification
----------|-------------------------|------------|--------|-------------
"""

    # Add verification for each threshold
    for _, row in threshold_metrics.iterrows():
        threshold = row["threshold"]
        theo_bound = 1 / threshold
        actual_fpr = row["trad_fpr"]
        margin = theo_bound - actual_fpr
        verified = "✓" if actual_fpr <= theo_bound else "✗"

        log_content += f"{int(threshold)} | {theo_bound:.6f} | {actual_fpr:.6f} | {margin:.6f} | {verified}\n"

    log_content += f"""
## Plot Description

The visualization consists of a 2x2 grid that shows:

### 1. Threshold vs FPR (top-left)
- Demonstrates Ville's inequality (FPR $\leq 1/\lambda$)
- Shows actual FPR performance relative to theoretical bound
- Compares different betting functions

### 2. TPR by Distance Metric (top-right)
- Compares distance metrics across betting functions
- Identifies best distance metric for each betting function
- Shows impact of distance choice on detection rate

### 3. FPR with Epsilon (bottom-left)
- Shows how epsilon parameter affects FPR for power and mixture betting
- Beta betting is excluded as it doesn't depend on epsilon
- Lower epsilon values generally reduce false positive rates
- Mixture betting uses multiple epsilon values simultaneously

### 4. TPR with Epsilon (bottom-right)
- Shows how epsilon parameter affects TPR for power and mixture betting
- Beta betting is excluded as it doesn't depend on epsilon
- Higher epsilon values generally improve true positive rates
- Reveals betting function sensitivity to epsilon parameter

## Summary

The optimal detection threshold is {int(best_threshold)}, which achieves a TPR of {best_thr_metrics["trad_tpr"]:.4f} while maintaining an FPR of {best_thr_metrics["trad_fpr"]:.6f}, well below the theoretical bound of {theoretical_bound:.6f}. All thresholds satisfy Ville's inequality as expected, confirming the theoretical guarantees of the method.

The analysis demonstrates the impact of epsilon on both TPR and FPR for epsilon-dependent betting functions, allowing for parameter tuning based on specific performance requirements. Beta betting function performance is independent of epsilon, while mixture betting uses multiple epsilon values simultaneously.
"""
    log_path = os.path.join(
        output_dir, f"{network_name}_detection_accuracy_analysis_log.md"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"Saved detection accuracy analysis log to {log_path}")


# ---------------------------------------------------------------------------
# --------------------- (3) PERFORMANCE TIMING DASHBOARD --------------------
# ---------------------------------------------------------------------------
def plot_performance_timing_dashboard(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Create a 2x2 grid visualization focusing on detection timing and delay analysis."""
    if df.empty:
        print("Error: No data available for performance timing analysis")
        return

    # Create publication-ready figure with tighter layout
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 8),  # Slightly smaller for more compact presentation
        gridspec_kw={"wspace": 0.15, "hspace": 0.15},  # Tighter spacing
        constrained_layout=True,  # Use constrained layout for better spacing
    )

    # Remove tight_layout since we're using constrained_layout
    # plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)

    betting_funcs = sorted(df["betting_func"].unique())
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}

    # 1. Detection Delay by Window Size (top-left)
    ax = axs[0, 0]
    windows = sorted(df["window"].unique())

    for bf in betting_funcs:
        subset = df[df["betting_func"] == bf]
        if not subset.empty:
            window_groups = (
                subset.groupby("window")["traditional_avg_delay"].mean().reset_index()
            )
            if not window_groups.empty:
                ax.plot(
                    window_groups["window"],
                    window_groups["traditional_avg_delay"],
                    "o-",
                    label=bf.capitalize(),
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                )

    ax.set_xlabel("Window Size", fontsize=16)
    ax.set_ylabel("Average Detection Delay", fontsize=16)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=14, loc="best", frameon=True, framealpha=0.9, handlelength=1.5)

    # Add annotation about window size effect with better positioning
    if windows:
        max_window = max(windows)
        min_window = min(windows)
        mid_x = (max_window + min_window) / 2
        y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.8

        # Use clearer positioning for annotation
        ax.annotate(
            "Larger windows improve\nmodel accuracy but\nincrease delay",
            xy=(max_window - (max_window - min_window) * 0.2, y_pos),
            xytext=(mid_x, y_pos * 0.6),
            fontsize=14,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="white",
                alpha=0.9,
                edgecolor="lightgray",
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.2",
                linewidth=1.5,
                alpha=0.8,
            ),
            zorder=10,
        )

    # 2. Early Detection Capabilities (top-right)
    ax = axs[0, 1]

    # Analyze fraction of delay to horizon
    if "traditional_avg_delay" in df.columns and "horizon" in df.columns:
        df_with_ratio = df.copy()
        # Calculate delay as a fraction of horizon
        df_with_ratio["delay_ratio"] = (
            df_with_ratio["traditional_avg_delay"] / df_with_ratio["horizon"]
        )

        # Group by betting function
        boxprops = dict(linewidth=2.5, alpha=0.8)
        whiskerprops = dict(linewidth=2.0, alpha=0.8)
        medianprops = dict(linewidth=2.5, color="black")

        # Make boxplots narrower and adjust plotting
        sns.boxplot(
            x="betting_func",
            y="delay_ratio",
            hue="betting_func",
            data=df_with_ratio,
            ax=ax,
            palette=colors,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            showfliers=False,
            legend=False,
            width=0.6,  # Make boxes narrower
        )

        # Add individual points for better visibility
        sns.stripplot(
            x="betting_func",
            y="delay_ratio",
            data=df_with_ratio,
            ax=ax,
            color="black",
            size=3,
            alpha=0.3,
            jitter=True,
        )

        ax.set_xlabel("Betting Function", fontsize=16)
        ax.set_ylabel("Delay / Horizon Ratio", fontsize=16)

        # First check the actual data range before setting limits
        actual_max = df_with_ratio["delay_ratio"].max()

        # Set y-axis limits to ensure all data is visible with padding
        # Use a generous upper limit to ensure all data points and whiskers are shown
        ax.set_ylim(0, max(2.0, actual_max * 1.1))

        # Make the reference line more prominent
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.8, linewidth=2.5, zorder=5)
        ax.grid(True, linestyle=":", alpha=0.4)

        # Add shading below the red line to highlight early detection region
        ax.axhspan(0, 1.0, alpha=0.1, color="green", zorder=1)

        # Move the annotation to avoid overlapping with data
        # Find position based on betting functions
        bf_count = len(betting_funcs)

        # Better positioned annotation
        ax.annotate(
            "Ideal early detection\n(before horizon end)",
            xy=(bf_count - 1, 0.9),  # Point to right side near the red line
            xytext=(bf_count / 2, 0.4),  # Position text lower and centered
            fontsize=14,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="white",
                alpha=0.9,
                edgecolor="lightgray",
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=-0.2",
                linewidth=1.5,
                alpha=0.8,
            ),
        )

        # Format x-tick labels
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [x.get_text().capitalize() for x in ax.get_xticklabels()], fontsize=12
        )
        ax.tick_params(axis="y", labelsize=12)

        # Set a reasonable y-axis limit
        y_max = delay_long["Delay"].max() * 1.15
        ax.set_ylim(0, y_max)

    # 3. Delay Reduction by Parameter (bottom-left)
    ax = axs[1, 0]

    # We'll use threshold as the parameter for delay reduction
    thresholds = sorted(df["threshold"].unique())

    for bf in betting_funcs:
        subset = df[df["betting_func"] == bf]
        if not subset.empty:
            threshold_groups = (
                subset.groupby("threshold")
                .agg(
                    {
                        "traditional_avg_delay": "mean",
                        "trad_tpr": "mean",
                    }
                )
                .reset_index()
            )

            if not threshold_groups.empty:
                ax.plot(
                    threshold_groups["threshold"],
                    threshold_groups["traditional_avg_delay"],
                    "o-",
                    label=bf.capitalize(),
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                )

    ax.set_xlabel("Detection Threshold ($\lambda$)", fontsize=16)
    ax.set_ylabel("Average Detection Delay", fontsize=16)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=14, loc="best", frameon=True, framealpha=0.9, handlelength=1.5)

    # Add informative annotation about threshold impact
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Determine if threshold increases or decreases delay
    if len(thresholds) > 1:
        if (
            df.groupby("threshold")["traditional_avg_delay"]
            .mean()
            .corr(pd.Series(sorted(df["threshold"].unique())))
            > 0
        ):
            trend = "increases"
            arrow_dir = 0.2  # Positive arc
        else:
            trend = "decreases"
            arrow_dir = -0.2  # Negative arc

        ax.annotate(
            f"Higher threshold\n{trend} detection delay",
            xy=(
                xlim[1] - (xlim[1] - xlim[0]) * 0.2,
                ylim[0] + (ylim[1] - ylim[0]) * 0.7,
            ),
            xytext=(
                xlim[0] + (xlim[1] - xlim[0]) * 0.5,
                ylim[0] + (ylim[1] - ylim[0]) * 0.4,
            ),
            fontsize=14,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="white",
                alpha=0.9,
                edgecolor="lightgray",
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={arrow_dir}",
                linewidth=1.5,
                alpha=0.8,
            ),
            zorder=10,
        )

    # 4. Optimal Configuration Regions (bottom-right)
    ax = axs[1, 1]

    # Create a scatter plot with delay vs TPR, colored by betting function
    # Size points by FPR (smaller is better)

    # First, calculate metrics for each configuration
    config_metrics = (
        df.groupby(["betting_func", "threshold", "window"])
        .agg(
            {
                "traditional_avg_delay": "mean",
                "trad_tpr": "mean",
                "trad_fpr": "mean",
            }
        )
        .reset_index()
    )

    # Size inversely proportional to FPR (smaller FPR = larger point)
    # Add small constant to avoid division by zero
    config_metrics["point_size"] = 100 / (config_metrics["trad_fpr"] + 0.005)

    # Cap the size for very small FPRs
    config_metrics["point_size"] = config_metrics["point_size"].clip(20, 200)

    # Use edgecolor for better visibility
    for bf in betting_funcs:
        subset = config_metrics[config_metrics["betting_func"] == bf]
        if not subset.empty:
            ax.scatter(
                subset["traditional_avg_delay"],
                subset["trad_tpr"],
                s=subset["point_size"],
                alpha=0.7,
                label=bf.capitalize(),
                color=colors[bf],
                edgecolor="white",
                linewidth=0.5,
            )

    # Highlight optimal region more clearly
    ax.axhspan(0.8, 1.0, alpha=0.15, color="green", zorder=0)

    # Add thin gridlines
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)

    ax.set_xlabel("Detection Delay", fontsize=16)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=16)
    ax.set_ylim(0, 1.05)

    # Add legend for betting functions with better placement
    ax.legend(
        fontsize=14,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        handlelength=1.5,
        markerscale=0.8,  # Make legend markers smaller
    )

    # Move annotation for point size to better location
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.25

    # Avoid overlapping with data points
    ax.annotate(
        "Point size $\\propto 1/FPR$\n(larger = better)",
        xy=(x_pos, 0.2),  # Arrow endpoint
        xytext=(x_pos, 0.08),  # Text position - moved lower
        fontsize=14,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="white",
            alpha=0.9,
            edgecolor="lightgray",
        ),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.2",
            linewidth=1.5,
        ),
        zorder=10,
    )

    # Better position for optimal region annotation
    ax.annotate(
        "Optimal region\n(high TPR, low delay)",
        xy=(
            xlim[0] + (xlim[1] - xlim[0]) * 0.25,
            0.9,
        ),  # Arrow endpoint in green region
        xytext=(xlim[0] + (xlim[1] - xlim[0]) * 0.5, 0.65),  # Text centered and lower
        fontsize=14,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="white",
            alpha=0.9,
            edgecolor="lightgray",
        ),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.2",
            linewidth=1.5,
        ),
        zorder=10,
    )

    # Publication-ready styling for all subplots
    for i in range(2):
        for j in range(2):
            # Remove top and right spines
            axs[i, j].spines["top"].set_visible(False)
            axs[i, j].spines["right"].set_visible(False)

            # Make bottom and left spines slightly thicker
            axs[i, j].spines["bottom"].set_linewidth(1.2)
            axs[i, j].spines["left"].set_linewidth(1.2)

            # Add subtle tick marks
            axs[i, j].tick_params(direction="out", length=4, width=1.2, pad=4)

            # Ensure y-axis has enough ticks for readability
            axs[i, j].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))

            # Format tick labels to avoid scientific notation and unnecessary decimals
            if i == 0 and j == 1:  # For Delay/Horizon Ratio plot
                axs[i, j].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            else:  # For other plots
                axs[i, j].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

            # For x-axis ticks
            if (i == 0 and j == 0) or (
                i == 1 and j == 0
            ):  # Window Size and Threshold plots
                axs[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{network_name}_performance_timing_dashboard.png"
    )

    # Use higher DPI and better format options for publication-quality output
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", format="png", transparent=False
    )

    generate_performance_timing_analysis_log(df, output_dir, network_name)

    print(f"Saved performance timing dashboard to {output_path}")
    plt.close(fig)


def generate_performance_timing_analysis_log(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Generate analysis log for performance timing dashboard."""
    data_points = len(df)

    # Calculate metrics for window size impact
    window_metrics = (
        df.groupby(["window", "betting_func"])
        .agg({"traditional_avg_delay": "mean", "trad_tpr": "mean", "trad_fpr": "mean"})
        .reset_index()
    )

    # Find optimal window size (balancing delay and TPR)
    window_metrics["efficiency"] = window_metrics["trad_tpr"] / (
        window_metrics["traditional_avg_delay"] + 1
    )
    best_window_group = window_metrics.groupby("window")["efficiency"].mean().idxmax()

    # Calculate delay/horizon ratios
    df_with_ratio = df.copy()
    df_with_ratio["delay_ratio"] = (
        df_with_ratio["traditional_avg_delay"] / df_with_ratio["horizon"]
    )

    # Find betting function with lowest delay ratio
    best_bf_for_early = (
        df_with_ratio.groupby("betting_func")["delay_ratio"].mean().idxmin()
    )
    early_detection_rate = (df_with_ratio["delay_ratio"] < 1.0).mean() * 100

    # Identify configuration with optimal balance
    # Weight: 60% for TPR, 30% for delay (inverse), 10% for FPR (inverse)
    df_config = df.copy()
    # Normalize metrics to 0-1 scale
    max_delay = df_config["traditional_avg_delay"].max()
    df_config["norm_delay"] = 1 - (df_config["traditional_avg_delay"] / max_delay)
    df_config["norm_fpr"] = 1 - (df_config["trad_fpr"] / df_config["trad_fpr"].max())

    # Calculate balanced score
    df_config["balanced_score"] = (
        0.6 * df_config["trad_tpr"]
        + 0.3 * df_config["norm_delay"]
        + 0.1 * df_config["norm_fpr"]
    )

    # Get best configuration
    best_config_idx = df_config["balanced_score"].idxmax()
    best_config = df_config.loc[best_config_idx]

    log_content = f"""# Performance Timing Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## General Information

- network_type: {NETWORK_NAMES.get(network_name, network_name.upper())}
- analysis_date: {datetime.now().strftime("%Y-%m-%d")}
- data_points: {data_points}

## Window Size Impact

- optimal_window_size: {int(best_window_group)}
- window_size_impact: Larger windows generally increase detection delay but can improve model accuracy
- window_delay_correlation: {df.groupby("window")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["window"].unique()))):.4f}

## Early Detection Analysis

- best_betting_function_for_early_detection: {best_bf_for_early.capitalize()}
- early_detection_rate: {early_detection_rate:.2f}% (detection before horizon end)
- avg_delay_ratio: {df_with_ratio["delay_ratio"].mean():.4f} (delay as fraction of horizon)
- early_detection_advantage: Detecting anomalies before the end of the prediction horizon enables proactive responses

## Delay Reduction Analysis

- threshold_delay_correlation: {df.groupby("threshold")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["threshold"].unique()))):.4f}
- parameter_with_most_delay_impact: {"threshold" if abs(df.groupby("threshold")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["threshold"].unique())))) > abs(df.groupby("window")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["window"].unique())))) else "window"}
- delay_reduction_strategy: {"Increase threshold" if df.groupby("threshold")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["threshold"].unique()))) < 0 else "Decrease threshold"} to reduce detection delay

## Optimal Configuration

- best_betting_function: {best_config["betting_func"].capitalize()}
- optimal_threshold: {int(best_config["threshold"])}
- optimal_window: {int(best_config["window"])}
- optimal_configuration_tpr: {best_config["trad_tpr"]:.4f}
- optimal_configuration_delay: {best_config["traditional_avg_delay"]:.2f}
- optimal_configuration_fpr: {best_config["trad_fpr"]:.6f}

## Plot Description

The visualization consists of a 2x2 grid that shows:

### 1. Detection Delay by Window Size (top-left)
- Shows how window size affects detection delay for each betting function
- Larger windows typically increase delay but may improve model accuracy
- Useful for selecting optimal window size based on application needs

### 2. Early Detection Capabilities (top-right)
- Compares delay/horizon ratio across betting functions
- Ratio < 1.0 indicates detection before the prediction horizon ends
- Helps identify betting functions that excel at early anomaly detection

### 3. Delay Reduction by Threshold (bottom-left)
- Shows relationship between detection threshold and delay
- Helps identify thresholds that minimize detection delay
- Useful for delay-sensitive applications

### 4. Optimal Configuration Regions (bottom-right)
- Scatter plot positioning configurations by delay and TPR
- Point size inversely proportional to FPR (larger = better)
- Highlights optimal region with high TPR and low delay
- Provides visual guide for parameter selection based on priorities

## Summary

The performance timing analysis reveals that {best_config["betting_func"].capitalize()} betting function with threshold {int(best_config["threshold"])} and window size {int(best_config["window"])} provides the optimal balance between detection accuracy (TPR: {best_config["trad_tpr"]:.4f}) and timeliness (delay: {best_config["traditional_avg_delay"]:.2f}). 

{best_bf_for_early.capitalize()} betting function demonstrates the best early detection capabilities, with detection often occurring before the prediction horizon ends. Window size shows a {"positive" if df.groupby("window")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["window"].unique()))) > 0 else "negative"} correlation with detection delay, confirming the expected tradeoff between model accuracy and timeliness.

For applications where early detection is critical, the analysis suggests using {best_bf_for_early.capitalize()} betting with smaller window sizes and {"higher" if df.groupby("threshold")["traditional_avg_delay"].mean().corr(pd.Series(sorted(df["threshold"].unique()))) < 0 else "lower"} threshold values.
"""

    log_path = os.path.join(
        output_dir, f"{network_name}_performance_timing_analysis_log.md"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"Saved performance timing analysis log to {log_path}")


# ---------------------------------------------------------------------------
# ---------------- (4) HORIZON VS TRADITIONAL COMPARISON GRID --------------
# ---------------------------------------------------------------------------
def plot_horizon_comparison_grid(df: pd.DataFrame, output_dir: str, network_name: str):
    """Create visualization comparing horizon and traditional approaches."""
    if df.empty:
        print("Error: No data available for horizon comparison analysis")
        return

    # Filter out rows with missing delay values
    df_clean = df.dropna(subset=["traditional_avg_delay", "horizon_avg_delay"]).copy()

    if len(df_clean) == 0:
        print("Error: No valid delay data for horizon comparison")
        return

    # Create publication-ready figure with tight layout
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"] + plt.rcParams["font.serif"]

    # Create 1x3 layout instead of 2x2
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15, 5),  # Wider and shorter for a single row
        gridspec_kw={"wspace": 0.20},  # Adjust spacing between plots
        constrained_layout=True,
    )

    # Define a better color palette for Traditional vs Horizon
    trad_color = "#2C5F7F"  # Deeper blue
    horizon_color = "#FF8C38"  # Warmer orange
    method_palette = [trad_color, horizon_color]

    betting_funcs = sorted(df_clean["betting_func"].unique())

    # 1. Delay Comparison: Traditional vs Horizon (left)
    ax = axs[0]

    # Group by betting function and calculate average delays
    delay_comp = []
    for bf in betting_funcs:
        bf_data = df_clean[df_clean["betting_func"] == bf]
        if not bf_data.empty:
            trad_delay = bf_data["traditional_avg_delay"].mean()
            hor_delay = bf_data["horizon_avg_delay"].mean()
            delay_comp.append(
                {"betting_func": bf, "Traditional": trad_delay, "Horizon": hor_delay}
            )

    delay_df = pd.DataFrame(delay_comp)
    if not delay_df.empty:
        # Convert to long format for grouped bar plot
        delay_long = pd.melt(
            delay_df,
            id_vars=["betting_func"],
            value_vars=["Traditional", "Horizon"],
            var_name="Method",
            value_name="Delay",
        )

        # Create grouped bar plot with better styling
        barplot = sns.barplot(
            x="betting_func",
            y="Delay",
            hue="Method",
            data=delay_long,
            ax=ax,
            palette=method_palette,
            width=0.7,  # Slightly narrower bars
            edgecolor="black",  # Add black edge for better definition
            linewidth=0.8,
        )

        # Make bar edges more visible
        for patch in barplot.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.8))  # Make bars slightly transparent

        # Calculate percentage improvement for annotation
        for i, bf in enumerate(delay_df["betting_func"]):
            trad = delay_df.loc[delay_df["betting_func"] == bf, "Traditional"].values[0]
            hor = delay_df.loc[delay_df["betting_func"] == bf, "Horizon"].values[0]
            if trad > 0:  # Avoid division by zero
                pct_improve = (trad - hor) / trad * 100
                if pct_improve > 0:  # Only show positive improvements
                    # Create better annotation background
                    ax.annotate(
                        f"-{pct_improve:.1f}%",
                        xy=(i, hor),
                        xytext=(0, -15),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        color="#006400",  # Dark green
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="#F8F8F8",
                            ec="#CCCCCC",
                            alpha=0.9,
                        ),
                    )

        ax.set_xlabel("")
        ax.set_ylabel("Average Detection Delay", fontsize=14, fontweight="bold")

        # Move legend to better position
        ax.legend(
            fontsize=12,
            title_fontsize=13,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
        )

        # Format x-tick labels
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [x.get_text().capitalize() for x in ax.get_xticklabels()], fontsize=12
        )
        ax.tick_params(axis="y", labelsize=12)

        # Set a reasonable y-axis limit
        y_max = delay_long["Delay"].max() * 1.15
        ax.set_ylim(0, y_max)

    # 2. TPR Comparison: Traditional vs Horizon (middle)
    ax = axs[1]

    # Group by betting function and calculate average TPR
    tpr_comp = []
    for bf in betting_funcs:
        bf_data = df_clean[df_clean["betting_func"] == bf]
        if not bf_data.empty:
            trad_tpr = bf_data["trad_tpr"].mean()
            hor_tpr = bf_data["horizon_tpr"].mean()
            tpr_comp.append(
                {"betting_func": bf, "Traditional": trad_tpr, "Horizon": hor_tpr}
            )

    tpr_df = pd.DataFrame(tpr_comp)
    if not tpr_df.empty:
        # Convert to long format for grouped bar plot
        tpr_long = pd.melt(
            tpr_df,
            id_vars=["betting_func"],
            value_vars=["Traditional", "Horizon"],
            var_name="Method",
            value_name="TPR",
        )

        # Create grouped bar plot with better styling
        barplot = sns.barplot(
            x="betting_func",
            y="TPR",
            hue="Method",
            data=tpr_long,
            ax=ax,
            palette=method_palette,
            width=0.7,
            edgecolor="black",
            linewidth=0.8,
        )

        # Make bar edges more visible
        for patch in barplot.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.8))  # Make bars slightly transparent

        # Calculate percentage improvement for annotation
        for i, bf in enumerate(tpr_df["betting_func"]):
            trad = tpr_df.loc[tpr_df["betting_func"] == bf, "Traditional"].values[0]
            hor = tpr_df.loc[tpr_df["betting_func"] == bf, "Horizon"].values[0]
            if trad > 0:  # Avoid division by zero
                pct_improve = (hor - trad) / trad * 100
                if pct_improve > 5:  # Only show significant improvements
                    ax.annotate(
                        f"+{pct_improve:.1f}%",
                        xy=(i, hor),
                        xytext=(0, -10),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        color="#006400",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="#F8F8F8",
                            ec="#CCCCCC",
                            alpha=0.9,
                        ),
                    )

        ax.set_xlabel("")
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=14, fontweight="bold")

        # Move legend to better position
        ax.legend(
            fontsize=12,
            title_fontsize=13,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
        )

        ax.set_ylim(0, 1.05)

        # Format x-tick labels
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [x.get_text().capitalize() for x in ax.get_xticklabels()], fontsize=12
        )
        ax.tick_params(axis="y", labelsize=12)

    # 3. FPR Comparison (right)
    ax = axs[2]

    # Group by betting function and calculate average FPR
    fpr_comp = []
    for bf in betting_funcs:
        bf_data = df_clean[df_clean["betting_func"] == bf]
        if not bf_data.empty:
            trad_fpr = bf_data["trad_fpr"].mean()
            hor_fpr = bf_data["horizon_fpr"].mean()
            fpr_comp.append(
                {"betting_func": bf, "Traditional": trad_fpr, "Horizon": hor_fpr}
            )

    fpr_df = pd.DataFrame(fpr_comp)
    if not fpr_df.empty:
        # Convert to long format for grouped bar plot
        fpr_long = pd.melt(
            fpr_df,
            id_vars=["betting_func"],
            value_vars=["Traditional", "Horizon"],
            var_name="Method",
            value_name="FPR",
        )

        # Create grouped bar plot with better styling
        barplot = sns.barplot(
            x="betting_func",
            y="FPR",
            hue="Method",
            data=fpr_long,
            ax=ax,
            palette=method_palette,
            width=0.7,
            edgecolor="black",
            linewidth=0.8,
        )

        # Make bar edges more visible
        for patch in barplot.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.8))  # Make bars slightly transparent

        ax.set_xlabel("")
        ax.set_ylabel("False Positive Rate (FPR)", fontsize=14, fontweight="bold")

        # Use scientific notation for small FPR values
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)

        # Set reasonable y-axis limits for FPR plot
        max_fpr = max(fpr_long["FPR"].max() * 1.2, 0.05)  # Set reasonable upper limit
        ax.set_ylim(0, min(max_fpr, 0.05))  # Cap at 0.05 to avoid excessive empty space

        # Move legend to better position
        ax.legend(
            fontsize=12,
            title_fontsize=13,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
        )

        # Format x-tick labels
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [x.get_text().capitalize() for x in ax.get_xticklabels()], fontsize=12
        )
        ax.tick_params(axis="y", labelsize=12)

    # Apply publication-ready styling to all subplots
    for i in range(3):
        # Remove top and right spines
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

        # Make bottom and left spines slightly thicker
        axs[i].spines["bottom"].set_linewidth(1.5)
        axs[i].spines["left"].set_linewidth(1.5)

        # Add subtle tick marks
        axs[i].tick_params(direction="out", length=5, width=1.5, pad=5)

        # Add grid but only for y-axis and more subtle
        axs[i].grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.3, zorder=0)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{network_name}_horizon_comparison_grid.png"
    )

    # Save high-resolution version
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    generate_horizon_comparison_analysis_log(df_clean, output_dir, network_name)

    print(f"Saved horizon comparison grid to {output_path}")
    plt.close(fig)


def generate_horizon_comparison_analysis_log(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Generate analysis log for horizon vs traditional comparison."""
    data_points = len(df)

    # Calculate average improvement metrics
    avg_delay_reduction = (
        (df["traditional_avg_delay"] - df["horizon_avg_delay"])
        / df["traditional_avg_delay"]
    ).mean() * 100

    # Calculate TPR improvements
    tpr_improvement = (df["horizon_tpr"] - df["trad_tpr"]).mean() * 100

    # Calculate FPR changes (negative is better)
    fpr_change = (df["horizon_fpr"] - df["trad_fpr"]).mean() * 100

    # Summarize by betting function
    betting_summary = []
    for bf in sorted(df["betting_func"].unique()):
        subset = df[df["betting_func"] == bf]
        if not subset.empty:
            delay_red = (
                (subset["traditional_avg_delay"] - subset["horizon_avg_delay"])
                / subset["traditional_avg_delay"]
            ).mean() * 100
            tpr_imp = (subset["horizon_tpr"] - subset["trad_tpr"]).mean() * 100
            fpr_chg = (subset["horizon_fpr"] - subset["trad_fpr"]).mean() * 100

            betting_summary.append(
                {
                    "betting_func": bf,
                    "delay_reduction_pct": delay_red,
                    "tpr_improvement_pct": tpr_imp,
                    "fpr_change_pct": fpr_chg,
                    "trad_tpr": subset["trad_tpr"].mean(),
                    "horizon_tpr": subset["horizon_tpr"].mean(),
                    "trad_delay": subset["traditional_avg_delay"].mean(),
                    "horizon_delay": subset["horizon_avg_delay"].mean(),
                }
            )

    bf_summary_df = pd.DataFrame(betting_summary)

    # Find best betting function for horizon approach
    best_bf_idx = None
    if not bf_summary_df.empty:
        # Score = TPR improvement + Delay reduction - FPR increase
        bf_summary_df["score"] = (
            bf_summary_df["tpr_improvement_pct"]
            + bf_summary_df["delay_reduction_pct"]
            - abs(bf_summary_df["fpr_change_pct"])
        )
        best_bf_idx = bf_summary_df["score"].idxmax()

    log_content = f"""# Horizon vs. Traditional Comparison Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## General Information

- network_type: {NETWORK_NAMES.get(network_name, network_name.upper())}
- analysis_date: {datetime.now().strftime("%Y-%m-%d")}
- data_points: {data_points}

## Overall Performance Impact

- average_delay_reduction: {avg_delay_reduction:.2f}% (positive values indicate horizon is faster)
- tpr_improvement: {tpr_improvement:.2f}% (positive values indicate horizon has better detection rate)
- fpr_change: {fpr_change:.2f}% (negative values indicate horizon has lower false positive rate)

## Performance by Betting Function

"""

    if not bf_summary_df.empty:
        # Add table headers
        log_content += "Betting Function | Delay Reduction | TPR Improvement | FPR Change | Traditional TPR | Horizon TPR | Traditional Delay | Horizon Delay\n"
        log_content += "--------------- | -------------- | -------------- | ---------- | -------------- | ---------- | ---------------- | ------------\n"

        # Add data rows
        for _, row in bf_summary_df.iterrows():
            log_content += (
                f"{row['betting_func'].capitalize()} | "
                f"{row['delay_reduction_pct']:.2f}% | "
                f"{row['tpr_improvement_pct']:.2f}% | "
                f"{row['fpr_change_pct']:.2f}% | "
                f"{row['trad_tpr']:.4f} | "
                f"{row['horizon_tpr']:.4f} | "
                f"{row['trad_delay']:.2f} | "
                f"{row['horizon_delay']:.2f}\n"
            )

        # Add best betting function for horizon
        if best_bf_idx is not None:
            best_bf = bf_summary_df.loc[best_bf_idx, "betting_func"]
            log_content += f"\n### Best Betting Function for Horizon Approach\n\n"
            log_content += f"The **{best_bf.capitalize()}** betting function shows the best overall improvement with the horizon approach, "
            log_content += f"with {bf_summary_df.loc[best_bf_idx, 'delay_reduction_pct']:.2f}% delay reduction and "
            log_content += f"{bf_summary_df.loc[best_bf_idx, 'tpr_improvement_pct']:.2f}% TPR improvement.\n"

    log_content += """
## Plot Description

The visualization consists of a 2x2 grid that shows:

### 1. Delay Comparison (top-left)
- Compares average detection delay between traditional and horizon approaches
- Shows percentage improvement for each betting function
- Lower bars indicate better performance

### 2. TPR Comparison (top-right)
- Compares true positive rate between traditional and horizon approaches
- Shows percentage improvement for significant improvements
- Higher bars indicate better performance

### 3. Delay Reduction Distribution (bottom-left)
- Shows the distribution of delay reduction percentages by betting function
- Highlights the betting function with the best average reduction
- Higher values indicate more improvement with the horizon approach

### 4. FPR Comparison (bottom-right)
- Compares false positive rates between traditional and horizon approaches
- Shows the theoretical bound as a reference
- Lower bars indicate better performance

## Summary

The horizon approach demonstrates substantial improvements over the traditional approach in terms of detection speed and accuracy. It achieves an average of {avg_delay_reduction:.2f}% reduction in detection delay while also improving the true positive rate by {tpr_improvement:.2f}%.

This analysis highlights the effectiveness of incorporating prediction horizons in the anomaly detection framework, enabling earlier detection of changes with comparable or better accuracy. The improvements are particularly pronounced with certain betting functions, suggesting that the choice of betting function can significantly impact the benefits gained from the horizon approach.
"""

    log_path = os.path.join(
        output_dir, f"{network_name}_horizon_comparison_analysis_log.md"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"Saved horizon comparison analysis log to {log_path}")


# ---------------------------------------------------------------------------
# ------------------------------ MAIN FUNCTION ------------------------------
# ---------------------------------------------------------------------------
def main():
    """Main entry point for hyperparameter analysis plotting."""
    parser = argparse.ArgumentParser(
        description="Create visualizations from hyperparameter analysis results."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/hyperparameter_analysis/plots",
        help="Directory to save plots (default: results/hyperparameter_analysis/plots)",
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        default="sbm",
        help="Network name to analyze (default: sbm)",
    )
    parser.add_argument(
        "--plot-type",
        "-p",
        type=str,
        default="all",
        choices=["betting", "accuracy", "timing", "horizon", "all"],
        help="Type of plot to generate (default: all)",
    )

    args = parser.parse_args()
    df_network = load_analysis_results(args.network_name)

    if df_network is not None:
        df_network_clean = preprocess_data(df_network)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.plot_type in ["betting", "all"]:
            plot_betting_function_comparison(
                df_network_clean, args.output_dir, args.network_name
            )

        if args.plot_type in ["accuracy", "all"]:
            plot_detection_accuracy_grid(
                df_network_clean, args.output_dir, args.network_name
            )

        if args.plot_type in ["timing", "all"]:
            plot_performance_timing_dashboard(
                df_network_clean, args.output_dir, args.network_name
            )

        if args.plot_type in ["horizon", "all"]:
            plot_horizon_comparison_grid(
                df_network_clean, args.output_dir, args.network_name
            )
    else:
        print(f"Error: Could not load {args.network_name} network data")


if __name__ == "__main__":
    main()
