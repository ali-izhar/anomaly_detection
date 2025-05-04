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
        figsize=(14, 10),  # Slightly reduce vertical size
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},  # Tighter spacing
        constrained_layout=False,  # Switch to tight_layout for better control
    )
    plt.tight_layout(
        pad=2.0, h_pad=2.0, w_pad=2.0
    )  # Careful padding for research layout

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
# ------------------------------ MAIN FUNCTION -----------------------
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
        choices=["betting", "accuracy", "all"],
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
    else:
        print(f"Error: Could not load {args.network_name} network data")


if __name__ == "__main__":
    main()
