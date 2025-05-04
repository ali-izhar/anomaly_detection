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
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.titlepad"] = 15
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

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


# ----------------------- (1) BETTING FUNCTION COMPARISON -----------------------
# -------------------------------------------------------------------------------
def plot_betting_function_comparison(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Compare the impact of different betting functions on performance."""
    if "betting_func" not in df.columns:
        print("Error: No betting function data available")
        return

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    betting_funcs = sorted(df["betting_func"].unique())
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}
    plot_df = df.copy()
    plot_df["betting_func"] = pd.Categorical(
        plot_df["betting_func"], categories=betting_funcs
    )

    # 1. Plot TPR by betting function
    ax = axs[0]
    boxprops = dict(linewidth=2.0, alpha=0.8)
    whiskerprops = dict(linewidth=2.0, alpha=0.8)
    medianprops = dict(linewidth=2.0, color="black")
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
    with open(log_path, "w") as f:
        f.write(log_content)
    print(f"Saved analysis log to {log_path}")


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

    args = parser.parse_args()
    df_network = load_analysis_results(args.network_name)

    if df_network is not None:
        df_network_clean = preprocess_data(df_network)
        os.makedirs(args.output_dir, exist_ok=True)
        plot_betting_function_comparison(
            df_network_clean, args.output_dir, args.network_name
        )
    else:
        print(f"Error: Could not load {args.network_name} network data")


if __name__ == "__main__":
    main()
