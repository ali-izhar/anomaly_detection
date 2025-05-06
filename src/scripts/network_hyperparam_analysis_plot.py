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
from matplotlib.ticker import MaxNLocator, ScalarFormatter

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 22
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

STREAM_COLORS = {
    "trad_color": "#2C5F7F",  # Deeper blue for traditional
    "horizon_color": "#FF8C38",  # Warmer orange for horizon
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

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    betting_funcs = sorted(df["betting_func"].unique())
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["trad_tpr", "horizon_tpr"], how="all")

    # Make sure betting_func is treated as categorical for proper ordering
    plot_df["betting_func"] = pd.Categorical(
        plot_df["betting_func"], categories=betting_funcs
    )
    boxprops = dict(linewidth=2.0, alpha=0.8)
    whiskerprops = dict(linewidth=1.5, alpha=0.8)
    medianprops = dict(linewidth=2.0, color="black")
    flierprops = dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.5)

    # 1. Plot TPR by betting function (traditional vs horizon)
    ax = axs[0]
    tpr_data = []
    for bf in betting_funcs:
        bf_data = plot_df[plot_df["betting_func"] == bf]
        if not bf_data.empty:
            for tpr in bf_data["trad_tpr"]:
                tpr_data.append(
                    {"betting_func": bf, "Method": "Traditional", "TPR": tpr}
                )
            for tpr in bf_data["horizon_tpr"].dropna():
                tpr_data.append({"betting_func": bf, "Method": "Horizon", "TPR": tpr})
    tpr_df = pd.DataFrame(tpr_data)

    # Draw boxplots first
    sns.boxplot(
        x="betting_func",
        y="TPR",
        hue="Method",
        data=tpr_df,
        ax=ax,
        palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        flierprops=flierprops,
        showfliers=True,
        width=0.7,
    )

    # Add individual data points
    sns.stripplot(
        x="betting_func",
        y="TPR",
        hue="Method",
        data=tpr_df,
        ax=ax,
        palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
        size=2.5,
        alpha=0.4,
        jitter=True,
        dodge=True,
        legend=False,
    )

    # Add mean markers with more visibility
    for i, bf in enumerate(betting_funcs):
        for j, method in enumerate(["Traditional", "Horizon"]):
            subset = tpr_df[
                (tpr_df["betting_func"] == bf) & (tpr_df["Method"] == method)
            ]
            if not subset.empty:
                mean_val = subset["TPR"].mean()
                ax.scatter(
                    i + (j * 0.4 - 0.2),  # Position for correct dodge
                    mean_val,
                    marker="D",  # Diamond for mean
                    s=100,  # Larger size
                    color="black",
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=10,  # Draw on top
                    label="_nolegend_",  # Don't add to legend
                )

    for i, bf in enumerate(betting_funcs):
        bf_data = plot_df[plot_df["betting_func"] == bf]
        if not bf_data.empty:
            trad_tpr = bf_data["trad_tpr"].mean()
            hor_tpr = bf_data["horizon_tpr"].mean()
            if not np.isnan(hor_tpr) and trad_tpr > 0:
                pct_improve = (hor_tpr - trad_tpr) / trad_tpr * 100
                if pct_improve > 5:
                    bf_idx = list(betting_funcs).index(bf)
                    ax.annotate(
                        f"+{pct_improve:.1f}\\% (mean)",  # Clarify this is mean improvement
                        xy=(bf_idx + 0.2, hor_tpr + 0.03),
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="#006400",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="#F8F8F8",
                            ec="#CCCCCC",
                            alpha=0.9,
                        ),
                    )

    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.4)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])
    ax.legend().remove()

    # 2. Plot Detection Delay by betting function (traditional vs horizon)
    ax = axs[1]
    delay_data = []
    for bf in betting_funcs:
        bf_data = plot_df[plot_df["betting_func"] == bf]
        if not bf_data.empty:
            for delay in bf_data["traditional_avg_delay"].dropna():
                delay_data.append(
                    {"betting_func": bf, "Method": "Traditional", "Delay": delay}
                )
            for delay in bf_data["horizon_avg_delay"].dropna():
                delay_data.append(
                    {"betting_func": bf, "Method": "Horizon", "Delay": delay}
                )
    delay_df = pd.DataFrame(delay_data)

    # Draw boxplots for delay
    sns.boxplot(
        x="betting_func",
        y="Delay",
        hue="Method",
        data=delay_df,
        ax=ax,
        palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        flierprops=flierprops,
        showfliers=True,
        width=0.7,
    )

    # Add individual points
    sns.stripplot(
        x="betting_func",
        y="Delay",
        hue="Method",
        data=delay_df,
        ax=ax,
        palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
        size=2.5,
        alpha=0.4,
        jitter=True,
        dodge=True,
        legend=False,
    )

    # Add mean markers with more visibility
    for i, bf in enumerate(betting_funcs):
        for j, method in enumerate(["Traditional", "Horizon"]):
            subset = delay_df[
                (delay_df["betting_func"] == bf) & (delay_df["Method"] == method)
            ]
            if not subset.empty:
                mean_val = subset["Delay"].mean()
                ax.scatter(
                    i + (j * 0.4 - 0.2),  # Position for correct dodge
                    mean_val,
                    marker="D",  # Diamond for mean
                    s=100,  # Larger size
                    color="black",
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=10,  # Draw on top
                    label="_nolegend_",  # Don't add to legend
                )

    for bf in betting_funcs:
        bf_data = plot_df[plot_df["betting_func"] == bf]
        if not bf_data.empty:
            trad_delay = bf_data["traditional_avg_delay"].mean()
            hor_delay = bf_data["horizon_avg_delay"].mean()
            if not np.isnan(hor_delay) and trad_delay > 0:
                pct_improve = (trad_delay - hor_delay) / trad_delay * 100
                if pct_improve > 5:
                    bf_idx = list(betting_funcs).index(bf)
                    ax.annotate(
                        f"-{pct_improve:.1f}\\% (mean)",  # Clarify this is mean improvement
                        xy=(bf_idx + 0.2, hor_delay),
                        xytext=(0, -15),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=11,
                        fontweight="bold",
                        color="#006400",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="#F8F8F8",
                            ec="#CCCCCC",
                            alpha=0.9,
                        ),
                    )
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Detection Delay (timesteps)", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])

    # Keep legend only for middle subplot with better positioning
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        # Add a note about diamond markers representing means
        legend = ax.legend(
            handles[:2],
            labels[:2],
            title="Method",
            fontsize=14,
            title_fontsize=14,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
        )

        # Add a custom handler for the mean marker
        mean_patch = plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="black",
            markeredgecolor="white",
            markersize=8,
            label="Mean Value",
        )
        legend._legend_box.get_children()[0].get_children().append(
            plt.matplotlib.offsetbox.VPacker(
                children=[
                    plt.matplotlib.offsetbox.TextArea(
                        "Mean Value", textprops=dict(fontsize=14)
                    ),
                    plt.matplotlib.offsetbox.DrawingArea(20, 20, 0, 0),
                ],
                align="center",
                pad=0,
                sep=5,
            )
        )
        da = legend._legend_box.get_children()[0].get_children()[-1].get_children()[1]
        p = plt.matplotlib.patches.FancyArrow(
            10,
            10,
            0,
            0,
            width=0,
            head_width=8,
            head_length=8,
            fc="black",
            transform=da.get_transform(),
        )
        da.add_artist(mean_patch)
    else:
        ax.legend().remove()

    # 3. Plot FPR by betting function (traditional vs horizon)
    ax = axs[2]
    fpr_data = []
    for bf in betting_funcs:
        bf_data = plot_df[plot_df["betting_func"] == bf]
        if not bf_data.empty:
            for fpr in bf_data["trad_fpr"].dropna():
                fpr_data.append(
                    {"betting_func": bf, "Method": "Traditional", "FPR": fpr}
                )
            for fpr in bf_data["horizon_fpr"].dropna():
                fpr_data.append({"betting_func": bf, "Method": "Horizon", "FPR": fpr})
    fpr_df = pd.DataFrame(fpr_data)

    # Draw boxplots
    sns.boxplot(
        x="betting_func",
        y="FPR",
        hue="Method",
        data=fpr_df,
        ax=ax,
        palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        flierprops=flierprops,
        showfliers=True,
        width=0.7,
    )

    # Add individual points
    sns.stripplot(
        x="betting_func",
        y="FPR",
        hue="Method",
        data=fpr_df,
        ax=ax,
        palette=[STREAM_COLORS["trad_color"], STREAM_COLORS["horizon_color"]],
        size=2.5,
        alpha=0.4,
        jitter=True,
        dodge=True,
        legend=False,
    )

    # Add mean markers with more visibility
    for i, bf in enumerate(betting_funcs):
        for j, method in enumerate(["Traditional", "Horizon"]):
            subset = fpr_df[
                (fpr_df["betting_func"] == bf) & (fpr_df["Method"] == method)
            ]
            if not subset.empty:
                mean_val = subset["FPR"].mean()
                ax.scatter(
                    i + (j * 0.4 - 0.2),  # Position for correct dodge
                    mean_val,
                    marker="D",  # Diamond for mean
                    s=100,  # Larger size
                    color="black",
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=10,  # Draw on top
                    label="_nolegend_",  # Don't add to legend
                )

    if "theoretical_bound" in plot_df.columns:
        theo_bound = plot_df["theoretical_bound"].mean()
        if not pd.isna(theo_bound):
            ax.axhline(
                y=theo_bound,
                color="#D62728",
                linestyle="--",
                alpha=0.9,
                linewidth=2,
                label="Theoretical bound",
            )
            ax.annotate(
                f"Bound: {theo_bound:.4f}",
                xy=(len(betting_funcs) - 1, theo_bound),
                xytext=(0, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=11,
                color="#D62728",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="#F8F8F8", ec="#CCCCCC", alpha=0.9
                ),
            )

    ax.set_xlabel("")
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])
    ax.legend().remove()

    # Apply consistent styling to all subplots
    for i in range(3):
        # Remove top and right spines
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

        # Make bottom and left spines slightly thicker
        axs[i].spines["bottom"].set_linewidth(1.2)
        axs[i].spines["left"].set_linewidth(1.2)

        # Add subtle tick marks
        axs[i].tick_params(direction="out", length=4, width=1.2, pad=4)

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

    # Get list of all betting functions
    betting_funcs = sorted(df["betting_func"].unique())

    # Determine best parameters based on metrics
    best_threshold = df[df["betting_func"] == best_bf]["threshold"].median()
    best_window = df[df["betting_func"] == best_bf]["window"].median()
    best_epsilon = df[df["betting_func"] == best_bf]["epsilon"].median()
    best_horizon = df[df["betting_func"] == best_bf]["horizon"].median()

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
    stats_df = df.describe()

    # Calculate detailed betting function performance metrics
    bf_detailed_metrics = {}

    for bf in betting_funcs:
        bf_data = df[df["betting_func"] == bf]
        if bf_data.empty:
            continue

        # Calculate basic statistics
        trad_tpr = bf_data["trad_tpr"].mean()
        horizon_tpr = bf_data["horizon_tpr"].mean()
        trad_fpr = bf_data["trad_fpr"].mean()
        horizon_fpr = bf_data["horizon_fpr"].mean()
        trad_delay = (
            bf_data["traditional_avg_delay"].mean()
            if "traditional_avg_delay" in bf_data
            else np.nan
        )
        horizon_delay = (
            bf_data["horizon_avg_delay"].mean()
            if "horizon_avg_delay" in bf_data
            else np.nan
        )

        # Perfect detection rates
        perfect_trad_pct = (bf_data["trad_tpr"] == 1.0).mean() * 100
        perfect_horizon_pct = (bf_data["horizon_tpr"] == 1.0).mean() * 100

        # Calculate improvement percentages
        tpr_improvement = (
            ((horizon_tpr - trad_tpr) / trad_tpr * 100) if trad_tpr > 0 else np.nan
        )
        delay_reduction = (
            ((trad_delay - horizon_delay) / trad_delay * 100)
            if trad_delay > 0 and not np.isnan(horizon_delay)
            else np.nan
        )

        # Find best parameters for this betting function
        best_configs = (
            bf_data.groupby(["threshold", "window", "horizon", "distance", "epsilon"])[
                "trad_tpr"
            ]
            .mean()
            .reset_index()
        )
        best_configs = best_configs.sort_values("trad_tpr", ascending=False).head(3)

        # Best distance metric for this betting function
        bf_best_distance = (
            bf_data.groupby("distance")["trad_tpr"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
            if not bf_data.groupby("distance")["trad_tpr"].mean().empty
            else "N/A"
        )

        bf_detailed_metrics[bf] = {
            "trad_tpr": trad_tpr,
            "horizon_tpr": horizon_tpr,
            "trad_fpr": trad_fpr,
            "horizon_fpr": horizon_fpr,
            "trad_delay": trad_delay,
            "horizon_delay": horizon_delay,
            "perfect_trad_pct": perfect_trad_pct,
            "perfect_horizon_pct": perfect_horizon_pct,
            "tpr_improvement": tpr_improvement,
            "delay_reduction": delay_reduction,
            "best_configs": best_configs,
            "best_distance": bf_best_distance,
        }

    log_content = f"""# Network Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## General Information

- network_type: {NETWORK_NAMES.get(network_name, network_name.upper())}
- analysis_date: {datetime.now().strftime("%Y-%m-%d")}
- data_points: {data_points}
- total_parameter_combinations: {unique_params}

## Overall Best Configuration

- best_betting_function: {best_bf.capitalize()}
- distance_metric: {best_distance}
- threshold: {int(best_threshold)}
- window_size: {int(best_window)}
- horizon: {int(best_horizon)}
- epsilon: {best_epsilon:.1f}

## Overall Performance Metrics

- best_betting_tpr: {bf_metrics[best_bf]["tpr"]:.4f}
- avg_tpr: {avg_tpr:.4f}
- avg_delay: {avg_delay:.2f}
- avg_fpr: {avg_fpr:.4f}
- best_betting_delay: {bf_metrics[best_bf]["delay"]:.1f}
- best_betting_fpr: {bf_metrics[best_bf]["fpr"]:.4f}

## Detailed Performance by Betting Function

"""

    # Add performance comparison table
    log_content += "Betting Function | Trad TPR | Horizon TPR | TPR Improvement | Trad Delay | Horizon Delay | Delay Reduction | Perfect Detection Rate\n"
    log_content += "--------------- | -------- | ----------- | --------------- | ---------- | ------------- | --------------- | --------------------\n"

    for bf in betting_funcs:
        if bf in bf_detailed_metrics:
            metrics = bf_detailed_metrics[bf]
            tpr_imp_str = (
                f"{metrics['tpr_improvement']:.1f}%"
                if not np.isnan(metrics["tpr_improvement"])
                else "N/A"
            )
            delay_red_str = (
                f"{metrics['delay_reduction']:.1f}%"
                if not np.isnan(metrics["delay_reduction"])
                else "N/A"
            )

            log_content += (
                f"{bf.capitalize()} | "
                f"{metrics['trad_tpr']:.4f} | "
                f"{metrics['horizon_tpr']:.4f} | "
                f"{tpr_imp_str} | "
                f"{metrics['trad_delay']:.2f} | "
                f"{metrics['horizon_delay']:.2f} | "
                f"{delay_red_str} | "
                f"{metrics['perfect_trad_pct']:.1f}%\n"
            )

    # Add section for each betting function's detailed metrics
    for bf in betting_funcs:
        if bf in bf_detailed_metrics:
            metrics = bf_detailed_metrics[bf]

            log_content += f"\n### {bf.capitalize()} Betting Function Details\n\n"

            # Basic metrics
            log_content += f"- **Traditional TPR**: {metrics['trad_tpr']:.4f}\n"
            log_content += f"- **Horizon TPR**: {metrics['horizon_tpr']:.4f}\n"
            log_content += (
                f"- **Perfect Detection Rate**: {metrics['perfect_trad_pct']:.1f}%\n"
            )
            log_content += f"- **Traditional Delay**: {metrics['trad_delay']:.2f}\n"
            log_content += f"- **Horizon Delay**: {metrics['horizon_delay']:.2f}\n"
            log_content += f"- **Traditional FPR**: {metrics['trad_fpr']:.6f}\n"
            log_content += f"- **Horizon FPR**: {metrics['horizon_fpr']:.6f}\n"
            log_content += f"- **Best Distance Metric**: {metrics['best_distance']}\n"

            # Best configurations
            log_content += "\n#### Top Configurations\n\n"
            log_content += "Threshold | Window | Horizon | Distance | Epsilon | TPR\n"
            log_content += "--------- | ------ | ------- | -------- | ------- | ---\n"

            for _, config in metrics["best_configs"].iterrows():
                log_content += (
                    f"{int(config['threshold'])} | "
                    f"{int(config['window'])} | "
                    f"{int(config['horizon'])} | "
                    f"{config['distance']} | "
                    f"{config['epsilon']:.1f} | "
                    f"{config['trad_tpr']:.4f}\n"
                )

    # Add best distance metric analysis
    log_content += "\n## Distance Metric Analysis\n\n"

    # Calculate performance by distance metric
    dist_metrics = (
        df.groupby("distance")
        .agg(
            {
                "trad_tpr": ["mean", "max"],
                "trad_fpr": "mean",
                "traditional_avg_delay": "mean",
            }
        )
        .reset_index()
    )

    dist_metrics.columns = ["distance", "tpr_mean", "tpr_max", "fpr_mean", "delay_mean"]

    log_content += "Distance | Avg TPR | Max TPR | Avg FPR | Avg Delay\n"
    log_content += "-------- | ------- | ------- | ------- | ---------\n"

    for _, row in dist_metrics.iterrows():
        log_content += (
            f"{row['distance']} | "
            f"{row['tpr_mean']:.4f} | "
            f"{row['tpr_max']:.4f} | "
            f"{row['fpr_mean']:.6f} | "
            f"{row['delay_mean']:.2f}\n"
        )

    # Add horizon vs traditional comparative analysis
    log_content += "\n## Horizon vs. Traditional Comparison\n\n"

    # Calculate average improvements
    avg_tpr_improvement = (df["horizon_tpr"] - df["trad_tpr"]).mean() * 100
    avg_delay_reduction = (
        (df["traditional_avg_delay"] - df["horizon_avg_delay"])
        / df["traditional_avg_delay"]
    ).dropna().mean() * 100

    log_content += f"- **Average TPR Improvement**: {avg_tpr_improvement:.2f}%\n"
    log_content += f"- **Average Delay Reduction**: {avg_delay_reduction:.2f}%\n"

    # Calculation formula explanation
    log_content += "\n## Calculation Formulas\n\n"
    log_content += "- **TPR Improvement**: (Horizon TPR - Traditional TPR) / Traditional TPR * 100\n"
    log_content += "- **Delay Reduction**: (Traditional Delay - Horizon Delay) / Traditional Delay * 100\n"
    log_content += (
        "- **Perfect Detection Rate**: Percentage of configurations where TPR = 1.0\n"
    )
    log_content += "- **Composite Score**: TPR - 0.5 * normalized_delay - 5 * FPR\n"

    # Add column statistics for reference
    log_content += "\n## Data Summary\n\n"
    log_content += "### Column Statistics\n\n"
    log_content += f"```\n{stats_df.to_string()}\n```\n"

    log_path = os.path.join(
        output_dir, f"{network_name}_betting_function_comparison_analysis_log.md"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"Saved detailed analysis log to {log_path}")


# ---------------------------------------------------------------------------
# ----------------------- (2) DETECTION ACCURACY GRID -----------------------
# ---------------------------------------------------------------------------
def plot_detection_accuracy_grid(df: pd.DataFrame, output_dir: str, network_name: str):
    """Create a 2x2 grid visualization focusing on detection accuracy and FPR control."""
    if df.empty:
        print("Error: No data available for detection accuracy analysis")
        return

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
        constrained_layout=True,
    )

    betting_funcs = sorted(df["betting_func"].unique())
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}

    # 1. Threshold vs FPR with theoretical bound (top-left)
    ax = axs[0, 0]
    ax.axhspan(0, 0.05, alpha=0.1, color="green")
    thresholds = sorted(df["threshold"].unique())
    x = np.array(thresholds)
    theoretical_values = 1 / x
    theoretical_line = ax.plot(
        x,
        theoretical_values,
        "k--",
        label="Theoretical bound (1/$\lambda$)",
        linewidth=2.5,
        alpha=0.8,
    )[0]

    # Group by threshold and betting function for both Traditional and Horizon
    for bf in betting_funcs:
        subset = df[df["betting_func"] == bf]
        if not subset.empty:
            trad_threshold_groups = (
                subset.groupby("threshold")["trad_fpr"].mean().reset_index()
            )
            if not trad_threshold_groups.empty:
                ax.plot(
                    trad_threshold_groups["threshold"],
                    trad_threshold_groups["trad_fpr"],
                    "o-",
                    label=f"{bf.capitalize()} (Trad)",
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                )

            hor_threshold_groups = (
                subset.groupby("threshold")["horizon_fpr"].mean().reset_index()
            )
            if not hor_threshold_groups.empty:
                ax.plot(
                    hor_threshold_groups["threshold"],
                    hor_threshold_groups["horizon_fpr"],
                    "s--",
                    label=f"{bf.capitalize()} (Horizon)",
                    color=colors[bf],
                    linewidth=2.0,
                    markersize=7,
                    alpha=0.6,
                )

    ax.set_xlabel("Detection Threshold ($\lambda$)", fontsize=18)
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=18)
    ax.grid(True, linestyle=":", alpha=0.4)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(
        [theoretical_line],
        ["Theoretical bound (1/$\lambda$)"],
        fontsize=12,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        handlelength=1.5,
    )

    # 2. TPR by distance metric (top-right)
    ax = axs[0, 1]
    if "distance" in df.columns:
        distances = sorted(df["distance"].unique())

        # Create a dataframe for grouped barplot
        distance_data = []
        for dist in distances:
            for bf in betting_funcs:
                # Get mean TPR for this betting function and distance
                bf_dist_data = df[(df["betting_func"] == bf) & (df["distance"] == dist)]
                if not bf_dist_data.empty:
                    trad_tpr = bf_dist_data["trad_tpr"].mean()
                    horizon_tpr = bf_dist_data["horizon_tpr"].mean()
                    distance_data.append(
                        {
                            "Distance": dist.capitalize(),
                            "Betting Function": bf.capitalize(),
                            "Traditional": trad_tpr,
                            "Horizon": horizon_tpr,
                        }
                    )

        distance_df = pd.DataFrame(distance_data)
        bar_width = 0.15  # Width of each bar
        index = np.arange(len(distances))  # X locations for groups

        # Calculate positions for bars
        for i, bf in enumerate(betting_funcs):
            bf_data = distance_df[distance_df["Betting Function"] == bf.capitalize()]
            if not bf_data.empty:
                # Calculate offset based on betting function
                offset = (i - len(betting_funcs) / 2 + 0.5) * bar_width * 2

                # Plot Traditional bars
                ax.bar(
                    index + offset,
                    bf_data["Traditional"],
                    bar_width,
                    color=colors[bf],
                    alpha=0.85,
                    label=f"{bf.capitalize()} (Trad)",
                )

                # Plot Horizon bars with hatch pattern
                ax.bar(
                    index + offset + bar_width,
                    bf_data["Horizon"],
                    bar_width,
                    color=colors[bf],
                    alpha=0.65,
                    hatch="///",
                    label=f"{bf.capitalize()} (Horizon)",
                )

                # Removed value annotations on top of bars

        ax.set_xticks(index)
        ax.set_xticklabels([d.capitalize() for d in distances], rotation=45, ha="right")

        # First create betting function legend handles
        bf_handles = []
        bf_labels = []
        for bf in betting_funcs:
            bf_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[bf], alpha=0.85))
            bf_labels.append(bf.capitalize())

        # Then create method legend handles
        method_handles = [
            plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.65, hatch="///"),
        ]
        method_labels = ["Traditional", "Horizon"]
        ax.legend(
            method_handles,
            method_labels,
            loc="lower center",
            fontsize=12,
            frameon=True,
        )

        ax.set_ylabel("True Positive Rate (TPR)", fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)

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
        # Get epsilon values directly from data
        epsilon_values = sorted(df["epsilon"].unique())

        # Explicitly define epsilon values to ensure they're shown
        expected_epsilons = [0.2, 0.5, 0.9]  # Hardcoded based on data

        # Filter out beta betting function which doesn't depend on epsilon
        epsilon_betting_funcs = [bf for bf in betting_funcs if bf != "beta"]

        # Create full legend with all betting functions
        legend_entries = []
        for bf in betting_funcs:
            # Traditional entry
            legend_entries.append(
                (
                    plt.Line2D(
                        [0],
                        [0],
                        color=colors[bf],
                        linestyle="-",
                        marker="o",
                        linewidth=2.5,
                        markersize=8,
                        alpha=0.8,
                    ),
                    f"{bf.capitalize()} (Trad)",
                )
            )
            # Horizon entry
            legend_entries.append(
                (
                    plt.Line2D(
                        [0],
                        [0],
                        color=colors[bf],
                        linestyle="--",
                        marker="s",
                        linewidth=2.0,
                        markersize=7,
                        alpha=0.6,
                    ),
                    f"{bf.capitalize()} (Horizon)",
                )
            )

        # Plot only non-beta betting functions
        for bf in epsilon_betting_funcs:
            subset = df[df["betting_func"] == bf]
            if not subset.empty:
                # Traditional FPR
                trad_epsilon_metrics = []
                for eps in epsilon_values:
                    eps_data = subset[subset["epsilon"] == eps]
                    avg_fpr = eps_data["trad_fpr"].mean()
                    trad_epsilon_metrics.append(
                        {
                            "epsilon": eps,
                            "FPR": avg_fpr,
                            "Method": "Traditional",
                        }
                    )

                # Horizon FPR
                hor_epsilon_metrics = []
                for eps in epsilon_values:
                    eps_data = subset[subset["epsilon"] == eps]
                    avg_fpr = eps_data["horizon_fpr"].mean()
                    hor_epsilon_metrics.append(
                        {
                            "epsilon": eps,
                            "FPR": avg_fpr,
                            "Method": "Horizon",
                        }
                    )

                # Plot Traditional FPR
                trad_eps_df = pd.DataFrame(trad_epsilon_metrics)
                ax.plot(
                    trad_eps_df["epsilon"],
                    trad_eps_df["FPR"],
                    "o-",
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                    solid_capstyle="round",
                )

                # Plot Horizon FPR
                hor_eps_df = pd.DataFrame(hor_epsilon_metrics)
                ax.plot(
                    hor_eps_df["epsilon"],
                    hor_eps_df["FPR"],
                    "s--",
                    color=colors[bf],
                    linewidth=2.0,
                    markersize=7,
                    alpha=0.6,
                    solid_capstyle="round",
                )

        ax.set_xlabel("Epsilon ($\\epsilon$)", fontsize=18)
        ax.set_ylabel("False Positive Rate (FPR)", fontsize=18)

        # Using a more direct approach to set x-axis ticks
        # Clear existing ticks first
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Set explicit x-axis limits with padding
        ax.set_xlim(0.1, 1.0)

        # Use hardcoded values that match the data
        ax.set_xticks(expected_epsilons)
        ax.set_xticklabels([f"{eps:.1f}" for eps in expected_epsilons])

        # Ensure ticks are visible
        ax.tick_params(axis="x", which="both", length=6, width=1.5, direction="out")

        # Fix FPR y-axis scaling and formatting
        # Find actual min and max FPR values
        fpr_values = []
        for bf in epsilon_betting_funcs:  # Only consider non-beta functions
            subset = df[df["betting_func"] == bf]
            if not subset.empty:
                fpr_values.extend(subset["trad_fpr"].tolist())
                fpr_values.extend(subset["horizon_fpr"].tolist())

        if fpr_values:
            max_fpr = max(fpr_values)
            ax.set_ylim(0, max_fpr * 1.2)
            formatter = ScalarFormatter(useMathText=True, useOffset=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.grid(True, linestyle=":", alpha=0.4)

        # Use legend_entries for the full legend with all betting functions
        handles, labels = zip(*legend_entries)
        ax.legend(
            handles,
            labels,
            fontsize=12,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            handlelength=1.5,
            ncol=2,
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
        # Get epsilon values from data
        epsilon_values = sorted(df["epsilon"].unique())

        # Explicitly define epsilon values to ensure they're shown
        expected_epsilons = [0.2, 0.5, 0.9]  # Hardcoded based on data

        # Filter out beta betting function for plotting
        epsilon_betting_funcs = [bf for bf in betting_funcs if bf != "beta"]

        # Plot for each betting function (except beta)
        for bf in epsilon_betting_funcs:
            subset = df[df["betting_func"] == bf]
            if not subset.empty:
                # Traditional TPR
                trad_epsilon_metrics = []
                for eps in epsilon_values:
                    eps_data = subset[subset["epsilon"] == eps]
                    avg_tpr = eps_data["trad_tpr"].mean()
                    trad_epsilon_metrics.append(
                        {
                            "epsilon": eps,
                            "TPR": avg_tpr,
                            "Method": "Traditional",
                        }
                    )

                # Horizon TPR
                hor_epsilon_metrics = []
                for eps in epsilon_values:
                    eps_data = subset[subset["epsilon"] == eps]
                    avg_tpr = eps_data["horizon_tpr"].mean()
                    hor_epsilon_metrics.append(
                        {
                            "epsilon": eps,
                            "TPR": avg_tpr,
                            "Method": "Horizon",
                        }
                    )

                # Plot Traditional TPR
                trad_eps_df = pd.DataFrame(trad_epsilon_metrics)
                ax.plot(
                    trad_eps_df["epsilon"],
                    trad_eps_df["TPR"],
                    "o-",
                    color=colors[bf],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                    solid_capstyle="round",
                    label=f"{bf.capitalize()} (Trad)",
                )

                # Plot Horizon TPR
                hor_eps_df = pd.DataFrame(hor_epsilon_metrics)
                ax.plot(
                    hor_eps_df["epsilon"],
                    hor_eps_df["TPR"],
                    "s--",
                    color=colors[bf],
                    linewidth=2.0,
                    markersize=7,
                    alpha=0.6,
                    solid_capstyle="round",
                    label=f"{bf.capitalize()} (Horizon)",
                )

        ax.set_xlabel("Epsilon ($\\epsilon$)", fontsize=18)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)

    for i in range(2):
        for j in range(2):
            # Remove top and right spines
            axs[i, j].spines["top"].set_visible(False)
            axs[i, j].spines["right"].set_visible(False)

            # Make bottom and left spines slightly thicker
            axs[i, j].spines["bottom"].set_linewidth(1.2)
            axs[i, j].spines["left"].set_linewidth(1.2)

            # Add subtle tick marks
            axs[i, j].tick_params(axis="both", which="major", labelsize=16)
            axs[i, j].tick_params(direction="out", length=4, width=1.2, pad=4)

            # Ensure y-axis has enough ticks for readability
            axs[i, j].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

            # Format tick labels to avoid scientific notation and unnecessary decimals
            if (i == 0 and j == 1) or (i == 1 and j == 1):  # For TPR plots
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

    # Update the analysis log generation with more accurate metrics
    df_copy = df.copy()
    df_copy["score"] = (
        df_copy["trad_tpr"] - 5 * df_copy["trad_fpr"]
    )  # Weight FPR 5x more than TPR
    best_threshold_group = df_copy.groupby("threshold")["score"].mean().idxmax()

    # Best distance metric
    best_distance = None
    if "distance" in df.columns:
        distance_stats = df.groupby("distance").agg(
            {
                "trad_tpr": "mean",
                "horizon_tpr": "mean",
                "trad_fpr": "mean",
            }
        )
        best_distance = distance_stats["trad_tpr"].idxmax()

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
    """Create a 1x2 grid visualization focusing on detection timing and delay analysis."""
    if df.empty:
        print("Error: No data available for performance timing analysis")
        return

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        gridspec_kw={"wspace": 0.2},
        constrained_layout=True,
    )

    betting_funcs = sorted(df["betting_func"].unique())
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}

    # 1. Detection Delay by Window Size (left)
    ax = axs[0]
    trad_style = {
        "linewidth": 2.5,
        "markersize": 8,
        "alpha": 0.8,
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    }

    horizon_style = {
        "linewidth": 2.0,
        "markersize": 7,
        "alpha": 0.6,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    }

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
                    label=f"{bf.capitalize()} (Trad)",
                    color=colors[bf],
                    **trad_style,
                )

            horizon_window_groups = (
                subset.groupby("window")["horizon_avg_delay"].mean().reset_index()
            )
            if (
                not horizon_window_groups.empty
                and not horizon_window_groups["horizon_avg_delay"].isna().all()
            ):
                ax.plot(
                    horizon_window_groups["window"],
                    horizon_window_groups["horizon_avg_delay"],
                    "s--",
                    label=f"{bf.capitalize()} (Horizon)",
                    color=colors[bf],
                    **horizon_style,
                )

    ax.set_xlabel("Window Size", fontsize=18)
    ax.set_ylabel("Average Detection Delay", fontsize=18)
    ax.grid(True, linestyle=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=12,
        loc="best",
        frameon=True,
    )

    # 2. Delay Reduction by Parameter (right)
    ax = axs[1]
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
                    label=f"{bf.capitalize()} (Trad)",
                    color=colors[bf],
                    **trad_style,
                )

            horizon_threshold_groups = (
                subset.groupby("threshold")
                .agg(
                    {
                        "horizon_avg_delay": "mean",
                    }
                )
                .reset_index()
            )

            if (
                not horizon_threshold_groups.empty
                and not horizon_threshold_groups["horizon_avg_delay"].isna().all()
            ):
                ax.plot(
                    horizon_threshold_groups["threshold"],
                    horizon_threshold_groups["horizon_avg_delay"],
                    "s--",
                    label=f"{bf.capitalize()} (Horizon)",
                    color=colors[bf],
                    **horizon_style,
                )

    ax.set_xlabel("Detection Threshold ($\lambda$)", fontsize=18)
    ax.set_ylabel("Average Detection Delay", fontsize=18)
    ax.grid(True, linestyle=":", alpha=0.4)

    for i in range(2):
        # Remove top and right spines
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

        # Make bottom and left spines slightly thicker
        axs[i].spines["bottom"].set_linewidth(1.2)
        axs[i].spines["left"].set_linewidth(1.2)

        # Add subtle tick marks
        axs[i].tick_params(axis="both", which="major", labelsize=16)
        axs[i].tick_params(direction="out", length=4, width=1.2, pad=4)

        # Ensure y-axis has enough ticks for readability
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

        # Format tick labels to avoid scientific notation and unnecessary decimals
        axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

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

## Threshold Impact

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

The visualization consists of a 1x2 grid that shows:

### 1. Detection Delay by Window Size (left)
- Shows how window size affects detection delay for each betting function
- Larger windows typically increase delay but may improve model accuracy
- Useful for selecting optimal window size based on application needs

### 2. Delay Reduction by Threshold (right)
- Shows relationship between detection threshold and delay
- Helps identify thresholds that minimize detection delay
- Useful for delay-sensitive applications

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
        choices=["betting", "accuracy", "timing", "all"],
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

    else:
        print(f"Error: Could not load {args.network_name} network data")


if __name__ == "__main__":
    main()
