#!/usr/bin/env python3
"""Network Hyperparameter Analysis Plotting.
This script creates concise, informative visualizations from hyperparameter analysis results."""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Set plot styles - use modern aesthetics
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")  # Larger context for better readability in presentations
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande"]
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["axes.titlepad"] = 15

# Define a modern color palette based on network types
NETWORK_NAMES = {
    "sbm": "SBM",
    "ba": "BA",
    "er": "ER",
    "ws": "NWS",
}

# Professional color palette inspired by scientific publications
NETWORK_COLORS = {
    "sbm": "#3182bd",  # Blue
    "ba": "#e6550d",  # Orange
    "er": "#31a354",  # Green
    "ws": "#756bb1",  # Purple
}

# Betting function color palette (complementary to network colors)
BETTING_COLORS = {
    "beta": "#fd8d3c",  # Light orange
    "mixture": "#637939",  # Olive green
    "power": "#7b4173",  # Purple
    "normal": "#3182bd",  # Blue
}

# Custom colormap for heatmaps
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_diverging", ["#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d"], N=256
)


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


def apply_style_to_axes(ax, title=None, xlabel=None, ylabel=None, legend=True):
    """Apply consistent styling to plot axes."""
    if title:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Apply grid styling
    ax.grid(True, linestyle="--", alpha=0.7)

    # Style the spines
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    if legend:
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, edgecolor="#cccccc")

    return ax


def plot_key_parameter_effects(output_dir: str):
    """Create publication-quality plots showing key parameter effects across networks."""
    # Load data for all network types
    networks = ["sbm", "ba", "er", "ws"]
    network_data = {}

    for network in networks:
        df = load_analysis_results(network)
        if df is not None:
            network_data[network] = preprocess_data(df)

    if not network_data:
        print("Error: No data available for any network type")
        return

    # Create figure with 2x2 subplots using GridSpec for better control
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

    fig.suptitle(
        "Key Parameter Effects on Network Anomaly Detection",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Add a subtle footnote
    fig.text(
        0.5,
        0.02,
        "Analysis of detection performance across different network types and parameters",
        ha="center",
        fontsize=10,
        style="italic",
        color="#666666",
    )

    # Plot A: Threshold vs. FPR with theoretical bound
    ax = fig.add_subplot(gs[0, 0])

    # Add reference zone for typical operating range
    ax.axhspan(0, 0.05, alpha=0.1, color="green", label="Acceptable FPR Range")

    # Theoretical bound λ = 1/threshold
    thresholds = np.linspace(20, 100, 100)
    theoretical_bound = 1 / thresholds
    ax.plot(
        thresholds,
        theoretical_bound,
        "k--",
        label="Theoretical bound (1/λ)",
        linewidth=2,
        alpha=0.8,
    )

    network_markers = ["o", "s", "^", "d"]  # Different markers for each network
    line_styles = ["-", "--", "-.", ":"]  # Different line styles

    # Plot each network with distinct markers and colors
    for i, (network, df) in enumerate(network_data.items()):
        if "trad_fpr" in df.columns and "threshold" in df.columns:
            threshold_groups = (
                df.groupby("threshold", observed=True)["trad_fpr"].mean().reset_index()
            )

            if not threshold_groups.empty:
                ax.plot(
                    threshold_groups["threshold"],
                    threshold_groups["trad_fpr"],
                    marker=network_markers[i % len(network_markers)],
                    linestyle=line_styles[i % len(line_styles)],
                    color=NETWORK_COLORS[network],
                    linewidth=2,
                    markersize=8,
                    alpha=0.8,
                    label=f"{NETWORK_NAMES.get(network, network.upper())}",
                )

    apply_style_to_axes(
        ax,
        title="A. Threshold vs. False Positive Rate",
        xlabel="Detection Threshold (λ)",
        ylabel="False Positive Rate",
    )

    # Add annotations to highlight key insights
    if any(["sbm" in network_data]):
        min_threshold = min(network_data["sbm"]["threshold"])
        max_threshold = max(network_data["sbm"]["threshold"])
        ax.annotate(
            "Higher thresholds\nreduce false alarms",
            xy=(max_threshold * 0.8, 0.04),
            xytext=(max_threshold * 0.6, 0.07),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            ha="center",
        )

    # Plot B: Window Size vs. Delay
    ax = fig.add_subplot(gs[0, 1])

    for i, (network, df) in enumerate(network_data.items()):
        if "traditional_avg_delay" in df.columns and "window" in df.columns:
            window_groups = (
                df.groupby("window", observed=True)["traditional_avg_delay"]
                .mean()
                .reset_index()
            )

            if not window_groups.empty:
                ax.plot(
                    window_groups["window"],
                    window_groups["traditional_avg_delay"],
                    marker=network_markers[i % len(network_markers)],
                    linestyle=line_styles[i % len(line_styles)],
                    color=NETWORK_COLORS[network],
                    linewidth=2,
                    markersize=8,
                    alpha=0.8,
                    label=f"{NETWORK_NAMES.get(network, network.upper())}",
                )

    apply_style_to_axes(
        ax,
        title="B. Window Size vs. Detection Delay",
        xlabel="History Window Size (w)",
        ylabel="Average Detection Delay (timesteps)",
    )

    # Add shaded region to highlight optimal window size
    if any(["window" in df.columns for df in network_data.values()]):
        min_window = min(
            [
                df["window"].min()
                for df in network_data.values()
                if "window" in df.columns
            ]
        )
        max_window = max(
            [
                df["window"].max()
                for df in network_data.values()
                if "window" in df.columns
            ]
        )

        # Find a reasonable "sweet spot" for window size based on available data
        sweet_spot = min_window + (max_window - min_window) // 3
        ax.axvspan(
            sweet_spot - 1,
            sweet_spot + 1,
            alpha=0.15,
            color="blue",
            label="Optimal window range",
        )

        ax.annotate(
            "Optimal balance",
            xy=(sweet_spot, 15),
            xytext=(sweet_spot + 2, 10),
            arrowprops=dict(
                facecolor="blue", shrink=0.05, width=1, headwidth=7, alpha=0.6
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            ha="center",
            fontsize=10,
        )

    # Plot C: Horizon vs. Detection Rate
    ax = fig.add_subplot(gs[1, 0])

    for i, (network, df) in enumerate(network_data.items()):
        if "horizon_tpr" in df.columns and "horizon" in df.columns:
            horizon_groups = (
                df.groupby("horizon", observed=True)["horizon_tpr"].mean().reset_index()
            )

            if not horizon_groups.empty:
                ax.plot(
                    horizon_groups["horizon"],
                    horizon_groups["horizon_tpr"],
                    marker=network_markers[i % len(network_markers)],
                    linestyle=line_styles[i % len(line_styles)],
                    color=NETWORK_COLORS[network],
                    linewidth=2,
                    markersize=8,
                    alpha=0.8,
                    label=f"{NETWORK_NAMES.get(network, network.upper())}",
                )

    apply_style_to_axes(
        ax,
        title="C. Prediction Horizon vs. Detection Rate",
        xlabel="Prediction Horizon (h)",
        ylabel="True Positive Rate",
    )

    # Add reference lines
    ax.axhline(y=0.8, linestyle=":", color="green", alpha=0.6, linewidth=1.5)
    ax.text(5, 0.81, "Good detection (TPR = 0.8)", fontsize=9, color="green")

    # Plot D: Distance Metric Comparison
    ax = fig.add_subplot(gs[1, 1])

    # Collect data for all networks and distance metrics
    distance_metrics = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
    data = []

    for network in networks:
        if network in network_data:
            df = network_data[network]
            for distance in distance_metrics:
                subset = df[df["distance"] == distance]
                if not subset.empty:
                    tpr = subset["trad_tpr"].mean()
                    data.append(
                        {
                            "network": NETWORK_NAMES.get(network, network.upper()),
                            "distance": distance.capitalize(),
                            "tpr": tpr,
                        }
                    )

    if data:
        distance_df = pd.DataFrame(data)

        # Plot grouped bar chart with improved aesthetics
        bar_width = 0.2
        x = np.arange(len(distance_metrics))

        for i, network in enumerate(NETWORK_NAMES.values()):
            network_data = distance_df[distance_df["network"] == network]
            if not network_data.empty:
                # Create a mapping from distance to TPR
                distance_to_tpr = {
                    row["distance"]: row["tpr"] for _, row in network_data.iterrows()
                }

                # Get TPR values in the correct order
                tpr_values = [
                    distance_to_tpr.get(d.capitalize(), 0) for d in distance_metrics
                ]

                # Plot bars with appropriate offset and style
                bars = ax.bar(
                    x + (i - 1.5) * bar_width,
                    tpr_values,
                    bar_width,
                    label=network,
                    color=NETWORK_COLORS[network.lower()],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                )

                # Add value labels on top of bars
                for bar_idx, bar in enumerate(bars):
                    if (
                        tpr_values[bar_idx] > 0.1
                    ):  # Only label bars with significant values
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02,
                            f"{tpr_values[bar_idx]:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=0,
                        )

        apply_style_to_axes(
            ax,
            title="D. Distance Metric Comparison",
            xlabel="Distance Metric",
            ylabel="True Positive Rate",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in distance_metrics])
        ax.set_ylim(0, 1.05)  # Leave room for bar value labels

    # Add a text box with key findings
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc")
    textstr = "\nKey Findings:\n\n"
    textstr += "• Higher thresholds reduce false positives but may delay detection\n"
    textstr += "• Window size affects detection delay - smaller windows react faster\n"
    textstr += "• Horizon prediction improves early detection in most networks\n"
    textstr += "• Distance metrics impact performance differently across networks"

    fig.text(
        0.5,
        0.48,
        textstr,
        transform=fig.transFigure,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=props,
    )

    # Save the figure with high resolution
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_path = os.path.join(output_dir, "key_parameter_effects.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Also save as PDF for publication quality
    pdf_path = os.path.join(output_dir, "key_parameter_effects.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()

    print(f"Comparative parameter analysis saved to {output_path}")


def plot_network_specific_analysis(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Create focused analysis plots for a specific network using consolidated grids."""
    # Create the consolidated grid layouts
    plot_network_parameter_grid(df, output_dir, network_name)

    # Optionally create a betting function comparison plot
    if (
        "betting_func" in df.columns
        and "trad_tpr" in df.columns
        and "traditional_avg_delay" in df.columns
    ):
        plot_betting_function_comparison(df, output_dir, network_name)

    # Create performance tradeoff visualization
    plot_performance_tradeoff(df, output_dir, network_name)


def plot_performance_tradeoff(df: pd.DataFrame, output_dir: str, network_name: str):
    """Create a visualization showing the tradeoff between TPR and detection delay."""
    if "trad_tpr" not in df.columns or "traditional_avg_delay" not in df.columns:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color mapping based on threshold
    norm = plt.Normalize(df["threshold"].min(), df["threshold"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # Create scatter plot with size based on TPR and color based on threshold
    scatter = ax.scatter(
        df["traditional_avg_delay"],
        df["trad_tpr"],
        c=df["threshold"],
        cmap="viridis",
        s=df["trad_tpr"] * 200 + 50,  # Size based on TPR
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Threshold (λ)", rotation=270, labelpad=20)

    # Annotate the Pareto frontier
    # Find points that are not dominated by any other point (higher TPR and lower delay)
    pareto_points = []

    for _, row in df.iterrows():
        tpr = row["trad_tpr"]
        delay = row["traditional_avg_delay"]

        # Check if this point is dominated
        dominated = False
        for _, other_row in df.iterrows():
            other_tpr = other_row["trad_tpr"]
            other_delay = other_row["traditional_avg_delay"]

            # If another point has higher TPR and lower delay, this point is dominated
            if (
                other_tpr >= tpr
                and other_delay <= delay
                and (other_tpr > tpr or other_delay < delay)
            ):
                dominated = True
                break

        if not dominated:
            pareto_points.append((delay, tpr, row))

    # Sort by delay for connecting line
    pareto_points.sort(key=lambda x: x[0])

    # Draw Pareto frontier line
    if pareto_points:
        pareto_delays = [p[0] for p in pareto_points]
        pareto_tprs = [p[1] for p in pareto_points]
        ax.plot(
            pareto_delays,
            pareto_tprs,
            "r--",
            linewidth=2,
            alpha=0.7,
            label="Pareto Frontier",
        )

        # Annotate a few key points on the Pareto frontier
        for i, (delay, tpr, row) in enumerate(pareto_points):
            if i % max(1, len(pareto_points) // 3) == 0:  # Annotate only a few points
                config_text = f"T={row['threshold']}, W={row['window']}"
                ax.annotate(
                    config_text,
                    xy=(delay, tpr),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

    # Add optimal region shading and annotations
    optimal_x = df["traditional_avg_delay"].min() * 1.5
    optimal_y = 0.8

    # Add quadrant labels
    ax.text(
        df["traditional_avg_delay"].max() * 0.8,
        0.2,
        "Poor Performance\n(Low TPR, High Delay)",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8),
    )

    ax.text(
        df["traditional_avg_delay"].min() * 1.5,
        0.9,
        "Optimal Region\n(High TPR, Low Delay)",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", fc="mintcream", alpha=0.8),
    )

    # Style the plot
    ax.set_title(
        f"Performance Tradeoff Analysis for {NETWORK_NAMES.get(network_name, network_name.upper())} Network",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Detection Delay (timesteps)")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend for betting functions if available
    if "betting_func" in df.columns:
        betting_funcs = df["betting_func"].unique()

        # Create legend elements
        legend_elements = []
        for bf in betting_funcs:
            color = BETTING_COLORS.get(bf, "#999999")
            legend_elements.append(
                Patch(
                    facecolor=color, edgecolor="white", alpha=0.7, label=bf.capitalize()
                )
            )

        # Add second legend for betting functions
        ax.legend(
            handles=legend_elements,
            title="Betting Function",
            loc="lower right",
            frameon=True,
            framealpha=0.9,
        )

    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{network_name}_performance_tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_network_parameter_grid(df: pd.DataFrame, output_dir: str, network_name: str):
    """Create consolidated grid layouts of parameter effects for better visualization."""
    # Create a 2x2 grid showing key parameter effects
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

    fig.suptitle(
        f"Parameter Effects on {NETWORK_NAMES.get(network_name, network_name.upper())} Network Detection",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Add a subtle explanatory subheading
    fig.text(
        0.5,
        0.02,
        f"Analysis of how different parameters affect detection performance in {NETWORK_NAMES.get(network_name, network_name.upper())} networks",
        ha="center",
        fontsize=10,
        style="italic",
        color="#666666",
    )

    # Grid 1 (Top Left): Threshold vs FPR with theoretical bound
    ax = fig.add_subplot(gs[0, 0])

    # Add reference zone for typical operating range
    ax.axhspan(0, 0.05, alpha=0.1, color="green", label="Acceptable FPR Range")

    if "trad_fpr" in df.columns and "theoretical_bound" in df.columns:
        # Theoretical bound
        thresholds = sorted(df["threshold"].unique())
        if thresholds:
            x = np.array(thresholds)
            theoretical_values = (
                df.groupby("threshold")["theoretical_bound"].first().values
            )
            ax.plot(
                x,
                theoretical_values,
                "k--",
                label="Theoretical bound",
                linewidth=2,
                alpha=0.8,
            )

        # Group by threshold for actual values
        threshold_groups = (
            df.groupby("threshold", observed=True)["trad_fpr"].mean().reset_index()
        )

        ax.plot(
            threshold_groups["threshold"],
            threshold_groups["trad_fpr"],
            "o-",
            color=NETWORK_COLORS[network_name],
            linewidth=2.5,
            markersize=8,
            label="Traditional",
        )

        if "horizon_fpr" in df.columns:
            h_groups = (
                df.groupby("threshold", observed=True)["horizon_fpr"]
                .mean()
                .reset_index()
            )
            ax.plot(
                h_groups["threshold"],
                h_groups["horizon_fpr"],
                "o--",
                color=NETWORK_COLORS[network_name],
                alpha=0.7,
                markersize=8,
                label="Horizon",
            )

        apply_style_to_axes(
            ax,
            title="A. Threshold vs. False Positive Rate",
            xlabel="Detection Threshold (λ)",
            ylabel="False Positive Rate",
        )

        # Add an annotation explaining the theoretical bound
        ax.annotate(
            "Theoretical\nbound",
            xy=(max(thresholds) * 0.9, theoretical_values[-1] * 1.2),
            xytext=(max(thresholds) * 0.7, theoretical_values[-1] * 3),
            arrowprops=dict(facecolor="black", shrink=0.05, alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            ha="center",
        )

    # Grid 2 (Top Right): Window Size vs Delay by Betting Function
    ax = fig.add_subplot(gs[0, 1])

    if "traditional_avg_delay" in df.columns:
        betting_funcs = sorted(df["betting_func"].unique())
        cmap = plt.cm.tab10

        for i, betting_func in enumerate(betting_funcs):
            subset = df[df["betting_func"] == betting_func]
            w_groups = (
                subset.groupby("window", observed=True)["traditional_avg_delay"]
                .mean()
                .reset_index()
            )

            # Use specific colors from our betting function palette
            color = BETTING_COLORS.get(betting_func, cmap(i % 10))

            ax.plot(
                w_groups["window"],
                w_groups["traditional_avg_delay"],
                "o-",
                label=betting_func.capitalize(),
                color=color,
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )

        apply_style_to_axes(
            ax,
            title="B. Window Size vs. Detection Delay",
            xlabel="History Window Size (w)",
            ylabel="Average Detection Delay (timesteps)",
        )

        # Add annotation highlighting the trend
        if len(betting_funcs) > 0:
            ax.annotate(
                "Larger windows\nincrease delay",
                xy=(df["window"].max() * 0.8, df["traditional_avg_delay"].max() * 0.7),
                xytext=(
                    df["window"].max() * 0.5,
                    df["traditional_avg_delay"].max() * 0.9,
                ),
                arrowprops=dict(facecolor="black", shrink=0.05, alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                ha="center",
            )

    # Grid 3 (Bottom Left): Horizon vs TPR
    ax = fig.add_subplot(gs[1, 0])

    if "horizon_tpr" in df.columns:
        for i, betting_func in enumerate(betting_funcs):
            subset = df[df["betting_func"] == betting_func]
            h_groups = (
                subset.groupby("horizon", observed=True)["horizon_tpr"]
                .mean()
                .reset_index()
            )

            # Use specific colors from our betting function palette
            color = BETTING_COLORS.get(betting_func, cmap(i % 10))

            ax.plot(
                h_groups["horizon"],
                h_groups["horizon_tpr"],
                "o-",
                label=betting_func.capitalize(),
                color=color,
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )

        apply_style_to_axes(
            ax,
            title="C. Prediction Horizon vs. Detection Rate",
            xlabel="Prediction Horizon (h)",
            ylabel="True Positive Rate",
        )

    # Grid 4 (Bottom Right): Distance Metric Comparison
    ax = fig.add_subplot(gs[1, 1])

    if "distance" in df.columns and "trad_tpr" in df.columns:
        # Prepare data for distance effects
        distance_data = []

        # Get all unique betting functions and distances
        betting_funcs = df["betting_func"].unique()
        distances = df["distance"].unique()

        for betting_func in betting_funcs:
            for distance in distances:
                subset = df[
                    (df["betting_func"] == betting_func) & (df["distance"] == distance)
                ]
                if not subset.empty:
                    tpr = subset["trad_tpr"].mean()
                    distance_data.append(
                        {"betting_func": betting_func, "distance": distance, "tpr": tpr}
                    )

        if distance_data:
            distance_df = pd.DataFrame(distance_data)

            # Create custom barplot with better aesthetics
            plt.sca(ax)

            # Create the barplot with enhanced aesthetics
            bars = sns.barplot(
                x="distance",
                y="tpr",
                hue="betting_func",
                data=distance_df,
                palette=BETTING_COLORS,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                ax=ax,
            )

            # Add value labels on the bars
            for container in bars.containers:
                bars.bar_label(container, fmt="%.2f", fontsize=8)

            apply_style_to_axes(
                ax,
                title="D. Distance Metric Comparison",
                xlabel="Distance Metric",
                ylabel="True Positive Rate",
                legend=False,
            )

            # Improve legend - place it in a better position
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                [l.capitalize() for l in labels],
                title="Betting Function",
                loc="upper left",
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                mode="expand",
                borderaxespad=0,
                ncol=len(betting_funcs),
                frameon=True,
                fancybox=True,
            )

            ax.set_ylim(0, 1.05)  # Make room for bar labels

    # Add a text box with key observations
    textstr = f"Key {NETWORK_NAMES.get(network_name, network_name.upper())} Network Observations:\n\n"

    # Add insights based on the data
    observations = []

    if "trad_tpr" in df.columns and df["trad_tpr"].mean() > 0.5:
        observations.append(
            f"• Good overall detection rate (avg TPR: {df['trad_tpr'].mean():.2f})"
        )
    else:
        observations.append(
            f"• Moderate detection rate (avg TPR: {df['trad_tpr'].mean():.2f})"
        )

    if "traditional_avg_delay" in df.columns:
        observations.append(
            f"• Average detection delay: {df['traditional_avg_delay'].mean():.1f} timesteps"
        )

    if "betting_func" in df.columns:
        best_bf = df.groupby("betting_func")["trad_tpr"].mean().idxmax()
        observations.append(
            f"• '{best_bf.capitalize()}' betting function performs best"
        )

    if "distance" in df.columns:
        best_dist = df.groupby("distance")["trad_tpr"].mean().idxmax()
        observations.append(
            f"• '{best_dist.capitalize()}' distance metric is most effective"
        )

    textstr += "\n".join(observations)

    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc")
    fig.text(
        0.5,
        0.48,
        textstr,
        transform=fig.transFigure,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=props,
    )

    # Save the figure with high resolution
    output_path = os.path.join(output_dir, f"{network_name}_parameter_effects_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Network parameter analysis saved to {output_path}")


def plot_betting_function_comparison(
    df: pd.DataFrame, output_dir: str, network_name: str
):
    """Compare the impact of different betting functions on performance."""
    # Create a more professional visualization with multiple metrics
    if "betting_func" not in df.columns:
        return

    # Create figure with subfigures
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    fig.suptitle(
        f"Impact of Betting Functions on {NETWORK_NAMES.get(network_name, network_name.upper())} Network Performance",
        fontsize=16,
        fontweight="bold",
    )

    # Get unique betting functions
    betting_funcs = sorted(df["betting_func"].unique())

    # Create colors dictionary for consistent coloring
    colors = {bf: BETTING_COLORS.get(bf, f"C{i}") for i, bf in enumerate(betting_funcs)}

    # Create a copy of the dataframe with betting_func as category for proper ordering
    plot_df = df.copy()
    plot_df["betting_func"] = pd.Categorical(
        plot_df["betting_func"], categories=betting_funcs
    )

    # 1. Plot TPR by betting function
    ax = axs[0]

    # Create a more aesthetically pleasing boxplot
    boxprops = dict(linewidth=2, alpha=0.8)
    whiskerprops = dict(linewidth=2, alpha=0.8)
    medianprops = dict(linewidth=2, color="black")

    # Fix: use hue parameter correctly
    bplot = sns.boxplot(
        x="betting_func",
        y="trad_tpr",
        hue="betting_func",  # Add hue parameter
        data=plot_df,
        ax=ax,
        palette=colors,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showfliers=False,
        legend=False,  # Don't show legend since hue is same as x
    )

    # Add individual data points for more detail
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

    apply_style_to_axes(
        ax,
        title="A. True Positive Rate",
        xlabel="Betting Function",
        ylabel="True Positive Rate",
        legend=False,
    )

    # Add mean value labels above boxplots
    for i, bf in enumerate(betting_funcs):
        mean_val = df[df["betting_func"] == bf]["trad_tpr"].mean()
        ax.text(i, 1.02, f"μ={mean_val:.2f}", ha="center", fontsize=9)

    ax.set_ylim(0, 1.1)

    # 2. Plot Detection Delay by betting function
    ax = axs[1]

    # Fix: use hue parameter correctly
    bplot = sns.boxplot(
        x="betting_func",
        y="traditional_avg_delay",
        hue="betting_func",  # Add hue parameter
        data=plot_df,
        ax=ax,
        palette=colors,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showfliers=False,
        legend=False,  # Don't show legend since hue is same as x
    )

    # Add individual data points
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

    apply_style_to_axes(
        ax,
        title="B. Detection Delay",
        xlabel="Betting Function",
        ylabel="Average Detection Delay (timesteps)",
        legend=False,
    )

    # Add mean value labels above boxplots
    for i, bf in enumerate(betting_funcs):
        mean_val = df[df["betting_func"] == bf]["traditional_avg_delay"].mean()
        if not np.isnan(mean_val):
            ax.text(
                i,
                df["traditional_avg_delay"].max() * 1.05,
                f"μ={mean_val:.1f}",
                ha="center",
                fontsize=9,
            )

    # 3. Plot FPR by betting function
    ax = axs[2]

    # Fix: use hue parameter correctly
    bplot = sns.boxplot(
        x="betting_func",
        y="trad_fpr",
        hue="betting_func",  # Add hue parameter
        data=plot_df,
        ax=ax,
        palette=colors,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showfliers=False,
        legend=False,  # Don't show legend since hue is same as x
    )

    # Add individual data points
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

    apply_style_to_axes(
        ax,
        title="C. False Positive Rate",
        xlabel="Betting Function",
        ylabel="False Positive Rate",
        legend=False,
    )

    # Add mean value labels above boxplots
    for i, bf in enumerate(betting_funcs):
        mean_val = df[df["betting_func"] == bf]["trad_fpr"].mean()
        max_fpr = max(0.05, df["trad_fpr"].max() * 1.1)  # Ensure enough space for label
        ax.text(i, max_fpr, f"μ={mean_val:.3f}", ha="center", fontsize=9)

    ax.set_ylim(0, max(0.05, df["trad_fpr"].max() * 1.2))  # Set reasonable y limit

    # Add reference line for acceptable FPR
    ax.axhline(y=0.05, linestyle="--", color="red", alpha=0.5, linewidth=1.5)
    ax.text(
        len(betting_funcs) / 2 - 0.5,
        0.052,
        "Acceptable FPR threshold",
        fontsize=8,
        color="red",
        ha="center",
    )

    # Add best betting function recommendation
    # Calculate a composite score (high TPR, low delay, low FPR)
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

        # Normalize delay to 0-1 range for fair comparison
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

    # Add text annotation with recommendation
    textstr = f"Recommended Betting Function: '{best_bf.capitalize()}'\n\n"
    textstr += f"• TPR: {bf_metrics[best_bf]['tpr']:.2f}\n"
    textstr += f"• Delay: {bf_metrics[best_bf]['delay']:.1f} timesteps\n"
    textstr += f"• FPR: {bf_metrics[best_bf]['fpr']:.4f}"

    props = dict(
        boxstyle="round", facecolor="white", alpha=0.9, edgecolor=colors[best_bf]
    )
    fig.text(
        0.5,
        0.02,
        textstr,
        transform=fig.transFigure,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="center",
        bbox=props,
    )

    # Save the figure
    output_path = os.path.join(
        output_dir, f"{network_name}_betting_function_comparison.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Betting function comparison saved to {output_path}")


def main():
    """Main entry point for network hyperparameter analysis plotting."""
    parser = argparse.ArgumentParser(
        description="Create focused, informative plots for hyperparameter analysis."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Network type to analyze (e.g., sbm, ba, er, ws), or 'all' for comparative analysis",
    )

    args = parser.parse_args()

    if args.network.lower() == "all":
        # Create output directory for comparative plots
        output_dir = "results/hyperparameter_analysis/plots/comparative"
        os.makedirs(output_dir, exist_ok=True)
        print("Creating key parameter effects plot comparing all network types...")
        plot_key_parameter_effects(output_dir)
        print(f"Comparative plots saved to {output_dir}")
    else:
        # Load and preprocess data for specific network
        results_df = load_analysis_results(args.network)
        if results_df is None:
            return

        df_clean = preprocess_data(results_df)
        output_dir = f"results/hyperparameter_analysis/plots/{args.network}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating focused analysis plots for network '{args.network}'...")
        plot_network_specific_analysis(df_clean, output_dir, args.network)
        print(f"Network-specific plots saved to {output_dir}")


if __name__ == "__main__":
    main()
