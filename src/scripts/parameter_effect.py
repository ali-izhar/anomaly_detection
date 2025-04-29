#!/usr/bin/env python3
"""Generate parameter effect visualizations for the paper.

This script creates a 2x2 grid of plots showing how different parameters
affect the performance of the change detection framework:
(A) Detection threshold vs. false alarm rate
(B) History window vs. detection delay
(C) Prediction horizon vs. detection accuracy
(D) Distance metric comparison across network types
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up paper-style formatting
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Define paths
PARAMETER_EFFECTS_DIR = "results/best_parameters/parameter_effects"
OUTPUT_DIR = "paper/Figures"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load parameter effect data."""
    data = {}

    # Load threshold effects
    data["threshold"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "threshold_effect.csv")
    )
    data["threshold_by_network"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "threshold_effect_by_network.csv")
    )

    # Load window effects
    data["window"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "window_effect.csv")
    )
    data["window_by_network"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "window_effect_by_network.csv")
    )

    # Load horizon effects
    data["horizon"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "horizon_effect.csv")
    )
    data["horizon_by_network"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "horizon_effect_by_network.csv")
    )

    # Load distance metric effects
    data["distance"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "distance_effect.csv")
    )
    data["distance_by_network"] = pd.read_csv(
        os.path.join(PARAMETER_EFFECTS_DIR, "distance_effect_by_network.csv")
    )

    return data


def create_parameter_effect_plot(data):
    """Create a 2x2 grid of parameter effect plots."""
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Network name mapping for pretty printing
    network_names = {
        "sbm": "SBM",
        "ba": "BA",
        "er": "ER",
        "ws": "NWS",
    }

    # Use a more professional, research-oriented color palette
    # palette = sns.color_palette("viridis", 4)
    palette = sns.color_palette("colorblind", 4)
    network_colors = {
        "sbm": palette[0],
        "ba": palette[1],
        "er": palette[2],
        "ws": palette[3],
    }

    # Define marker styles for consistency
    markers = {"sbm": "o", "ba": "s", "er": "^", "ws": "D"}

    # Add background grid for better readability
    grid_params = {
        "linestyle": "--",
        "alpha": 0.3,
        "color": "gray",
        "linewidth": 0.5,
    }

    # Create subplot A: Detection threshold vs. false positive rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.grid(True, **grid_params)

    # Calculate FPR as 1/threshold (theoretical upper bound)
    thresholds = np.array([20, 50, 60, 100])
    theoretical_fpr = 1 / thresholds

    # Plot theoretical line
    ax1.plot(thresholds, theoretical_fpr, "k--", label="Theoretical bound (1/λ)")

    # Extract FPR data from the complete dataset - since it's not directly in the CSV
    # We'll extract it from the all_results dataframe
    all_results = pd.read_csv("results/best_parameters/all_results.csv")

    # Group by threshold and calculate average FPR
    threshold_fpr = all_results.groupby("threshold")["fpr"].mean().reset_index()

    # Plot empirical FPR
    ax1.plot(
        threshold_fpr["threshold"],
        threshold_fpr["fpr"],
        "bo-",
        label="Traditional",
        linewidth=2,
        markersize=8,
    )

    # Also plot horizon FPR if available
    if "horizon_fpr" in all_results.columns:
        threshold_horizon_fpr = (
            all_results.groupby("threshold")["horizon_fpr"].mean().reset_index()
        )
        ax1.plot(
            threshold_horizon_fpr["threshold"],
            threshold_horizon_fpr["horizon_fpr"],
            "ro-",
            label="Horizon",
            linewidth=2,
            markersize=8,
        )

    ax1.set_xlabel("Detection Threshold (λ)")
    ax1.set_ylabel("False Positive Rate")
    ax1.set_title("(A) Threshold vs. FPR")
    ax1.legend(frameon=True, fancybox=True, shadow=True)

    # Create subplot B: History window vs. detection delay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.grid(True, **grid_params)

    # Plot for each network type
    for network in ["sbm", "ba", "er", "ws"]:
        network_data = data["window_by_network"][
            data["window_by_network"]["network"] == network
        ]
        ax2.plot(
            network_data["window"],
            network_data["avg_delay"],
            "-",
            marker=markers[network],
            label=network_names[network],
            color=network_colors[network],
            linewidth=2,
            markersize=8,
        )

    ax2.set_xlabel("History Window Size (w)")
    ax2.set_ylabel("Average Detection Delay")
    ax2.set_title("(B) Window Size vs. Delay")
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    # Create subplot C: Prediction horizon vs. TPR (detection rate)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.grid(True, **grid_params)

    # Plot for each network type
    for network in ["sbm", "ba", "er", "ws"]:
        network_data = data["horizon_by_network"][
            data["horizon_by_network"]["network"] == network
        ]
        ax3.plot(
            network_data["horizon"],
            network_data["tpr"],
            "-",
            marker=markers[network],
            label=network_names[network],
            color=network_colors[network],
            linewidth=2,
            markersize=8,
        )

    ax3.set_xlabel("Prediction Horizon (h)")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("(C) Horizon vs. Detection Rate")
    ax3.legend(frameon=True, fancybox=True, shadow=True)

    # Create subplot D: Distance metric comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.grid(True, axis="y", **grid_params)

    # Prepare data for grouped bar chart
    distance_metrics = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
    metric_pretty = {
        "euclidean": "Euclidean",
        "mahalanobis": "Mahalanobis",
        "cosine": "Cosine",
        "chebyshev": "Chebyshev",
    }

    # Define hatching patterns for better distinction in black and white printing
    hatches = ["", "///", "...", "xxx"]

    # Filter by distance and network
    network_distance_tpr = {}

    for network in ["sbm", "ba", "er", "ws"]:
        network_distance_tpr[network] = []
        for dist in distance_metrics:
            filtered = data["distance_by_network"][
                (data["distance_by_network"]["network"] == network)
                & (data["distance_by_network"]["distance"] == dist)
            ]
            if not filtered.empty:
                network_distance_tpr[network].append(filtered["tpr"].values[0])
            else:
                network_distance_tpr[network].append(0)  # Default if no data

    # Set width of bars - make more compact
    barWidth = 0.17  # reduced from 0.2
    r1 = np.arange(len(distance_metrics))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Create grouped bar chart
    bars = []
    bars.append(
        ax4.bar(
            r1,
            network_distance_tpr["sbm"],
            width=barWidth,
            label=network_names["sbm"],
            color=network_colors["sbm"],
            edgecolor="black",
            linewidth=1,
            hatch=hatches[0],
        )
    )
    bars.append(
        ax4.bar(
            r2,
            network_distance_tpr["ba"],
            width=barWidth,
            label=network_names["ba"],
            color=network_colors["ba"],
            edgecolor="black",
            linewidth=1,
            hatch=hatches[1],
        )
    )
    bars.append(
        ax4.bar(
            r3,
            network_distance_tpr["er"],
            width=barWidth,
            label=network_names["er"],
            color=network_colors["er"],
            edgecolor="black",
            linewidth=1,
            hatch=hatches[2],
        )
    )
    bars.append(
        ax4.bar(
            r4,
            network_distance_tpr["ws"],
            width=barWidth,
            label=network_names["ws"],
            color=network_colors["ws"],
            edgecolor="black",
            linewidth=1,
            hatch=hatches[3],
        )
    )

    # Add labels and legend
    ax4.set_xlabel("Distance Metric")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title("(D) Distance Metric Comparison")
    ax4.set_xticks([r + barWidth * 1.5 for r in range(len(distance_metrics))])
    ax4.set_xticklabels([metric_pretty[m] for m in distance_metrics])
    # Set specific y-axis ticks
    ax4.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    # Position legend inside the plot at the bottom
    ax4.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        fontsize=8,
        borderaxespad=0.1,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "parameter_effects.png"), dpi=300)
    print(f"Figure saved to {os.path.join(OUTPUT_DIR, 'parameter_effects.png')}")


def main():
    """Main function to generate parameter effect plot."""
    data = load_data()
    create_parameter_effect_plot(data)


if __name__ == "__main__":
    main()
