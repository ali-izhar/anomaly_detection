import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import seaborn as sns

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_ba_graphs import generate_ba_graphs
from src.graph.features import adjacency_to_graph
from src.changepoint import ChangePointDetector


def visualize_martingale_analysis(
    graphs: List[np.ndarray],
    time_points: List[int],
    martingales: Dict[str, Dict[str, Any]],
    martingales_cumulative: Dict[str, Dict[str, Any]],
    change_points: List[int],
    output_dir: str = "martingale_outputs",
) -> None:
    """Create visualization combining graph structure and martingale analysis."""
    # Set style for professional plotting
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create figure with 3 rows
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # Top row: Graph structure evolution
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(gs[0, i])
        G = adjacency_to_graph(graphs[t])

        # Node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [1000 * (v + 1) / max(degrees.values()) for v in degrees.values()]

        # Color nodes by degree centrality
        node_colors = list(degrees.values())

        # Spring layout for consistent positioning
        pos = nx.spring_layout(G, k=1, iterations=50)

        nx.draw(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=6,
            edge_color="gray",
            alpha=0.7,
            ax=ax,
            cmap=plt.cm.viridis,
        )

        stats = (
            f"t={t}\nN={G.number_of_nodes()}, E={G.number_of_edges()}\n"
            f"Avg deg={np.mean(list(degrees.values())):.1f}"
        )
        ax.set_title(stats, fontsize=8)

    # Middle row: Reset Martingales (normal scale)
    ax_reset = fig.add_subplot(gs[1, :])
    colors = sns.color_palette("husl", len(martingales))

    # Plot individual martingales
    for (name, results), color in zip(martingales.items(), colors):
        martingale_values = np.array(results["martingales"])
        ax_reset.plot(  # Changed back to regular plot
            martingale_values,
            color=color,
            label=name.capitalize(),
            linewidth=1,
            alpha=0.7,
        )

    # Plot combined martingales with bold style
    martingale_arrays = [np.array(m["martingales"]) for m in martingales.values()]
    M_sum = np.sum(martingale_arrays, axis=0)
    M_avg = M_sum / len(martingales)

    ax_reset.plot(
        M_sum,
        color="black",
        label="Sum",
        linewidth=2.5,  # Regular plot
        linestyle="-",
        alpha=0.8,
    )
    ax_reset.plot(
        M_avg,
        color="red",
        label="Average",
        linewidth=2.5,  # Regular plot
        linestyle="--",
        alpha=0.8,
    )

    # Add vertical lines for true change points
    for cp in change_points:
        ax_reset.axvline(x=cp, color="red", linestyle=":", alpha=0.5)

    # Customize reset plot
    ax_reset.set_xlabel("Time Steps", fontsize=10)
    ax_reset.set_ylabel("Martingale Values", fontsize=10)  # Removed log scale note
    ax_reset.set_title("Martingale Measures (with Reset)", fontsize=12, pad=10)
    ax_reset.legend(fontsize=8, ncol=2, loc="upper right")
    ax_reset.grid(True, alpha=0.3)
    ax_reset.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 0)
    )  # Scientific notation
    ax_reset.yaxis.major.formatter._useMathText = True

    # Bottom row: Cumulative Martingales (log scale)
    ax_cumulative = fig.add_subplot(gs[2, :])

    # Plot cumulative martingales with log scale
    for (name, results), color in zip(martingales_cumulative.items(), colors):
        martingale_values = np.array(results["martingales"])
        ax_cumulative.semilogy(  # Keep log scale for cumulative plot
            martingale_values,
            color=color,
            label=name.capitalize(),
            linewidth=1,
            alpha=0.7,
        )

    # Plot combined cumulative martingales
    martingale_arrays = [
        np.array(m["martingales"]) for m in martingales_cumulative.values()
    ]
    M_sum = np.sum(martingale_arrays, axis=0)
    M_avg = M_sum / len(martingales_cumulative)

    ax_cumulative.semilogy(
        M_sum,
        color="black",
        label="Sum",
        linewidth=2.5,  # Keep log scale
        linestyle="-",
        alpha=0.8,
    )
    ax_cumulative.semilogy(
        M_avg,
        color="red",
        label="Average",
        linewidth=2.5,  # Keep log scale
        linestyle="--",
        alpha=0.8,
    )

    # Add vertical lines for true change points
    for cp in change_points:
        ax_cumulative.axvline(x=cp, color="red", linestyle=":", alpha=0.5)

    # Customize cumulative plot
    ax_cumulative.set_xlabel("Time Steps", fontsize=10)
    ax_cumulative.set_ylabel("Cumulative Martingale Values (log scale)", fontsize=10)
    ax_cumulative.set_title("Cumulative Martingale Measures", fontsize=12, pad=10)
    ax_cumulative.legend(fontsize=8, ncol=2, loc="upper right")
    ax_cumulative.grid(True, alpha=0.3)
    ax_cumulative.yaxis.major.formatter._useMathText = True

    # Overall figure adjustments
    plt.tight_layout()

    # Add a super title
    fig.suptitle("Barabási-Albert Graph Change Point Analysis", fontsize=14, y=0.95)

    # Save with high quality
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/ba_martingale_analysis.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def save_martingale_results(
    martingales: Dict[str, Dict[str, Any]],
    change_points: List[int],
    output_dir: str = "martingale_outputs",
) -> None:
    """Save martingale analysis results to a text file."""
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/ba_martingale_results.txt", "w") as f:
        # Write true change points
        f.write("True Change Points:\n")
        f.write(f"{change_points}\n\n")

        # Write results for each centrality measure
        f.write("Martingale Analysis Results:\n")
        f.write("=" * 50 + "\n\n")

        for name, results in martingales.items():
            f.write(f"Centrality Measure: {name.upper()}\n")
            f.write("-" * 30 + "\n")

            # Detected change points
            detected_points = results["change_detected_instant"]
            f.write(f"Detected change points: {detected_points}\n")

            # Handle martingale values
            martingale_values = results["martingales"]
            # Convert array of arrays to flat array
            flat_values = np.array(
                [
                    x.item() if isinstance(x, np.ndarray) else x
                    for x in martingale_values
                ]
            )

            if flat_values.size > 0:
                max_val = np.max(flat_values)
                mean_val = np.mean(flat_values)
                std_val = np.std(flat_values)

                f.write(f"Maximum martingale value: {max_val:.3f}\n")
                f.write(f"Average martingale value: {mean_val:.3f}\n")
                f.write(f"Standard deviation: {std_val:.3f}\n")

            # Detection accuracy
            if detected_points:
                detected = np.array(detected_points)
                true_cp = np.array(change_points)

                # Find closest true change point for each detection
                min_distances = [min(abs(d - true_cp)) for d in detected]
                avg_distance = np.mean(min_distances)
                f.write(
                    f"Average distance to nearest true change point: {avg_distance:.1f}\n"
                )

                # Count detections within ±5 time steps of true change points
                tolerance = 5
                accurate_detections = sum(
                    min(abs(d - true_cp)) <= tolerance for d in detected
                )
                f.write(
                    f"Accurate detections (±{tolerance} steps): {accurate_detections}/{len(detected)}\n"
                )

            f.write("\n")

        # Write combined martingale statistics
        f.write("\nCombined Martingale Analysis:\n")
        f.write("=" * 50 + "\n")

        # Handle combined martingales
        all_martingales = []
        for m in martingales.values():
            values = m["martingales"]
            flat_values = np.array(
                [x.item() if isinstance(x, np.ndarray) else x for x in values]
            )
            all_martingales.append(flat_values)

        if all_martingales:
            M_sum = np.sum(all_martingales, axis=0)
            M_avg = M_sum / len(all_martingales)

            f.write(f"Maximum combined martingale: {np.max(M_sum):.3f}\n")
            f.write(f"Average combined martingale: {np.mean(M_sum):.3f}\n")
            f.write(f"Standard deviation: {np.std(M_sum):.3f}\n")


def main():
    # Generate BA graphs
    result = generate_ba_graphs()
    graphs = result["graphs"]
    change_points = result["change_points"]

    # Select key time points to visualize
    time_points = [
        0,  # Start
        change_points[0] - 10,  # Before first change
        change_points[0] + 10,  # After first change
        change_points[-1] + 10,  # After last change
    ]

    # Initialize detector
    detector = ChangePointDetector()
    detector.initialize(graphs)

    # Extract features and compute martingales
    centralities = detector.extract_features()

    threshold = 15

    # Compute martingales with reset
    martingales = {}
    # Compute cumulative martingales (without reset)
    martingales_cumulative = {}

    for name, values in centralities.items():
        # Normalize values
        values_array = np.array(values)
        normalized_values = (values_array - np.mean(values_array, axis=0)) / np.std(
            values_array, axis=0
        )

        # Compute martingale with reset
        martingales[name] = detector.martingale_test(
            data=normalized_values, threshold=threshold, epsilon=0.8, reset=True
        )

        # For cumulative martingale, we need to ensure it's monotonically increasing
        cumulative_result = detector.martingale_test(
            data=normalized_values, threshold=threshold, epsilon=0.8, reset=False
        )

        # Convert to cumulative sum
        cumulative_values = np.array(cumulative_result["martingales"])
        cumulative_result["martingales"] = np.cumsum(cumulative_values)
        martingales_cumulative[name] = cumulative_result

    # Create visualization
    visualize_martingale_analysis(
        graphs=graphs,
        time_points=time_points,
        martingales=martingales,
        martingales_cumulative=martingales_cumulative,
        change_points=change_points,
    )

    # Save detailed results to text file
    save_martingale_results(martingales, change_points)


if __name__ == "__main__":
    import networkx as nx  # Import here to avoid circular imports

    main()
