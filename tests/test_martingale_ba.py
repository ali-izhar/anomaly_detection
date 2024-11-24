import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from create_ba_graphs import generate_ba_graphs
from src.graph.features import adjacency_to_graph
from src.changepoint import ChangePointDetector

THRESHOLD = 30


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
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Use a modern color palette
    colors = sns.color_palette("Set2", len(martingales))

    # Create figure with 3 rows and more vertical spacing
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.5, wspace=0.3)

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

    # Add shaded background for change points
    for cp in change_points:
        ax_reset.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

    # Plot individual martingales with improved styling
    max_value = 0  # Track maximum value for y-axis scaling
    for (name, results), color in zip(martingales.items(), colors):
        martingale_values = np.array(results["martingales"])
        max_value = max(max_value, np.max(martingale_values))
        ax_reset.plot(
            martingale_values,
            color=color,
            label=name.capitalize(),
            linewidth=1.5,
            alpha=0.6,
            linestyle="-",
        )

    # Plot combined martingales with bold style
    martingale_arrays = [np.array(m["martingales"]) for m in martingales.values()]
    M_sum = np.sum(martingale_arrays, axis=0)
    M_avg = M_sum / len(martingales)
    max_value = max(max_value, np.max(M_sum), np.max(M_avg))

    # Add vertical lines for detected changes (where sum or average exceeds threshold)
    detected_changes = set()
    for t in range(len(M_sum)):
        if M_sum[t] > THRESHOLD or M_avg[t] > THRESHOLD:
            ax_reset.axvline(x=t, color='purple', linestyle='--', alpha=0.3, zorder=1)
            detected_changes.add(t)
    if detected_changes:  # Add to legend only if there were detections
        ax_reset.plot([], [], color='purple', linestyle='--', alpha=0.3, 
                     label='Detected Change')

    # Add subtle grid lines
    ax_reset.grid(True, linestyle="--", alpha=0.3)

    # Plot average with emphasis
    ax_reset.plot(
        M_avg,
        color="#FF4B4B",  # Bright red
        label="Average",
        linewidth=2.5,
        linestyle="-",
        alpha=0.9,
        zorder=5,
    )

    # Plot sum with different style
    ax_reset.plot(
        M_sum,
        color="#2F2F2F",  # Dark gray
        label="Sum",
        linewidth=2.5,
        linestyle="-.",
        alpha=0.8,
        zorder=4,
    )

    # Use EXACTLY the same legend style as cumulative plot
    legend = ax_reset.legend(
        fontsize=10,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1, 1.02),
        frameon=True,
        facecolor='none',  # Transparent background
        edgecolor='none',
        shadow=False,
    )
    # Add these lines to ensure transparency
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_alpha(0)

    # Set y-axis limits with some padding
    y_max = max_value * 1.1  # Add 10% padding
    y_min = 0
    ax_reset.set_ylim(y_min, y_max)

    # Add minor ticks for better readability
    ax_reset.yaxis.set_minor_locator(AutoMinorLocator())

    # Format y-axis ticks to be more readable
    ax_reset.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

    # Customize reset plot with smaller title
    ax_reset.set_xlabel("Time Steps", fontsize=12, labelpad=10)
    ax_reset.set_ylabel("Martingale Values", fontsize=12, labelpad=10)
    ax_reset.set_title(
        "Reset Martingale Measures",  # Shorter title
        fontsize=12,  # Smaller font
        pad=20,
    )

    # Bottom row: Cumulative Martingales (log scale)
    ax_cumulative = fig.add_subplot(gs[2, :])

    # Add shaded background for change points
    for cp in change_points:
        ax_cumulative.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

    # Plot cumulative martingales with improved styling
    for (name, results), color in zip(martingales_cumulative.items(), colors):
        martingale_values = np.array(results["martingales"])
        ax_cumulative.semilogy(
            martingale_values,
            color=color,
            label=name.capitalize(),
            linewidth=1.5,
            alpha=0.6,
            linestyle="-",
        )

    # Plot combined cumulative martingales
    martingale_arrays = [
        np.array(m["martingales"]) for m in martingales_cumulative.values()
    ]
    M_sum = np.sum(martingale_arrays, axis=0)
    M_avg = M_sum / len(martingales_cumulative)

    # Add subtle grid lines
    ax_cumulative.grid(True, linestyle="--", alpha=0.3)

    # Plot average with emphasis
    ax_cumulative.semilogy(
        M_avg,
        color="#FF4B4B",  # Bright red
        label="Average",
        linewidth=2.5,
        linestyle="-",
        alpha=0.9,
        zorder=5,
    )

    # Plot sum with different style
    ax_cumulative.semilogy(
        M_sum,
        color="#2F2F2F",  # Dark gray
        label="Sum",
        linewidth=2.5,
        linestyle="-.",
        alpha=0.8,
        zorder=4,
    )

    # Original cumulative plot legend style
    legend = ax_cumulative.legend(
        fontsize=10,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        frameon=True,
        facecolor='none',  # Transparent background
        edgecolor='none',
        shadow=False,
    )
    # Add these lines to ensure transparency
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_alpha(0)

    # Customize cumulative plot with smaller title
    ax_cumulative.set_xlabel("Time Steps", fontsize=12, labelpad=15)
    ax_cumulative.set_ylabel(
        "Cumulative Martingale Values\n(log scale)", fontsize=12, labelpad=10
    )
    ax_cumulative.set_title(
        "Cumulative Martingale Measures",  # Shorter title
        fontsize=12,  # Smaller font
        pad=20,
    )

    # Overall figure adjustments with more spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a super title
    fig.suptitle(
        "Barabási-Albert Graph Change Point Analysis",  # Removed subtitle
        fontsize=14,  # Slightly smaller
        y=0.98,
    )

    # Save with high quality
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/ba_martingale_analysis.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.2,
    )
    plt.close()


def save_martingale_results(
    martingales: Dict[str, Dict[str, Any]],
    change_points: List[int],
    output_dir: str = "martingale_outputs",
    threshold: float = 0.0,
) -> None:
    """Save martingale analysis results to a text file."""
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/ba_martingale_results.txt", "w") as f:
        # Write true change points
        f.write("True Change Points:\n")
        f.write(f"{change_points}\n\n")

        # Write results for each centrality measure
        f.write("Individual Martingale Analysis Results:\n")
        f.write("=" * 50 + "\n\n")

        for name, results in martingales.items():
            f.write(f"Centrality Measure: {name.upper()}\n")
            f.write("-" * 30 + "\n")

            # Detected change points with their martingale values
            detected_points = results["change_detected_instant"]
            martingale_values = results["martingales"]
            
            f.write("Detected change points and their martingale values:\n")
            if detected_points:
                for point in detected_points:
                    value = martingale_values[point]
                    if isinstance(value, np.ndarray):
                        value = value.item()
                    f.write(f"Time step {point}: {value:.3f}\n")
            else:
                f.write("No change points detected\n")
            f.write("\n")

            # Handle martingale values statistics
            flat_values = np.array(
                [x.item() if isinstance(x, np.ndarray) else x for x in martingale_values]
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

                min_distances = [min(abs(d - true_cp)) for d in detected]
                avg_distance = np.mean(min_distances)
                f.write(
                    f"Average distance to nearest true change point: {avg_distance:.1f}\n"
                )

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

            # Detect changes using sum model
            sum_changes = detect_changes(M_sum, threshold=threshold)
            f.write("\nSum Model Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write("Detected change points and their martingale values:\n")
            for point in sum_changes:
                f.write(f"Time step {point}: {M_sum[point]:.3f}\n")
            f.write("\n")
            if sum_changes:
                sum_accuracy = analyze_detection_accuracy(sum_changes, change_points)
                f.write(f"Detection accuracy metrics:\n{sum_accuracy}\n")

            # Detect changes using average model
            avg_changes = detect_changes(M_avg, threshold=threshold)
            f.write("\nAverage Model Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write("Detected change points and their martingale values:\n")
            for point in avg_changes:
                f.write(f"Time step {point}: {M_avg[point]:.3f}\n")
            f.write("\n")
            if avg_changes:
                avg_accuracy = analyze_detection_accuracy(avg_changes, change_points)
                f.write(f"Detection accuracy metrics:\n{avg_accuracy}\n")

            # Overall statistics
            f.write("\nOverall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Maximum combined martingale (sum): {np.max(M_sum):.3f}\n")
            f.write(f"Average combined martingale (sum): {np.mean(M_sum):.3f}\n")
            f.write(f"Standard deviation (sum): {np.std(M_sum):.3f}\n")
            f.write(f"Maximum combined martingale (avg): {np.max(M_avg):.3f}\n")
            f.write(f"Average combined martingale (avg): {np.mean(M_avg):.3f}\n")
            f.write(f"Standard deviation (avg): {np.std(M_avg):.3f}\n")


def detect_changes(martingale_values: np.ndarray, threshold: float) -> List[int]:
    """Detect change points from martingale values."""
    changes = []
    for t in range(1, len(martingale_values)):
        if martingale_values[t] > threshold:
            changes.append(t)
    return changes


def analyze_detection_accuracy(detected: List[int], true_changes: List[int]) -> str:
    """Analyze accuracy of detected change points."""
    if not detected:
        return "No changes detected"

    detected = np.array(detected)
    true_changes = np.array(true_changes)

    # Find closest true change point for each detection
    min_distances = [min(abs(d - true_changes)) for d in detected]
    avg_distance = np.mean(min_distances)

    # Count detections within ±5 time steps
    tolerance = 5
    accurate_detections = sum(min(abs(d - true_changes)) <= tolerance for d in detected)

    result = (
        f"Total detections: {len(detected)}\n"
        f"Average distance to nearest true change point: {avg_distance:.1f}\n"
        f"Accurate detections (±{tolerance} steps): {accurate_detections}/{len(detected)}\n"
        f"False positives: {len(detected) - accurate_detections}\n"
        f"Missed changes: {len(true_changes) - accurate_detections if accurate_detections <= len(true_changes) else 0}"
    )
    return result


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
            data=normalized_values, threshold=THRESHOLD, epsilon=0.8, reset=True
        )

        # For cumulative martingale, we need to ensure it's monotonically increasing
        cumulative_result = detector.martingale_test(
            data=normalized_values, threshold=THRESHOLD, epsilon=0.8, reset=False
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
    save_martingale_results(martingales, change_points, threshold=THRESHOLD)


if __name__ == "__main__":
    import networkx as nx  # Import here to avoid circular imports

    main()
