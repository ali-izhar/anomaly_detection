# examples/direct_martingale_visualization.py

"""Visualize martingale-based change detection on network sequences.
Usage:
    python examples/direct_martingale_visualization.py ba
    python examples/direct_martingale_visualization.py ws
    python examples/direct_martingale_visualization.py er
    python examples/direct_martingale_visualization.py sbm
"""

from pathlib import Path

import sys
import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns

from matplotlib.gridspec import GridSpec

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import ChangePointDetector
from src.changepoint.visualizer import MartingaleVisualizer
from src.configs.loader import get_config
from src.graph.features import NetworkFeatureExtractor
from src.graph.generator import GraphGenerator
from src.graph.visualizer import NetworkVisualizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def extract_numeric_features(feature_dict: dict) -> np.ndarray:
    """Extract numeric features from feature dictionary in a consistent order."""
    # Extract basic metrics
    degrees = feature_dict.get("degrees", [])
    avg_degree = np.mean(degrees) if degrees else 0.0
    density = feature_dict.get("density", 0.0)
    clustering = feature_dict.get("clustering", [])
    avg_clustering = np.mean(clustering) if clustering else 0.0

    # Extract centrality metrics
    betweenness = feature_dict.get("betweenness", [])
    avg_betweenness = np.mean(betweenness) if betweenness else 0.0
    eigenvector = feature_dict.get("eigenvector", [])
    avg_eigenvector = np.mean(eigenvector) if eigenvector else 0.0
    closeness = feature_dict.get("closeness", [])
    avg_closeness = np.mean(closeness) if closeness else 0.0

    # Extract spectral metrics
    singular_values = feature_dict.get("singular_values", [])
    largest_sv = max(singular_values) if singular_values else 0.0
    laplacian_eigenvalues = feature_dict.get("laplacian_eigenvalues", [])
    smallest_nonzero_le = (
        min(x for x in laplacian_eigenvalues if x > 1e-10)
        if laplacian_eigenvalues
        else 0.0
    )

    return np.array(
        [
            avg_degree,
            density,
            avg_clustering,
            avg_betweenness,
            avg_eigenvector,
            avg_closeness,
            largest_sv,
            smallest_nonzero_le,
        ]
    )


def visualize_network_states(
    model_name: str,
    graphs: list,
    true_change_points: list,
    detected_change_points: list,
    output_dir: str = "examples",
):
    """Create visualizations of network states at key points."""
    os.makedirs(output_dir, exist_ok=True)
    viz = NetworkVisualizer()

    # Create network state visualizations at key points
    key_points = sorted(
        list(set([0] + true_change_points + detected_change_points + [len(graphs) - 1]))
    )
    n_points = len(key_points)

    fig, axes = plt.subplots(
        n_points,
        2,
        figsize=(viz.SINGLE_COLUMN_WIDTH, viz.STANDARD_HEIGHT * n_points / 2),
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Network States",
        fontsize=viz.TITLE_SIZE,
        y=0.98,
    )

    # Prepare node colors for SBM
    node_color = None
    if "stochastic_block_model" in model_name.lower():
        graph = nx.from_numpy_array(graphs[0])
        n = graph.number_of_nodes()
        num_blocks = int(np.sqrt(n))  # Estimate number of blocks
        block_sizes = [n // num_blocks] * (num_blocks - 1)
        block_sizes.append(n - sum(block_sizes))
        node_color = []
        for j, size in enumerate(block_sizes):
            node_color.extend([f"C{j}"] * size)

    for i, time_idx in enumerate(key_points):
        # Plot network state
        point_type = (
            "Initial State"
            if time_idx == 0
            else (
                "Final State"
                if time_idx == len(graphs) - 1
                else (
                    "True Change Point"
                    if time_idx in true_change_points
                    else (
                        "Detected Change Point"
                        if time_idx in detected_change_points
                        else "State"
                    )
                )
            )
        )

        viz.plot_network(
            graphs[time_idx],
            ax=axes[i, 0],
            title=f"Network {point_type} at t={time_idx}",
            layout="spring",
            node_color=node_color,
        )

        # Plot adjacency matrix
        viz.plot_adjacency(
            graphs[time_idx], ax=axes[i, 1], title=f"Adjacency Matrix at t={time_idx}"
        )

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_states.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def visualize_results(
    model_name: str,
    graphs: list,
    features_raw: list,
    true_change_points: list,
    detected_change_points: list,
    feature_results: dict,
    output_dir: str = "examples",
):
    """Create visualizations of network states and features at key points."""
    os.makedirs(output_dir, exist_ok=True)
    viz = NetworkVisualizer()

    # Set paper-style parameters
    plt.style.use("seaborn-v0_8-paper")
    sns.set_style("whitegrid", {"grid.linestyle": ":"})
    sns.set_context("paper", font_scale=1.0)

    # Special colors for indicators
    line_colors = {
        "actual": "#1f77b4",  # Blue
        "predicted": "#ff7f0e",  # Orange
        "threshold": "#666666",  # Gray
        "changepoint": "#FF9999",  # Light red
    }

    # Create network state visualizations
    visualize_network_states(
        model_name, graphs, true_change_points, detected_change_points, output_dir
    )

    # Create feature evolution visualization
    fig = plt.figure(figsize=(viz.SINGLE_COLUMN_WIDTH, viz.STANDARD_HEIGHT * 2))
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(1.0)

    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Get list of features to plot
    feature_names = [
        "degrees",
        "density",
        "clustering",
        "betweenness",
        "eigenvector",
        "closeness",
        "singular_values",
        "laplacian_eigenvalues",
    ]

    for i, feature_name in enumerate(feature_names):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        time = range(len(features_raw))

        # Extract feature values
        if isinstance(features_raw[0][feature_name], list):
            # For list features (like degrees), compute mean and std
            mean_values = [
                np.mean(f[feature_name]) if len(f[feature_name]) > 0 else 0
                for f in features_raw
            ]
            std_values = [
                np.std(f[feature_name]) if len(f[feature_name]) > 0 else 0
                for f in features_raw
            ]

            # Plot mean line with std band
            ax.plot(
                time,
                mean_values,
                color=line_colors["actual"],
                alpha=0.8,
                linewidth=1.0,
            )
            ax.fill_between(
                time,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                color=line_colors["actual"],
                alpha=0.1,
            )
        else:
            # For scalar features (like density)
            values = [f[feature_name] for f in features_raw]
            ax.plot(
                time,
                values,
                color=line_colors["actual"],
                alpha=0.8,
                linewidth=1.0,
            )

        # Add true change points
        for cp in true_change_points:
            ax.axvline(
                cp,
                color=line_colors["changepoint"],
                linestyle="--",
                alpha=0.5,
                linewidth=0.8,
            )

        # Add detected change points
        for cp in detected_change_points:
            if isinstance(features_raw[0][feature_name], list):
                y_val = mean_values[cp]
            else:
                y_val = features_raw[cp][feature_name]
            ax.plot(
                cp,
                y_val,
                "o",
                color=line_colors["predicted"],
                markersize=6,
                alpha=0.8,
                markeredgewidth=1,
            )

        # Set title and labels
        ax.set_title(
            feature_name.replace("_", " ").title(),
            fontsize=viz.TITLE_SIZE,
            pad=4,
        )
        ax.set_xlabel("Time" if row == 3 else "", fontsize=viz.LABEL_SIZE, labelpad=2)
        ax.set_ylabel("Value" if col == 0 else "", fontsize=viz.LABEL_SIZE, labelpad=2)
        ax.tick_params(labelsize=viz.TICK_SIZE, pad=1)
        ax.grid(True, alpha=0.15, linewidth=0.5, linestyle=":")

        # Set x-axis ticks at intervals of 50
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

    plt.suptitle(
        f"{model_name.replace('_', ' ').title()} Feature Evolution\n"
        + "True Change Points (Red), Detected Change Points (Orange)",
        fontsize=viz.TITLE_SIZE,
        y=0.98,
    )
    plt.tight_layout(pad=0.3)
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_features.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.02,
    )
    plt.close()


def run_visualization(model_alias: str, threshold: float = 60.0, epsilon: float = 0.7):
    """Run change point detection visualization on network sequence from specified model."""
    # 1. Get full model name and configuration
    model_name = get_full_model_name(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__

    # 2. Generate network sequence with actual change points
    generator = GraphGenerator(model_alias)
    logger.info(f"Generating {model_name} network sequence...")
    result = generator.generate_sequence(params)

    graphs = result["graphs"]  # List of adjacency matrices
    true_change_points = result["change_points"]

    # 3. Extract features from each graph using all available extractors
    logger.info("Extracting network features...")
    feature_extractor = NetworkFeatureExtractor()
    features_raw = []  # Store raw feature dictionaries for visualization
    features_numeric = []  # Store numeric features for change detection

    for adj_matrix in graphs:
        graph = nx.from_numpy_array(adj_matrix)
        # Get all available feature types
        feature_dict = feature_extractor.get_features(graph)
        features_raw.append(feature_dict)
        numeric_features = extract_numeric_features(feature_dict)
        features_numeric.append(numeric_features)

    # Convert features to numpy array
    data = np.array(features_numeric)
    logger.info(f"Extracted feature matrix of shape {data.shape}")

    # 4. Create a ChangePointDetector instance
    cpd = ChangePointDetector()

    # 5. Run the detector using direct martingale test
    logger.info(
        "Running direct martingale change detection with threshold=%f, epsilon=%f",
        threshold,
        epsilon,
    )

    # Run detector on the sequence
    result = cpd.detect_changes(
        data=data,
        threshold=threshold,
        epsilon=epsilon,
        reset=True,
        max_window=None,
        random_state=42,
    )

    # Get detected change points
    detected_change_points = result["change_points"]

    # Calculate individual feature martingales
    individual_results = []
    for i in range(data.shape[1]):  # For each feature dimension
        feature_result = cpd.detect_changes(
            data=data[:, i : i + 1],  # Single feature
            threshold=threshold,
            epsilon=epsilon,
            reset=True,
            max_window=None,
            random_state=42,
        )
        individual_results.append(feature_result)

    # Define feature names for individual features
    feature_names = [
        "degree",
        "density",
        "clustering",
        "betweenness",
        "eigenvector",
        "closeness",
        "singular_value",
        "laplacian",
    ]

    # Prepare feature results for visualization
    feature_results = {}

    # Create individual feature entries with their own martingales
    for i, feature_name in enumerate(feature_names):
        feature_results[feature_name] = {
            "change_points": individual_results[i]["change_points"],
            "martingales": individual_results[i]["martingales"],
            "p_values": individual_results[i]["p_values"],
            "strangeness": individual_results[i]["strangeness"],
        }

    # Add combined martingales as a special feature
    feature_results["combined"] = {
        "martingales": result["martingales"],
        "p_values": result["p_values"],
        "strangeness": result["strangeness"],
        "martingale_sum": result["martingales"],  # For compatibility with visualizer
        "martingale_avg": result["martingales"]
        / len(feature_names),  # Average over features
    }

    # 6. Print results and compare with true change points
    print(
        f"\n==== Direct Martingale Change Detection on {model_name.replace('_', ' ').title()} Network ===="
    )
    print(f"Network parameters: {params}")
    print(f"\nFeatures used for detection:")
    print("- Average degree")
    print("- Density")
    print("- Average clustering coefficient")
    print("- Average betweenness centrality")
    print("- Average eigenvector centrality")
    print("- Average closeness centrality")
    print("- Largest singular value")
    print("- Smallest non-zero Laplacian eigenvalue")

    print(f"\nTrue change points: {true_change_points}")
    print("\nDetected change points by feature:")
    for i, feature_name in enumerate(feature_names):
        print(f"- {feature_name}: {individual_results[i]['change_points']}")
    print(f"\nCombined martingale change points: {detected_change_points}")

    # Print martingale statistics
    print("\nMartingale Statistics:")
    print("Individual feature martingales:")
    for i, feature_name in enumerate(feature_names):
        max_martingale = np.max(individual_results[i]["martingales"])
        final_martingale = individual_results[i]["martingales"][-1]
        print(f"- {feature_name}:")
        print(f"  - Final value: {final_martingale:.2f}")
        print(f"  - Maximum value: {max_martingale:.2f}")

    print("\nCombined martingale:")
    print(f"- Final value: {result['martingales'][-1]:.2f}")
    print(f"- Maximum value: {np.max(result['martingales']):.2f}")

    # Calculate detection accuracy based on sum martingale
    if true_change_points and detected_change_points:
        # For each true change point, find the closest detected point from sum martingale
        errors = []
        for true_cp in true_change_points:
            closest_detected = min(
                detected_change_points, key=lambda x: abs(x - true_cp)
            )
            error = abs(closest_detected - true_cp)
            errors.append(error)

        avg_error = np.mean(errors)
        print(f"\nDetection Performance (based on combined martingale):")
        print(f"Average detection delay: {avg_error:.2f} time steps")

    # 7. Create visualizations
    logger.info("Creating visualizations...")
    output_dir = f"examples/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize network states and features
    visualize_results(
        model_name=model_name,
        graphs=graphs,
        features_raw=features_raw,
        true_change_points=true_change_points,
        detected_change_points=detected_change_points,
        feature_results=feature_results,
        output_dir=output_dir,
    )

    # Use MartingaleVisualizer for martingale visualization
    martingale_viz = MartingaleVisualizer(
        martingales=feature_results,
        change_points=true_change_points,
        threshold=threshold,
        epsilon=epsilon,
        output_dir=output_dir,
    )
    martingale_viz.create_visualization()

    logger.info(f"Visualizations saved to {output_dir}/")


def main():
    """Run visualization based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize change point detection on network evolution."
    )
    parser.add_argument(
        "model",
        choices=["ba", "ws", "er", "sbm"],
        help="Model to visualize: ba (Barabási-Albert), ws (Watts-Strogatz), "
        "er (Erdős-Rényi), sbm (Stochastic Block Model)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Detection threshold for martingale (default: 10.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.7,
        help="Sensitivity parameter for martingale (default: 0.7)",
    )

    args = parser.parse_args()
    run_visualization(args.model, args.threshold, args.epsilon)


if __name__ == "__main__":
    main()
