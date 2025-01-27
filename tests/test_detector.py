#!/usr/bin/env python3
# tests/test_detector.py

"""
Test script for martingale-based change detection on a sequence of
NetworkX graphs using actual network data generation.

Example usage:
    python test_detector.py ba    # Test with Barabási-Albert model
    python test_detector.py ws    # Test with Watts-Strogatz model
    python test_detector.py er    # Test with Erdős-Rényi model
    python test_detector.py sbm   # Test with Stochastic Block Model
"""

import sys
import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import ChangePointDetector
from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor
from src.graph.visualizer import NetworkVisualizer
from src.changepoint.visualizer import MartingaleVisualizer
from src.configs.loader import get_config

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
    """Extract numeric features from feature dictionary in a consistent order.

    Args:
        feature_dict: Dictionary of network features
    Returns:
        numpy array of numeric features
    """
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
    output_dir: str = "test_results",
):
    """Create visualizations of network states at key points.

    Args:
        model_name: Full name of the network model
        graphs: List of adjacency matrices
        true_change_points: List of true change point indices
        detected_change_points: List of detected change point indices
        output_dir: Directory to save visualizations
    """
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
    output_dir: str = "test_results",
):
    """Create visualizations of network states and features at key points.

    Args:
        model_name: Full name of the network model
        graphs: List of adjacency matrices
        features_raw: List of raw feature dictionaries
        true_change_points: List of true change point indices
        detected_change_points: List of detected change point indices
        feature_results: Dictionary of detection results for each feature
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    viz = NetworkVisualizer()

    # 1. Create network state visualizations at key points
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

    # 2. Plot feature evolution
    fig, axes = plt.subplots(4, 2, figsize=(viz.SINGLE_COLUMN_WIDTH, viz.GRID_HEIGHT))
    axes = axes.flatten()

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
        ax = axes[i]
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
                color=viz.COLORS["actual"],
                alpha=viz.LINE_ALPHA,
                linewidth=viz.LINE_WIDTH,
            )
            ax.fill_between(
                time,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                color=viz.COLORS["actual"],
                alpha=0.1,
            )
        else:
            # For scalar features (like density)
            values = [f[feature_name] for f in features_raw]
            ax.plot(
                time,
                values,
                color=viz.COLORS["actual"],
                alpha=viz.LINE_ALPHA,
                linewidth=viz.LINE_WIDTH,
            )

        # Add true change points as red vertical lines
        for cp in true_change_points:
            ax.axvline(
                cp,
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=viz.LINE_WIDTH * 0.8,
            )

        # Add feature-specific detected change points as orange dots
        # Map feature names to their detection result keys
        feature_mapping = {
            "degrees": "degree",
            "density": "density",
            "clustering": "clustering",
            "betweenness": "betweenness",
            "eigenvector": "eigenvector",
            "closeness": "closeness",
            "singular_values": "singular_value",
            "laplacian_eigenvalues": "laplacian",
        }

        feature_key = feature_mapping.get(feature_name)
        if feature_key and feature_key in feature_results:
            feature_cps = feature_results[feature_key]["change_points"]
            for cp in feature_cps:
                if isinstance(features_raw[0][feature_name], list):
                    y_val = mean_values[cp]
                else:
                    y_val = features_raw[cp][feature_name]
                ax.plot(
                    cp,
                    y_val,
                    "o",
                    color="orange",
                    markersize=6,  # Increased size for better visibility
                    alpha=0.8,
                    markeredgewidth=1,
                )

        # Set title and labels
        ax.set_title(
            feature_name.replace("_", " ").title(), fontsize=viz.TITLE_SIZE, pad=4
        )
        ax.set_xlabel("Time", fontsize=viz.LABEL_SIZE, labelpad=2)
        ax.set_ylabel("Value", fontsize=viz.LABEL_SIZE, labelpad=2)
        ax.tick_params(labelsize=viz.TICK_SIZE, pad=1)
        ax.grid(True, alpha=viz.GRID_ALPHA, linewidth=viz.GRID_WIDTH)

    plt.suptitle(
        f"{model_name.replace('_', ' ').title()} Feature Evolution\n"
        + "True Change Points (Red), Feature-Specific Detected Change Points (Orange)",
        fontsize=viz.TITLE_SIZE,
        y=0.98,
    )
    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_features.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def run_change_detection(
    model_alias: str, threshold: float = 60.0, epsilon: float = 0.7
):
    """Run change point detection on network sequence from specified model.

    Args:
        model_alias: Short name of the model ('ba', 'ws', 'er', 'sbm')
        threshold: Detection threshold for martingale
        epsilon: Sensitivity parameter for martingale (0,1)
    """
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

    # Define feature names
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

    # 4. Create a ChangePointDetector instance
    cpd = ChangePointDetector()

    # 5. Run the detector using multiview martingale test
    logger.info(
        "Running multiview change detection with threshold=%f, epsilon=%f",
        threshold,
        epsilon,
    )

    # Run detector on all features together
    multiview_result = cpd.detect_changes_multiview(
        data=[
            data[:, i : i + 1] for i in range(data.shape[1])
        ],  # Split features into separate views
        threshold=threshold,  # Threshold only for combined evidence
        epsilon=epsilon,
        max_window=None,
        random_state=42,
    )

    # Store results for visualization
    feature_results = {}

    # Use individual martingales from multiview result
    for i, feature_name in enumerate(feature_names):
        feature_results[feature_name] = {
            "change_points": multiview_result["change_points"],
            "martingales": multiview_result["individual_martingales"][
                i
            ],  # Use individual martingales from multiview
            "p_values": multiview_result["p_values"][i],
            "strangeness": multiview_result["strangeness"][i],
        }

    # Combined change points are the same as multiview result
    combined_change_points = multiview_result["change_points"]

    # 6. Print results and compare with true change points
    print(
        f"\n==== Multiview Martingale Change Detection on {model_name.replace('_', ' ').title()} Network ===="
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
    print(f"Detected change points: {combined_change_points}")

    # Print martingale statistics
    print("\nMartingale Statistics:")
    print(
        f"- Final sum martingale value: {multiview_result['martingales_sum'][-1]:.2f}"
    )
    print(
        f"- Final average martingale value: {multiview_result['martingales_avg'][-1]:.2f}"
    )
    print(
        f"- Maximum sum martingale value: {np.max(multiview_result['martingales_sum']):.2f}"
    )
    print(
        f"- Maximum average martingale value: {np.max(multiview_result['martingales_avg']):.2f}"
    )

    # Calculate detection accuracy
    if true_change_points and combined_change_points:
        # Simple metric: for each true change point, find the closest detected point
        errors = []
        for true_cp in true_change_points:
            closest_detected = min(
                combined_change_points, key=lambda x: abs(x - true_cp)
            )
            error = abs(closest_detected - true_cp)
            errors.append(error)

        avg_error = np.mean(errors)
        print(f"\nDetection Performance:")
        print(f"Average detection delay: {avg_error:.2f} time steps")

    # 7. Create visualizations
    logger.info("Creating visualizations...")
    output_dir = f"tests/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize network states and features
    visualize_results(
        model_name=model_name,
        graphs=graphs,
        features_raw=features_raw,
        true_change_points=true_change_points,
        detected_change_points=combined_change_points,
        feature_results=feature_results,
        output_dir=output_dir,
    )

    # Prepare feature martingales for visualization
    feature_martingales = {
        name: {
            "martingales": results["martingales"],  # Individual feature martingales
            "p_values": results["p_values"],
            "strangeness": results["strangeness"],
        }
        for name, results in feature_results.items()
    }

    # Add combined martingales as a special feature
    feature_martingales["combined"] = {
        "martingales": multiview_result[
            "martingales_sum"
        ],  # Use sum for main martingale line
        "p_values": [1.0] * len(multiview_result["martingales_sum"]),  # Dummy p-values
        "strangeness": [0.0]
        * len(multiview_result["martingales_sum"]),  # Dummy strangeness
        "martingale_sum": multiview_result["martingales_sum"],
        "martingale_avg": multiview_result["martingales_avg"],
    }

    # Visualize martingale analysis
    martingale_viz = MartingaleVisualizer(
        martingales=feature_martingales,
        change_points=true_change_points,
        threshold=threshold,
        epsilon=epsilon,
        output_dir=output_dir,
    )
    martingale_viz.create_visualization()

    logger.info(f"Visualizations saved to {output_dir}/")


def main():
    """Run change detection test based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test change point detection on network evolution."
    )
    parser.add_argument(
        "model",
        choices=["ba", "ws", "er", "sbm"],
        help="Model to test: ba (Barabási-Albert), ws (Watts-Strogatz), "
        "er (Erdős-Rényi), sbm (Stochastic Block Model)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Detection threshold for martingale (default: 60.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.7,
        help="Sensitivity parameter for martingale (default: 0.7)",
    )

    args = parser.parse_args()
    run_change_detection(args.model, args.threshold, args.epsilon)


if __name__ == "__main__":
    main()
