# tests/test_graph/test_features_vis.py

"""Demonstrating network evolution and feature dynamics visualization."""

import os
import sys
import argparse
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import matplotlib.pyplot as plt

from src.configs.loader import get_config
from src.graph.features import NetworkFeatureExtractor
from src.graph.generator import GraphGenerator
from src.plot.plot_graph import NetworkVisualizer


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def calculate_block_sizes(n: int, num_blocks: int) -> list:
    """Calculate block sizes for SBM."""
    base_size = n // num_blocks
    remainder = n % num_blocks
    sizes = [base_size] * num_blocks
    # Distribute remainder across blocks
    for i in range(remainder):
        sizes[i] += 1
    return sizes


def visualize_network_evolution(
    model_alias: str, output_dir: str = "tests/test_graph/output"
):
    """Generate and visualize network evolution with features."""
    # Get full model name for config loading
    model_name = get_full_model_name(model_alias)

    # Load model configuration
    config = get_config(model_name)
    params = config["params"].__dict__

    # Initialize components
    viz = NetworkVisualizer()
    generator = GraphGenerator(model_alias)
    feature_extractor = NetworkFeatureExtractor()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate network sequence
    print(f"Generating {model_name} network sequence...")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    change_points = result["change_points"]

    # Extract features for each graph
    print("Extracting network features...")
    features = []
    for adj_matrix in graphs:
        graph = nx.from_numpy_array(adj_matrix)
        features.append(feature_extractor.get_features(graph))

    # Create network state visualizations at key points
    print("Creating network state visualizations...")
    key_points = [0] + change_points + [len(graphs) - 1]
    n_points = len(key_points)

    # Create figure for network states
    fig = plt.figure()
    title = f"{model_name.replace('_', ' ').title()} Network States"

    # Plot each key point
    for i, time_idx in enumerate(key_points):
        # Create a new figure for each time point
        plt.figure()

        # Prepare node colors for SBM
        node_color = None
        if model_alias == "sbm":
            block_sizes = calculate_block_sizes(params["n"], params["num_blocks"])
            node_color = []
            for j, size in enumerate(block_sizes):
                node_color.extend([f"C{j}"] * size)

        # Create visualization with both network and adjacency matrix
        state_title = f"Network State at t={time_idx}" + (
            " (Change Point)" if time_idx in change_points else ""
        )
        viz.plot_network_with_adjacency(
            graphs[time_idx], title=state_title, layout="spring", node_color=node_color
        )

        # Save each state visualization
        state_file = os.path.join(output_dir, f"{model_name}_state_{time_idx}.png")
        plt.savefig(state_file, bbox_inches="tight", dpi=300)
        plt.close()

    print("Creating feature evolution plots...")
    # Plot feature evolution
    fig, _ = viz.plot_all_features(features, change_points=change_points, n_cols=2)

    # Save feature evolution plot
    feature_file = os.path.join(output_dir, f"{model_name}_features.png")
    plt.savefig(feature_file, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Done! Visualizations have been saved to {output_dir}/")


def main():
    """Run network evolution visualization based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize network evolution and feature dynamics."
    )
    parser.add_argument(
        "model",
        choices=["ba", "ws", "er", "sbm"],
        help="Model to visualize: ba (Barabási-Albert), ws (Watts-Strogatz), "
        "er (Erdős-Rényi), sbm (Stochastic Block Model)",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/test_graph/output",
        help="Directory to save visualizations (default: tests/test_graph/output)",
    )

    args = parser.parse_args()
    visualize_network_evolution(args.model, args.output_dir)


if __name__ == "__main__":
    main()
