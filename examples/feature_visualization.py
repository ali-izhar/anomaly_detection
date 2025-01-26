# examples/feature_visualization.py

"""Example script demonstrating network evolution and feature dynamics visualization."""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import matplotlib.pyplot as plt

from src.graph.visualizer import NetworkVisualizer
from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor
from src.configs.loader import get_config


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


def visualize_network_evolution(model_alias: str, output_dir: str = "examples"):
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

    for i, time_idx in enumerate(key_points):
        # Prepare node colors for SBM
        node_color = None
        if model_alias == "sbm":
            block_sizes = calculate_block_sizes(params["n"], params["num_blocks"])
            node_color = []
            for j, size in enumerate(block_sizes):
                node_color.extend([f"C{j}"] * size)

        # Plot network state
        viz.plot_network(
            graphs[time_idx],
            ax=axes[i, 0],
            title=f"Network State at t={time_idx}"
            + (" (Change Point)" if time_idx in change_points else ""),
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

    # Plot feature evolution
    print("Creating feature evolution plots...")
    fig, _ = viz.plot_all_features(features, change_points=change_points, n_cols=2)

    plt.suptitle(
        f"{model_name.replace('_', ' ').title()} Feature Evolution",
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
        default="examples",
        help="Directory to save visualizations (default: examples)",
    )

    args = parser.parse_args()
    visualize_network_evolution(args.model, args.output_dir)


if __name__ == "__main__":
    main()
