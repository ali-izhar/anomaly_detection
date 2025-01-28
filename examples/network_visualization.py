# examples/network_visualization.py

"""Example script demonstrating network generation and visualization for different models."""

from pathlib import Path

import os
import sys
import argparse
import matplotlib.pyplot as plt

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.configs.loader import get_config
from src.graph.generator import GraphGenerator
from src.graph.visualizer import NetworkVisualizer


def calculate_block_sizes(n: int, num_blocks: int) -> list:
    """Calculate block sizes for SBM.

    Args:
        n: Total number of nodes
        num_blocks: Number of blocks
    Returns:
        List of block sizes
    """
    base_size = n // num_blocks
    remainder = n % num_blocks
    sizes = [base_size] * num_blocks
    # Distribute remainder across blocks
    for i in range(remainder):
        sizes[i] += 1
    return sizes


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias.

    Args:
        alias: Short model alias (ba, ws, er, sbm)
    Returns:
        Full model name
    """
    # Reverse mapping of MODEL_ALIASES
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def visualize_network(model_alias: str, output_dir: str = "examples"):
    """Generate and visualize a network model.

    Args:
        model_alias: Short name of the model to visualize (ba, ws, er, sbm)
        output_dir: Directory to save visualizations
    """
    # Get full model name for config loading
    model_name = get_full_model_name(model_alias)

    # Load model configuration
    config = get_config(model_name)
    params = config["params"].__dict__

    # Initialize visualizer and generator
    viz = NetworkVisualizer()
    generator = GraphGenerator(model_alias)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate network
    result = generator.generate_sequence(params)
    adj_matrix = result["graphs"][0]  # This is a numpy array

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(viz.SINGLE_COLUMN_WIDTH, viz.STANDARD_HEIGHT / 2)
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Network",
        fontsize=viz.TITLE_SIZE,
        y=0.95,
    )

    # Prepare node colors for SBM
    node_color = None
    if model_alias == "sbm":
        block_sizes = calculate_block_sizes(params["n"], params["num_blocks"])
        node_color = []
        for i, size in enumerate(block_sizes):
            node_color.extend([f"C{i}"] * size)

    # Plot network
    viz.plot_network(
        adj_matrix, ax=ax1, title="Network View", layout="spring", node_color=node_color
    )

    # Plot adjacency matrix
    viz.plot_adjacency(adj_matrix, ax=ax2, title="Adjacency Matrix")

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_dir, f"{model_name}_network.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Done! Network visualization has been saved to {output_file}")


def main():
    """Run network visualization based on command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize different network models.")
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
    visualize_network(args.model, args.output_dir)


if __name__ == "__main__":
    main()
