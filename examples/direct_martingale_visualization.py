# examples/direct_martingale_visualization.py

"""Visualize single-view martingale-based change detection on network sequences.
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

from src.changepoint.pipeline import MartingalePipeline
from src.changepoint.visualizer import MartingaleVisualizer
from src.configs.loader import get_config
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

    # 3. Create and run the pipeline
    logger.info("Running single-view change detection pipeline...")
    pipeline = MartingalePipeline(
        martingale_method="single_view",
        threshold=threshold,
        epsilon=epsilon,
        random_state=42,
        feature_set="all",  # Using all features but will be combined into single view
        reset=True,
        max_window=None,
    )

    # Run pipeline directly on adjacency matrices
    pipeline_result = pipeline.run(
        data=graphs,
        data_type="adjacency",  # Specify that we're passing adjacency matrices
    )

    # 4. Print results and compare with true change points
    print(
        f"\n==== Single-View Martingale Change Detection on {model_name.replace('_', ' ').title()} Network ===="
    )
    print(f"Network parameters: {params}")
    print(f"\nTrue change points: {true_change_points}")
    print(f"Detected change points: {pipeline_result['change_points']}")

    # Print martingale statistics
    print("\nMartingale Statistics:")
    print(f"- Final martingale value: {pipeline_result['martingales'][-1]:.2f}")
    print(f"- Maximum martingale value: {np.max(pipeline_result['martingales']):.2f}")

    # Calculate detection accuracy
    if true_change_points and pipeline_result["change_points"]:
        errors = []
        for true_cp in true_change_points:
            closest_detected = min(
                pipeline_result["change_points"], key=lambda x: abs(x - true_cp)
            )
            error = abs(closest_detected - true_cp)
            errors.append(error)

        avg_error = np.mean(errors)
        print(f"\nDetection Performance:")
        print(f"Average detection delay: {avg_error:.2f} time steps")

    # 5. Create visualizations
    logger.info("Creating visualizations...")
    output_dir = f"examples/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize network states
    visualize_network_states(
        model_name=model_name,
        graphs=graphs,
        true_change_points=true_change_points,
        detected_change_points=pipeline_result["change_points"],
        output_dir=output_dir,
    )

    # Create martingale visualizer
    martingale_viz = MartingaleVisualizer(
        martingales={"combined": pipeline_result},  # Pass pipeline results directly
        change_points=true_change_points,
        threshold=threshold,
        epsilon=epsilon,
        output_dir=output_dir,
        skip_shap=True,  # Skip SHAP for single-view
    )
    martingale_viz.create_visualization()

    logger.info(f"Visualizations saved to {output_dir}/")


def main():
    """Run visualization based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize single-view change point detection on network evolution."
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
        help="Detection threshold for martingale (default: 60.0)",
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
