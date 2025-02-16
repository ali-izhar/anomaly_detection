# tests/test_martingale/singleview_vis.py

"""Visualize single-view martingale-based change detection on network sequences.
Usage:
    python tests/test_martingale/singleview_vis.py <model>
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

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.visualizer import MartingaleVisualizer
from src.configs.loader import get_config
from src.graph.generator import GraphGenerator
from src.graph.visualizer import NetworkVisualizer
from src.changepoint.detector import ChangePointDetector, DetectorConfig
from src.graph.features import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Paper-style figure sizes (in inches)
SINGLE_COLUMN_WIDTH = 8.0
DOUBLE_COLUMN_WIDTH = 12.0
STANDARD_HEIGHT = 6.0
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10


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
    output_dir: str = "tests/test_martingale/output",
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
        figsize=(SINGLE_COLUMN_WIDTH, STANDARD_HEIGHT * n_points / 2),
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Network States",
        fontsize=TITLE_SIZE,
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
    output_dir: str = "tests/test_martingale/output",
):
    """Create visualizations of network states and features at key points."""
    os.makedirs(output_dir, exist_ok=True)

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

    # Create feature evolution visualization
    fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, STANDARD_HEIGHT * 2))
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
        if feature_name == "density":
            values = [f[feature_name] for f in features_raw]
        else:
            # For list features, take the mean
            values = [
                np.mean(f[feature_name]) if f[feature_name] else 0.0
                for f in features_raw
            ]
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
            ax.plot(
                cp,
                values[cp],
                "o",
                color=line_colors["predicted"],
                markersize=6,
                alpha=0.8,
                markeredgewidth=1,
            )

        # Set title and labels
        title = feature_name.replace("_", " ")
        if feature_name != "density":
            title = "Mean " + title
        if feature_name in ["singular_values", "laplacian_eigenvalues"]:
            title = title.replace("values", "value").replace(
                "eigenvalues", "eigenvalue"
            )
        ax.set_title(
            title.title(),
            fontsize=TITLE_SIZE,
            pad=4,
        )
        ax.set_xlabel("Time" if row == 3 else "", fontsize=LABEL_SIZE, labelpad=2)
        ax.set_ylabel("Value" if col == 0 else "", fontsize=LABEL_SIZE, labelpad=2)
        ax.tick_params(labelsize=TICK_SIZE, pad=1)
        ax.grid(True, alpha=0.15, linewidth=0.5, linestyle=":")

        # Set x-axis ticks at intervals of 50
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_xlim(0, 200)

    plt.suptitle(
        f"{model_name.replace('_', ' ').title()} Feature Evolution\n"
        + "True Change Points (Red), Detected Change Points (Orange)",
        fontsize=TITLE_SIZE,
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

    # 3. Extract features
    logger.info("Extracting features...")
    feature_extractor = NetworkFeatureExtractor()
    features_raw = []
    features_numeric = []

    for adj_matrix in graphs:
        graph = adjacency_to_graph(adj_matrix)
        raw_features = feature_extractor.get_features(graph)
        numeric_features = feature_extractor.get_numeric_features(graph)
        features_raw.append(raw_features)
        features_numeric.append(
            [
                numeric_features["mean_degree"],
                numeric_features["density"],
                numeric_features["mean_clustering"],
                numeric_features["mean_betweenness"],
                numeric_features["mean_eigenvector"],
                numeric_features["mean_closeness"],
                numeric_features["max_singular_value"],
                numeric_features["min_nonzero_laplacian"],
            ]
        )

    features_numeric = np.array(features_numeric)

    # Combine features into a single value for single-view detection
    logger.info("Combining features into single view...")
    # Normalize each feature
    features_normalized = (
        features_numeric - np.mean(features_numeric, axis=0)
    ) / np.std(features_numeric, axis=0)
    # Take mean across features for each time point
    features_combined = np.mean(features_normalized, axis=1)

    # 4. Initialize and run detector
    logger.info("Running single-view change detection...")
    detector_config = DetectorConfig(
        method="single_view",
        threshold=threshold,
        history_size=10,
        batch_size=1000,
        reset=True,
        max_window=None,
        betting_func_config={"name": "power", "params": {"epsilon": epsilon}},
        distance_measure="euclidean",
        distance_p=2.0,
        random_state=42,
    )

    detector = ChangePointDetector(detector_config)
    detection_result = detector.run(
        data=features_combined,  # Use the combined features
        predicted_data=None,  # No predictions for basic visualization
    )

    # 5. Print results and compare with true change points
    print(
        f"\n==== Single-View Martingale Change Detection on {model_name.replace('_', ' ').title()} Network ===="
    )
    print(f"Network parameters: {params}")
    print(f"\nTrue change points: {true_change_points}")
    print(f"Detected change points: {detection_result['traditional_change_points']}")

    # Print martingale statistics
    print("\nMartingale Statistics:")
    print(
        f"- Final martingale value: {detection_result['traditional_martingales'][-1]:.2f}"
    )
    print(
        f"- Maximum martingale value: {np.max(detection_result['traditional_martingales']):.2f}"
    )

    # Calculate detection accuracy
    if true_change_points and detection_result["traditional_change_points"]:
        errors = []
        for true_cp in true_change_points:
            closest_detected = min(
                detection_result["traditional_change_points"],
                key=lambda x: abs(x - true_cp),
            )
            error = abs(closest_detected - true_cp)
            errors.append(error)

        avg_error = np.mean(errors)
        print(f"\nDetection Performance:")
        print(f"Average detection delay: {avg_error:.2f} time steps")

    # 6. Create visualizations
    logger.info("Creating visualizations...")
    output_dir = f"tests/test_martingale/output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize network states
    visualize_network_states(
        model_name=model_name,
        graphs=graphs,
        true_change_points=true_change_points,
        detected_change_points=detection_result["traditional_change_points"],
        output_dir=output_dir,
    )

    # Create martingale visualizer
    martingale_viz = MartingaleVisualizer(
        martingales={
            "traditional": detection_result["traditional_martingales"],
            "horizon": detection_result.get("horizon_martingales", None),
        },
        change_points=true_change_points,
        threshold=threshold,
        betting_config={"function": "power", "params": {"power": {"epsilon": epsilon}}},
        output_dir=output_dir,
        prefix="",
        skip_shap=True,  # Skip SHAP for single-view
        method="single_view",
    )
    martingale_viz.create_visualization()

    # Visualize feature evolution
    visualize_results(
        model_name=model_name,
        graphs=graphs,
        features_raw=features_raw,
        true_change_points=true_change_points,
        detected_change_points=detection_result["traditional_change_points"],
        feature_results=detection_result,
        output_dir=output_dir,
    )

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
