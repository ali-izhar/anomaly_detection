# tests/test_predictor/predictor_vis.py

"""Use all predictors on the same graph network (ba, sbm, er, ws) to compare their performance."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import argparse
from typing import Dict, Any, List

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.configs.loader import get_config
from src.graph.features import NetworkFeatureExtractor
from src.graph.generator import GraphGenerator
from src.graph.visualizer import NetworkVisualizer
from src.predictor.factory import PredictorFactory
from src.graph.metrics import (
    compute_feature_metrics,
    compute_feature_distribution_metrics,
    FeatureMetrics,
    DistributionMetrics,
)
from src.configs.plotting import (
    FIGURE_DIMENSIONS as FD,
    TYPOGRAPHY as TYPO,
    LINE_STYLE as LS,
    COLORS,
)


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def generate_network_and_features():
    """Generate network sequence and compute features once."""
    # Get full model name and config
    model_name = get_full_model_name(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__

    # Initialize components
    generator = GraphGenerator(model_alias)
    feature_extractor = NetworkFeatureExtractor()

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

    return model_name, params, graphs, change_points, features


def test_network_feature_visualization(
    model_name, params, graphs, change_points, features
):
    """Test network feature visualization for different graph models."""
    output_dir = "tests/test_predictor/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    viz = NetworkVisualizer()

    # Create network state visualizations
    print("Creating network state visualizations...")
    key_points = [0] + change_points + [len(graphs) - 1]
    n_points = len(key_points)

    fig, axes = plt.subplots(
        n_points,
        2,
        figsize=(FD["SINGLE_COLUMN_WIDTH"], FD["STANDARD_HEIGHT"] * n_points / 2),
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Network States",
        fontsize=TYPO["TITLE_SIZE"],
        y=0.98,
    )

    for i, time_idx in enumerate(key_points):
        # Prepare node colors for SBM
        node_color = None
        if model_alias == "sbm":
            block_sizes = [params["n"] // params["num_blocks"]] * (
                params["num_blocks"] - 1
            )
            block_sizes.append(params["n"] - sum(block_sizes))
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
            graphs[time_idx],
            ax=axes[i, 1],
            title=f"Adjacency Matrix at t={time_idx}",
        )

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    plt.savefig(
        Path(output_dir) / f"{model_name}_states.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(f"Done! Network state visualizations have been saved to {output_dir}/")


def compare_predictors(
    model_name: str,
    graphs: List[np.ndarray],
    change_points: List[int],
    features: List[Dict[str, Any]],
    predictor_configs: Dict[str, Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare all predictors on the same data."""
    output_dir = "tests/test_predictor/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize components
    viz = NetworkVisualizer()
    feature_extractor = NetworkFeatureExtractor()

    # Get all predictor types
    predictor_types = list(PredictorFactory.PREDICTOR_TYPES.keys())

    # Store results for each predictor
    results = {}

    # Initial warmup period
    warmup = 50

    # Run each predictor on the same data
    for predictor_type in predictor_types:
        print(f"\nTesting {predictor_type} predictor...")

        # Create predictor
        config = predictor_configs.get(predictor_type) if predictor_configs else None
        predictor = PredictorFactory.create(predictor_type, config)
        print(f"Using config:", config or "default")

        # Initialize history
        history = []
        predicted_features = []

        # Warmup period
        for t in range(warmup):
            state = {"adjacency": graphs[t], "time": t}
            history.append(state)
            predictor.update_state(state)

        # Generate predictions
        for t in range(warmup, len(graphs) - 1):
            # Update current state
            state = {"adjacency": graphs[t], "time": t}
            history.append(state)
            if len(history) > predictor.history_size:
                history = history[-predictor.history_size :]

            # Get predicted next state features
            pred_adjs = predictor.predict(history, horizon=1)
            pred_graph = nx.from_numpy_array(pred_adjs[0])
            predicted_features.append(feature_extractor.get_features(pred_graph))

            # Update predictor state
            predictor.update_state(state)

        # Get actual features for comparison
        actual_features = features[warmup + 1 : len(graphs)]

        # Compute metrics
        basic_metrics = compute_feature_metrics(actual_features, predicted_features)
        dist_metrics = compute_feature_distribution_metrics(
            actual_features, predicted_features
        )

        # Store results
        results[predictor_type] = {
            "basic_metrics": basic_metrics,
            "distribution_metrics": dist_metrics,
            "predicted_features": predicted_features,
        }

        # Plot individual predictor results
        plot_prediction_comparison(
            model_name,
            predictor_type,
            actual_features,
            predicted_features,
            basic_metrics,
            dist_metrics,
            change_points,
            warmup,
            viz,
            output_dir,
        )

    # Create comparison plot across predictors
    plot_predictor_comparison(model_name, results, viz, output_dir)

    return results


def plot_prediction_comparison(
    model_name: str,
    predictor_type: str,
    actual_features: List[Dict[str, Any]],
    predicted_features: List[Dict[str, Any]],
    basic_metrics: Dict[str, FeatureMetrics],
    dist_metrics: Dict[str, DistributionMetrics],
    change_points: List[int],
    warmup: int,
    viz: NetworkVisualizer,
    output_dir: str,
):
    """Plot comparison for a single predictor."""
    fig, axes = plt.subplots(
        4, 2, figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["GRID_HEIGHT"] * 2)
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Feature Prediction Comparison\n({predictor_type} predictor)",
        fontsize=TYPO["TITLE_SIZE"],
        y=0.98,
    )
    axes = axes.flatten()

    # Plot each feature comparison
    feature_names = list(actual_features[0].keys())
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        time = np.arange(len(actual_features))

        # For list features (like degrees), plot mean and std
        if isinstance(actual_features[0][feature], list):
            # Actual values
            actual_means = [np.mean(f[feature]) for f in actual_features]
            actual_stds = [np.std(f[feature]) for f in actual_features]
            ax.plot(time, actual_means, label="Actual", color=COLORS["actual"])
            ax.fill_between(
                time,
                np.array(actual_means) - np.array(actual_stds),
                np.array(actual_means) + np.array(actual_stds),
                color=COLORS["actual"],
                alpha=0.1,
            )

            # Predicted values
            pred_means = [np.mean(f[feature]) for f in predicted_features]
            pred_stds = [np.std(f[feature]) for f in predicted_features]
            ax.plot(time, pred_means, label="Predicted", color=COLORS["predicted"])
            ax.fill_between(
                time,
                np.array(pred_means) - np.array(pred_stds),
                np.array(pred_means) + np.array(pred_stds),
                color=COLORS["predicted"],
                alpha=0.1,
            )

            # Add distribution metrics if available
            if feature in dist_metrics:
                metrics_text = (
                    f"KL: {dist_metrics[feature].kl_divergence:.3f}\n"
                    f"JS: {dist_metrics[feature].js_divergence:.3f}\n"
                    f"W: {dist_metrics[feature].wasserstein:.3f}"
                )
            else:
                metrics_text = ""
        else:
            # For scalar features
            actual_values = [f[feature] for f in actual_features]
            pred_values = [f[feature] for f in predicted_features]
            ax.plot(time, actual_values, label="Actual", color=COLORS["actual"])
            ax.plot(time, pred_values, label="Predicted", color=COLORS["predicted"])
            metrics_text = ""

        # Add basic metrics
        basic_text = (
            f"RMSE: {basic_metrics[feature].rmse:.3f}\n"
            f"MAE: {basic_metrics[feature].mae:.3f}\n"
            f"R²: {basic_metrics[feature].r2:.3f}"
        )

        # Combine metrics text
        full_text = basic_text
        if metrics_text:
            full_text += "\n" + metrics_text

        # Add metrics text to plot
        ax.text(
            0.02,
            0.98,
            full_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=TYPO["ANNOTATION_SIZE"],
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        # Mark change points
        for cp in change_points:
            if cp >= warmup and cp < len(actual_features) + warmup:
                ax.axvline(
                    cp - warmup,
                    color=COLORS["change_point"],
                    linestyle="--",
                    alpha=0.5,
                )

        ax.set_title(feature.replace("_", " ").title(), fontsize=TYPO["TITLE_SIZE"])
        ax.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"])
        ax.set_ylabel("Value", fontsize=TYPO["LABEL_SIZE"])
        ax.tick_params(labelsize=TYPO["TICK_SIZE"])
        ax.grid(True, alpha=LS["GRID_ALPHA"])
        ax.legend(fontsize=TYPO["LEGEND_SIZE"])

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    plt.savefig(
        Path(output_dir) / f"{model_name}_{predictor_type}_prediction_features.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_predictor_comparison(
    model_name: str,
    results: Dict[str, Dict[str, Any]],
    viz: NetworkVisualizer,
    output_dir: str,
):
    """Plot comparison across all predictors."""
    # Get feature names from first predictor's results
    first_predictor = next(iter(results.values()))
    feature_names = list(first_predictor["basic_metrics"].keys())

    # Create comparison plots
    fig, axes = plt.subplots(
        len(feature_names),
        1,
        figsize=(FD["DOUBLE_COLUMN_WIDTH"], FD["STANDARD_HEIGHT"] * len(feature_names)),
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Predictor Comparison",
        fontsize=TYPO["TITLE_SIZE"],
        y=0.98,
    )

    if len(feature_names) == 1:
        axes = [axes]

    # Plot comparison for each feature
    for i, feature in enumerate(feature_names):
        ax = axes[i]

        # Bar plot of RMSE and R² for each predictor
        x = np.arange(len(results))
        width = 0.35

        rmse_values = [results[p]["basic_metrics"][feature].rmse for p in results]
        r2_values = [results[p]["basic_metrics"][feature].r2 for p in results]

        ax.bar(x - width / 2, rmse_values, width, label="RMSE")
        ax.bar(x + width / 2, r2_values, width, label="R²")

        ax.set_title(
            f"{feature.replace('_', ' ').title()}", fontsize=TYPO["TITLE_SIZE"]
        )
        ax.set_xticks(x)
        ax.set_xticklabels(results.keys())
        ax.legend()
        ax.grid(True, alpha=LS["GRID_ALPHA"])

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    plt.savefig(
        Path(output_dir) / f"{model_name}_predictor_comparison.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test network predictors")
    parser.add_argument(
        "model", choices=["ba", "ws", "er", "sbm"], help="Network model type"
    )
    parser.add_argument(
        "--configs", type=str, help="JSON string of predictor configs for each type"
    )
    args = parser.parse_args()

    model_alias = args.model

    # Parse configs if provided
    predictor_configs = None
    if args.configs:
        import json

        try:
            predictor_configs = json.loads(args.configs)
        except json.JSONDecodeError:
            print("Error: Invalid JSON config string")
            sys.exit(1)

    # Generate network and compute features once
    model_name, params, graphs, change_points, features = (
        generate_network_and_features()
    )

    # Run visualizations
    test_network_feature_visualization(
        model_name, params, graphs, change_points, features
    )

    # Compare all predictors
    results = compare_predictors(
        model_name, graphs, change_points, features, predictor_configs
    )

    # Print summary metrics for all predictors
    print("\nPrediction Performance Summary:")
    for predictor_type, metrics in results.items():
        print(f"\n{predictor_type} Predictor:")
        print(
            "Average RMSE across features:",
            np.mean([m.rmse for m in metrics["basic_metrics"].values()]),
        )
        print(
            "Average R² across features:",
            np.mean([m.r2 for m in metrics["basic_metrics"].values()]),
        )
