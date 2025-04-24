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
from src.utils.plot_graph import NetworkVisualizer
from src.predictor.factory import PredictorFactory
from src.graph.metrics import (
    compute_feature_metrics,
    compute_feature_distribution_metrics,
    FeatureMetrics,
    DistributionMetrics,
)
from src.utils.plotting_config import (
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

    print(
        f"Generated {len(graphs)} graphs with {len(change_points)} change points at: {change_points}"
    )

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
    # This function is temporarily disabled
    print("Network state visualization is disabled.")
    return

    # Original code below
    """
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
    """


def create_prediction_comparison_matrix(
    actual_adj: np.ndarray, predicted_adj: np.ndarray
):
    """Create a color-coded matrix comparing actual and predicted adjacency matrices.

    Colors:
    - Green: Correct predictions (true positives)
    - Red: False positives
    - Gray: Missed edges (false negatives)
    - Black: Correct non-edges (true negatives)

    Returns:
        RGB matrix and prediction metrics
    """
    # Create RGB matrix (initialize with black for correct non-edges)
    height, width = actual_adj.shape
    rgb_matrix = np.zeros((height, width, 3))

    # Green for correct predictions (true positives)
    true_positives = (actual_adj == 1) & (predicted_adj == 1)
    rgb_matrix[true_positives, 0] = 0  # R
    rgb_matrix[true_positives, 1] = 1  # G
    rgb_matrix[true_positives, 2] = 0  # B

    # Red for false positives
    false_positives = (actual_adj == 0) & (predicted_adj == 1)
    rgb_matrix[false_positives, 0] = 1  # R
    rgb_matrix[false_positives, 1] = 0  # G
    rgb_matrix[false_positives, 2] = 0  # B

    # Gray for missed edges (false negatives)
    false_negatives = (actual_adj == 1) & (predicted_adj == 0)
    rgb_matrix[false_negatives, 0] = 0.7  # R
    rgb_matrix[false_negatives, 1] = 0.7  # G
    rgb_matrix[false_negatives, 2] = 0.7  # B

    # Compute metrics
    tp = np.sum(true_positives)
    fp = np.sum(false_positives)
    fn = np.sum(false_negatives)

    # Avoid division by zero
    if tp + fn > 0:
        coverage = tp / (tp + fn)
    else:
        coverage = 0

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # False positive rate
    total_negatives = np.sum(actual_adj == 0)
    if total_negatives > 0:
        fpr = fp / total_negatives
    else:
        fpr = 0

    # Overall score (harmonic mean of coverage and 1-FPR)
    if coverage > 0 and fpr < 1:
        score = 2 * coverage * (1 - fpr) / (coverage + (1 - fpr))
    else:
        score = 0

    metrics = {"coverage": coverage, "fpr": fpr, "score": score}

    return rgb_matrix, metrics


def plot_adjacency_comparison_around_change_points(
    model_name: str,
    predictor_type: str,
    graphs: List[np.ndarray],
    predicted_adjs: List[np.ndarray],
    change_points: List[int],
    warmup: int,
    viz: NetworkVisualizer,
    output_dir: str,
):
    """Plot actual vs predicted adjacency matrices before and after change points."""
    print("Creating adjacency comparison visualizations around change points...")

    # Filter change points that occur after warmup
    valid_change_points = [cp for cp in change_points if cp > warmup + 1]

    if not valid_change_points:
        print("No change points available after warmup period for comparison.")
        return

    # Only use the first valid change point
    cp = valid_change_points[0]
    print(f"Creating visualization for first change point at t={cp}")

    # Calculate indices for actual and predicted adjacency matrices
    before_idx = cp - 1
    after_idx = cp + 1
    stable_idx = cp + 30  # 30 steps after the change point to show stabilization

    # Check if indices are within range
    if stable_idx >= len(graphs):
        print(f"Warning: t+30 is out of range, using last available point instead")
        stable_idx = len(graphs) - 1

    # Actual matrices
    actual_before = graphs[before_idx]
    actual_after = graphs[after_idx]
    actual_stable = graphs[stable_idx]

    # Predicted matrices (offset by warmup+1)
    pred_before_idx = before_idx - (warmup + 1)
    pred_after_idx = after_idx - (warmup + 1)
    pred_stable_idx = stable_idx - (warmup + 1)

    # Check if we have predictions for these indices
    if pred_before_idx < 0 or pred_before_idx >= len(predicted_adjs):
        print(f"Skip change point at t={cp} due to prediction index out of range")
        return

    if pred_after_idx >= len(predicted_adjs) or pred_stable_idx >= len(predicted_adjs):
        print(f"Skip change point at t={cp} due to prediction index out of range")
        return

    pred_before = predicted_adjs[pred_before_idx]
    pred_after = predicted_adjs[pred_after_idx]
    pred_stable = predicted_adjs[pred_stable_idx]

    # Create comparison matrices with color coding
    before_comparison, before_metrics = create_prediction_comparison_matrix(
        actual_before, pred_before
    )
    after_comparison, after_metrics = create_prediction_comparison_matrix(
        actual_after, pred_after
    )
    stable_comparison, stable_metrics = create_prediction_comparison_matrix(
        actual_stable, pred_stable
    )

    # Increased font sizes
    TITLE_SIZE = 14
    METRICS_SIZE = 13
    LEGEND_SIZE = 12

    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Plot actual matrices on top row
    axes[0, 0].imshow(actual_before, cmap="Blues")
    axes[0, 0].set_title(f"Actual t={before_idx}\n(before change)", fontsize=TITLE_SIZE)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(actual_after, cmap="Blues")
    axes[0, 1].set_title(f"Actual t={after_idx}\n(after change)", fontsize=TITLE_SIZE)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(actual_stable, cmap="Blues")
    axes[0, 2].set_title(f"Actual t={stable_idx}\n(stabilized)", fontsize=TITLE_SIZE)
    axes[0, 2].axis("off")

    # Plot comparison matrices on bottom row
    axes[1, 0].imshow(before_comparison)
    metrics_text = f"Coverage: {before_metrics['coverage']:.3f}\nFPR: {before_metrics['fpr']:.3f}\nScore: {before_metrics['score']:.3f}"
    axes[1, 0].set_title(
        f"Prediction t={before_idx}\n{metrics_text}", fontsize=METRICS_SIZE
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(after_comparison)
    metrics_text = f"Coverage: {after_metrics['coverage']:.3f}\nFPR: {after_metrics['fpr']:.3f}\nScore: {after_metrics['score']:.3f}"
    axes[1, 1].set_title(
        f"Prediction t={after_idx}\n{metrics_text}", fontsize=METRICS_SIZE
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(stable_comparison)
    metrics_text = f"Coverage: {stable_metrics['coverage']:.3f}\nFPR: {stable_metrics['fpr']:.3f}\nScore: {stable_metrics['score']:.3f}"
    axes[1, 2].set_title(
        f"Prediction t={stable_idx}\n{metrics_text}", fontsize=METRICS_SIZE
    )
    axes[1, 2].axis("off")

    # Add legend for color coding
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="green", label="Correct Prediction"),
        plt.Rectangle((0, 0), 1, 1, color="red", label="False Positive"),
        plt.Rectangle((0, 0), 1, 1, color="gray", label="Missed Edge"),
        plt.Rectangle((0, 0), 1, 1, color="black", label="Correct Non-edge"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        fontsize=LEGEND_SIZE,
    )

    # Add padding for better layout with larger text
    plt.tight_layout(pad=2.0)
    plt.savefig(
        Path(output_dir)
        / f"{model_name}_{predictor_type}_change_point_{cp}_comparison.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(f"Done! Adjacency comparison visualization has been saved to {output_dir}/")


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
    # predictor_types = list(PredictorFactory.PREDICTOR_TYPES.keys())
    predictor_types = ["graph"]

    # Store results for each predictor
    results = {}

    # Minimum warmup period to have enough history (smaller than first change point)
    warmup = 10

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
        predicted_adjs = []  # Store predicted adjacency matrices

        # Minimum warmup period (just enough for history requirements)
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
            pred_adjs_list = predictor.predict(history, horizon=1)
            pred_adj = pred_adjs_list[0]
            pred_graph = nx.from_numpy_array(pred_adj)

            # Store predictions
            predicted_adjs.append(pred_adj)
            predicted_features.append(feature_extractor.get_features(pred_graph))

            # Update predictor state
            predictor.update_state(state)

        # Get all actual features for full timeline display
        actual_features = features

        # Get actual features for metrics calculation - should align with predictions (warmup+1 to end)
        metrics_actual_features = features[warmup + 1 : len(graphs)]

        # Compute metrics
        basic_metrics = compute_feature_metrics(
            metrics_actual_features, predicted_features
        )
        dist_metrics = compute_feature_distribution_metrics(
            metrics_actual_features, predicted_features
        )

        # Store results
        results[predictor_type] = {
            "basic_metrics": basic_metrics,
            "distribution_metrics": dist_metrics,
            "predicted_features": predicted_features,
            "predicted_adjs": predicted_adjs,  # Store predicted adjacency matrices
            "actual_features": actual_features,
            "warmup": warmup,
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

        # Plot adjacency comparisons around change points
        plot_adjacency_comparison_around_change_points(
            model_name,
            predictor_type,
            graphs,
            predicted_adjs,
            change_points,
            warmup,
            viz,
            output_dir,
        )

    # Create comparison plot across predictors - DISABLED
    # plot_predictor_comparison(model_name, results, viz, output_dir)

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
    # Remove suptitle
    # fig.suptitle(
    #     f"{model_name.replace('_', ' ').title()} Feature Prediction Comparison\n({predictor_type} predictor)",
    #     fontsize=TYPO["TITLE_SIZE"],
    #     y=0.98,
    # )
    axes = axes.flatten()

    # Create references for legend (will use plot from first subplot)
    legend_lines = []
    legend_labels = []

    # Plot each feature comparison
    feature_names = list(actual_features[0].keys())
    for i, feature in enumerate(feature_names):
        ax = axes[i]

        # Full time range for x-axis
        full_time = np.arange(len(actual_features))

        # For list features (like degrees), plot mean and std
        if isinstance(actual_features[0][feature], list):
            # Actual values (full timeline)
            actual_means = [np.mean(f[feature]) for f in actual_features]
            actual_stds = [np.std(f[feature]) for f in actual_features]
            (actual_line,) = ax.plot(full_time, actual_means, color=COLORS["actual"])
            ax.fill_between(
                full_time,
                np.array(actual_means) - np.array(actual_stds),
                np.array(actual_means) + np.array(actual_stds),
                color=COLORS["actual"],
                alpha=0.1,
            )

            # Predicted values (starting from warmup+1)
            pred_means = [np.mean(f[feature]) for f in predicted_features]
            pred_stds = [np.std(f[feature]) for f in predicted_features]

            # Create time points for predictions (starting from warmup+1)
            pred_time = np.arange(warmup + 1, warmup + 1 + len(predicted_features))

            (pred_line,) = ax.plot(pred_time, pred_means, color=COLORS["predicted"])
            ax.fill_between(
                pred_time,
                np.array(pred_means) - np.array(pred_stds),
                np.array(pred_means) + np.array(pred_stds),
                color=COLORS["predicted"],
                alpha=0.1,
            )

            # Save lines for legend (only from first subplot)
            if i == 0:
                legend_lines = [actual_line, pred_line]
                legend_labels = ["Actual", "Predicted"]

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

            # Create time points for predictions (starting from warmup+1)
            pred_time = np.arange(warmup + 1, warmup + 1 + len(predicted_features))

            (actual_line,) = ax.plot(full_time, actual_values, color=COLORS["actual"])
            (pred_line,) = ax.plot(pred_time, pred_values, color=COLORS["predicted"])

            # Save lines for legend (only from first subplot)
            if i == 0:
                legend_lines = [actual_line, pred_line]
                legend_labels = ["Actual", "Predicted"]

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

        # Remove the metrics text box display
        # ax.text(
        #     0.02,
        #     0.98,
        #     full_text,
        #     transform=ax.transAxes,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        #     fontsize=TYPO["ANNOTATION_SIZE"],
        #     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        # )

        # Mark change points at their absolute positions
        for cp in change_points:
            ax.axvline(
                cp,
                color=COLORS["change_point"],
                linestyle="--",
                alpha=0.5,
            )

        ax.set_title(feature.replace("_", " ").title(), fontsize=TYPO["TITLE_SIZE"])
        ax.set_xlabel("Time", fontsize=TYPO["LABEL_SIZE"])
        ax.set_ylabel("Value", fontsize=TYPO["LABEL_SIZE"])
        ax.tick_params(labelsize=TYPO["TICK_SIZE"])
        ax.grid(True, alpha=LS["GRID_ALPHA"])

        # Don't add legend to individual subplots
        # ax.legend(fontsize=TYPO["LEGEND_SIZE"])

    # Add a single legend for the entire figure
    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower right",
        fontsize=TYPO["LEGEND_SIZE"],
        framealpha=0.7,
    )

    plt.tight_layout()
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
    # This function is temporarily disabled
    print("Predictor comparison is disabled.")
    return

    # Original code below
    """
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
    """


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

    # Run visualizations - DISABLED
    # test_network_feature_visualization(
    #     model_name, params, graphs, change_points, features
    # )

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
