"""Tests for comparing actual vs predicted graph features."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.predictor.weighted import EnhancedWeightedPredictor
from src.graph.generator import GraphGenerator
from src.configs.loader import get_config


def compute_graph_features(G: nx.Graph) -> Dict[str, float]:
    """Compute comprehensive set of graph features with better handling of edge cases."""
    features = {}

    # Basic features
    n = G.number_of_nodes()
    features["n_nodes"] = n
    features["n_edges"] = G.number_of_edges()
    features["density"] = nx.density(G)

    # Degree statistics with error checking
    degrees = np.array([d for _, d in G.degree()])
    features["avg_degree"] = float(np.mean(degrees))
    features["degree_std"] = float(np.std(degrees)) if len(degrees) > 1 else 0.0
    features["max_degree"] = float(np.max(degrees)) if len(degrees) > 0 else 0.0

    # Clustering with error handling
    try:
        features["avg_clustering"] = float(nx.average_clustering(G))
    except:
        features["avg_clustering"] = 0.0

    # Component analysis
    components = list(nx.connected_components(G))
    features["n_components"] = len(components)
    features["largest_cc_size"] = len(max(components, key=len)) if components else 0

    # Only compute expensive metrics for smaller graphs
    if n <= 100:
        # Get largest component for path-based metrics
        largest_cc = max(components, key=len) if components else set()
        largest_cc_graph = G.subgraph(largest_cc).copy() if largest_cc else None

        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G)
            features["avg_betweenness"] = float(np.mean(list(betweenness.values())))
        except:
            features["avg_betweenness"] = 0.0

        # Path length metrics (computed only on largest component)
        if largest_cc_graph and nx.is_connected(largest_cc_graph):
            try:
                features["avg_path_length"] = float(
                    nx.average_shortest_path_length(largest_cc_graph)
                )
                features["diameter"] = float(nx.diameter(largest_cc_graph))
            except:
                features["avg_path_length"] = -1.0
                features["diameter"] = -1.0
        else:
            features["avg_path_length"] = -1.0
            features["diameter"] = -1.0

    return features


def plot_feature_comparison(
    actual_features: List[Dict[str, float]],
    predicted_features: List[Dict[str, float]],
    time_points: List[int],
    change_points: List[int],
    output_dir: Path,
) -> None:
    """Plot comparison with better handling of special values."""
    features = list(actual_features[0].keys())
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    axes_flat = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes_flat[idx]

        # Extract and clean feature values
        actual_vals = np.array([f[feature] for f in actual_features])
        pred_vals = np.array([f[feature] for f in predicted_features])

        # Handle special values
        valid_mask = (
            (actual_vals != float("inf"))
            & (actual_vals != -1.0)
            & (pred_vals != float("inf"))
            & (pred_vals != -1.0)
            & ~np.isnan(actual_vals)
            & ~np.isnan(pred_vals)
        )

        if np.any(valid_mask):
            plot_times = np.array(time_points)[valid_mask]
            actual_plot = actual_vals[valid_mask]
            pred_plot = pred_vals[valid_mask]

            # Plot with error handling
            ax.plot(plot_times, actual_plot, "b-", label="Actual", alpha=0.7)
            ax.plot(plot_times, pred_plot, "r--", label="Predicted", alpha=0.7)

            # Compute metrics only for valid values
            try:
                correlation = np.corrcoef(actual_plot, pred_plot)[0, 1]
                mse = np.mean((actual_plot - pred_plot) ** 2)
                title = f'{feature.replace("_", " ").title()}\nCorr: {correlation:.2f}, MSE: {mse:.2e}'
            except:
                title = feature.replace("_", " ").title()
        else:
            title = f'{feature.replace("_", " ").title()}\n(Insufficient Data)'

        # Add change points
        for cp in change_points:
            ax.axvline(x=cp, color="gray", linestyle=":", alpha=0.5)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

        ax.tick_params(axis="x", rotation=45)

    # Remove empty subplots
    for idx in range(len(features), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()
    plt.savefig(output_dir / "feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_correlations(
    actual_features: List[Dict[str, float]],
    predicted_features: List[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot correlations with better handling of special values."""
    features = list(actual_features[0].keys())
    correlations = []
    valid_features = []

    for feature in features:
        actual_vals = np.array([f[feature] for f in actual_features])
        pred_vals = np.array([f[feature] for f in predicted_features])

        # Handle special values
        valid_mask = (
            (actual_vals != float("inf"))
            & (actual_vals != -1.0)
            & (pred_vals != float("inf"))
            & (pred_vals != -1.0)
            & ~np.isnan(actual_vals)
            & ~np.isnan(pred_vals)
        )

        if np.any(valid_mask):
            actual_vals = actual_vals[valid_mask]
            pred_vals = pred_vals[valid_mask]

            try:
                if (
                    len(actual_vals) >= 2
                    and np.std(actual_vals) > 1e-10
                    and np.std(pred_vals) > 1e-10
                ):
                    corr = np.corrcoef(actual_vals, pred_vals)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        valid_features.append(feature)
            except:
                continue

    if correlations:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations, y=valid_features)
        plt.title("Feature Prediction Correlations")
        plt.xlabel("Correlation Coefficient")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "feature_correlations.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def run_single_trial(
    generator: GraphGenerator, params: dict, predictor: EnhancedWeightedPredictor
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[int], List[int]]:
    """Run a single trial of feature prediction."""
    # Generate sequence
    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    change_points = result["change_points"]

    # Initialize storage
    actual_features = []
    predicted_features = []
    time_points = []

    # Prepare history format
    history = [
        {"adjacency": adj, "graph": nx.from_numpy_array(adj)}
        for adj in graphs[:3]  # Initial history
    ]

    # Make predictions and compute features
    for t in range(3, len(graphs) - 1):
        # Make prediction
        predicted_adj = predictor.predict(history, horizon=1)[0]

        # Convert to graphs
        actual_graph = nx.from_numpy_array(graphs[t + 1])
        pred_graph = nx.from_numpy_array(predicted_adj)

        # Compute features
        actual_feats = compute_graph_features(actual_graph)
        pred_feats = compute_graph_features(pred_graph)

        actual_features.append(actual_feats)
        predicted_features.append(pred_feats)
        time_points.append(t + 1)

        # Update history
        history.append(
            {"adjacency": graphs[t], "graph": nx.from_numpy_array(graphs[t])}
        )

    return actual_features, predicted_features, time_points, change_points


def compute_trial_metrics(
    actual_features: List[Dict[str, float]], predicted_features: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Compute metrics with better handling of edge cases and numerical stability."""
    metrics = {}
    features = list(actual_features[0].keys())

    for feature in features:
        actual_vals = np.array([f[feature] for f in actual_features])
        pred_vals = np.array([f[feature] for f in predicted_features])

        # Handle special values
        valid_mask = (
            (actual_vals != float("inf"))
            & (actual_vals != -1.0)
            & (pred_vals != float("inf"))
            & (pred_vals != -1.0)
            & ~np.isnan(actual_vals)
            & ~np.isnan(pred_vals)
        )

        if not np.any(valid_mask):
            continue

        actual_vals = actual_vals[valid_mask]
        pred_vals = pred_vals[valid_mask]

        # Skip if not enough variation for correlation
        if (
            len(actual_vals) < 2
            or np.std(actual_vals) < 1e-10
            or np.std(pred_vals) < 1e-10
        ):
            continue

        # Compute metrics with error handling
        try:
            correlation = float(np.corrcoef(actual_vals, pred_vals)[0, 1])
            if np.isnan(correlation):
                continue

            mse = float(np.mean((actual_vals - pred_vals) ** 2))
            mae = float(np.mean(np.abs(actual_vals - pred_vals)))

            metrics[feature] = {"correlation": correlation, "mse": mse, "mae": mae}
        except:
            continue

    return metrics


def plot_averaged_feature_comparison(
    all_trials_data: List[
        Tuple[List[Dict[str, float]], List[Dict[str, float]], List[int], List[int]]
    ],
    output_dir: Path,
) -> None:
    """Plot averaged feature comparisons with standard deviation bands."""
    # Select features to plot (only averages)
    features_to_plot = [
        "avg_degree",
        "avg_clustering",
        "avg_betweenness",
        "avg_path_length",
        "density",
    ]

    n_features = len(features_to_plot)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Get time points (assuming same across all trials)
    time_points = all_trials_data[0][2]

    for idx, feature in enumerate(features_to_plot):
        ax = axes_flat[idx]

        # Collect values across all trials
        actual_vals_trials = []
        pred_vals_trials = []

        for actual_features, predicted_features, _, _ in all_trials_data:
            actual_vals = np.array([f[feature] for f in actual_features])
            pred_vals = np.array([f[feature] for f in predicted_features])

            # Handle special values
            valid_mask = (
                (actual_vals != float("inf"))
                & (actual_vals != -1.0)
                & (pred_vals != float("inf"))
                & (pred_vals != -1.0)
                & ~np.isnan(actual_vals)
                & ~np.isnan(pred_vals)
            )

            if np.any(valid_mask):
                actual_vals_trials.append(actual_vals[valid_mask])
                pred_vals_trials.append(pred_vals[valid_mask])

        if actual_vals_trials:
            # Convert to numpy arrays with padding
            max_len = max(len(v) for v in actual_vals_trials)
            actual_array = np.full((len(actual_vals_trials), max_len), np.nan)
            pred_array = np.full((len(pred_vals_trials), max_len), np.nan)

            for i, (act, pred) in enumerate(zip(actual_vals_trials, pred_vals_trials)):
                actual_array[i, : len(act)] = act
                pred_array[i, : len(pred)] = pred

            # Compute means and stds
            actual_mean = np.nanmean(actual_array, axis=0)
            actual_std = np.nanstd(actual_array, axis=0)
            pred_mean = np.nanmean(pred_array, axis=0)
            pred_std = np.nanstd(pred_array, axis=0)

            # Plot means and std bands
            plot_times = time_points[: len(actual_mean)]

            ax.plot(plot_times, actual_mean, "b-", label="Actual", alpha=0.7)
            ax.fill_between(
                plot_times,
                actual_mean - actual_std,
                actual_mean + actual_std,
                color="b",
                alpha=0.2,
            )

            ax.plot(plot_times, pred_mean, "r--", label="Predicted", alpha=0.7)
            ax.fill_between(
                plot_times,
                pred_mean - pred_std,
                pred_mean + pred_std,
                color="r",
                alpha=0.2,
            )

            # Compute metrics on means
            valid_mask = ~np.isnan(actual_mean) & ~np.isnan(pred_mean)
            if np.any(valid_mask):
                correlation = np.corrcoef(
                    actual_mean[valid_mask], pred_mean[valid_mask]
                )[0, 1]
                mse = np.mean((actual_mean[valid_mask] - pred_mean[valid_mask]) ** 2)
                title = f'{feature.replace("_", " ").title()}\nCorr: {correlation:.2f}, MSE: {mse:.2e}'
            else:
                title = feature.replace("_", " ").title()
        else:
            title = f'{feature.replace("_", " ").title()}\n(Insufficient Data)'

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

        ax.tick_params(axis="x", rotation=45)

    # Remove empty subplots
    for idx in range(len(features_to_plot), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()
    plt.savefig(
        output_dir / "feature_comparison_averaged.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def test_feature_prediction(n_trials: int = 5):
    """Test and visualize how well the predictor preserves network features across multiple trials."""

    print(f"\nRunning {n_trials} trials...")

    # 1. Load configuration
    config = get_config("stochastic_block_model")
    params = config["params"].__dict__

    # Override with test parameters
    params.update(
        {
            "n": 50,  # Network size
            "seq_len": 200,  # Sequence length
            "min_changes": 1,
            "max_changes": 2,
            "min_segment": 40,
            "intra_prob": 0.8,
            "inter_prob": 0.1,
        }
    )

    # Initialize generator and predictor
    generator = GraphGenerator("sbm")
    predictor = EnhancedWeightedPredictor(
        n_history=3,
        spectral_reg=0.3,
        community_reg=0.3,
        n_communities=2,
        temporal_window=5,
    )

    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    # Storage for all trials
    all_trials_data = []
    all_metrics = []

    # Run trials
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")

        # Run single trial
        trial_data = run_single_trial(generator, params, predictor)
        all_trials_data.append(trial_data)

        # Compute metrics for this trial
        actual_features, predicted_features, _, _ = trial_data
        trial_metrics = compute_trial_metrics(actual_features, predicted_features)
        all_metrics.append(trial_metrics)

    # Create averaged visualizations
    print("\nCreating averaged visualizations...")
    plot_averaged_feature_comparison(all_trials_data, output_dir)

    # Compute and print averaged metrics across all trials
    print("\nAveraged Feature Prediction Summary:")

    # Aggregate metrics across trials
    aggregated_metrics = defaultdict(lambda: defaultdict(list))
    for trial_metrics in all_metrics:
        for feature, metrics in trial_metrics.items():
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    aggregated_metrics[feature][metric_name].append(value)

    # Print averaged metrics
    for feature in aggregated_metrics:
        print(f"\n{feature}:")
        for metric_name in ["correlation", "mse", "mae"]:
            values = aggregated_metrics[feature][metric_name]
            if values:  # Only print if we have valid values
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric_name.upper()}: {mean_val:.3f} ± {std_val:.3f}")


if __name__ == "__main__":
    test_feature_prediction(n_trials=30)
