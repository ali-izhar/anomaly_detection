"""Tests for the graph predictor module."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import networkx as nx
import seaborn as sns
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.predictor.predictor import GraphPredictor
from src.graph.generator import GraphGenerator
from src.graph.visualizer import NetworkVisualizer
from src.configs.loader import get_config

logger = logging.getLogger(__name__)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute prediction accuracy metrics.

    Parameters
    ----------
    actual : np.ndarray
        True adjacency matrix
    predicted : np.ndarray
        Predicted adjacency matrix

    Returns
    -------
    dict
        Dictionary containing accuracy metrics:
        - accuracy: Overall accuracy
        - recall: Edge coverage (% of true edges predicted)
        - fpr: False positive rate
        - precision: % of predicted edges that are correct
    """
    # Convert to binary
    actual_bin = actual.astype(bool)
    pred_bin = predicted.astype(bool)

    # True positives, false positives, etc
    tp = np.sum((actual_bin & pred_bin))
    fp = np.sum((~actual_bin & pred_bin))
    tn = np.sum((~actual_bin & ~pred_bin))
    fn = np.sum((actual_bin & ~pred_bin))

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "accuracy": accuracy,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "true_density": np.mean(actual_bin),
        "pred_density": np.mean(pred_bin),
    }


def plot_network_performance(
    actual_adj: np.ndarray, predicted_adj: np.ndarray, metrics: dict, t: int
) -> None:
    """Create a comprehensive visualization of network prediction performance."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Network Prediction Performance Analysis - Stochastic Block Model Model",
        y=0.95,
    )

    # Create grid layout
    gs = plt.GridSpec(2, 4, height_ratios=[1, 0.3])

    # 1. Plot actual network
    ax1 = fig.add_subplot(gs[0, 0])
    G_actual = nx.from_numpy_array(actual_adj)
    pos = nx.spring_layout(G_actual, seed=42)
    nx.draw_networkx(
        G_actual,
        pos=pos,
        node_color="lightblue",
        node_size=200,
        with_labels=True,
        font_size=8,
    )
    ax1.set_title(f"Best Actual Network (t={t})")

    # 2. Plot predicted network
    ax2 = fig.add_subplot(gs[0, 1])
    G_pred = nx.from_numpy_array(predicted_adj)
    nx.draw_networkx(
        G_pred,
        pos=pos,
        node_color="lightgreen",
        node_size=200,
        with_labels=True,
        font_size=8,
    )
    ax2.set_title("Best Predicted Network")

    # Add metrics text
    metrics_text = f"Coverage: {metrics['recall']:.3f}\nFPR: {metrics['fpr']:.3f}\nScore: {metrics['accuracy']:.3f}"
    ax2.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=8,
    )

    # 3. Plot actual adjacency matrix
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(actual_adj, cmap="Blues", square=True, cbar=False, ax=ax3)
    ax3.set_title("Best Actual Adjacency")

    # 4. Plot predicted adjacency matrix with error highlighting
    ax4 = fig.add_subplot(gs[0, 3])
    # Create colored matrix for visualization
    error_matrix = np.zeros_like(actual_adj)
    error_matrix = np.where(
        (actual_adj == 1) & (predicted_adj == 1),
        2,  # True Positive (Green)
        np.where(
            (actual_adj == 0) & (predicted_adj == 1),
            1,  # False Positive (Red)
            np.where(
                (actual_adj == 1) & (predicted_adj == 0), 3, 0  # Missed Edge (Black)
            ),
        ),
    )  # True Negative (White)

    cmap = plt.cm.colors.ListedColormap(["white", "red", "green", "black"])
    sns.heatmap(
        error_matrix,
        cmap=cmap,
        square=True,
        cbar=True,
        cbar_kws={"label": "Prediction Types"},
        ax=ax4,
    )
    ax4.set_title("Best Predicted Adjacency")

    # Add legend for prediction types
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="green", label="True Positive"),
        plt.Rectangle((0, 0), 1, 1, facecolor="red", label="False Positive"),
        plt.Rectangle((0, 0), 1, 1, facecolor="black", label="Missed Edge"),
    ]
    ax4.legend(
        handles=legend_elements,
        title="Prediction Types",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()
    return fig


def plot_performance_evolution(
    time_points, metrics_over_time, change_points, output_dir
):
    """Plot the evolution of performance metrics over time."""
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract metrics over time
    accuracies = np.array([m["accuracy"] for m in metrics_over_time])
    recalls = np.array([m["recall"] for m in metrics_over_time])
    fprs = np.array([m["fpr"] for m in metrics_over_time])

    # Apply moving average smoothing
    window = 5  # Window size for moving average
    accuracies_smooth = np.convolve(accuracies, np.ones(window) / window, mode="valid")
    recalls_smooth = np.convolve(recalls, np.ones(window) / window, mode="valid")
    fprs_smooth = np.convolve(fprs, np.ones(window) / window, mode="valid")

    # Adjust time points for smoothed data
    smooth_time_points = time_points[window - 1 :]

    # Plot smoothed metrics with original data as light points
    ax.plot(time_points, accuracies, "o", color="#0072B2", alpha=0.2, markersize=2)
    ax.plot(time_points, recalls, "o", color="#009E73", alpha=0.2, markersize=2)
    ax.plot(time_points, fprs, "o", color="#D55E00", alpha=0.2, markersize=2)

    ax.plot(
        smooth_time_points,
        accuracies_smooth,
        label="Accuracy",
        color="#0072B2",
        linewidth=2,
        alpha=0.9,
    )
    ax.plot(
        smooth_time_points,
        recalls_smooth,
        label="Coverage",
        color="#009E73",
        linewidth=2,
        alpha=0.9,
    )
    ax.plot(
        smooth_time_points,
        fprs_smooth,
        label="FPR",
        color="#D55E00",
        linewidth=2,
        alpha=0.9,
    )

    # Calculate and plot standard deviation bands
    def rolling_std(data, window):
        result = np.zeros(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.std(data[i : i + window])
        return result

    acc_std = rolling_std(accuracies, window)
    recall_std = rolling_std(recalls, window)
    fpr_std = rolling_std(fprs, window)

    ax.fill_between(
        smooth_time_points,
        accuracies_smooth - acc_std,
        accuracies_smooth + acc_std,
        color="#0072B2",
        alpha=0.2,
    )
    ax.fill_between(
        smooth_time_points,
        recalls_smooth - recall_std,
        recalls_smooth + recall_std,
        color="#009E73",
        alpha=0.2,
    )
    ax.fill_between(
        smooth_time_points,
        fprs_smooth - fpr_std,
        fprs_smooth + fpr_std,
        color="#D55E00",
        alpha=0.2,
    )

    # Add change points with improved visibility
    for cp in change_points:
        ax.axvline(
            x=cp,
            color="#CC79A7",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label="Change Point" if cp == change_points[0] else "",
        )
        ax.annotate(
            f"CP (t={cp})",
            xy=(cp, 0.1),
            xytext=(cp - 2, 0.05),
            fontsize=8,
            color="#CC79A7",
            ha="right",
        )

    # Customize grid
    ax.grid(True, linestyle=":", alpha=0.6)

    # Set axis labels and title with improved fonts
    ax.set_xlabel("Time", fontsize=10, labelpad=8)
    ax.set_ylabel("Metric Value", fontsize=10, labelpad=8)
    ax.set_title("Performance Metrics Evolution", fontsize=12, pad=10)

    # Customize axis ranges
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(time_points) - 1, max(time_points) + 1)

    # Add mean values to legend
    mean_acc = np.mean(accuracies)
    mean_recall = np.mean(recalls)
    mean_fpr = np.mean(fprs)

    legend_elements = [
        plt.Line2D([0], [0], color="#0072B2", label=f"Accuracy (μ={mean_acc:.3f})"),
        plt.Line2D([0], [0], color="#009E73", label=f"Coverage (μ={mean_recall:.3f})"),
        plt.Line2D([0], [0], color="#D55E00", label=f"FPR (μ={mean_fpr:.3f})"),
        plt.Line2D([0], [0], color="#CC79A7", linestyle="--", label="Change Point"),
    ]

    # Customize legend
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    # Add horizontal reference lines
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=0.25, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=0.75, color="gray", linestyle=":", alpha=0.3)

    # Customize tick labels
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure with high quality
    plt.savefig(
        output_dir / "metrics_evolution.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def plot_performance_extremes(best_case, avg_case, worst_case, output_dir):
    """Create a comprehensive visualization of prediction performance across cases."""
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        "Network Prediction Performance Analysis - Stochastic Block Model Model", y=0.95
    )

    # Create 3x4 grid
    gs = plt.GridSpec(3, 4, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # Create NetworkVisualizer instance
    viz = NetworkVisualizer()

    # Function to plot a single case
    def plot_case(case, row, case_name):
        # Network plots
        ax_net_actual = fig.add_subplot(gs[row, 0])
        ax_net_pred = fig.add_subplot(gs[row, 1])
        ax_adj_actual = fig.add_subplot(gs[row, 2])
        ax_adj_pred = fig.add_subplot(gs[row, 3])

        # Get consistent layout for both actual and predicted
        G_actual = nx.from_numpy_array(case["actual"])
        pos = nx.spring_layout(G_actual, seed=42)

        # Plot actual network
        viz.plot_network(
            case["actual"],
            ax=ax_net_actual,
            title=f"{case_name} Actual Network (t={case['time']})",
            layout_params={"pos": pos},
        )

        # Plot predicted network with metrics
        viz.plot_network(
            case["predicted"],
            ax=ax_net_pred,
            title=f"{case_name} Predicted Network",
            layout_params={"pos": pos},
        )

        # Add metrics text
        metrics_text = (
            f"Coverage: {case['metrics']['recall']:.3f}\n"
            f"FPR: {case['metrics']['fpr']:.3f}\n"
            f"Score: {case['metrics']['accuracy']:.3f}"
        )
        ax_net_pred.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax_net_pred.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            verticalalignment="top",
            fontsize=8,
        )

        # Plot actual adjacency
        viz.plot_adjacency(
            case["actual"], ax=ax_adj_actual, title=f"{case_name} Actual Adjacency"
        )

        # Plot predicted adjacency with error highlighting
        error_matrix = np.zeros_like(case["actual"])
        error_matrix = np.where(
            (case["actual"] == 1) & (case["predicted"] == 1),
            2,  # True Positive (Green)
            np.where(
                (case["actual"] == 0) & (case["predicted"] == 1),
                1,  # False Positive (Red)
                np.where(
                    (case["actual"] == 1) & (case["predicted"] == 0),
                    3,  # Missed Edge (Black)
                    0,
                ),
            ),
        )  # True Negative (White)

        cmap = plt.cm.colors.ListedColormap(["white", "red", "green", "black"])
        sns.heatmap(error_matrix, cmap=cmap, square=True, cbar=False, ax=ax_adj_pred)
        ax_adj_pred.set_title(f"{case_name} Predicted Adjacency")

    # Plot all cases
    plot_case(best_case, 0, "Best")
    plot_case(avg_case, 1, "Average")
    plot_case(worst_case, 2, "Worst")

    # Add prediction types legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="green", label="True Positive"),
        plt.Rectangle((0, 0), 1, 1, facecolor="red", label="False Positive"),
        plt.Rectangle((0, 0), 1, 1, facecolor="black", label="Missed Edge"),
    ]
    fig.legend(
        handles=legend_elements,
        title="Prediction Types",
        bbox_to_anchor=(0.98, 0.02),
        loc="lower right",
        fontsize=8,
    )

    plt.savefig(output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_network_prediction_metrics():
    """Test and visualize network prediction performance."""
    # 1. Load and generate test data
    config = get_config("stochastic_block_model")
    params = config["params"].__dict__
    params.update(
        {
            "n": 20,
            "seq_len": 50,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 20,
            "intra_prob": 0.8,
            "inter_prob": 0.1,
        }
    )

    # 2. Generate sequence
    generator = GraphGenerator("sbm")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    change_points = result["change_points"]

    # 3. Initialize predictor
    predictor = GraphPredictor(k=10, alpha=0.8, initial_gamma=0.1, initial_beta=0.5)

    # 4. Track metrics over time and extremes
    best_case = {"metrics": {"accuracy": 0}}
    worst_case = {"metrics": {"accuracy": 1}}
    avg_case = None
    all_cases = []
    metrics_over_time = []
    time_points = []

    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    # 5. Make predictions and compute metrics
    for t in range(10, len(graphs) - 3):
        history = graphs[max(0, t - 10) : t]
        current = graphs[t]

        predicted_adjs = predictor.forecast(
            history_adjs=history, current_adj=current, h=3
        )
        predicted_adjs = (predicted_adjs > 0.5).astype(float)

        for step in range(min(3, len(predicted_adjs))):
            if t + step + 1 < len(graphs):
                actual_adj = graphs[t + step + 1]
                pred_adj = predicted_adjs[step]
                metrics = compute_metrics(actual_adj, pred_adj)

                case = {
                    "metrics": metrics,
                    "actual": actual_adj,
                    "predicted": pred_adj,
                    "time": t + step + 1,
                }
                all_cases.append(case)

                metrics_over_time.append(metrics)
                time_points.append(t + step + 1)

                if metrics["accuracy"] > best_case["metrics"]["accuracy"]:
                    best_case = case
                if metrics["accuracy"] < worst_case["metrics"]["accuracy"]:
                    worst_case = case

    # Find average case (closest to mean accuracy)
    mean_accuracy = np.mean([c["metrics"]["accuracy"] for c in all_cases])
    avg_case = min(
        all_cases, key=lambda x: abs(x["metrics"]["accuracy"] - mean_accuracy)
    )

    # 6. Create visualizations
    plot_performance_evolution(
        time_points, metrics_over_time, change_points, output_dir
    )
    plot_performance_extremes(best_case, avg_case, worst_case, output_dir)

    # 7. Print summary statistics
    print("\nPrediction Performance Summary:")
    print(f"Best Accuracy: {best_case['metrics']['accuracy']:.3f}")
    print(
        f"Average Accuracy: {np.mean([m['accuracy'] for m in metrics_over_time]):.3f}"
    )
    print(f"Worst Accuracy: {worst_case['metrics']['accuracy']:.3f}")

    return (
        best_case["metrics"],
        np.mean([m["accuracy"] for m in metrics_over_time]),
        worst_case["metrics"],
    )


def test_predictor_visualization():
    """Test and visualize how the predictor forecasts graph evolution."""

    # 1. Load base configuration and override for visualization
    config = get_config("stochastic_block_model")
    params = config["params"].__dict__

    # Override with visualization-friendly parameters
    params.update(
        {
            "n": 20,  # Small number of nodes for clear visualization
            "seq_len": 50,  # Longer sequence to ensure enough history
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 20,  # Longer segments
            # Keep other SBM parameters from config but ensure clear community structure
            "intra_prob": 0.8,  # High intra-community probability
            "inter_prob": 0.1,  # Low inter-community probability
        }
    )

    # 2. Generate sequence
    generator = GraphGenerator("sbm")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    change_points = result["change_points"]

    # 3. Initialize predictor with test parameters
    predictor = GraphPredictor(
        k=10,  # Larger window for better temporal patterns
        alpha=0.8,
        initial_gamma=0.1,
        initial_beta=0.5,
    )

    # 4. Create visualizer
    viz = NetworkVisualizer()

    # 5. Select points around change point for visualization
    change_point = change_points[0]
    test_points = [
        max(15, change_point - 3),  # Ensure at least 15 graphs of history
        change_point,
        min(len(graphs) - 4, change_point + 3),  # Ensure we can see 3 steps ahead
    ]

    print(f"\nChange point at t={change_point}")
    print(f"Testing prediction at points: {test_points}")

    # Create output directory
    os.makedirs("test_results", exist_ok=True)

    # Track metrics across all predictions
    all_metrics = []

    # 6. For each test point, make predictions and visualize
    for t in test_points:
        # Get history for prediction
        history = graphs[max(0, t - 10) : t]  # Use full k=10 history when possible
        current = graphs[t]

        print(f"\nPredicting at t={t} with {len(history)} graphs in history")
        print(f"History densities: {[np.mean(g) for g in history]}")
        print(f"Current density: {np.mean(current)}")

        # Make predictions
        predicted_adjs = predictor.forecast(
            history_adjs=history, current_adj=current, h=3  # Predict 3 steps ahead
        )

        # Print prediction stats before thresholding
        print(
            "Prediction densities before threshold:",
            [np.mean(p) for p in predicted_adjs],
        )

        # Threshold predictions to binary (0 or 1)
        predicted_adjs = (predicted_adjs > 0.5).astype(float)
        print(
            "Prediction densities after threshold:",
            [np.mean(p) for p in predicted_adjs],
        )

        # Create visualization grid
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle(
            f"Graph Evolution at t={t} (Change Point: {change_point})", fontsize=12
        )

        # Compute layout once using current graph
        G_current = nx.from_numpy_array(current)
        pos = nx.spring_layout(G_current, k=1, iterations=50, seed=42)

        # Plot current state
        viz.plot_network(
            current, ax=axes[0, 0], title=f"Current (t={t})", layout_params={"pos": pos}
        )
        viz.plot_adjacency(current, ax=axes[1, 0], title="Current Adjacency")

        # Plot predictions
        for i, pred_adj in enumerate(predicted_adjs):
            viz.plot_network(
                pred_adj,
                ax=axes[0, i + 1],
                title=f"Predicted t+{i+1}",
                layout_params={"pos": pos},
            )
            viz.plot_adjacency(
                pred_adj,
                ax=axes[1, i + 1],
                title=f"Predicted Adj t+{i+1}",
                show_values=True,  # Show the binary values
            )

        plt.tight_layout()
        plt.savefig(f"test_results/prediction_t{t}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Also create a comparison with actual future states if available
        if t + 3 < len(graphs):
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f"Prediction vs Actual at t={t}", fontsize=12)

            for i in range(3):
                # Plot predicted (thresholded)
                viz.plot_adjacency(
                    predicted_adjs[i],
                    ax=axes[0, i],
                    title=f"Predicted t+{i+1}",
                    show_values=True,
                )

                # Plot actual
                viz.plot_adjacency(
                    graphs[t + i + 1],
                    ax=axes[1, i],
                    title=f"Actual t+{i+1}",
                    show_values=True,
                )

            plt.tight_layout()
            plt.savefig(
                f"test_results/comparison_t{t}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # For each prediction step
        for step in range(min(3, len(predicted_adjs))):
            if t + step + 1 < len(graphs):  # If we have actual future graph
                actual_adj = graphs[t + step + 1]
                pred_adj = predicted_adjs[step]

                # Compute metrics
                metrics = compute_metrics(actual_adj, pred_adj)
                all_metrics.append(metrics)

                # Log metrics for this prediction
                logger.info(f"\nPrediction metrics for t={t}, step={step+1}:")
                logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
                logger.info(f"Edge Coverage (Recall): {metrics['recall']:.3f}")
                logger.info(f"False Positive Rate: {metrics['fpr']:.3f}")
                logger.info(f"Precision: {metrics['precision']:.3f}")
                logger.info(f"True Density: {metrics['true_density']:.3f}")
                logger.info(f"Predicted Density: {metrics['pred_density']:.3f}")

    # Print average metrics
    if all_metrics:
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()
        }
        logger.info("\nAverage metrics across all predictions:")
        logger.info(f"Accuracy: {avg_metrics['accuracy']:.3f}")
        logger.info(f"Edge Coverage (Recall): {avg_metrics['recall']:.3f}")
        logger.info(f"False Positive Rate: {avg_metrics['fpr']:.3f}")
        logger.info(f"Precision: {avg_metrics['precision']:.3f}")
        logger.info(f"True Density: {avg_metrics['true_density']:.3f}")
        logger.info(f"Predicted Density: {avg_metrics['pred_density']:.3f}")


def test_predictor_basic():
    """Test basic functionality of the predictor."""
    # Create a simple predictor
    predictor = GraphPredictor(k=3, alpha=0.8)

    # Create a simple sequence of graphs
    n = 5  # number of nodes
    A1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    A2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    # Test forecast
    history = [A1, A2]
    current = A2
    predictions = predictor.forecast(history, current, h=2)

    # Basic checks
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2, 3, 3)  # h=2 predictions for 3x3 matrices
    assert np.all(predictions >= 0) and np.all(predictions <= 1)

    # Test symmetry
    for pred in predictions:
        assert np.allclose(pred, pred.T)
        assert np.allclose(np.diag(pred), 0)


if __name__ == "__main__":
    test_network_prediction_metrics()
