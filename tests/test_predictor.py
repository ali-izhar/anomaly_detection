"""Tests for the graph predictor module."""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import networkx as nx

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
