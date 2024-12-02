# tests/test_spatiotemporal.py

import sys
import os
import torch
import logging
import numpy as np
from typing import Tuple, Dict, Any, List
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import extract_centralities, compute_embeddings
from src.models import (
    GraphConvLayer,
    TemporalAttention,
    STEncoder,
    TemporalDecoder,
    SpatioTemporalPredictor,
    STModelConfig,
)
from tests.create_ba_graphs import generate_ba_graphs, BA_CONFIG
from visualize_graphs import GraphVisualizer
from tests.visualize_ba_martingales import BAMartingaleAnalyzer, BA_MARTINGALE_CONFIG
from tests.visualize_ba_martingales import MartingaleVisualizer
from tests.predict_changepoints import ChangePointPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Normalize adjacency matrix: D^(-1/2)AD^(-1/2)."""
    # Add self-loops
    adj = adj + torch.eye(adj.size(0))

    # Compute D^(-1/2)
    d = adj.sum(1)
    d = torch.diag(torch.pow(d, -0.5))

    # Compute D^(-1/2)AD^(-1/2)
    return torch.matmul(torch.matmul(d, adj), d)


def generate_synthetic_data(
    n_nodes: int = 20, n_graphs: int = 100, config: Dict = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[np.ndarray]]:
    """Generate synthetic graph sequence with features."""

    # Generate graph sequence
    logger.info(
        f"Generating synthetic graph sequence: {n_nodes} nodes, {n_graphs} graphs"
    )
    result = generate_ba_graphs(config)
    graphs = result["graphs"]

    # Extract features
    logger.info("Extracting graph features")
    centralities = extract_centralities(graphs)
    embeddings = compute_embeddings(graphs, method="svd", n_components=2)

    # Combine features
    features = []
    for i in range(len(graphs)):
        graph_features = []
        for cent_type in centralities.values():
            graph_features.extend(cent_type[i])
        graph_features.extend(embeddings[i])
        features.append(graph_features)

    # Convert to tensors
    features_tensor = torch.FloatTensor(features)
    adj_tensor = normalize_adjacency(torch.FloatTensor(graphs[0]))

    # Reshape features to [n_graphs, n_nodes, features_per_node]
    n_features_per_node = features_tensor.shape[-1] // n_nodes
    features_tensor = features_tensor.reshape(n_graphs, n_nodes, n_features_per_node)

    metadata = {
        "n_centrality_features": len(centralities),
        "n_embedding_features": len(embeddings[0]) // n_nodes,
        "features_per_node": n_features_per_node,
        "total_timesteps": n_graphs,
    }

    return features_tensor, adj_tensor, metadata, graphs


def test_graph_conv_layer(
    features: torch.Tensor, adj: torch.Tensor, in_dim: int, out_dim: int
) -> None:
    """Test GraphConvLayer."""
    logger.info("\nTesting GraphConvLayer")
    layer = GraphConvLayer(in_dim=in_dim, out_dim=out_dim)

    # Take a batch of features from different timesteps
    batch_size = 2
    x = features[:batch_size]  # [batch_size, n_nodes, features_per_node]

    # Forward pass
    start_time = time.time()
    output = layer(x, adj)
    elapsed = time.time() - start_time

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Forward pass time: {elapsed:.4f}s")


def test_temporal_attention(features: torch.Tensor, hidden_dim: int) -> None:
    """Test TemporalAttention."""
    logger.info("\nTesting TemporalAttention")
    attention = TemporalAttention(hidden_dim=hidden_dim)

    # Prepare input: use a sequence of node features
    batch_size = 2
    seq_len = 10
    n_nodes = features.shape[1]

    # Create random hidden representations
    x = torch.randn(batch_size, seq_len, n_nodes, hidden_dim)

    # Forward pass
    start_time = time.time()
    output = attention(x)
    elapsed = time.time() - start_time

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Forward pass time: {elapsed:.4f}s")


def test_encoder(
    features: torch.Tensor, adj: torch.Tensor, in_channels: int, hidden_dim: int
) -> None:
    """Test STEncoder."""
    logger.info("\nTesting STEncoder")
    encoder = STEncoder(
        in_channels=in_channels, hidden_dim=hidden_dim, num_layers=2, dropout=0.1
    )

    # Prepare input: use a sequence of features
    batch_size = 2
    seq_len = 10
    x = features[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Forward pass
    start_time = time.time()
    encoded, hidden_states = encoder(x, adj)
    elapsed = time.time() - start_time

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Encoded shape: {encoded.shape}")
    logger.info(f"Hidden states: {len(hidden_states)}")
    logger.info(f"Forward pass time: {elapsed:.4f}s")


def test_decoder(hidden_dim: int, num_features: int, n_nodes: int = 20) -> None:
    """Test TemporalDecoder."""
    logger.info("\nTesting TemporalDecoder")
    decoder = TemporalDecoder(
        hidden_dim=hidden_dim,
        num_features=num_features,
        num_nodes=n_nodes,
        num_layers=2,
        dropout=0.1,
    )

    # Prepare input
    batch_size = 2
    seq_len = 10

    # Create input with correct dimensions [batch_size, seq_len, hidden_dim]
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    start_time = time.time()
    predictions = decoder(x, steps=5)
    elapsed = time.time() - start_time

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Forward pass time: {elapsed:.4f}s")


def test_full_model(
    features: torch.Tensor, adj: torch.Tensor, num_features: int
) -> None:
    """Test complete SpatioTemporalPredictor."""
    logger.info("\nTesting Full Model")

    # Get dimensions from features tensor
    n_nodes = features.shape[1]  # [n_timesteps, n_nodes, n_features]

    config = STModelConfig(
        hidden_dim=64, num_layers=2, window_size=10, forecast_horizon=5
    )

    model = SpatioTemporalPredictor(
        config=config, num_nodes=n_nodes, num_features=num_features
    )

    # Prepare input
    batch_size = 2
    seq_len = config.window_size
    x = features[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Forward pass
    start_time = time.time()
    predictions = model(x, adj)
    elapsed = time.time() - start_time

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Forward pass time: {elapsed:.4f}s")


def visualize_synthetic_data(
    graphs: List[np.ndarray], metadata: Dict[str, Any]
) -> None:
    """Visualize the generated synthetic graph sequence, martingales, and predictions."""
    logger.info("\nVisualizing graph sequence")

    # Calculate change points based on the sequence lengths
    graphs_per_segment = metadata["total_timesteps"] // 4
    change_points = [graphs_per_segment, graphs_per_segment * 2, graphs_per_segment * 3]

    # Graph structure visualization
    visualizer = GraphVisualizer(
        graphs=graphs, change_points=change_points, graph_type="BA"
    )
    visualizer.create_dashboard()

    # Martingale visualization
    logger.info("\nComputing and visualizing martingales")
    analyzer = BAMartingaleAnalyzer(
        threshold=BA_MARTINGALE_CONFIG["threshold"],
        epsilon=BA_MARTINGALE_CONFIG["epsilon"],
        output_dir="test_outputs/martingales",
    )

    martingales = analyzer.compute_martingales(graphs)
    martingale_visualizer = MartingaleVisualizer(
        graphs=graphs,
        change_points=change_points,
        martingales=martingales,
        graph_type="BA",
        threshold=BA_MARTINGALE_CONFIG["threshold"],
        epsilon=BA_MARTINGALE_CONFIG["epsilon"],
        output_dir="test_outputs/martingales",
    )
    martingale_visualizer.create_dashboard()
    martingale_visualizer.save_results()

    # Change point prediction
    logger.info("\nPredicting and visualizing change points")
    predictor = ChangePointPredictor(
        window_size=10,
        forecast_horizon=5,
        threshold=20.0,
        epsilon=0.8,
    )

    # Train on first segment (before first change point)
    predictor.train(graphs, true_change_point=change_points[0])

    # Track predictions
    predictions_log = []

    # Make predictions using sliding window
    for t in range(change_points[0], len(graphs) - predictor.config.window_size):
        # Get current window of graphs
        current_window = graphs[t : t + predictor.config.window_size]

        # Extract features
        centralities = extract_centralities(current_window)
        embeddings = compute_embeddings(current_window, method="svd", n_components=2)

        # Combine features
        features = []
        n_nodes = current_window[0].shape[0]

        for i in range(len(current_window)):
            node_features = []
            for cent_type in centralities.values():
                cent_values = np.array(cent_type[i]).reshape(n_nodes, 1)
                node_features.append(cent_values)
            node_features.append(embeddings[i].reshape(n_nodes, -1))
            timestep_features = np.hstack(node_features)
            features.append(timestep_features)

        # Convert to tensor
        features_tensor = torch.FloatTensor(np.array(features))
        features_tensor = features_tensor.unsqueeze(0)

        # Make predictions
        predictions = predictor.predict(current_window, features_tensor)

        # Log prediction details
        pred_info = {
            "timestep": t,
            "martingale_values": predictions["predicted_martingales"].mean(axis=-1),
            "max_martingale": predictions["predicted_martingales"].max(),
            "detection_times": predictions["detection_times"],
            "confidence": predictions["max_confidence"],
        }
        predictions_log.append(pred_info)

    # Visualize predictions
    plt.figure(figsize=(15, 8))

    # Plot actual change points
    for cp in change_points:
        plt.axvline(x=cp, color="r", linestyle="--", label="Actual Change Point")

    # Plot predicted martingale values
    timesteps = [p["timestep"] for p in predictions_log]
    martingale_values = [p["max_martingale"] for p in predictions_log]
    plt.plot(timesteps, martingale_values, label="Predicted Martingale")

    # Plot confidence scores
    confidence_scores = [p["confidence"] for p in predictions_log]
    plt.plot(timesteps, confidence_scores, label="Confidence Score")

    plt.axhline(
        y=predictor.threshold, color="g", linestyle=":", label="Detection Threshold"
    )

    plt.title("Change Point Prediction Results")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Save prediction visualization
    plt.savefig("test_outputs/predictions/prediction_results.png")
    plt.close()


def main():
    # Configure graph generation
    n_nodes = 20
    n_graphs = 100
    custom_config = BA_CONFIG.copy()
    custom_config["n"] = n_nodes
    graphs_per_segment = n_graphs // 4
    custom_config["sequence_length"] = {
        "before_change": graphs_per_segment,
        "after_change1": graphs_per_segment,
        "after_change2": graphs_per_segment,
        "after_change3": graphs_per_segment,
    }

    # Generate synthetic data
    features, adj, metadata, graphs = generate_synthetic_data(
        n_nodes=n_nodes, n_graphs=n_graphs, config=custom_config
    )

    logger.info(f"Generated features shape: {features.shape}")
    logger.info(f"Adjacency matrix shape: {adj.shape}")
    logger.info(f"Feature metadata: {metadata}")

    # Create output directories
    os.makedirs("test_outputs/martingales", exist_ok=True)
    os.makedirs("test_outputs/predictions", exist_ok=True)

    # Visualize the graphs, martingales, and predictions
    visualize_synthetic_data(graphs, metadata)

    # Test individual components
    hidden_dim = 64
    test_graph_conv_layer(features, adj, metadata["features_per_node"], hidden_dim)
    test_temporal_attention(features, hidden_dim)
    test_encoder(features, adj, metadata["features_per_node"], hidden_dim)
    test_decoder(hidden_dim, metadata["features_per_node"])

    # Test full model
    test_full_model(features, adj, metadata["features_per_node"])


if __name__ == "__main__":
    main()
