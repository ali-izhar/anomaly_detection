import sys
import os
import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.syn_data_generator import SyntheticDataGenerator, GenerationConfig
from src.graph.graph_generator import GraphGenerator
from src.graph.features import extract_centralities, compute_embeddings
from src.models.layers import GraphConvLayer, TemporalAttention
from src.models.encoder import STEncoder
from src.models.decoder import TemporalDecoder
from src.models.spatiotemporal import SpatioTemporalPredictor, STModelConfig

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
    n_nodes: int = 20,
    n_graphs: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Generate synthetic graph sequence with features."""
    
    # Generate graph sequence
    logger.info(f"Generating synthetic graph sequence: {n_nodes} nodes, {n_graphs} graphs")
    generator = GraphGenerator()
    config = GenerationConfig(
        graph_type="BA",
        n=n_nodes,
        params_before={"m": 2},
        params_after={"m": 4},
        n_graphs_before=n_graphs//2,
        n_graphs_after=n_graphs//2
    )
    
    # Generate BA graphs
    graphs = generator.barabasi_albert(
        n=config.n,
        m1=config.params_before["m"],
        m2=config.params_after["m"],
        set1=config.n_graphs_before,
        set2=config.n_graphs_after
    )
    
    # Extract features
    logger.info("Extracting graph features")
    centralities = extract_centralities(graphs)
    embeddings = compute_embeddings(graphs, method='svd', n_components=2)
    
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
        "total_timesteps": n_graphs
    }
    
    return features_tensor, adj_tensor, metadata

def test_graph_conv_layer(
    features: torch.Tensor,
    adj: torch.Tensor,
    in_dim: int,
    out_dim: int
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

def test_temporal_attention(
    features: torch.Tensor,
    hidden_dim: int
) -> None:
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
    features: torch.Tensor,
    adj: torch.Tensor,
    in_channels: int,
    hidden_dim: int
) -> None:
    """Test STEncoder."""
    logger.info("\nTesting STEncoder")
    encoder = STEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.1
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

def test_decoder(
    hidden_dim: int,
    num_features: int,
    n_nodes: int = 20
) -> None:
    """Test TemporalDecoder."""
    logger.info("\nTesting TemporalDecoder")
    decoder = TemporalDecoder(
        hidden_dim=hidden_dim,
        num_features=num_features,
        num_nodes=n_nodes,
        num_layers=2,
        dropout=0.1
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
    features: torch.Tensor,
    adj: torch.Tensor,
    num_features: int
) -> None:
    """Test complete SpatioTemporalPredictor."""
    logger.info("\nTesting Full Model")
    
    # Get dimensions from features tensor
    n_nodes = features.shape[1]  # [n_timesteps, n_nodes, n_features]
    
    config = STModelConfig(
        hidden_dim=64,
        num_layers=2,
        window_size=10,
        forecast_horizon=5
    )
    
    model = SpatioTemporalPredictor(
        config=config,
        num_nodes=n_nodes,
        num_features=num_features
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

def main():
    # Generate synthetic data
    features, adj, metadata = generate_synthetic_data()
    logger.info(f"Generated features shape: {features.shape}")
    logger.info(f"Adjacency matrix shape: {adj.shape}")
    logger.info(f"Feature metadata: {metadata}")
    
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