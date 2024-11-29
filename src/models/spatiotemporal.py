# src/models/spatiotemporal.py

"""Spatio-temporal GNN model for graph sequence prediction."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .encoder import STEncoder
from .decoder import TemporalDecoder


@dataclass
class STModelConfig:
    """Configuration for spatio-temporal model."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    window_size: int = 10
    forecast_horizon: int = 5
    learning_rate: float = 0.001


class SpatioTemporalPredictor(nn.Module):
    """Complete spatio-temporal prediction model.

    Architecture:
    1. STEncoder: Processes graph sequences using GCN + temporal attention
    2. TemporalDecoder: Generates multi-step predictions

    Input features from features.py:
    - Centrality measures (degree, betweenness, eigenvector, closeness)
    - Graph embeddings (SVD, LSVD)
    """

    def __init__(self, config: STModelConfig, num_nodes: int, num_features: int):
        """
        Args:
            config: Model configuration
            num_nodes: Number of nodes in the graph
            num_features: Number of features per node
        """
        super().__init__()

        self.encoder = STEncoder(
            in_channels=num_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        # Project encoder output to decoder input dimension
        self.projection = nn.Linear(num_nodes * config.hidden_dim, config.hidden_dim)

        self.decoder = TemporalDecoder(
            hidden_dim=config.hidden_dim,
            num_features=num_features,
            num_nodes=num_nodes,
            num_layers=2,
            dropout=config.dropout,
        )

        self.config = config
        self.num_nodes = num_nodes
        self.num_features = num_features

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, num_nodes, num_features]
               Features are extracted using features.py functions
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Predictions [batch_size, forecast_horizon, num_nodes, num_features]
        """
        # Encode spatio-temporal features
        encoded, _ = self.encoder(x, adj)  # [batch_size, seq_len, num_nodes, hidden_dim]

        batch_size, seq_len, num_nodes, hidden_dim = encoded.size()

        # Reshape and project encoder output
        encoder_output = encoded.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, num_nodes * hidden_dim]
        decoder_input = self.projection(encoder_output)  # [batch_size, seq_len, hidden_dim]

        # Generate predictions
        predictions = self.decoder(
            decoder_input, 
            steps=self.config.forecast_horizon
        )  # [batch_size, forecast_horizon, num_nodes, num_features]

        return predictions

    def predict(
        self,
        features: torch.Tensor,
        adj_matrix: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make predictions with optional anomaly detection."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(features, adj_matrix)

        result = {"predictions": predictions.numpy()}

        if threshold is not None:
            anomalies = predictions > threshold
            result["anomalies"] = anomalies.numpy()

        return result
