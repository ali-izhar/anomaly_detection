# src/models/spatiotemporal.py

"""Spatio-temporal model combining GNN and LSTM components."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder import STEncoder
from .decoder import TemporalDecoder


@dataclass
class STModelConfig:
    """Configuration for the spatio-temporal model."""

    # Model dimensions
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2

    # GNN parameters
    gnn_type: str = "gcn"
    attention_heads: int = 4

    # LSTM parameters
    lstm_layers: int = 2
    bidirectional: bool = True

    # Dataset-specific constants
    num_nodes: int = 100
    max_seq_length: int = 200  # Updated to match dataset
    window_size: int = 20  # Sliding window size for local temporal patterns
    forecast_horizon: int = 10

    # Feature dimensions
    centrality_dim: int = 4  # degree, betweenness, closeness, eigenvector
    svd_dim: int = 2  # SVD embedding dimension
    lsvd_dim: int = 16  # LSVD embedding dimension

    # Total feature dimension
    num_features: int = centrality_dim + svd_dim + lsvd_dim  # 22 total

    # Change point specific
    num_change_points: int = 2  # Each sequence has exactly 2 change points


class SpatioTemporalPredictor(nn.Module):
    """Combined model for spatio-temporal graph prediction."""

    def __init__(self, config: STModelConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Encoder for processing input sequences
        self.encoder = STEncoder(
            in_channels=config.num_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_nodes=config.num_nodes,
            window_size=config.window_size,
        )

        # Decoder for generating predictions
        self.decoder = TemporalDecoder(
            hidden_dim=config.hidden_dim * (2 if config.bidirectional else 1),
            num_features=config.num_features,
            num_nodes=config.num_nodes,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
        )

        # Optional: Feature-specific projection layers
        self.feature_projections = nn.ModuleDict(
            {
                "centrality": nn.Linear(config.hidden_dim, 4),  # 4 centrality features
                "svd": nn.Linear(config.hidden_dim, 2),  # 2-dim SVD
                "lsvd": nn.Linear(config.hidden_dim, 16),  # 16-dim LSVD
            }
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        adj: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Dictionary of input features:
               - centrality measures: (batch_size, seq_len, num_nodes)
               - svd: (batch_size, seq_len, num_nodes, 2)
               - lsvd: (batch_size, seq_len, num_nodes, 16)
            adj: Adjacency matrix (batch_size, seq_len, num_nodes, num_nodes)
            hidden: Optional hidden state
        """
        batch_size = next(iter(x.values())).size(0)
        seq_len = next(iter(x.values())).size(1)
        device = next(iter(x.values())).device

        # Process centrality features
        centrality_features = []
        for feat in ["degree", "betweenness", "closeness", "eigenvector"]:
            if feat in x:
                centrality_features.append(x[feat].unsqueeze(-1))
        centrality_tensor = torch.cat(
            centrality_features, dim=-1
        )  # (batch, seq_len, nodes, 4)

        # Process SVD and LSVD features - already have correct shapes
        svd_tensor = x["svd"]  # (batch, seq_len, nodes, 2)
        lsvd_tensor = x["lsvd"]  # (batch, seq_len, nodes, 16)

        # Combine all features
        input_features = torch.cat(
            [centrality_tensor, svd_tensor, lsvd_tensor], dim=-1
        )  # (batch, seq_len, nodes, 22)

        # Ensure all tensors require grad
        input_features.requires_grad_(True)

        # Encode sequence
        encoded, _ = self.encoder(input_features, adj)

        # Generate predictions
        decoded = self.decoder(
            encoded, steps=self.config.forecast_horizon, hidden=hidden
        )

        # Split predictions into feature groups
        predictions = {}

        # Split centrality predictions
        centrality_preds = decoded[..., :4]
        predictions.update(
            {
                "degree": centrality_preds[..., 0],
                "betweenness": centrality_preds[..., 1],
                "closeness": centrality_preds[..., 2],
                "eigenvector": centrality_preds[..., 3],
            }
        )

        # Split embedding predictions
        predictions["svd"] = decoded[..., 4:6]  # 2D SVD
        predictions["lsvd"] = decoded[..., 6:22]  # 16D LSVD

        return predictions
