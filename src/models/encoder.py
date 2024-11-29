# src/models/encoder.py

"""Spatio-temporal encoder for graph sequences."""

import torch
import torch.nn as nn
from typing import List, Tuple
from .layers import GraphConvLayer, TemporalAttention
import torch.nn.functional as F


class STEncoder(nn.Module):
    """Spatio-temporal encoder combining GCN and temporal attention.

    Processes a sequence of graph snapshots by:
    1. Applying GCN to each snapshot to capture spatial dependencies
    2. Using temporal attention to capture dependencies across time

    The features processed include:
    - Node centrality measures (degree, betweenness, eigenvector, closeness)
    - Graph embeddings (SVD, LSVD)
    """

    def __init__(
        self, in_channels: int, hidden_dim: int, num_layers: int, dropout: float
    ):
        """
        Args:
            in_channels: Number of input features per node
                        (centralities + embeddings)
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__()

        # GCN layers
        self.gc_layers = nn.ModuleList(
            [
                GraphConvLayer(
                    in_dim=in_channels if i == 0 else hidden_dim, out_dim=hidden_dim
                )
                for i in range(num_layers)
            ]
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Node features [batch_size, seq_len, num_nodes, num_features]
               Features include centralities and embeddings from features.py
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            - Encoded sequence with temporal attention
            - List of intermediate GCN outputs
        """
        batch_size, seq_len, num_nodes, _ = x.size()

        # Process each timestep with GCN
        hidden_states = []
        for t in range(seq_len):
            h = x[:, t]  # Features at time t

            # Apply graph convolutions
            for gc_layer in self.gc_layers:
                h = F.relu(gc_layer(h, adj))
                h = self.dropout(h)

            hidden_states.append(h)

        # Stack temporal sequence
        hidden_seq = torch.stack(hidden_states, dim=1)

        # Apply temporal attention
        attended = self.temporal_attention(hidden_seq)

        return attended, hidden_states
