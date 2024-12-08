# src/models/encoder.py

"""Spatio-temporal encoder for graph sequences."""


from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvLayer, TemporalAttention


class STEncoder(nn.Module):
    """Spatio-temporal encoder combining GCN and temporal attention.

    Processes a sequence of graph snapshots by:
    1. Applying GCN to each snapshot to capture spatial dependencies.
    2. Using temporal attention to capture dependencies across time steps.

    The input features typically include node-level centralities and graph embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_nodes: int = 100,
        window_size: int = 20,
    ):
        """
        Args:
            in_channels: Number of input features per node (22 total):
                - 4 centrality features (degree, betweenness, closeness, eigenvector)
                - 2-dim SVD embeddings
                - 16-dim LSVD embeddings
            hidden_dim: Hidden layer dimension for GCN and attention
            num_layers: Number of sequential GCN layers
            dropout: Dropout rate for regularization
            num_nodes: Fixed number of nodes (100 for this dataset)
            window_size: Local temporal window size for processing
                        (smaller than max_seq_length=200 for efficiency)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size

        # GCN layers with residual connections
        self.gc_layers = nn.ModuleList(
            [
                GraphConvLayer(
                    in_dim=in_channels if i == 0 else hidden_dim,
                    out_dim=hidden_dim,
                    activation=None,
                )
                for i in range(num_layers)
            ]
        )

        # Layer normalization for better training stability
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.temporal_attention = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.final_layernorm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.use_checkpointing = True

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Node features [batch_size, seq_len, num_nodes, in_channels]
               where in_channels = 22 (4 centrality + 2 SVD + 16 LSVD)
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            attended: Tensor [batch_size, seq_len, num_nodes, hidden_dim]
            hidden_states: List of intermediate node embeddings
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        assert num_nodes == self.num_nodes
        assert (
            seq_len <= self.window_size
        ), f"Input sequence length {seq_len} exceeds window size {self.window_size}"

        # Expand adjacency matrix if needed
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        hidden_states = []
        for t in range(seq_len):
            h = x[:, t]  # [batch_size, num_nodes, in_channels]

            # Process through GCN layers with residual connections
            for gc_layer, layernorm in zip(self.gc_layers, self.layernorms):
                if self.use_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        self._forward_layer, h, adj, gc_layer, layernorm
                    )
                else:
                    h = self._forward_layer(h, adj, gc_layer, layernorm)
            hidden_states.append(h)

        # Stack temporal sequence: [batch_size, seq_len, num_nodes, hidden_dim]
        hidden_seq = torch.stack(hidden_states, dim=1)

        # Apply temporal attention and final transformations
        attended = self.temporal_attention(hidden_seq)
        attended = self.proj(attended)
        attended = self.final_layernorm(attended)
        attended = self.dropout(attended)

        return attended, hidden_states

    def _forward_layer(self, h, adj, layer, layernorm):
        h_in = h
        h = layer(h, adj)
        h = layernorm(h)
        h = F.relu(h)
        h = self.dropout(h)
        if h_in.size() == h.size():
            h = h + h_in  # Residual connection
        return h
