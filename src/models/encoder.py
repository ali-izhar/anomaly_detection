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
        in_channels: int,  # Total feature dimensions (centrality + spectral)
        hidden_dim: int,   # Hidden dimension (e.g., 128)
        num_layers: int,   # Number of GCN layers (e.g., 3)
        dropout: float,    # Dropout rate (e.g., 0.2)
        num_nodes: int = 100,  # Fixed number of nodes
        window_size: int = 20,  # Sequence window size
    ):
        """
        Args:
            in_channels: Number of input features per node (centralities + embeddings).
            hidden_dim: Hidden layer dimension for GCN and attention.
            num_layers: Number of sequential GCN layers.
            dropout: Dropout rate for regularization.
            num_nodes: Fixed number of nodes.
            window_size: Sequence window size.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        
        # Input feature validation
        self.in_channels = in_channels  # Should be sum of:
        # 4 centrality features (degree, betweenness, closeness, eigenvector)
        # 2-dim SVD embeddings
        # 16-dim LSVD embeddings
        # Total: 4 + 2 + 16 = 22 features per node
        
        # GCN layers
        self.gc_layers = nn.ModuleList([
            GraphConvLayer(
                in_dim=in_channels if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                activation=None,
            )
            for i in range(num_layers)
        ])

        self.temporal_attention = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.use_checkpointing = True

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Node features [B, T, N, F]
            adj: Adjacency matrix [N, N] or [B, N, N]

        Returns:
            attended: Tensor [B, T, N, H] where H is hidden_dim
            hidden_states: List of node embeddings
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        assert num_nodes == self.num_nodes
        assert seq_len == self.window_size
        
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        hidden_states = []
        for t in range(seq_len):
            h = x[:, t]  # [B, N, F]
            
            for gc_layer in self.gc_layers:
                if self.use_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        self._forward_layer,
                        h, adj, gc_layer
                    )
                else:
                    h = self._forward_layer(h, adj, gc_layer)
            hidden_states.append(h)

        # Stack: [B, T, N, H]
        hidden_seq = torch.stack(hidden_states, dim=1)
        
        # Apply attention and return [B, T, N, H]
        attended = self.temporal_attention(hidden_seq)
        attended = self.proj(attended)
        attended = self.layernorm(attended)

        return attended, hidden_states

    def _forward_layer(self, h, adj, layer):
        h_in = h
        h = layer(h, adj)
        h = F.relu(h)
        h = self.dropout(h)
        if h_in.size() == h.size():
            h = h + h_in
        return h
