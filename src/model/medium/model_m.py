# src/model/medium/model_m.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.attention.astgcn import ASTGCN


class MediumASTGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_nodes: int = 30,
        window_size: int = 10,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        """
        Medium complexity ASTGCN model for spatio-temporal graph data.
        """
        super(MediumASTGCN, self).__init__()

        self.num_nodes = num_nodes

        # ASTGCN configuration
        self.astgcn = ASTGCN(
            nb_block=num_layers,
            in_channels=in_channels,
            K=3,  # Order of Chebyshev polynomials
            nb_chev_filter=hidden_channels,
            nb_time_filter=out_channels,
            time_strides=1,
            num_for_predict=1,  # We only predict one adjacency matrix
            len_input=window_size,
            num_of_vertices=num_nodes,
            normalization="sym",
            bias=True,
        )

        # Final layers to convert ASTGCN output to adjacency matrix
        self.dropout = nn.Dropout(dropout)

        # From astgcn.py, output shape is [batch_size, num_nodes, time_prediction]
        self.fc1 = nn.Linear(1, hidden_channels)  # Changed from num_nodes to 1
        self.fc2 = nn.Linear(hidden_channels, num_nodes)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the model.

        Args:
            x: Node features [batch_size, sequence_length, num_nodes, features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges], optional

        Returns:
            Predicted adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size = x.size(0)

        # ASTGCN expects input shape: [batch_size, num_nodes, features, time_steps]
        x = x.permute(0, 2, 3, 1)

        # Apply ASTGCN
        # From astgcn.py, output shape is [batch_size, num_nodes, time_prediction]
        x = self.astgcn(x, edge_index)  # [batch_size, num_nodes, 1]

        # Transform each node's prediction
        h = self.fc1(x)  # [batch_size, num_nodes, hidden_channels]
        h = self.layer_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Create adjacency matrix
        h = self.fc2(h)  # [batch_size, num_nodes, num_nodes]

        # Make it symmetric using matrix operations
        adj_matrix = 0.5 * (h + h.transpose(-2, -1))  # Fixed transpose dimensions
        adj_matrix = torch.sigmoid(adj_matrix)

        return adj_matrix
