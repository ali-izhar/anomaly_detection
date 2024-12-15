# src/model/small/model_s.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.attention.stgcn import STConv


class SmallSTGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32,
        out_channels: int = 32,
        num_nodes: int = 30,
        window_size: int = 10,
        dropout: float = 0.2,
    ):
        """
        Small STGCN model for spatio-temporal graph data.
        """
        super(SmallSTGCN, self).__init__()

        # Calculate temporal size after convolutions
        # Each STConv reduces sequence length by 2*(kernel_size-1)
        self.temporal_size = window_size - 4  # Account for kernel_size=3 twice
        self.num_nodes = num_nodes

        # STGCN blocks with ChebConv
        self.st_conv1 = STConv(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            K=2,  # Chebyshev filter size
        )

        self.st_conv2 = STConv(
            num_nodes=num_nodes,
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            K=2,
        )

        # From the debug output, we can see the actual flattened size is 1920
        # After st_conv2 shape: torch.Size([32, 2, 30, 32])
        # 2 * 30 * 32 = 1920
        self.flatten_size = 2 * num_nodes * out_channels

        # Final layers with adjusted sizes
        self.dropout = nn.Dropout(dropout)
        hidden_size = 256  # Reduced hidden size
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_nodes * num_nodes)

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

        # Apply STGCN blocks
        x = self.st_conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.st_conv2(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = F.relu(x)

        # Reshape for fully connected layers
        x = x.reshape(batch_size, -1)  # Flatten all dimensions except batch
        
        # Apply final layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Reshape to adjacency matrix
        x = x.view(batch_size, self.num_nodes, self.num_nodes)
        adj_matrix = torch.sigmoid(x)

        return adj_matrix
